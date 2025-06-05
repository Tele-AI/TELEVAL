#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vector quantizer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import typing as tp
from einops import rearrange, repeat

def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int): 
    num_samples, device = samples.shape[0], samples.device
 
    if num_samples >= num: 
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device) 

    return samples[indices]


def kmeans(accelerator, samples_one_device, num_clusters: int, num_iters: int = 10): # * kmeans 初始化多卡问题得到解决
    if accelerator is not None:
        # Gather samples from all processes
        samples = accelerator.gather(samples_one_device)
        # Flatten the gathered samples: (num_processes * Batch_size, Dim)
        samples = samples.view(-1, samples_one_device.shape[-1])
    else:
        samples = samples_one_device

    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        logging.info(f"codebook kmeans init iter: {_}")
        # diffs = rearrange(samples, "n d -> n () d") - rearrange( # 有问题！！！！   这里强行 broadcast 空间为 n * m * d, 会 OOM
        #     means, "c d -> () c d"
        # )
        dists = -(samples.pow(2).sum(1, keepdim=True) - 2 * samples @ means.t() + means.t().pow(2).sum(0, keepdim=True))
        # dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class VectorQuantize(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        train_codebook, # ! 是否要训 codebook !!!!!!, 第三阶段不能训 ！！！！！！！！
        dim: int,
        codebook_size: int,
        kmeans_init: int = True,
        kmeans_iters: int = 50,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code=None, 
        commitment = 1,
    ):
        super().__init__()
        if kmeans_init:
            assert kmeans_iters > 0
        assert threshold_ema_dead_code != None
        self.train_codebook = train_codebook
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.commitment = commitment

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.accelerator = None
        
    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size = kmeans(self.accelerator, data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        #broadcast_tensors(self.buffers())

    def patch_accelerator(self, accelerator):
        self.accelerator = accelerator
        logging.info(f"[in layers/vq_module.py VectorQuantize] patch accelerator success")

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples_one_device):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        # if not torch.any(expired_codes):
        #     return
        
        logging.info(f"expire codebook center number = {expired_codes.sum()}")

        batch_samples_one_device = rearrange(batch_samples_one_device, "... d -> (...) d")
        
        batch_samples = self.accelerator.gather(batch_samples_one_device)
        
        bs = batch_samples.shape[0]
        logging.info(f"sum cluster_size = {torch.sum(self.cluster_size)}")
        logging.info(f"sum embed = {torch.sum(self.embed)}")
        logging.info(f"sum embed_avg = {torch.sum(self.embed_avg)}")
        
        # if bs / self.codebook_size < self.threshold_ema_dead_code:
        #     assert False, \
        #         f"batch size, codebook size, threshold_ema_dead_code must satisfy: batch size / codebook size > threshold_ema_dead_code \n" + \
        #         f"but found {bs} / {self.codebook_size} < {self.threshold_ema_dead_code}" \
            
        self.replace_(batch_samples, mask=expired_codes)
        #broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -( 
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind): 
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)
        return embed_ind, quantize

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, input):
        shape, dtype = input.shape, input.dtype
        x = self.preprocess(input)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)  # 可能会导致显存问题
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)
        quantize = input + (quantize - input).detach()
        commit_loss = F.mse_loss(quantize.detach(), input) * self.commitment
        avg_probs = torch.mean(embed_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        embed_num = embed_onehot.sum(0)

        if self.training and self.train_codebook and torch.is_grad_enabled():
            logging.info("Try To Update Codebook, Please Make Sure That Codebook Is Not Fixed In Current Stage!")

            # 计算 embed_onehot.sum(0) 和 embed_sum
            embed_sum = x.t() @ embed_onehot  # shape: (dim, codebook_size)

            # 如果使用了 Accelerate 且进程数大于1，则进行同步
            if self.accelerator is not None:
                # 同步 embed_onehot_sum
                embed_num = self.accelerator.reduce(embed_num, reduction="sum")
                # 同步 embed_sum
                embed_sum = self.accelerator.reduce(embed_sum, reduction="sum")

            logging.info("reduce embed_num and embed_sum success")

            # 更新 self.cluster_size 和 self.embed_avg
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_num, self.decay)
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)  # embed_sum.t() 的 shape: (codebook_size, dim)

            # 计算 cluster_size 并进行 Laplace 平滑
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon) * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_num, commit_loss, perplexity

class ResidualVQ(nn.Module):
    """ Residual VQ following algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x):
        quantized_out = 0.
        residual = x
        all_losses = []
        all_perplexities = []
        all_embed_nums = []
        all_layers_output = []
        for layer in self.layers:
            quantized, embed_num, loss, perplexity = layer(residual)
            # Issue: https://github.com/lucidrains/vector-quantize-pytorch/issues/33
            # We found considering only the 1st layer VQ's graident results in better performance
            #residual = residual - quantized.detach() # considering all layers' graidents
            residual = residual - quantized # considering only the first layer's graident 
            quantized_out = quantized_out + quantized
            all_embed_nums.append(embed_num)
            all_losses.append(loss)
            all_perplexities.append(perplexity)
            all_layers_output.append(quantized)
        all_losses, all_perplexities, all_embed_nums = map(torch.stack, (all_losses, all_perplexities, all_embed_nums))
        return quantized_out, all_embed_nums, all_losses, all_perplexities, all_layers_output

    def patch_accelerator(self, accelerator):
        for i, layer in enumerate(self.layers):
            logging.info(f"[in layers.vq_module.py ResidualVQ] rvq layer {i} patch accelerator !")
            layer.patch_accelerator(accelerator)

    def forward_index(self, x, flatten_idx=False):
        quantized_out = 0.
        residual = x
        all_indices = []
        for i, layer in enumerate(self.layers):
            quantized, indices = layer.forward_index(residual)
            #residual = residual - quantized.detach()
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            if flatten_idx:
                indices += (self.codebook_size * i)
            all_indices.append(indices)
        all_indices= torch.stack(all_indices)
        return quantized_out, all_indices.squeeze(1)
    
    def encode(self, x): # 给 quantizer
        residual = x
        all_indices = []
        for i, layer in enumerate(self.layers):
            indices, quantized = layer.encode(residual)
            #residual = residual - quantized.detach()
            residual = residual - quantized
            all_indices.append(indices)
        all_indices = torch.stack(all_indices)
        return all_indices # (num_layers, bs, len)
    
    def decode(self, indices): # 给 quantizer
        quantized_out = 0 
        for i, layer in enumerate(self.layers):
            quantized = layer.decode(indices[i])  
            quantized_out = quantized_out + quantized
        return quantized_out # (bs, length, dim)