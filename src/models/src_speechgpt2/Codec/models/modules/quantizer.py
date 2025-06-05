#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import logging

from .vq_module import ResidualVQ

class Quantizer(torch.nn.Module):
    def __init__(
            self,
            train_codebook,
            code_dim,
            codebook_num,
            codebook_size,
            kmeans_init,
            kmeans_iters,
            decay,
            threshold_ema_dead_code,
            model,
        ):
        self.quantizer_type = model
        super().__init__()
        # speech
        if model == 'residual_vq':
            self.codebook = ResidualVQ(
                train_codebook=train_codebook, 
                dim=code_dim, 
                num_quantizers=codebook_num, 
                codebook_size=codebook_size,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                decay=decay,
                threshold_ema_dead_code=threshold_ema_dead_code
            )
        else:
            raise NotImplementedError(f"Model ({model}) is not supported!")

    def patch_accelerator(self, accelerator):
        logging.info(f"[in models/melvqgan/modules/quantizer.py/ Quantizer] patch accelerator !")
        self.codebook.patch_accelerator(accelerator)

    def initial(self):
        self.codebook.initial()    
    
    def forward(self, z):
        zq, embed_nums, vqloss, perplexity, all_layers_output = self.codebook(z.transpose(2, 1))
        all_layers_output = [output.transpose(2, 1) for output in all_layers_output]
        zq = zq.transpose(2, 1)        
        return zq, embed_nums, vqloss, perplexity, all_layers_output
    
    def inference(self, z):  
        zq, indices = self.codebook.forward_index(z.transpose(2, 1))
        zq = zq.transpose(2, 1)
        return zq, indices
    
    def encode(self, z): # 给 model 
        indices = self.codebook.encode(z.transpose(2, 1))
        return indices # (num_layers, bs, len)
    
    def decode(self, indices):  # 给 model     (num_layers, bs, len)
        zq = self.codebook.decode(indices)
        zq = zq.transpose(1, 2)
        return zq # (bs, length, dim)
