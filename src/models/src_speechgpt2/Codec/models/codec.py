#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import logging
import yaml
from torch import nn
from typing import List
from contextlib import contextmanager

# AudioDec Module
from .modules.conv_layers import CausalConv1d, CausalConvTranspose1d
from .modules.mel_transform import MelSpectrogram
from .modules.encoder import Encoder as Convnext_Encoder
from .modules.decoder import Decoder as HifiGAN_Decoder
from .modules.projector import Projector
from .modules.quantizer import Quantizer


# Moshi Module
from .moshi_modules.resample import ConvDownsample1d as Moshi_ConvDownsample1d
from .moshi_modules.transformer import ProjectedTransformer
from .moshi_modules.streaming import StreamingModule


class Generator(torch.nn.Module):
    """AudioDec generator."""

    def __init__(
        self,
        mel_kwargs,
        encoder_convnext_kwargs,
        encoder_transformer_kwargs,
        encoder_extra_down_sample_kwargs,
        quantizer_type,
        quantizer_kwargs,
        post_projecter_kwargs,
        vq_transformer_kwargs,
        post_projecter_of_vq_transformer_kwargs,
        decoder_extra_up_sample_kwargs,
        decoder_transformer_kwargs,
        decoder_convnext_kwargs,
        decoder_hifigan_kwargs,
        mode,
        use_weight_norm,
    ):
        super().__init__()
        
        convnext = Convnext_Encoder
        decoder_hifigan = HifiGAN_Decoder
        
        self.mode = mode
        if self.mode == "causal":
            down_sample_conv = Moshi_ConvDownsample1d
            up_sample_conv = CausalConvTranspose1d
        else:
            assert False, f"mode of {self.mode} not support"
        
        self.input_channels = 1
        self.sampling_rate = mel_kwargs['fs']
        self.codebook_size = quantizer_kwargs['codebook_size']
        self.code_dim = quantizer_kwargs['code_dim']
        
        self.mel_downsample_factor = mel_kwargs['hop_size']
        self.mel_transform = MelSpectrogram(
            self.mode,
            **mel_kwargs
        )
        
        self.total_downsample_factor = 960
        
        self.encoder_convnext = convnext(
            mode=self.mode,
            **encoder_convnext_kwargs
        )

        self.encoder_transformer = ProjectedTransformer(
            **encoder_transformer_kwargs,
        )

        self.encoder_extra_down_sample_1 = down_sample_conv(
            **encoder_extra_down_sample_kwargs
        )

        self.encoder_extra_down_sample_2 = down_sample_conv(
            **encoder_extra_down_sample_kwargs
        )

        if quantizer_type == "residual_vq":
            self.quantizer = Quantizer(
                model=quantizer_type,
                **quantizer_kwargs
            )
        else:
            assert False, f"quantizer_type of {quantizer_type} not support"

        self.post_projector = Projector(
            mode=self.mode,
            **post_projecter_kwargs,
        ) 

        self.vq_transformer = ProjectedTransformer(
            **vq_transformer_kwargs
        )
        
        self.post_projector_of_vq_transformer = Projector(
            mode=self.mode,
            **post_projecter_of_vq_transformer_kwargs,
        )


        self.decoder_extra_up_sample_1 = up_sample_conv(
            **decoder_extra_up_sample_kwargs
        )
        self.decoder_extra_up_sample_2 = up_sample_conv(
            **decoder_extra_up_sample_kwargs
        )

        self.decoder_transformer = ProjectedTransformer(
            **decoder_transformer_kwargs
        )
    
        self.decoder_convnext = convnext(
            mode=self.mode,
            **decoder_convnext_kwargs
        )

        self.decoder_hifigan = decoder_hifigan(
            mode=self.mode,
            out_channels=1,
            **decoder_hifigan_kwargs,
        )

        # apply weight norm & reset parameters
        if use_weight_norm:
            self.apply_weight_norm()
            self.reset_parameters()        

        self._init_for_streaming_inference()


    def forward(self, x: torch.Tensor):
        (batch, channel, length) = x.size()
        if channel != self.input_channels: 
            x = x.reshape(-1, self.input_channels, length) # (B, C, T) -> (B', C', T)
        m = self.mel_transform(x)[:, :, :length // self.mel_downsample_factor]
        encoder_convnext_output = self.encoder_convnext(m)
        (encoder_transformer_output, ) = self.encoder_transformer(encoder_convnext_output)
        encoder_extra_down_sample_1_output = self.encoder_extra_down_sample_1(encoder_transformer_output)
        encoder_extra_down_sample_2_output = self.encoder_extra_down_sample_2(encoder_extra_down_sample_1_output)
        z = encoder_extra_down_sample_2_output
        zq, embed_nums, vqloss, perplexity, all_layers_of_quantized  = self.quantizer(z)
        post_projector_output = self.post_projector(zq)
        (vq_transformer_output, ) = self.vq_transformer(post_projector_output)
        post_projector_of_vq_transformer_output = self.post_projector_of_vq_transformer(vq_transformer_output)
        decoder_extra_up_sample_1_output = self.decoder_extra_up_sample_1(post_projector_of_vq_transformer_output)
        decoder_extra_up_sample_2_output = self.decoder_extra_up_sample_2(decoder_extra_up_sample_1_output)
        
        (decoder_transformer_output, ) = self.decoder_transformer(decoder_extra_up_sample_2_output)
        decoder_convnext_output = self.decoder_convnext(decoder_transformer_output)
        y = self.decoder_hifigan(decoder_convnext_output)
        
        return {
            "y": y,
            "zq": zq,
            "z": z,
            "vq_regression_output": post_projector_of_vq_transformer_output,
            "embed_nums": embed_nums,
            "vqloss": vqloss,
            "perplexity": perplexity
        }
    
    def reset_parameters(self): 
        def _reset_parameters(m):
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")
        self.apply(_reset_parameters)


    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)


    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or \
            isinstance(
                m, torch.nn.ConvTranspose1d
            ) or \
            isinstance(m, torch.nn.Linear): 
                torch.nn.utils.parametrizations.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")
        self.apply(_apply_weight_norm)




    def _apply_named(self, fn):

        def _handle_module(prefix: str, module: nn.Module):

            """
            def apply(self: T, fn: Callable[['Module'], None]) -> T:
                for module in self.children():
                    module.apply(fn)
                fn(self)
                return self   
            """
            for name, child in module.named_children():
                if prefix:
                    new_prefix = prefix + "." + name
                else:
                    new_prefix = name
                _handle_module(new_prefix, child)
            fn(prefix, module)
        

        _handle_module("", self)
    
    def _init_for_streaming_inference(self):

        encode_module_names = [
            'mel_transform',
            'encoder_convnext',
            'encoder_transformer',
            'encoder_extra_down_sample_1',
            'encoder_extra_down_sample_2',
            'quantizer',
        ]
        decode_module_names = [
            'post_projector',
            'vq_transformer',
            'post_projector_of_vq_transformer',
            'decoder_extra_up_sample_1',
            'decoder_extra_up_sample_2',
            'decoder_transformer',
            'decoder_convnext',
            'decoder_hifigan'
        ]

        self.modules_dict = {
            "encoder": [],
            "decoder": []
        }
        
        def append_encode_module(module_name, module):
            for name in encode_module_names:
                if name in module_name:
                    self.modules_dict['encoder'].append(module)
        self._apply_named(append_encode_module)
        
        def append_decode_module(module_name, module):
            for name in decode_module_names:
                if name in module_name:
                    self.modules_dict['decoder'].append(module)
        self._apply_named(append_decode_module)
        

        self.modules_for_streaming_inference = {
            "streaming_tokenize": {
                "module_of_audiodec": [],
                "module_of_moshi": [],
            },
            "streaming_detokenize": {
                "module_of_audiodec": [],
                "module_of_moshi": [],
            }
        }
        
        for module in self.modules_dict["encoder"]:
            # Moshi Module
            if isinstance(module, StreamingModule):
                self.modules_for_streaming_inference['streaming_tokenize']['module_of_moshi'].append(module)
            # AudioDec Module
            elif isinstance(module, CausalConv1d) or isinstance(module, CausalConvTranspose1d) or isinstance(module, MelSpectrogram):
                self.modules_for_streaming_inference['streaming_tokenize']['module_of_audiodec'].append(module)
            
        for module in self.modules_dict["decoder"]:
            # Moshi Module
            if isinstance(module, StreamingModule):
                self.modules_for_streaming_inference['streaming_detokenize']['module_of_moshi'].append(module)
            # AudioDec Module
            elif isinstance(module, CausalConv1d) or isinstance(module, CausalConvTranspose1d) or isinstance(module, MelSpectrogram):
                self.modules_for_streaming_inference['streaming_detokenize']['module_of_audiodec'].append(module)
    
    def _set_streaming_state(self, state, streaming_tokenize_or_streaming_detokenize):

        if state is None: # 如果状态为空，则要设置一遍状态为默认状态
            for module_type, module_list in self.modules_for_streaming_inference[streaming_tokenize_or_streaming_detokenize].items():
                if module_type == 'module_of_moshi':
                    for module in module_list:
                        module._streaming_state = module._init_streaming_state(batch_size=1) 
                elif module_type == 'module_of_audiodec':
                    for module in module_list:
                        module.reset_buffer(batch_size=1)
                else:
                    assert False, f"expect module_type in ['module_of_moshi', 'module_of_audiodec'], found module_type = {module_type}"
        else: # 如果状态非空，则要将输入状态放到对应的 module 里
            idx_of_state = 0
            for module_type, module_list in self.modules_for_streaming_inference[streaming_tokenize_or_streaming_detokenize].items():
                if module_type == 'module_of_moshi':
                    for module in module_list:
                        module._streaming_state = state[idx_of_state] 
                        idx_of_state = idx_of_state + 1
                elif module_type == 'module_of_audiodec':
                    for module in module_list:
                        module.set_buffer(state[idx_of_state])
                        idx_of_state = idx_of_state + 1
                else:
                    assert False, f"expect module_type in ['module_of_moshi', 'module_of_audiodec'], found module_type = {module_type}"
            
            assert idx_of_state == len(state), f"expect idx_of_state == len(state), found idx_of_state = {idx_of_state}, len(state) = {len(state)}"
            
    
    def _get_streaming_state(self, streaming_tokenize_or_streaming_detokenize):

        state = []
        for module_type, module_list in self.modules_for_streaming_inference[streaming_tokenize_or_streaming_detokenize].items():
            if module_type == 'module_of_moshi':
                for module in module_list:
                    state.append(module._streaming_state)
            elif module_type == 'module_of_audiodec':
                for module in module_list:
                    state.append(module.pad_buffer)
            else:
                assert False, f"expect module_type in ['module_of_moshi', 'module_of_audiodec'], found module_type = {module_type}"
        return state
    
    def streaming_inference_tokenize_with_state(self, x: torch.Tensor, input_state_of_streaming_inference_tokenize):

        self._set_streaming_state(input_state_of_streaming_inference_tokenize, "streaming_tokenize")
        tokens = self.streaming_inference_tokenize(x)
        updated_state_of_streaming_inference_tokenize = self._get_streaming_state("streaming_tokenize")
        return {
            "tokens": tokens,
            "updated_state_of_streaming_inference_tokenize": updated_state_of_streaming_inference_tokenize
        }
        
    def streaming_inference_detokenize_with_state(self, tokens, input_state_of_streaming_inference_detokenize):

        self._set_streaming_state(input_state_of_streaming_inference_detokenize, "streaming_detokenize")
        y = self.streaming_inference_detokenize(tokens)
        updated_state_of_streaming_inference_detokenize = self._get_streaming_state("streaming_detokenize")
        return {
            "y": y,
            "updated_state_of_streaming_inference_detokenize": updated_state_of_streaming_inference_detokenize
        }
    
    def _start_streaming(self, batch_size):

        # * AudioDec Module
        def _reset_buffer(m):
            if isinstance(m, CausalConv1d) or isinstance(m, CausalConvTranspose1d) or isinstance(m, MelSpectrogram):
                m.reset_buffer(batch_size)
        self.apply(_reset_buffer)
        
        # * Moshi Module
        def _start_streaming(module):
            if isinstance(module, StreamingModule):
                module._streaming_state = module._init_streaming_state(batch_size) 
        self.apply(_start_streaming)
    
    def _stop_streaming(self, batch_size):

        # * AudioDec Module
        def _reset_buffer(m):
            if isinstance(m, CausalConv1d) or isinstance(m, CausalConvTranspose1d) or isinstance(m, MelSpectrogram):
                m.reset_buffer(batch_size) 
        self.apply(_reset_buffer)
        
        # * Moshi Module
        def _stop_streaming(module):
            if isinstance(module, StreamingModule):
                module._streaming_state = None

        self.apply(_stop_streaming)
    
        
    @contextmanager
    def streaming(self, batch_size):

        self._start_streaming(batch_size)
        try:
            yield
        finally:
            self._stop_streaming(batch_size)
    
    def streaming_inference(self, x: torch.Tensor) -> torch.Tensor:

        assert self.mode == "causal"
        (batch, channel, length) = x.size()
        if channel != self.input_channels: 
            x = x.reshape(-1, self.input_channels, length) # (B, C, T) -> (B', C', T)
        m = self.mel_transform.inference(x)[:, :, :length // self.mel_downsample_factor]
        encoder_convnext_output = self.encoder_convnext.inference(m)
        (encoder_transformer_output, ) = self.encoder_transformer(encoder_convnext_output)
        encoder_extra_down_sample_1_output = self.encoder_extra_down_sample_1(encoder_transformer_output)
        encoder_extra_down_sample_2_output = self.encoder_extra_down_sample_2(encoder_extra_down_sample_1_output)
        z = encoder_extra_down_sample_2_output
        zq, embed_nums, vqloss, perplexity, all_layers_of_quantized  = self.quantizer(z)
        post_projector_output = self.post_projector(zq)
        (vq_transformer_output, ) = self.vq_transformer(post_projector_output)
        post_projector_of_vq_transformer_output = self.post_projector_of_vq_transformer(vq_transformer_output)
        decoder_extra_up_sample_1_output = self.decoder_extra_up_sample_1.inference(post_projector_of_vq_transformer_output)
        decoder_extra_up_sample_2_output = self.decoder_extra_up_sample_2.inference(decoder_extra_up_sample_1_output)
        (decoder_transformer_output, ) = self.decoder_transformer(decoder_extra_up_sample_2_output)
        decoder_convnext_output = self.decoder_convnext.inference(decoder_transformer_output)
        y = self.decoder_hifigan.inference(decoder_convnext_output)
        return y

    def streaming_inference_tokenize(self, x: torch.Tensor) -> torch.Tensor:

        assert self.mode == "causal"
        (batch, channel, length) = x.size()
        if channel != self.input_channels: 
            x = x.reshape(-1, self.input_channels, length) # (B, C, T) -> (B', C', T)
        m = self.mel_transform.inference(x)[:, :, :length // self.mel_downsample_factor]
        encoder_convnext_output = self.encoder_convnext.inference(m)
        (encoder_transformer_output, ) = self.encoder_transformer(encoder_convnext_output)
        encoder_extra_down_sample_1_output = self.encoder_extra_down_sample_1(encoder_transformer_output)
        encoder_extra_down_sample_2_output = self.encoder_extra_down_sample_2(encoder_extra_down_sample_1_output)
        z = encoder_extra_down_sample_2_output
        tokens = self.quantizer.encode(z)
        return tokens
        
    def streaming_inference_detokenize(self, tokens: torch.Tensor) -> torch.Tensor:

        assert self.mode == "causal"
        zq = self.quantizer.decode(tokens)
        post_projector_output = self.post_projector(zq)
        (vq_transformer_output, ) = self.vq_transformer(post_projector_output)
        post_projector_of_vq_transformer_output = self.post_projector_of_vq_transformer(vq_transformer_output)
        decoder_extra_up_sample_1_output = self.decoder_extra_up_sample_1.inference(post_projector_of_vq_transformer_output)
        decoder_extra_up_sample_2_output = self.decoder_extra_up_sample_2.inference(decoder_extra_up_sample_1_output)
        (decoder_transformer_output, ) = self.decoder_transformer(decoder_extra_up_sample_2_output)
        decoder_convnext_output = self.decoder_convnext.inference(decoder_transformer_output)
        y = self.decoder_hifigan.inference(decoder_convnext_output)
        return y

    def inference_tokenize(self, x: torch.Tensor) -> torch.Tensor:

        (batch, channel, length) = x.size()
        if channel != self.input_channels: 
            x = x.reshape(-1, self.input_channels, length) # (B, C, T) -> (B', C', T)
        m = self.mel_transform(x)[:, :, :length // self.mel_downsample_factor]
        encoder_convnext_output = self.encoder_convnext(m)
        (encoder_transformer_output, ) = self.encoder_transformer(encoder_convnext_output)
        encoder_extra_down_sample_1_output = self.encoder_extra_down_sample_1(encoder_transformer_output)
        encoder_extra_down_sample_2_output = self.encoder_extra_down_sample_2(encoder_extra_down_sample_1_output)
        z = encoder_extra_down_sample_2_output
        tokens = self.quantizer.encode(z)
        return tokens
    
    def inference_detokenize(self, tokens: torch.Tensor) -> torch.Tensor: # (num_layers, bs, len)

        zq = self.quantizer.decode(tokens)
        post_projector_output = self.post_projector(zq)
        (vq_transformer_output, ) = self.vq_transformer(post_projector_output)
        post_projector_of_vq_transformer_output = self.post_projector_of_vq_transformer(vq_transformer_output)
        decoder_extra_up_sample_1_output = self.decoder_extra_up_sample_1(post_projector_of_vq_transformer_output)
        decoder_extra_up_sample_2_output = self.decoder_extra_up_sample_2(decoder_extra_up_sample_1_output)
        (decoder_transformer_output, ) = self.decoder_transformer(decoder_extra_up_sample_2_output)
        decoder_convnext_output = self.decoder_convnext(decoder_transformer_output)
        y = self.decoder_hifigan(decoder_convnext_output)
        return y


    @classmethod
    def load_from_checkpoint(cls, config_path: str, checkpoint_path: str):
        # Load Config
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        # Define Model
        codec = cls(**cfg['generator_params'])
        
        # Load ckpt
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        codec.load_state_dict(ckpt, strict=True)
        
        return codec   