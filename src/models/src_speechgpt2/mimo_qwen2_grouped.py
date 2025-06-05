from typing import List, Optional, Tuple, Union, Callable
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import numpy as np
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import *
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config as LlamaConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as LlamaModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel as LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import (
    AttentionMaskConverter,
    _CONFIG_FOR_DOC,
)
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    GenerationConfig,
    GenerateOutput,
    is_deepspeed_zero3_enabled,
)
import torch.distributed as dist
from dataclasses import dataclass, asdict
import warnings
import inspect
from dataclasses import dataclass, field

from transformers.cache_utils import Cache, DynamicCache, StaticCache
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass
class MIMOModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    speech_vocab_size: int = field(
        default=1025, metadata={"help": "vocab size of speech tokens"}
    )
    n_vq: int = field(default=3, metadata={"help": "number of input rvq tokens"})
    vocab_size: int = field(default=-1, metadata={"help": "vocab_size"})
    group_size: int = field(default=3, metadata={"help": "num speech tokens to group into one"})

    local_dim: int = field(default=1024, metadata={"help": "local transformer hidden size"})
    local_layers: int = field(default=8, metadata={"help": "local transformer num layers"})
    local_attn_heads: int = field(default=16, metadata={"help": "local transformer num heads"})
    local_ffn_dim: int = field(default=4096, metadata={"help": "local transformer feedforward dim"})
    local_attn_dropout: float = field(default=0.0, metadata={"help": "local transformer attention dropout"})

    def to_dict(self):
        return asdict(self)


@dataclass
class MIMOCausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[Tuple[torch.FloatTensor]] = None
    text_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    local_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    local_hidden_states: Optional[Tuple[torch.FloatTensor]] = None # Downcasted hidden states for local transformer generation
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MIMOLlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(
        self,
        config,
        padding_idx,
        sosp_idx,
        eosp_idx,
        args,
        token_loss_weights=None,
        audio_loss_weights=None,
        text_auxiliary_loss_inference_mode=False,
        empty_idx=-1
    ):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = args.vocab_size
        if self.vocab_size < 0:
            self.vocab_size = self.config.vocab_size
        self.padding_idx = padding_idx
        self.speech_vocab_size = config.speech_vocab_size = args.speech_vocab_size
        config.group_size = args.group_size
        self.n_vq = config.n_vq = args.n_vq
        self.sosp_idx = sosp_idx
        self.eosp_idx = eosp_idx
        self.zeroemb_idx = self.speech_vocab_size - 1
        assert self.zeroemb_idx == 1024

        self.empty_idx = empty_idx
        self.text_auxiliary_loss_inference_mode = text_auxiliary_loss_inference_mode
        if self.text_auxiliary_loss_inference_mode:
            assert self.empty_idx > 0

        self.loss_weights = None
        if token_loss_weights:
            self.loss_weights = torch.ones(self.vocab_size)
            for input_id in token_loss_weights:
                self.loss_weights[input_id] = token_loss_weights[input_id]
        self.audio_loss_weights = None
        if audio_loss_weights:
            self.audio_loss_weights = torch.ones(self.speech_vocab_size)
            for input_id in audio_loss_weights:
                self.audio_loss_weights[input_id] = audio_loss_weights[input_id]

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Construct local transformer
        local_transformer_config = copy.deepcopy(config)
        local_transformer_config.hidden_size = args.local_dim
        local_transformer_config.num_hidden_layers = args.local_layers
        local_transformer_config.num_attention_heads = args.local_attn_heads
        local_transformer_config.num_key_value_heads = args.local_attn_heads
        local_transformer_config.intermediate_size = args.local_ffn_dim
        local_transformer_config.attention_dropout = args.local_attn_dropout
        local_transformer_config.vocab_size = args.speech_vocab_size
        self.local_transformer_config = local_transformer_config

        self.local_transformer = LlamaModel(local_transformer_config)
        self.local_transformer.embed_tokens = None # This is not used

        self.local_transformer_lm_heads = nn.ModuleList([nn.Linear(
            local_transformer_config.hidden_size, local_transformer_config.vocab_size, bias=False
        ) for _ in range(self.n_vq)])

        self.speech_embedddings = nn.ModuleList([nn.Embedding(
            self.speech_vocab_size, local_transformer_config.hidden_size, padding_idx=self.zeroemb_idx
        ) for _ in range(self.n_vq)])
        self.speech_group_downcast = nn.Linear(local_transformer_config.hidden_size * config.group_size, config.hidden_size, bias=False)
        self.hidden_states_downcast = nn.Linear(config.hidden_size, local_transformer_config.hidden_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.mimo_tie_weights()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def mimo_tie_weights(self):
        for j in range(0, self.n_vq):
            self.local_transformer_lm_heads[j].weight = self.speech_embedddings[j].weight

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, MIMOCausalLMOutputWithCrossAttentions]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        B = input_ids.shape[0]
        if len(input_ids.shape) <= 2:
            input_ids = input_ids.reshape(B, -1, self.n_vq + 1).transpose(1, 2).contiguous()  #[B,vq, T*group_size]
        
        input_ids = input_ids.int()
        group_size = self.config.group_size
        
        text_input_ids = input_ids[:, 0, ::group_size]
        speech_input_ids = input_ids[:, 1:, :].view(B, self.n_vq, -1, group_size).transpose(1, 2) # [B, T//group_size, n_vq, group_size]
        
        speech_embeddings = 0
        for i in range(self.n_vq):
            speech_embeddings += self.speech_embedddings[i](speech_input_ids[:, :, i, :]) # [B, T//group_size, group_size, hidden_size]
        
        speech_grouped_embeddings = self.speech_group_downcast(speech_embeddings.view(B, -1, self.config.group_size * self.local_transformer_config.hidden_size))  # [B, T//group_size, hidden_size]

        text_embeds = self.model.embed_tokens(text_input_ids)  # Text input
        text_embeds = text_embeds.to(self.device)  
        speech_grouped_embeddings = speech_grouped_embeddings.to(self.device)

        inputs_embeds = text_embeds + speech_grouped_embeddings
        del text_embeds, speech_grouped_embeddings

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0] # [B, T, hidden_size]

        text_logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        shift_hidden_states = self.hidden_states_downcast(hidden_states[:, -1, :].unsqueeze(1)) # [B, 1, hidden_size]
        # We directly pass the hidden_states of the model as the output. Autoregressive generation of the local transformer will be handled in the forward_local method.

        return MIMOCausalLMOutputWithCrossAttentions(
            text_logits=text_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            local_hidden_states=shift_hidden_states,
            attentions=outputs.attentions,
        )
    
    def forward_local(self,
                      local_last_hidden_states: torch.FloatTensor, # [B, 1, hidden_size]
                      input_ids: torch.Tensor, # [B, T_local, n_vq]
                      past_key_values: Optional[Cache] = None,
                      **kwargs):
        # Get shape from past_key_values to determine how many new input_ids need to be sent into forward
        cached_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        # Construct sequence for generation
        input_embs = local_last_hidden_states  #[B, 1, hidden_size]
        if input_ids.numel():
            speech_embeddings = 0
            for i in range(self.n_vq):
                speech_embeddings += self.speech_embedddings[i](input_ids[:, :, i]) # [B, T_local, hidden_size]
            input_embs = torch.cat([local_last_hidden_states, speech_embeddings], dim=1)
        B, T_local, hidden_size = input_embs.shape
        input_embs = input_embs.reshape((-1, T_local, hidden_size)) #  [B, T_local, hidden_size]
        # Keep only the new input_ids
        input_embs = input_embs[:, cached_len:, :]
        output = self.local_transformer(
            inputs_embeds=input_embs,
            past_key_values=past_key_values
        )
        local_last_hidden_states = output.last_hidden_state[:,-1,:] # [B, hidden_size]
        past_key_values = output.past_key_values
        local_logits_vq = [lm_head(local_last_hidden_states) for lm_head in self.local_transformer_lm_heads]
        return local_logits_vq, past_key_values

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[torch.Tensor],
        eos_token_id: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        # No information for attention mask inference -> return default attention mask
        default_attention_mask = torch.ones(
            (inputs.shape[0], inputs.shape[1] // (self.n_vq + 1) // self.config.group_size),  #zd modified
            dtype=torch.long,
            device=inputs.device,
        )
        return default_attention_mask

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        input_ids = input_ids.reshape(input_ids.shape[0], -1, (self.n_vq + 1) * self.config.group_size).permute(     #zd modified
            0, 2, 1
        )  #[B, vq*group_size, T]
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = (
                past_key_values[0][0].shape[2] if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_length,
                input_ids.shape[2],
                dtype=torch.long,
                device=input_ids.device,
            )

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if (
                inputs_embeds is not None or cache_position[-1] >= input_ids.shape[2]
            ):  # Exception 1 or Exception 3
                input_ids = input_ids[:, :, -cache_position.shape[0] :]
            elif (
                input_ids.shape[2] != cache_position.shape[0]
            ):  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, :, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = (
            "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        )
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(
                    memory_format=torch.contiguous_format
                )
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(
                memory_format=torch.contiguous_format
            )

        # 4. Create missing `position_ids` on the fly
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = (
                position_ids  # placed in kwargs for further processing (see below)
            )

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[2] :]
                    model_input = model_input.clone(
                        memory_format=torch.contiguous_format
                    )
                model_inputs[model_input_name] = model_input

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        #zd modified
        if model_inputs[input_ids_key] is not None:
            model_inputs[input_ids_key] = model_inputs[input_ids_key].permute(0,2,1).reshape(input_ids.shape[0], -1, (self.n_vq + 1)).permute(0,2,1)  #[B, vq, T*group_size]
            
        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        """Calculates `cache_position` for the pre-fill stage based on `input_ids` and optionally past length"""
        # `torch.compile`-friendly `torch.arange` from a shape -- the lines below are equivalent to `torch.arange`
        if "inputs_embeds" in model_kwargs:
            cache_position = (
                torch.ones_like(
                    model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64
                ).cumsum(0)
                - 1
            )
        else:
            cache_position = (
                torch.ones(
                    (input_ids.shape[1] // (self.n_vq + 1) // self.config.group_size,),  #zd modified
                    dtype=torch.int64,
                ).cumsum(0)
                - 1
            )

        past_length = 0
        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif (
                hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None
            ):
                past_length = cache.get_seq_length()

            # TODO(joao): this is not torch.compile-friendly, find a work-around. If the cache is not empty,
            # end-to-end compilation will yield bad results because `cache_position` will be incorrect.
            if not is_torchdynamo_compiling():
                cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        force_generated_text_channel_id: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        tokenizer = kwargs.pop(
            "tokenizer", None
        )  # Pull this out first, we only use it for stopping criteria
        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False

        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )

        accepts_attention_mask = "attention_mask" in set(
            inspect.signature(self.forward).parameters.keys()
        )
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(
            generation_config, kwargs_has_attention_mask, device=device
        )

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(
                    inputs_tensor[:, -1] == generation_config._pad_token_tensor
                )
                > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        if (
            not kwargs_has_attention_mask
            and requires_attention_mask
            and accepts_attention_mask
        ):
            model_kwargs["attention_mask"] = (
                self._prepare_attention_mask_for_generation(
                    inputs_tensor,
                    generation_config._pad_token_tensor,
                    generation_config._eos_token_tensor,
                )
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = (
                inputs_tensor
                if model_input_name == "input_ids"
                else model_kwargs.pop("input_ids")
            )

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        has_default_min_length = (
            kwargs.get("min_length") is None
            and generation_config.min_length is not None
        )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `num_logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if "num_logits_to_keep" not in model_kwargs:
            model_kwargs["num_logits_to_keep"] = 1

        self._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        # TODO (joao): remove `user_defined_cache` after v4.47 (remove default conversion to legacy format)
        cache_name = (
            "past_key_values"
            if "mamba" not in self.__class__.__name__.lower()
            else "cache_params"
        )
        user_defined_cache = model_kwargs.get(cache_name)
        max_cache_length = generation_config.max_length
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config,
            model_kwargs,
            assistant_model,
            batch_size,
            max_cache_length,
            device,
        )

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
            **kwargs,
        )

        # 11. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
        result = self.mimo_sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            force_generated_text_channel_id=force_generated_text_channel_id,
            **model_kwargs,
        )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache
            is not False  # Should check for `True` after v4.47
            and not is_torchdynamo_compiling()
            and hasattr(result, "past_key_values")
            and hasattr(result.past_key_values, "to_legacy_cache")
            and result.past_key_values.to_legacy_cache is not None
        ):
            # handle BC (convert by default if he user hasn't passed a cache AND the cache is of the default type)
            should_convert_cache = generation_config.return_legacy_cache
            is_user_defined_cache = user_defined_cache is not None
            is_default_cache_type = type(
                result.past_key_values
            ) == DynamicCache or (  # noqa E721
                isinstance(result.past_key_values, EncoderDecoderCache)
                and type(result.past_key_values.self_attention_cache)
                == DynamicCache  # noqa E721
                and type(result.past_key_values.cross_attention_cache)
                == DynamicCache  # noqa E721
            )
            if not is_user_defined_cache and is_default_cache_type:
                logger.warning_once(
                    "From v4.47 onwards, when a model cache is to be returned, `generate` will return a `Cache` "
                    "instance instead by default (as opposed to the legacy tuple of tuples format). If you want to "
                    "keep returning the legacy format, please set `return_legacy_cache=True`."
                )
                should_convert_cache = True
            if should_convert_cache:
                result.past_key_values = result.past_key_values.to_legacy_cache()
        return result

    def mimo_sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        force_generated_text_channel_id: Optional[int] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        cross_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions")
                if output_attentions
                else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(
            this_peer_finished,
            synced_gpus,
            device=input_ids.device,
            cur_len=cur_len,
            max_length=max_length,
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits_fst = (
                outputs.text_logits[:, -1, :].clone().float()
            )  # [B,vs]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits_fst)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            # if force_generated_text_channel_id and self.get_t:
            #     next_tokens_first = torch.full(
            #         (input_ids.shape[0],),
            #         force_generated_text_channel_id,
            #         dtype=torch.long,
            #         device=input_ids.device,
            #     )
            # else:
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens_first = torch.multinomial(probs, num_samples=1).squeeze(
                    1
                )  # [B]
            else:
                next_tokens_first = torch.argmax(next_token_scores, dim=-1) # [B]

            # Text auxiliary loss mode
            if self.text_auxiliary_loss_inference_mode:
                # Set text output between sosp and eosp to empty
                if (
                    self.get_token_modality(input_ids[0]) == "speech"
                    and next_tokens_first[0] != self.eosp_idx
                ):
                    # print("full empty")
                    next_tokens_first = torch.full_like(next_tokens_first, self.empty_idx)

            if (
                self.get_token_modality(input_ids[0]) == "text"
                or next_tokens_first[0] == self.eosp_idx
            ):
                # Fill audio channel with 1024
                local_sequence = torch.full((input_ids.size(0), self.config.group_size, self.n_vq), self.zeroemb_idx, dtype=input_ids.dtype, device=next_tokens_first.device)  #[B, group_size, vq]
            else:
                # Generate audio tokens by running local_forward group_size times
                local_past_key_values = DynamicCache()
                local_sequence = torch.zeros((input_ids.size(0), 0, self.n_vq), dtype=input_ids.dtype, device=input_ids.device)
                for t in range(self.config.group_size):
                    next_token_logits_audio, local_past_key_values = self.forward_local(
                        local_last_hidden_states=outputs.local_hidden_states,
                        input_ids=local_sequence,
                        past_key_values=local_past_key_values,
                    )
                    next_tokens_residual = []
                    for next_token_logits_res in next_token_logits_audio:
                        # pre-process distribution
                        next_token_scores = logits_processor(input_ids, next_token_logits_res)
                        # sampling
                        if do_sample:
                            next_token_scores[:, self.zeroemb_idx] = float('-inf') # Prevent sampling zeroemb token
                            probs = nn.functional.softmax(next_token_scores, dim=-1)
                            next_tokens_r = torch.multinomial(probs, num_samples=1).squeeze(
                                1
                            )  # [B]
                        else:
                            next_tokens_r = torch.argmax(next_token_scores, dim=-1)
                        next_tokens_residual.append(next_tokens_r)
                    next_tokens_residual = torch.stack(
                        next_tokens_residual, dim=0
                    ).t()  # [B, n_vq]
                    # print("next_tokens_residual", next_tokens_residual.shape)
                    local_sequence = torch.cat([local_sequence, next_tokens_residual.unsqueeze(1)], dim=1) # [B, T, n_vq]

                # finished sentences should have their next token be a padding token, [self.pad_token_id, self.zeroemb_idx, self.zeroemb_idx, self.zeroemb_idx, self.zeroemb_idx, self.zeroemb_idx, self.zeroemb_idx, self.zeroemb_idx, self.zeroemb_idx]
                if has_eos_stopping_criteria:
                    next_tokens_first = (
                        next_tokens_first * unfinished_sequences
                        + pad_token_id * (1 - unfinished_sequences)
                    )
                    next_tokens_residual = (
                        next_tokens_residual * unfinished_sequences
                        + torch.ones(input_ids.shape[0], (self.n_vq + 1) - 1).to(
                            input_ids.device
                        )
                        * self.zeroemb_idx
                        * (1 - unfinished_sequences)
                    )

            # Next tokens picking & padding.
            # If generating text tokens, only save tokens from the first lm_head. If generating speech tokens, save tokens from all n_io lm_heads.
            # To keep shape consistent in one batch, we should pad the generated text token using self.padding_idx.
            
            # Also pad the generated text token to [B, self.config.group_size] using -100
            next_tokens_first = torch.cat(
                [next_tokens_first.unsqueeze(1), torch.full((next_tokens_first.size(0), self.config.group_size - 1), -100, device=next_tokens_first.device, dtype=next_tokens_first.dtype)], dim=1
            ).unsqueeze(2)  # [B, group_size, 1]

            # generate speech tokens
            next_tokens = torch.cat(
                (next_tokens_first, local_sequence), dim=-1
            ).reshape(next_tokens_first.size(0), -1)    #[B, group_size * vq]

            # print("input_ids", input_ids.shape)
            # print("next_tokens", next_tokens.shape)
            # update generated ids, model inputs, and length for next step
            device = input_ids.device  # 获取 input_ids 的设备
            next_tokens = next_tokens.to(device)
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)  #[B, T*group_size*vq]
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(
                input_ids, scores
            )
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


    def get_token_modality(self, seq: torch.Tensor):
        """
        This function determines the type of the token to be generated, classifying it into one of the predefined categories: 'speech' or 'text'
        """
        sosp_poses = torch.where(seq == self.sosp_idx)[0]
        eosp_poses = torch.where(seq == self.eosp_idx)[0]
        if len(sosp_poses) == len(eosp_poses):
            return "text"
        return "speech"
