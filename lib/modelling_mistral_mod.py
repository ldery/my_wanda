# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Mistral model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pdb
from transformers import AutoConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.mistral import MistralConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with
# Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# Copied from
# transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with
# Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralMLP(nn.Module):
    def __init__(self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        layer_id: int
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]
        self.intermediate_size = intermediate_size
        self.layer_id = layer_id
        self.main_mask = None
        self.temp_mask = None
        self.is_using_main = True
        self.intermed_cache = None
        self.skip_computation = False
        self.prune_method = None
        self.ins_ = None
        self.computing_updated_bias = None

    def forward(self, x):
        if self.skip_computation:
            return torch.zeros_like(x)

        intermed_result = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        if self.is_using_main and (self.main_mask is not None):
            intermed_result = intermed_result * self.main_mask
        elif (self.temp_mask is not None):
            intermed_result = intermed_result * self.temp_mask

        last_dim = intermed_result.shape[-1]
        with torch.no_grad():
            if self.prune_method == "magnitude":
                self.intermed_cache = intermed_result.abs().view(-1, last_dim).mean(axis=0,
                                                          keepdims=True).view(1, 1, -1)
            elif self.prune_method == "wanda":
                if self.ins_ is None:
                    self.ins_ = self.down_proj.weight.data.to(torch.float32).abs()

                ins_ = self.ins_ * intermed_result.view(-1, last_dim).to(
                    torch.float32).pow(2).mean(0, keepdim=True).sqrt()
                self.intermed_cache = ins_.mean(axis=0).view(1, 1, -1)
                if self.intermed_cache.isinf().any() or self.intermed_cache.isnan().any():
                    print("We hit a nan or inf. Stopping")
                    self.intermed_cache = torch.zeros_like(self.intermed_cache)

            # elif self.prune_method == "fluct":
            #     ins_ = intermed_result.view(-1, last_dim)
            #     ins_ = (ins_ - ins_.mean(dim=0, keepdim=True)).to(torch.float32).pow_(2).mean(dim=0, keepdim=True)
            #     ins_ = (self.down_proj.weight.data.abs().to(torch.float32).pow_(2).mean(dim=0, keepdim=True)) * ins_
            #     self.intermed_cache = ins_.view(1, 1, -1)
            # elif self.prune_method == "random":
            #     self.intermed_cache = torch.rand((1, 1, self.intermediate_size)).to(x.device)
            # else:
            #     self.intermed_cache = torch.zeros((1, 1, self.intermediate_size)).to(x.device)

            if self.computing_updated_bias is not None:
                self.intermed_cache = intermed_result.view(-1, last_dim).mean(axis=0, keepdims=True) * (self.computing_updated_bias).squeeze(0)
                self.intermed_cache = self.down_proj.weight.matmul(self.intermed_cache.T)

        return self.down_proj(intermed_result)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim=hidden_states.shape
    if n_rep == 1: # to reflect the number of k/v groups within Mistral
        return hidden_states
    hidden_states=hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim)


class MistralAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        self.main_mask = None
        self.temp_mask = None
        self.is_using_main = True
        self.intermed_cache = None
        self.prune_method = None
        self.intermediate_size = self.num_key_value_heads
        self.skip_computation = False
        self.ins_ = None		
        self.computing_updated_bias = None


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.skip_computation:
            attn_output = torch.zeros_like(hidden_states)
            return attn_output, None, None

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        if self.is_using_main and (self.main_mask is not None):
            attn_output = attn_output * repeat_kv(self.main_mask.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)
        elif (self.temp_mask is not None):
            attn_output = attn_output * repeat_kv(self.temp_mask.transpose(1, 2), self.num_key_value_groups).transpose(1, 2)

        if self.prune_method == "magnitude":
            with torch.no_grad():
                self.intermed_cache = attn_output.abs().transpose(2, 3).reshape(-1, self.num_key_value_heads, self.num_key_value_groups).transpose(1, 2).reshape(-1, self.num_key_value_heads).mean(axis=0, keepdims=True).view(1, 1, self.num_key_value_heads, 1)
        elif self.prune_method == "wanda":
            with torch.no_grad():
                if self.ins_ is None:
                    self.ins_ = self.o_proj.weight.data.to(torch.float32).abs()
                ins_ = attn_output.reshape(bsz, q_len, self.hidden_size).reshape(-1, self.hidden_size).to(torch.float32)
                ins_ = self.ins_ * ins_.pow(2).mean(0, keepdim=True).sqrt()
                self.intermed_cache = ins_.mean(axis=0).view(1, 1, self.num_key_value_heads, -1).mean(axis=-1, keepdim=True)

                if self.intermed_cache.isinf().any() or self.intermed_cache.isnan().any():
                    print("We hit a nan or inf. Resettinig ")
                    self.intermed_cache = torch.zeros_like(self.intermed_cache)
        # elif self.prune_method == "fluct":
        #     ins_ = attn_output.reshape(bsz, q_len, self.hidden_size).reshape(-1, self.hidden_size).to(torch.float32)
        #     ins_ = (ins_ - ins_.mean(dim=0, keepdim=True)).pow_(2).mean(dim=0, keepdim=True)
        #     ins_ = (self.o_proj.weight.data.abs().to(torch.float32).pow_(2).mean(dim=0, keepdim=True)) * ins_
        #     self.intermed_cache = ins_.view(1, 1, self.num_heads, -1).mean(axis=-1, keepdim=True)
        # elif self.prune_method == "random":
        #     self.intermed_cache = torch.rand((1, 1, self.num_heads, 1)).to(attn_output.device)
        # else:
        #     self.intermed_cache = torch.zeros((1, 1, self.num_heads, 1)).to(attn_output.device)

        if self.computing_updated_bias is not None:
            shape = attn_output.shape[-2:]
            
            self.intermed_cache = attn_output.reshape(-1, shape[0], shape[1]).mean(axis=0, keepdims=True).unsqueeze(0)
            repeated_mask = self.computing_updated_bias.repeat(1, 1, self.num_key_value_groups, 1)
            self.intermed_cache = (self.intermed_cache * repeated_mask).view(1, -1)
            self.intermed_cache = self.intermed_cache.matmul(self.o_proj.weight.T)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_id:int = -1):
        super().__init__()
        self.hidden_size=config.hidden_size

        self.self_attn=MistralAttention(config=config)

        self.mlp=MistralMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            layer_id=layer_id,
        )
        self.input_layernorm=MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm=MistralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_value: Optional[Tuple[torch.Tensor]]=None,
        output_attentions: Optional[bool]=False,
        use_cache: Optional[bool]=False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual=hidden_states

        hidden_states=self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value=self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states=residual + hidden_states

        # Fully Connected
        residual=hidden_states
        hidden_states=self.post_attention_layernorm(hidden_states)
        hidden_states=self.mlp(hidden_states)
        hidden_states=residual + hidden_states

        outputs=(hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


MISTRAL_START_DOCSTRING=r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MistralConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@ add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralPreTrainedModel(PreTrainedModel):
    config_class=MistralConfig
    base_model_prefix="model"
    supports_gradient_checkpointing=True
    _no_split_modules=["MistralDecoderLayer"]
    _keys_to_ignore_on_load_unexpected=[r"decoder\.version"]

    def _init_weights(self, module):
        std=self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MistralModel):
            module.gradient_checkpointing=value


MISTRAL_INPUTS_DOCSTRING=r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@ add_start_docstrings(
    "The bare Mistral Model outputting raw hidden-states without any specific head on top.",
    MISTRAL_START_DOCSTRING,
)
class MistralModel(MistralPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.padding_idx=config.pad_token_id
        self.vocab_size=config.vocab_size

        self.embed_tokens=nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx)
        self.layers=nn.ModuleList([MistralDecoderLayer(
            config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self._attn_implementation=config._attn_implementation
        self.norm=MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Do some setup to compute size of important units here
        self.params_per_pruned_hidden=config.hidden_size * \
            3  # [gate_proj, up_proj, down_projs]

        head_dim=config.hidden_size // config.num_attention_heads
        # (num parameters per-head) * [3 (kqv) + 1(o)]
        self.params_per_pruned_head=(head_dim * config.hidden_size) * 2.5

        self.gradient_checkpointing=False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens=value

        # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask=None
        if input_shape[-1] > 1:
            combined_attention_mask=_make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask=_expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask=(
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + \
                    combined_attention_mask
            )

        return combined_attention_mask

    @ add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[List[torch.FloatTensor]]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions=output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states=(
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache=use_cache if use_cache is not None else self.config.use_cache

        return_dict=return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length=input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _=inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past=seq_length
        past_key_values_length=0

        if past_key_values is not None:
            past_key_values_length=past_key_values[0][0].shape[2]
            seq_length_with_past=seq_length_with_past + past_key_values_length

        if position_ids is None:
            device=input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids=torch.arange(
                past_key_values_length,
                seq_length +
                past_key_values_length,
                dtype=torch.long,
                device=device)
            position_ids=position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids=position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds=self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask=torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask=self._prepare_decoder_attention_mask(
            attention_mask, (batch_size,
                             seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states=inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache=False

        # decoder layers
        all_hidden_states=() if output_hidden_states else None
        all_self_attns=() if output_attentions else None
        next_decoder_cache=() if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value=past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs=torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs=decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states=layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states=self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache=next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys=["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model=MistralModel(config)
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens=value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head=new_embeddings

    def set_decoder(self, decoder):
        self.model=decoder

    def get_decoder(self):
        return self.model

    @ add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    @ replace_return_docstrings(output_type=CausalLMOutputWithPast,
                               config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[List[torch.FloatTensor]]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        labels: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions=output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states=(
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden,
        # dec_attn)
        outputs=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states=outputs[0]
        logits=self.lm_head(hidden_states)
        logits=logits.float()

        loss=None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits=logits[..., :-1, :].contiguous()
            shift_labels=labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct=CrossEntropyLoss()
            shift_logits=shift_logits.view(-1, self.config.vocab_size)
            shift_labels=shift_labels.view(-1)
            # Enable model parallelism
            shift_labels=shift_labels.to(shift_logits.device)
            loss=loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output=(logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length=past_key_values.get_seq_length()
                past_length=past_key_values.seen_tokens
                max_cache_length=past_key_values.get_max_length()
            else:
                cache_length=past_length=past_key_values[0][0].shape[2]
                max_cache_length=None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids=input_ids[:, -
                                      (attention_mask.shape[1] - past_length):]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids=input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume
            # input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to
            # crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask=attention_mask[:, -max_cache_length:]

        position_ids=kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids=attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids=position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st
        # generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs={"inputs_embeds": inputs_embeds}
        else:
            model_inputs={"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @ staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past=()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(
                        0,
                        beam_idx.to(
                            past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@ add_start_docstrings(
    """
    The Mistral Model transformer with a sequence classification head on top (linear layer).

    [`MistralForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MISTRAL_START_DOCSTRING,
)
# Copied from
# transformers.models.llama.modeling_llama.LlamaForSequenceClassification
# with Llama->Mistral, LLAMA->MISTRAL
class MistralForSequenceClassification(MistralPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels=config.num_labels
        self.model=MistralModel(config)
        self.score=nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens=value

    @ add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor=None,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[List[torch.FloatTensor]]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        labels: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs=self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states=transformer_outputs[0]
        logits=self.score(hidden_states)

        if input_ids is not None:
            batch_size=input_ids.shape[0]
        else:
            batch_size=inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths=-1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing
                # for ONNX compatibility
                sequence_lengths=torch.eq(
                    input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths=sequence_lengths % input_ids.shape[-1]
                sequence_lengths=sequence_lengths.to(logits.device)
            else:
                sequence_lengths=-1

        pooled_logits=logits[torch.arange(
            batch_size, device=logits.device), sequence_lengths]

        loss=None
        if labels is not None:
            labels=labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type="regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type="single_label_classification"
                else:
                    self.config.problem_type="multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct=MSELoss()
                if self.num_labels == 1:
                    loss=loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss=loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct=CrossEntropyLoss()
                loss=loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct=BCEWithLogitsLoss()
                loss=loss_fct(pooled_logits, labels)
        if not return_dict:
            output=(pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


if __name__ == "__main__":
    import numpy as np
    config=AutoConfig.from_pretrained('mistralai/Mistral-7B-v0.1')
    model=MistralForCausalLM(config)
    print('We just got here first!')
    k, v=list(model.named_parameters())[0]
    print(k, v.mean().item(), v.max().item())
