# Copyright 2025 The vLLM team.

from functools import partial
from typing import Callable, Optional
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLMultiModalProcessor, Qwen2_5_VLProcessingInfo,
        Qwen2_5_VLDummyInputsBuilder, Qwen2_5_VisionPatchMerger,
)

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention, Qwen2_5_VisionBlock,
    Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionTransformer,
    Qwen2_5_VLForConditionalGeneration
)


MIN_PAD_SIZE = 64
MAX_PAD_SIZE = 128


class CustomQwen2_5_VisionAttention(Qwen2_5_VisionAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            projection_size,
            quant_config,
            prefix,
        )
        self.embed_dim = embed_dim

    def pad_bias(self, bias):
        first_half = bias.reshape(-1, 3, self.hidden_size_per_attention_head)[:, :, :self.linear_num_half]
        second_half = bias.reshape(-1, 3, self.hidden_size_per_attention_head)[:, :, self.linear_num_half:]
        first_half_padded = torch.nn.functional.pad(first_half, (0, self.linear_pad_num_half))
        second_half_padded = torch.nn.functional.pad(second_half, (0, self.linear_pad_num_half))
        bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        bias_final = bias_padded.reshape(-1)
        return bias_final

    def pad_linear(self):
        self.qkv.update = True
        if not hasattr(self.qkv, 'linear_num_half'):
            self.linear_num_half = self.hidden_size_per_attention_head // 2
            self.linear_pad_num_half = (MAX_PAD_SIZE - self.hidden_size_per_attention_head) // 2
        qkv_weight_first_half = self.qkv.weight.data.reshape(-1,
                                                            3,
                                                            self.hidden_size_per_attention_head,
                                                            self.embed_dim)[:, :, :self.linear_num_half, :]
        qkv_weight_second_half = self.qkv.weight.data.reshape(-1,
                                                            3,
                                                            self.hidden_size_per_attention_head,
                                                            self.embed_dim)[:, :, self.linear_num_half:, :]

        qkv_weight_first_half_padded = torch.nn.functional.pad(qkv_weight_first_half, (0, 0, 0, self.linear_pad_num_half))
        qkv_weight_second_half_padded = torch.nn.functional.pad(qkv_weight_second_half, (0, 0, 0, self.linear_pad_num_half))
        qkv_weight_padded = torch.cat([qkv_weight_first_half_padded, qkv_weight_second_half_padded], dim=2)
        qkv_weight_final = qkv_weight_padded.reshape(-1, self.embed_dim)
        qkv_bias = self.pad_bias(self.qkv.bias)
        self.qkv.weight.data = qkv_weight_final
        self.qkv.bias = nn.Parameter(qkv_bias)
        out_weight = self.proj.weight.data
        out_weight = torch.nn.functional.pad(
                                            out_weight.reshape(self.embed_dim, -1, self.linear_num_half),
                                            (0, self.linear_pad_num_half, 0, 0)
                                        ).reshape(self.embed_dim, -1)
        self.hidden_size_per_attention_head = MAX_PAD_SIZE
        self.proj.weight.data = out_weight

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:

        if not hasattr(self.qkv, 'update'):
            self.pad_linear()

        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                   for x in (q, k, v))
        if rotary_pos_emb is not None:
            cos = rotary_pos_emb.cos() # [seqlen, rotary_dim / 2]
            sin = rotary_pos_emb.sin()
            cos = torch.nn.functional.pad(cos, (0, 24))
            sin = torch.nn.functional.pad(sin, (0, 24))

            interleaved = False
            if not interleaved:
                cos_new = torch.cat((cos, cos), dim=-1)
                sin_new = torch.cat((sin, sin), dim=-1)
            else:
                cos_new = rearrange(torch.stack((cos, cos), dim=-1), "... d two -> ...(d two)", two=2)
                sin_new = rearrange(torch.stack((sin, sin), dim=-1), "... d two -> ...(d two)", two=2)
            cos_new = cos_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
            sin_new = sin_new.reshape(1, -1, 1, self.hidden_size_per_attention_head)
            q = torch_npu.npu_rotary_mul(q, cos_new, sin_new)
            k = torch_npu.npu_rotary_mul(k, cos_new, sin_new)

        q, k, v = [
            rearrange(x, "b s h d -> (b s) h d").contiguous()
            for x in (q, k, v)
        ]

        context_layer = torch.torch.empty_like(q)

        # operator requires pta version >= 2.5.1.dev20250226
        torch_npu._npu_flash_attention_unpad(
            query=q,
            key=k,
            value=v,
            seq_len=cu_seqlens,
            scale_value=self.hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer)

        context_layer = rearrange(context_layer,
                                  "(b s) h d -> s b (h d)",
                                  b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


class CustomQwen2_5_VisionBlock(Qwen2_5_VisionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            mlp_hidden_dim,
            act_fn,
            norm_layer,
            quant_config,
            prefix
        )
        self.attn = CustomQwen2_5_VisionAttention(embed_dim=dim,
                                                num_heads=num_heads,
                                                projection_size=dim,
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.attn")


class CustomQwen2_5_VisionPatchEmbed(Qwen2_5_VisionPatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(
            self.proj.weight.data.view(self.hidden_size, -1).transpose(0, 1)
        )
        return x

class CustomQwen2_5_VisionPatchMerger(Qwen2_5_VisionPatchMerger):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        spatial_merge_size: int = 2,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            d_model,
            context_dim,
            norm_layer,
            spatial_merge_size,
            quant_config,
            prefix
        )


class CustomQwen2_5_VisionTransformer(Qwen2_5_VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vision_config,
            norm_eps,
            quant_config,
            prefix
        )
        norm_layer=partial(RMSNorm, eps=norm_eps)
        self.patch_embed = CustomQwen2_5_VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )
        self.blocks = nn.ModuleList([
            CustomQwen2_5_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])

        self.merger = CustomQwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_window_seqlens = torch.diff(cu_window_seqlens).cpu().to(torch.int32)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cpu().to(torch.int32)

        # transformers
        hidden_states = hidden_states.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens_now,
                                rotary_pos_emb=rotary_pos_emb)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states

@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder)
class CustomQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config: Qwen2_5_VLConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.visual = CustomQwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )
