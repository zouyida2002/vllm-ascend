#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Adapted from vllm/model_executor/models/qwen2_vl.py
# Copyright 2023 The vLLM team.
#
# This file is a part of the vllm-ascend project.
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

from functools import partial
from typing import Callable, Optional, Type

import torch
import torch.nn as nn
import torch_npu
from einops import rearrange
from transformers.models.qwen2_vl.configuration_qwen2_vl import \
    Qwen2VLVisionConfig
from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import QuickGELU
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VisionAttention, Qwen2VisionBlock, Qwen2VisionPatchEmbed,
    Qwen2VisionTransformer, Qwen2VLDummyInputsBuilder,
    Qwen2VLForConditionalGeneration, Qwen2VLMultiModalProcessor,
    Qwen2VLProcessingInfo, apply_rotary_pos_emb_vision)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY


class CustomQwen2VisionAttention(Qwen2VisionAttention):

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
        self.cu_seqlens = None
        self.embed_dim = embed_dim

    def pad_bias(self, bias):

        first_half = bias.reshape(self.num_attention_heads_per_partition, 3, 80)[:, :, :40]
        second_half = bias.reshape(self.num_attention_heads_per_partition, 3, 80)[:, :, 40:]
        first_half_padded = torch.nn.functional.pad(first_half, (0, 24))
        second_half_padded = torch.nn.functional.pad(second_half, (0, 24))
        bias_padded = torch.cat([first_half_padded, second_half_padded], dim=2)
        bias_final = bias_padded.reshape(self.num_attention_heads_per_partition * 128 * 3)
        return bias_final

    def pad_linear(self):

        self.qkv.update = True
        self.hidden_size_per_attention_head = 128
        qkv_weight_first_half = self.qkv.weight.data.reshape(self.num_attention_heads_per_partition, 3, 80, self.embed_dim)[:, :, :40, :]
        qkv_weight_second_half = self.qkv.weight.data.reshape(self.num_attention_heads_per_partition, 3, 80, self.embed_dim)[:, :, 40:, :]

        qkv_weight_first_half_padded = torch.nn.functional.pad(qkv_weight_first_half, (0, 0, 0, 24))
        qkv_weight_second_half_padded = torch.nn.functional.pad(qkv_weight_second_half, (0, 0, 0, 24))
        qkv_weight_padded = torch.cat([qkv_weight_first_half_padded, qkv_weight_second_half_padded], dim=2)
        qkv_weight_final = qkv_weight_padded.reshape(self.num_attention_heads_per_partition * 128 * 3, self.embed_dim)
        qkv_bias = self.pad_bias(self.qkv.bias)
        self.qkv.weight.data = qkv_weight_final
        self.qkv.bias = nn.Parameter(qkv_bias)
        out_weight = self.proj.weight.data
        out_weight = torch.nn.functional.pad(
                                            out_weight.reshape(self.embed_dim, self.num_attention_heads_per_partition * 2, 40), (0, 24, 0, 0)
                                        ).reshape(self.embed_dim, self.num_attention_heads_per_partition * 128)
        self.proj.weight.data = out_weight

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:

        if not hasattr(self.qkv, 'update'):
            self.pad_linear()
        self.cu_seqlens = cu_seqlens

        # [s, b, c] --> [s, b, 3 * head * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = [
            rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)
        ]

        interleaved = False
        if rotary_pos_emb is not None:
            cos = rotary_pos_emb.cos() # [seqlen, rotary_dim / 2]
            sin = rotary_pos_emb.sin()
            cos = torch.nn.functional.pad(cos, (0, 24))
            sin = torch.nn.functional.pad(sin, (0, 24))

            if not interleaved:
                cos_new = torch.cat((cos, cos), dim=-1)
                sin_new = torch.cat((sin, sin), dim=-1)
            else:
                cos_new = rearrange(torch.stack((cos,cos), dim=-1), "... d two -> ...(d two)", two=2)
                sin_new = rearrange(torch.stack((sin,sin), dim=-1), "... d two -> ...(d two)", two=2)

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
            seq_len=self.cu_seqlens,
            scale_value=self.hidden_size_per_attention_head**-0.5,
            num_heads=self.num_attention_heads_per_partition,
            num_kv_heads=self.num_attention_heads_per_partition,
            out=context_layer)
        context_layer = rearrange(context_layer,
                                  "(b s) h d -> s b (h d)",
                                  b=batch_size).contiguous()

        output, _ = self.proj(context_layer)
        return output


class CustomQwen2VisionBlock(Qwen2VisionBlock):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        act_layer: Type[nn.Module] = QuickGELU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(dim, num_heads, mlp_ratio, act_layer, norm_layer,
                         quant_config, prefix)
        self.attn = CustomQwen2VisionAttention(embed_dim=dim,
                                               num_heads=num_heads,
                                               projection_size=dim,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.attn")


class CustomQwen2VisionPatchEmbed(Qwen2VisionPatchEmbed):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.matmul(
            self.proj.weight.data.view(self.embed_dim, -1).transpose(0, 1))
        return x


class CustomQwen2VisionTransformer(Qwen2VisionTransformer):

    def __init__(
        self,
        vision_config: Qwen2VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix)

        self.patch_embed = CustomQwen2VisionPatchEmbed(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            in_channels=vision_config.in_channels,
            embed_dim=vision_config.embed_dim,
        )

        self.blocks = nn.ModuleList([
            CustomQwen2VisionBlock(dim=self.embed_dim,
                                   num_heads=self.num_heads,
                                   mlp_ratio=vision_config.mlp_ratio,
                                   norm_layer=partial(nn.LayerNorm,
                                                      eps=norm_eps),
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.blocks.{layer_idx}")
            for layer_idx in range(vision_config.depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # compute cu_seqlens and avoid cumsum to fit operator unpadFA
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:,
                                                      0]).cpu().to(torch.int32)

        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        # adapter
        x = self.merger(x)
        return x


@MULTIMODAL_REGISTRY.register_processor(Qwen2VLMultiModalProcessor,
                                        info=Qwen2VLProcessingInfo,
                                        dummy_inputs=Qwen2VLDummyInputsBuilder)
class CustomQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config)
        self.visual = CustomQwen2VisionTransformer(
            self.config.vision_config,
            norm_eps=getattr(self.config, "rms_norm_eps", 1e-6),
            quant_config=self._maybe_ignore_quant_config(
                vllm_config.quant_config),
            prefix=maybe_prefix(prefix, "visual"),
        )
