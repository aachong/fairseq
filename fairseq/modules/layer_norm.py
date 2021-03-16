# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.drc_fused_layernorm import HeadFusedLayerNorm as _HeadFusedLayerNorm
from fairseq.modules.drc_fused_layernorm import ResidualFusedLayerNorm as _ResidualFusedLayerNorm

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


class ResidualFusedLayerNorm(_ResidualFusedLayerNorm):
    @torch.jit.unused
    def forward(self, x):
        if not x.is_cuda:
            return super().forward(x)
        else:
            with torch.cuda.device(x.device):
                return super().forward(x)


class HeadFusedLayerNorm(_HeadFusedLayerNorm):
    @torch.jit.unused
    def forward(self, x):
        if not x.is_cuda:
            return super().forward(x)
        else:
            with torch.cuda.device(x.device):
                return super().forward(x)


def LayerNorm(normalized_shape, eps=1e-5,
              elementwise_affine=True,
              export=False, 
              need_drc_head=False, 
              need_drc_residual=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        if need_drc_head:
            return HeadFusedLayerNorm(normalized_shape, eps, elementwise_affine)
        if need_drc_residual:
            return ResidualFusedLayerNorm(normalized_shape, eps, elementwise_affine)
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
