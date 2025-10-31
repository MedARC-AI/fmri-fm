# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# SimSiam: https://github.com/facebookresearch/simsiam
# --------------------------------------------------------

import torch
import torch.nn as nn


from flat_mae.models_mae import MaskedViT

from simclr.loss import (
    SimCLRProjectionHead,
    SimSiamProjectionHead,
    SimSiamPredictionHead,
)

class ContrastiveModel(nn.Module):
    def __init__(self, backbone: MaskedViT, mode: str = "simclr", embed_dim: int = 384, model_kwargs: dict = None):
        super().__init__()
        if mode not in ["simclr", "simsiam"]:
            raise ValueError(f"Invalid contrastive mode: {mode}")

        self.mode = mode
        self.backbone = backbone

        if self.mode == "simclr":
            self.projection_head = SimCLRProjectionHead(in_dim=embed_dim)

        elif self.mode == "simsiam":
            self.projection_head = SimSiamProjectionHead(in_dim=embed_dim)
            self.prediction_head = SimSiamPredictionHead()

    def get_representation(self, x: torch.Tensor, mask_ratio: float):
        cls_embeds, _, _ = self.backbone.forward_embedding(x, mask_ratio=mask_ratio)
        return cls_embeds.squeeze(1)

    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor, mask_ratio: float):
        h1 = self.get_representation(view_1, mask_ratio)
        h2 = self.get_representation(view_2, mask_ratio)

        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        if self.mode == "simclr":
            return z1, z2

        elif self.mode == "simsiam":
            p1 = self.prediction_head(z1)
            p2 = self.prediction_head(z2)

            return p1, z2, p2, z1