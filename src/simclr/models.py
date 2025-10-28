import torch
import torch.nn as nn
from src.flat_mae.models_mae import MaskedViT # We reuse the encoder from the MAE implementation
from simclr.loss import (
    SimCLRProjectionHead,
    SimSiamProjectionHead,
    SimSiamPredictionHead,
    nt_xent_loss,
    simsiam_loss,
)

class ContrastiveModel(nn.Module):
    """
    A unified model for contrastive learning, supporting both SimCLR and SimSiam.
    """
    def __init__(self, backbone: MaskedViT, mode: str = "simclr", embed_dim: int = 384):
        """
        Args:
            backbone (MaskedViT): The pre-trained or randomly initialized backbone encoder.
            mode (str): The contrastive learning mode. Can be "simclr" or "simsiam".
            embed_dim (int): The output dimension of the backbone encoder.
        """
        super().__init__()
        if mode not in ["simclr", "simsiam"]:
            raise ValueError(f"Invalid contrastive mode: {mode}")
        
        self.mode = mode
        self.backbone = backbone

        if self.mode == "simclr":
            # For SimCLR, we only need a projection head.
            self.projection_head = SimCLRProjectionHead(in_dim=embed_dim)
        
        elif self.mode == "simsiam":
            # For SimSiam, we need both a projection head and a prediction head.
            self.projection_head = SimSiamProjectionHead(in_dim=embed_dim)
            self.prediction_head = SimSiamPredictionHead()

    def get_representation(self, x: torch.Tensor, mask_ratio: float):
        """
        A helper function to pass an input through the backbone and get the CLS token.
        """
        # The MAE backbone returns (cls_token, reg_tokens, patch_tokens, mask, ids_keep)
        # We only need the cls_token for contrastive learning.
        cls_embeds, _, _, _, _ = self.backbone(x, mask_ratio=mask_ratio)
        # The cls_embeds has a shape of [Batch, 1, Dim], so we squeeze it.
        return cls_embeds.squeeze(1)

    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor, mask_ratio: float):
        """
        The main forward pass. It takes two augmented views and computes the final loss.

        Args:
            view_1 (torch.Tensor): The first batch of augmented images.
            view_2 (torch.Tensor): The second batch of augmented images.
            mask_ratio (float): The ratio of patches to mask in the encoder.

        Returns:
            torch.Tensor: The final calculated loss for the batch.
        """
        
        # Get the representations (h1, h2) from the backbone for each view
        h1 = self.get_representation(view_1, mask_ratio)
        h2 = self.get_representation(view_2, mask_ratio)

        if self.mode == "simclr":
            # --- SimCLR Forward Pass ---
            # 1. Get the projections (z1, z2)
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            
            # 2. Calculate the loss
            loss = nt_xent_loss(z1, z2)
            return loss

        elif self.mode == "simsiam":
            # --- SimSiam Forward Pass ---
            # 1. Get the projections (z1, z2)
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)

            # 2. Get the predictions (p1, p2)
            p1 = self.prediction_head(z1)
            p2 = self.prediction_head(z2)

            # 3. Calculate the loss
            loss = simsiam_loss(p1, z2, p2, z1)
            return loss

