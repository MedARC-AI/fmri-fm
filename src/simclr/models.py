# src/flat_mae/simclr_models.py

import torch
import torch.nn as nn

class ProjectionHead(nn.Module):
    """
    The Projection Head (g(·)) for the SimCLR framework.
    As described in the paper, this is a small MLP that maps the representation (h)
    to the latent space where the contrastive loss is applied.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Args:
            input_dim (int): The feature dimension of the output from the base encoder (h).
            hidden_dim (int): The dimension of the hidden layer in the MLP.
            output_dim (int): The final output dimension of the projection (z).
        """
        super().__init__()
        
        # The MLP consists of a linear layer, a non-linearity (ReLU), and another linear layer.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Passes the representation vector through the MLP.
        Args:
            x (torch.Tensor): The representation vector (h) from the base encoder.
        Returns:
            torch.Tensor: The projected vector (z).
        """
        return self.mlp(x)


class SimCLRModel(nn.Module):
    """
    The complete SimCLR model, combining the base encoder and the projection head.
    """
    def __init__(self, backbone: nn.Module, projection_head: nn.Module):
        """
        Args:
            backbone (nn.Module): The base encoder network (f(·)), e.g., a ViT.
                                  It is expected to have a `forward_embedding` method.
            projection_head (nn.Module): The MLP projection head (g(·)).
        """
        super().__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor):
        """
        Performs the forward pass for both augmented views as shown in Figure 2 of the paper.

        Args:
            view_1 (torch.Tensor): The first batch of augmented images.
            view_2 (torch.Tensor): The second batch of augmented images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the two projected vectors (z1, z2).
        """
        
        # --- Process View 1 ---
        # 1. Get the representation vector (h1) from the backbone encoder.
        #    We use `forward_embedding` to get the feature representation. The output is a tuple,
        #    and we'll typically use the CLS token as the representation.
        _, _, h1_patches = self.backbone.forward_embedding(view_1)
        
        # For a ViT, a common representation is the average of all patch tokens.
        h1 = h1_patches.mean(dim=1)
        
        # 2. Get the projection (z1) by passing h1 through the projection head.
        z1 = self.projection_head(h1)

        # --- Process View 2 ---
        # Repeat the exact same process for the second view.
        _, _, h2_patches = self.backbone.forward_embedding(view_2)
        h2 = h2_patches.mean(dim=1)
        z2 = self.projection_head(h2)

        return z1, z2