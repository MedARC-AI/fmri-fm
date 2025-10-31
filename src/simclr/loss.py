# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# SimSiam: https://github.com/facebookresearch/simsiam
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)

class SimSiamProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        return self.head(x)

class SimSiamPredictionHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 2048):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5, distributed: bool = False):

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    all_z = torch.cat([z1, z2], dim=0)

    if distributed:
        all_z_dist = torch.cat(torch.distributed.nn.all_gather(all_z), dim=0)
        rank = int(os.getenv("LOCAL_RANK", 0))
    else:
        all_z_dist = all_z
        rank = 0

    logits = torch.matmul(all_z, all_z_dist.T) / temperature

    batch_size = z1.shape[0]
    labels_v1 = torch.arange(batch_size, device=z1.device) + batch_size
    labels_v2 = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels_v1, labels_v2], dim=0)

    labels = labels + (rank * 2 * batch_size)

    return F.cross_entropy(logits, labels)

def simsiam_loss(p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor):

    z1 = z1.detach()
    z2 = z2.detach()

    loss1 = -F.cosine_similarity(p1, z2, dim=-1).mean()
    loss2 = -F.cosine_similarity(p2, z1, dim=-1).mean()

    return (loss1 + loss2) / 2
