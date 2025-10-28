import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SimCLRProjectionHead(nn.Module):
    """ The g(·) projection head for SimCLR. """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)

class SimSiamProjectionHead(nn.Module):
    """ The g(·) projection head for SimSiam. """
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
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
    """ The h(·) prediction head for SimSiam. """
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.head(x)

def nt_xent_loss(z1, z2, temperature=0.5, distributed=False):
    """ The NT-Xent loss for SimCLR. """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Concatenate all representations
    all_z = torch.cat([z1, z2], dim=0)

    if distributed:
        # Gather representations from all GPUs in a distributed setting
        all_z_dist = torch.cat(torch.distributed.nn.all_gather(all_z), dim=0)
        world_size = int(os.getenv("WORLD_SIZE", 1))
        rank = int(os.getenv("LOCAL_RANK", 0))
    else:
        all_z_dist = all_z
        world_size = 1
        rank = 0

    # Calculate pairwise similarity
    logits = torch.matmul(all_z, all_z_dist.T) / temperature

    # Create labels
    batch_size = z1.shape[0]
    labels = torch.arange(batch_size, device=z1.device)
    labels = labels + (rank * batch_size)
    
    # The positive pair for z1[i] is z2[i], which is at index (i + batch_size) in all_z
    # And the positive pair for z2[i] is z1[i], which is at index i in all_z
    labels = torch.cat([labels + batch_size, labels], dim=0)

    # We need to mask out the similarity of an embedding with itself
    mask = ~torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
    logits = logits[mask].view(2 * batch_size, -1)
    
    # Adjust labels because of the mask
    labels_adjusted = labels.clone()
    for i in range(2 * batch_size):
        if labels[i] > i:
            labels_adjusted[i] -= 1
            
    loss = F.cross_entropy(logits, labels_adjusted, reduction="sum")
    return loss / (2 * batch_size)


def simsiam_loss(p1, z2, p2, z1):
    """ The loss for SimSiam. """
    # Stop-gradient: we don't want gradients to flow from z to the encoder
    z1 = z1.detach()
    z2 = z2.detach()

    loss1 = -F.cosine_similarity(p1, z2, dim=-1).mean()
    loss2 = -F.cosine_similarity(p2, z1, dim=-1).mean()

    return (loss1 + loss2) / 2

