# src/flat_mae/simclr_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):

    def __init__(self, temperature: float = 0.5):

        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):

        batch_size = z1.shape[0]

        representations = torch.cat([z1, z2], dim=0)

        similarity_matrix = self.similarity_f(representations.unsqueeze(1), representations.unsqueeze(0))

        labels = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ]).to(similarity_matrix.device)


        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(similarity_matrix.device)

        similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)

        labels_adjusted = labels.clone()
        for i in range(batch_size, batch_size * 2):
            if labels[i] > i:
                labels_adjusted[i] -= 1

        logits = similarity_matrix / self.temperature

        loss = self.criterion(logits, labels_adjusted)

        loss = loss / (2 * batch_size)

        return loss