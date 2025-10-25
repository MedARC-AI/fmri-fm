# This source code is licensed under the Apache License, Version 2.0
#
# References:
# capi: https://github.com/facebookresearch/capi/blob/main/eval_classification.py

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# backbone classification wrappers adapted from capi with minor changes


class ClassificationWrapper(nn.Module):
    """
    Wrap a backbone embedding model together with a grid of classifier heads.

    backbone: backbone model implementing forward_embedding
    classifiers: map of (feature_source, (lr_scale, weight_decay)) -> classifier
    """

    def __init__(
        self,
        backbone: nn.Module,
        classifiers: dict[tuple[str, tuple[float, float]], nn.Module],
    ):
        super().__init__()
        self.representations = {key[0] for key in classifiers}
        self.backbone = backbone

        # can't use ModuleDict bc of restrictions of keys (must be strings, no dots).
        self.classifier_keys = list(classifiers)
        self.classifiers = nn.ModuleList(list(classifiers.values()))

    def forward(self, *args, **kwargs) -> Tensor:
        cls_token, object_tokens, patch_tokens = self.backbone.forward_embedding(*args, **kwargs)
        backbone_out = pool_representations(
            cls_token, object_tokens, patch_tokens, self.representations
        )

        all_logit = []
        for ii, (feature_source, _) in enumerate(self.classifier_keys):
            clf = self.classifiers[ii]
            all_logit.append(clf(backbone_out[feature_source]))

        # [B, num_classes, num_classifiers]
        all_logit = torch.stack(all_logit, dim=-1)
        return all_logit


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cls_token):
        return self.linear(cls_token)


class AttnPoolClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or in_dim
        assert embed_dim % 64 == 0
        self.query_token = nn.Parameter(torch.empty(embed_dim))
        self.embed_dim = embed_dim
        self.num_heads = embed_dim // 64
        self.kv = nn.Linear(in_dim, embed_dim * 2)
        self.linear = nn.Linear(embed_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat_tokens):
        B, N, _ = feat_tokens.shape
        D = self.embed_dim

        q = self.query_token.expand(B, 1, -1)
        q = q.reshape(B, 1, self.num_heads, D // self.num_heads)  # [B, 1, head, D_head]
        q = q.permute(0, 2, 1, 3)  # [B, head, 1, D_head]

        kv = self.kv(feat_tokens).reshape(
            B, N, 2, self.num_heads, D // self.num_heads
        )  # [B, N, 2, head, D_head]
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, head, N, D_head]
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]

        x = F.scaled_dot_product_attention(q, k, v)  # [B, head, 1, D_head]
        x = x.reshape(B, D)  # [B, D]
        return self.linear(x)


def pool_representations(
    cls_token: Tensor | None,
    object_tokens: Tensor | None,
    patch_tokens: Tensor,
    representations: list[str],
):
    B, N, D = patch_tokens.shape

    if cls_token is not None:
        # nb, for connectome baseline the "cls_token" is a different shape. hack.
        assert cls_token.shape == (B, 1, cls_token.shape[-1])
        cls_token = cls_token.squeeze(1)

    if object_tokens is not None:
        R = object_tokens.shape[1]
        assert object_tokens.shape == (B, R, D)

    # Global features for the linear classifiers
    out: dict[str, Tensor] = {}
    if "cls" in representations:
        out["cls"] = cls_token  # [B, D]
    if "avg_patch" in representations:
        out["avg_patch"] = patch_tokens.mean(1)  # [B, D]
    if "cls_avg_patch" in representations:
        out["cls_avg_patch"] = torch.cat([cls_token, patch_tokens.mean(1)], dim=-1)  # [B, 2 * D]
    if "avg_objects" in representations:
        out["avg_objects"] = object_tokens.mean(1)  # [B, D]
    if "concat_objects" in representations:
        out["concat_objects"] = object_tokens.flatten(1, 2)  # [B, R * D]
    # Object features (registers) for the attention pooling classifiers
    if "objects" in representations:
        out["reg"] = object_tokens
    # Patch features for the attention pooling classifiers
    if "patch" in representations:
        out["patch"] = patch_tokens  # [B, h * w, D]
    return out


class CosineLoss(nn.Module):
    """
    Cosine similarity loss.

    Removes the target argument in `nn.CosineEmbeddingLoss` and adds a dimension to
    normalize over.
    """

    def __init__(self, dim: int = -1, reduction: Literal["none", "sum", "mean"] = "mean"):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = F.normalize(input, dim=self.dim)
        target = F.normalize(target, dim=self.dim)
        loss = 1 - (input * target).sum(dim=self.dim)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
