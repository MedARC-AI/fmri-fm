from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.layers import to_2tuple

from util import video_vit


class ClassificationWrapper(nn.Module):
    """
    Wrap a backbone embedding model together with a grid of classifier heads.

    backbone: backbone model implementing forward_embedding
    classifiers: map of (feature_source, (lr_scale, weight_decay)) -> classifier
    """
    def __init__(
        self,
        backbone: nn.Module,
        classifiers: dict[tuple[str, tuple[int, int]], nn.Module],
    ):
        super().__init__()
        self.representations = {key[0] for key in classifiers}
        self.backbone = backbone

        # can't use ModuleDict bc of restrictions of keys (must be strings, no dots).
        self.classifier_keys = list(classifiers)
        self.classifiers = nn.ModuleList(list(classifiers.values()))

    def forward(self, *args, **kwargs) -> Tensor:
        cls_token, object_tokens, patch_tokens = self.backbone.forward_embedding(
            *args, **kwargs
        )
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

        kv = self.kv(feat_tokens).reshape(B, N, 2, self.num_heads, D // self.num_heads)  # [B, N, 2, head, D_head]
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


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        num_frames=16,
        t_patch_size=4,
        patch_embed=video_vit.PatchEmbed,
        sep_pos_embed=False,
        trunc_init=False,
        mask_patch_embed=False,
        **kwargs,
    ):
        super().__init__()
        self.sep_pos_embed = sep_pos_embed
        self.trunc_init = trunc_init
        self.mask_patch_embed = mask_patch_embed

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
    def forward(
        self,
        imgs: torch.Tensor,
        img_mask: torch.Tensor | None = None,
    ):
        # imgs: [N, C, T, H, W]
        # x: [N, L, C]
        if img_mask is not None:
            img_mask = img_mask.expand_as(imgs)
        imgs = img_mask * imgs

        x = self.patch_embed(imgs, mask=img_mask if self.mask_patch_embed else None)
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_temporal[:, :, None] + self.pos_embed_spatial[:, None, :]
            pos_embed = pos_embed.flatten(1, 2)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        return x
    
    def forward_embedding(
        self,
        imgs: torch.Tensor,
        img_mask: torch.Tensor | None = None,
    ):
        x = self.forward(imgs, img_mask)  # [N, L, C]
        return None, None, x


class Connectome(nn.Module):
    parc_weight: Tensor

    def __init__(
        self,
        parcellation_path: str | Path,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        parc_weight = load_parcellation(parcellation_path)
        self.register_buffer("parc_weight", parc_weight)

    def forward_embedding(
        self,
        imgs: torch.Tensor,
        img_mask: torch.Tensor | None = None,
    ):
        N, C, T, H, W = imgs.shape
        assert C == 1
        latent = imgs.reshape(N, T, H*W)  # [N, T, D]

        # roi averaging
        latent = latent @ self.parc_weight.t()  # [N, T, R]

        # normalize to mean zero, unit norm
        x = latent
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (torch.norm(x, dim=1, keepdim=True) + self.eps)

        # R x R pearson connectome
        conn = x.transpose(1, 2) @ x  # [N, R, R]

        # flatten upper triangle
        R = conn.shape[1]
        row, col = torch.unbind(torch.triu_indices(R, R, offset=1, device=x.device))
        conn = conn[:, row, col]  # [N, R*(R-1)/2]

        cls_token = conn[:, None, :]

        return cls_token, None, latent


def load_parcellation(parcellation_path: str | Path) -> Tensor:
    # parcellation is shape (H, W), with values in [0, n_rois]
    # 0 is background
    parc = np.load(parcellation_path)
    parc = parc.flatten()

    # make one hot encoding, (n_rois, n_vertices)
    n_rois = parc.max()
    parc_one_hot = np.arange(1, n_rois + 1)[:, None] == parc

    # normalize to sum to one for roi averaging
    parc_weight = parc_one_hot / np.sum(parc_one_hot, axis=1, keepdims=True)

    # to tensor
    parc_weight = torch.from_numpy(parc_weight).float()
    return parc_weight


class PCA(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        num_frames: int = 16,
        embed_dim: int = 768,
        **kwargs,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        H, W = self.img_size
        self.spatial_linear = nn.Linear(H*W, embed_dim)
        self.temporal_linear = nn.Linear(num_frames, embed_dim, bias=False)

    def forward_embedding(
        self,
        imgs: torch.Tensor,
        img_mask: torch.Tensor | None = None,
    ):
        N, C, T, H, W = imgs.shape
        assert C == 1
        latent = imgs.reshape(N, T, H*W)  # [N, T, D]

        # spatial projection
        latent = self.spatial_linear(latent)  # [N, T, d]
        # temporal projection
        cls_token = torch.einsum("ntd,dt->nd", latent, self.temporal_linear.weight)
        # unsqueeze
        cls_token = cls_token[:, None, :]  # [N, 1, d]
        return cls_token, None, latent


class ImageFlatten(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_embedding(
        self,
        imgs: torch.Tensor,
        img_mask: torch.Tensor | None = None,
    ):
        N, C, T, H, W = imgs.shape
        assert C == 1
        latent = imgs.reshape(N, T, H*W)  # [N, T, D]
        cls_token = latent.mean(dim=1, keepdim=True)  # [N, D]
        return cls_token, None, latent


def patch_embed_small(**kwargs):
    return PatchEmbed(embed_dim=384, **kwargs)


def patch_embed_base(**kwargs):
    return PatchEmbed(embed_dim=768, **kwargs)


def connectome(**kwargs):
    return Connectome(**kwargs)


def pca_small(**kwargs):
    return PCA(embed_dim=384, **kwargs)


def pca_base(**kwargs):
    return PCA(embed_dim=768, **kwargs)


def image_flatten(**kwargs):
    return ImageFlatten(**kwargs)
