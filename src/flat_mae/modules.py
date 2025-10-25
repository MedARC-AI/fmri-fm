# This source code is licensed under the Apache License, Version 2.0
#
# References:
# capi: https://github.com/facebookresearch/capi/blob/main/model.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/vision_transformer.py
# vjepa2: https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/pos_embs.py

import math
from functools import partial
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from jaxtyping import Float, Int
from timm.layers import DropPath, to_2tuple, to_3tuple

Layer = Type[nn.Module]


# Transformer modules adapted from capi (but removed the efficient residual)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        context_dim = context_dim or dim

        # using separate q, k, v weights so that xavier init uses the correct dim.
        # although perhaps technically it should be initialized wrt the head dim..
        # but this is what original mae does.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def extra_repr(self):
        return f"num_heads={self.num_heads}"

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        if context is None:
            context = x
        B, N, D = x.shape
        _, M, _ = context.shape
        h = self.num_heads

        q = self.q(x).reshape(B, N, h, D // h).transpose(1, 2)
        k = self.k(context).reshape(B, M, h, D // h).transpose(1, 2)
        v = self.v(context).reshape(B, M, h, D // h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int | float = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# timm default eps=1e-6
LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        mlp_ratio: int | float = 4,
        drop_path: float = 0.0,
        norm_layer: Layer = LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            context_dim=context_dim,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=proj_bias,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        # should the context also be normalized? capi doesn't, so I guess not
        x = x + self.drop_path1(self.attn(self.norm1(x), context=context))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# Patching and position embedding modules


class Patchify2D(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans

        H, W = self.img_size
        p_h, p_w = self.patch_size
        self.grid_size = (H // p_h, W // p_w)
        self.num_patches = math.prod(self.grid_size)
        self.patch_dim = in_chans * math.prod(self.patch_size)

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B N P"]:
        x = patchify2d(x, self.patch_size)
        return x

    def unpatchify(self, x: Float[Tensor, "B N P"]) -> Float[Tensor, "B C H W"]:
        x = unpatchify2d(x, patch_size=self.patch_size, img_size=self.img_size)
        return x

    def extra_repr(self):
        return f"{self.img_size}, {self.patch_size}, in_chans={self.in_chans}"


class Patchify3D(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int, int],
        patch_size: int | tuple[int, int, int],
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        self.img_size = to_3tuple(img_size)
        self.patch_size = to_3tuple(patch_size)
        self.in_chans = in_chans

        T, H, W = self.img_size
        p_t, p_h, p_w = self.patch_size
        self.grid_size = (T // p_t, H // p_h, W // p_w)
        self.num_patches = math.prod(self.grid_size)
        self.patch_dim = in_chans * math.prod(self.patch_size)

    def forward(self, x: Float[Tensor, "B C T H W"]) -> Float[Tensor, "B N P"]:
        x = patchify3d(x, self.patch_size)
        return x

    def unpatchify(self, x: Float[Tensor, "B N P"]) -> Float[Tensor, "B C T H W"]:
        x = unpatchify3d(x, patch_size=self.patch_size, img_size=self.img_size)
        return x

    def extra_repr(self):
        return f"{self.img_size}, {self.patch_size}, in_chans={self.in_chans}"


class StridedPatchify3D(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int, int],
        patch_size: int | tuple[int, int, int],
        in_chans: int = 3,
        t_stride: int = 2,
    ) -> None:
        super().__init__()
        T, H, W = to_3tuple(img_size)
        p_t, p_h, p_w = to_3tuple(patch_size)
        assert (T % t_stride) == (p_t % t_stride) == 0, "invalid t_stride"

        self.img_size = (T, H, W)
        self.patch_size = (p_t // t_stride, p_h, p_w)
        self.in_chans = in_chans
        self.t_stride = t_stride

        self.grid_size = (T // p_t, H // p_h, W // p_w)
        self.num_patches = math.prod(self.grid_size)
        self.patch_dim = in_chans * math.prod(self.patch_size)

    def forward(self, x: Float[Tensor, "B C T H W"]) -> Float[Tensor, "B N P"]:
        x = x[:, :, :: self.t_stride]
        x = patchify3d(x, self.patch_size)
        return x

    def unpatchify(self, x: Float[Tensor, "B N P"]) -> Float[Tensor, "B C T H W"]:
        T, H, W = self.img_size
        x = unpatchify3d(x, patch_size=self.patch_size, img_size=(T // self.t_stride, H, W))
        x = torch.repeat_interleave(x, self.t_stride, dim=2)
        return x

    def extra_repr(self):
        return (
            f"{self.img_size}, {self.patch_size}, in_chans={self.in_chans}, "
            f"t_stride={self.t_stride}"
        )


def patchify2d(x: Tensor, patch_size: tuple[int, int]) -> Tensor:
    p_h, p_w = to_2tuple(patch_size)
    # channels first dimension so that we can load weights that use conv patch embed
    x = rearrange(x, "b c (h p) (w q) -> b (h w) (c p q)", p=p_h, q=p_w)
    return x


def unpatchify2d(
    x: Tensor,
    patch_size: tuple[int, int],
    img_size: tuple[int, int],
) -> Tensor:
    B, N, P = x.shape
    p_h, p_w = to_2tuple(patch_size)
    H, W = to_2tuple(img_size)
    x = rearrange(
        x,
        "b (h w) (c p q) -> b c (h p) (w q)",
        h=H // p_h,
        w=W // p_w,
        p=p_h,
        q=p_w,
    )
    return x


def patchify3d(x: Tensor, patch_size: tuple[int, int, int]) -> Tensor:
    p_t, p_h, p_w = to_3tuple(patch_size)
    B, C, T, H, W = x.shape
    x = rearrange(x, "b c (t u) (h p) (w q) -> b (t h w) (c u p q)", u=p_t, p=p_h, q=p_w)
    return x


def unpatchify3d(
    x: Tensor,
    patch_size: tuple[int, int, int],
    img_size: tuple[int, int, int],
) -> Tensor:
    B, N, P = x.shape
    p_t, p_h, p_w = to_3tuple(patch_size)
    T, H, W = to_3tuple(img_size)
    x = rearrange(
        x,
        "b (t h w) (c u p q) -> b c (t u) (h p) (w q)",
        t=T // p_t,
        h=H // p_h,
        w=W // p_w,
        u=p_t,
        p=p_h,
        q=p_w,
    )
    return x


class AbsolutePosEmbed(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, ...]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)

        self.weight = nn.Parameter(torch.empty(self.num_patches, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        pos_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        x = apply_pos_embed(x, self.weight, pos_ids=pos_ids)
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}"


class SeparablePosEmbed(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, ...]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)

        N_t, *grid_size_spatial = grid_size
        N_s = math.prod(grid_size_spatial)
        self.weight_spatial = nn.Parameter(torch.empty(1, N_s, embed_dim))
        self.weight_temporal = nn.Parameter(torch.empty(N_t, 1, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight_spatial, std=0.02)
        nn.init.trunc_normal_(self.weight_temporal, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        pos_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        B, N, D = x.shape
        weight = (self.weight_temporal + self.weight_spatial).flatten(0, 1)  # [N, D]
        x = apply_pos_embed(x, weight, pos_ids=pos_ids)
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}"


class SinCosPosEmbed2D(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, int]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)

        weight = get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size)
        self.weight = nn.Parameter(torch.from_numpy(weight).float(), requires_grad=False)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        pos_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        x = apply_pos_embed(x, self.weight, pos_ids=pos_ids)
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}"


class SinCosPosEmbed3D(nn.Module):
    def __init__(self, embed_dim: int, grid_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_patches = math.prod(grid_size)

        N_t, N_h, N_w = grid_size
        weight = get_3d_sincos_pos_embed(
            embed_dim=embed_dim,
            grid_size=(N_h, N_w),
            grid_depth=N_t,
            uniform_power=True,
        )
        self.weight = nn.Parameter(torch.from_numpy(weight).float(), requires_grad=False)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        pos_ids: Int[Tensor, "B L"] | None = None,
    ) -> Float[Tensor, "B L D"]:
        x = apply_pos_embed(x, self.weight, pos_ids=pos_ids)
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}"


# sincos pos embed utils from vjepa2, but fixed the confusing meshgrid indexing


def get_3d_sincos_pos_embed(embed_dim, grid_size, grid_depth, cls_token=False, uniform_power=False):
    """
    grid_size: tuple of int of the grid height and width
    grid_depth: int of the grid depth
    returns:
        pos_embed: [grid_depth*grid_height*grid_width, embed_dim] (w/o cls_token)
                or [1+grid_depth*grid_height*grid_width, embed_dim] (w/ cls_token)
    """
    grid_d = np.arange(grid_depth, dtype=float)
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid_d, grid_h, grid_w = np.meshgrid(grid_d, grid_h, grid_w, indexing="ij")

    if not uniform_power:
        h_embed_dim = embed_dim // 4
        w_embed_dim = embed_dim // 4
        d_embed_dim = embed_dim // 2
    else:
        h_embed_dim = w_embed_dim = d_embed_dim = int(np.ceil(embed_dim / 6) * 2)

    emb_h = get_1d_sincos_pos_embed_from_grid(h_embed_dim, grid_h)  # (T*H*W, D1)
    emb_w = get_1d_sincos_pos_embed_from_grid(w_embed_dim, grid_w)  # (T*H*W, D2)
    emb_d = get_1d_sincos_pos_embed_from_grid(d_embed_dim, grid_d)  # (T*H*W, D3)
    pos_embed = np.concatenate([emb_d, emb_h, emb_w], axis=1)
    pos_embed = pos_embed[:, :embed_dim]
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: tuple of int of the grid height and width
    returns:
        pos_embed: [grid_height*grid_width, embed_dim] (w/o cls_token)
                or [1+grid_height*grid_width, embed_dim] (w/ cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid_h, grid_w = np.meshgrid(grid_h, grid_w, indexing="ij")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w)  # (H*W, D/2)
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim] (w/o cls_token)
                or [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    returns: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def apply_pos_embed(
    x: Float[Tensor, "B L D"],
    weight: Float[Tensor, "N D"],
    pos_ids: Int[Tensor, "B L"] | None = None,
) -> Float[Tensor, "B L D"]:
    B, L, D = x.shape
    weight = weight.expand(B, -1, -1)
    if pos_ids is not None:
        weight = weight.gather(1, pos_ids.unsqueeze(-1).expand(-1, -1, D))
    x = x + weight
    return x


# (masked) normalization used for MAE target normalization


class Normalize(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, ...],
        dim: int | tuple[int, ...] | None = -1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        Normalize input sequence along dim(s) after reshaping to grid.
        Returns tuple of (x, mean, std).
        """
        B, N, D = x.shape
        x = x.reshape((B, *self.grid_size, D))
        if mask is not None:
            mask = mask.reshape((B, *self.grid_size, D))
            x, mean, std = masked_normalize(x, mask, dim=self.dim, eps=self.eps)
        else:
            x, mean, std = normalize(x, dim=self.dim, eps=self.eps)
        mean = mean.expand_as(x).reshape(B, N, D)
        std = std.expand_as(x).reshape(B, N, D)
        x = x.reshape(B, N, D)
        return x, mean, std

    def extra_repr(self):
        return f"{self.grid_size}, dim={self.dim}"


def masked_normalize(
    x: Tensor,
    mask: Tensor,
    dim: int | tuple[int, ...] | None = -1,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    num_obs = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    mean = (mask * x).sum(dim=dim, keepdim=True) / num_obs
    var = (mask * (x - mean) ** 2).sum(dim=dim, keepdim=True) / num_obs
    std = (var + eps) ** 0.5
    x = mask * (x - mean) / std
    return x, mean, std


def normalize(
    x: Tensor,
    dim: int | tuple[int, ...] | None = -1,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    mean = x.mean(dim=dim, keepdim=True)
    var = torch.var(x, dim=dim, keepdim=True, unbiased=False)
    std = (var + eps) ** 0.5
    x = (x - mean) / std
    return x, mean, std
