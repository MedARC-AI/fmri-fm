# This source code is licensed under the Apache License, Version 2.0
#
# References:
# capi: https://github.com/facebookresearch/capi/blob/main/data.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import default_collate
from jaxtyping import Float, Int
from timm.layers import to_2tuple

from .modules import Patchify2D, Patchify3D
from .utils import filter_kwargs


class RandomMasking(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        num_frames: int | None = None,
        t_patch_size: int | None = None,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        if num_frames:
            img_size = (num_frames, *img_size)
            patch_size = (t_patch_size, *patch_size)

        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = patch_size

        patchify_layer = {2: Patchify2D, 3: Patchify3D}[len(img_size)]
        self.patchify = patchify_layer(img_size, patch_size, in_chans=1)

    def extra_repr(self):
        return f"mask_ratio={self.mask_ratio}"

    def forward(
        self,
        img_mask: Float[Tensor, "H W"] | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        # [B, C, H, W] or [B, C, T, H, W]
        if img_mask is None:
            img_mask = torch.ones((1, 1, *self.img_size), device=device)
        else:
            img_mask = img_mask.expand((1, 1, *self.img_size))

        mask_patches = self.patchify(img_mask)
        patch_mask = mask_patches.any(dim=-1).float()
        patch_mask, _ = trim_patch_mask(patch_mask, mask_ratio=self.mask_ratio, shuffle=True)
        mask_patches = patch_mask.unsqueeze(-1).expand(-1, -1, mask_patches.shape[-1])
        mask = self.patchify.unpatchify(mask_patches)
        mask = mask.reshape(self.img_size)  # [H, W] or [T, H, W]
        return mask


class TubeMasking(RandomMasking):
    """
    tube masking is a special case of random masking where the mask is broadcasted
    across the first (time) dimension.
    """

    def __init__(
        self,
        mask_ratio: float,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        num_frames: int | None = None,
        t_patch_size: int | None = None,
    ):
        super().__init__(mask_ratio=mask_ratio, img_size=img_size, patch_size=patch_size)


def _inverse_block_masking(mask: torch.Tensor, block_size: int = 160) -> torch.Tensor:
    """Sample a block visible mask."""
    H, W = mask.shape
    h_idx = np.random.randint(0, H - block_size)
    w_idx = np.random.randint(0, W - block_size)

    mask[h_idx : h_idx + block_size, w_idx : w_idx + block_size] = 1
    return mask


class InverseBlockMasking(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int],
        block_size: int = 160,
        num_frames: int | None = None,
        **kwargs,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        if num_frames:
            img_size = (num_frames, *img_size)
        self.img_size = img_size
        self.block_size = block_size

    def forward(
        self,
        img_mask: Float[Tensor, "H W"] | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Sample a block visible mask."""
        mask = torch.zeros(self.img_size, device=device)
        return _inverse_block_masking(mask, self.block_size)

MASKING_DICT = {
    "random": RandomMasking,
    "tube": TubeMasking,
    "inverse": InverseBlockMasking,
}


def create_masking(name: str, **kwargs) -> RandomMasking:
    cls = MASKING_DICT[name]
    kwargs = filter_kwargs(cls, kwargs)
    mask_fn = cls(**kwargs)
    return mask_fn


def mask_collate(
    samples: list[dict[str, Tensor]], *, mask_fn: RandomMasking | None = None
) -> dict[str, Tensor]:
    """
    Generates a visible mask for each sample, and pads the shape with singleton
    dimensions for batching.
    """
    for sample in samples:
        image = sample["image"]
        img_mask = sample.get("img_mask")
        if mask_fn is not None:
            visible_mask = mask_fn(img_mask)
            sample["visible_mask"] = _unsqueeze_as(visible_mask, image)
        if img_mask is not None:
            sample["img_mask"] = _unsqueeze_as(img_mask, image)
    batch = default_collate(samples)
    return batch


def _unsqueeze_as(x: Tensor, other: Tensor) -> Tensor:
    assert other.ndim >= x.ndim
    x = x.reshape((1,) * (other.ndim - x.ndim) + x.shape)
    return x


def trim_patch_mask(
    patch_mask: Float[Tensor, "B N"],
    mask_ratio: float | None = None,
    len_keep: int | None = None,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "B N"], Int[Tensor, "B L"]]:
    """
    Trim a batch of patch masks to the same number of patches.
    Kept patches are selected randomly (shuffle=True) or sequentially (shuffle=False).
    """
    assert not (mask_ratio and len_keep), "can't set both mask_ratio and len_keep"
    B, N = patch_mask.shape
    device = patch_mask.device

    # override len_keep with mask_ratio
    if mask_ratio is not None:
        len_keep = int((1 - mask_ratio) * N)

    # shuffle patches for each sample
    if shuffle:
        noise = torch.rand(B, N, generator=generator, device=device)
        shuffle_ids = torch.argsort(noise, dim=1)
        restore_ids = torch.argsort(shuffle_ids, dim=1)
        patch_mask = patch_mask.gather(1, shuffle_ids)

    # all masks trimmed to have the same size, no bigger than the smallest mask
    min_count = patch_mask.sum(dim=1).min()
    len_keep = min_count if len_keep is None else min_count.clamp(max=len_keep)

    # discard extra patches
    patch_mask = patch_mask * (patch_mask.cumsum(dim=1) <= len_keep)

    # shuffle patches back to original order
    if shuffle:
        patch_mask = patch_mask.gather(1, restore_ids)

    mask_ids = patch_mask.nonzero(as_tuple=False)[:, 1].reshape(B, -1)
    return patch_mask, mask_ids


def pad_image_mask(mask: Float[Tensor, "... H W"], pad: int = 1):
    """
    dilate ("pad") an image mask by a few pixels.
    """
    dtype = mask.dtype
    device = mask.device
    *shape, H, W = mask.shape
    mask = mask.reshape(-1, 1, H, W)
    kernel_size = 2 * pad + 1
    weight = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype)
    out_mask = F.conv2d(mask, weight, padding="same")
    out_mask = (out_mask > 0).to(dtype)
    out_mask = out_mask.reshape((*shape, H, W))
    return out_mask
