# Copyright (c) Sophont, Inc
#
# This source code is licensed under the Apache License, Version 2.0

import io
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

plt.rcParams["figure.dpi"] = 150

# from rick betzel's figures
FC_COLORS = np.array(
    [
        [64, 80, 160],
        [64, 96, 176],
        [96, 192, 240],
        [144, 208, 224],
        [255, 255, 255],
        [240, 240, 96],
        [240, 208, 64],
        [224, 112, 64],
        [224, 64, 48],
    ],
    dtype=np.uint8,
)

FC_CMAP = LinearSegmentedColormap.from_list("fc", FC_COLORS / 255.0)
FC_CMAP.set_bad("gray")


def plot_mask_pred(
    target: Tensor,
    pred: Tensor,
    visible_mask: Tensor,
    pred_mask: Tensor | None,
    img_mask: Tensor | None = None,
    paste_visible: bool = True,
    stride: int = 2,
    nrow: int = 8,
    vmax: float = 3.0,
):
    # [B, C, H, W] or [B, C, T, H, W]
    assert target.ndim in {4, 5}, "invalid target shape"
    T = target.shape[2] if target.ndim == 5 else 1

    target = _prep_images(target, nrow, stride)
    pred = _prep_images(pred, nrow, stride)
    visible_mask = _prep_images(visible_mask, nrow, stride)
    pred_mask = _prep_images(pred_mask, nrow, stride)
    img_mask = _prep_images(img_mask, nrow, stride)

    if img_mask is None:
        img_mask = np.ones_like(visible_mask)
    if pred_mask is None:
        pred_mask = img_mask * (1 - visible_mask)

    target_masked = target * visible_mask
    pred_masked = pred * pred_mask
    if paste_visible:
        pred_masked = (1 - pred_mask) * target_masked + pred_mask * pred

    _, H, W, _ = target.shape
    ploth = 2.0
    plotw = (W / H) * ploth
    nrow = len(target)
    ncol = 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(plotw * ncol, ploth * nrow), squeeze=False)

    for ii in range(nrow):
        idx = ii * stride
        n_idx, t_idx = idx // T, idx % T

        plt.sca(axs[ii, 0])
        _imshow(target_masked[ii], mask=img_mask[ii], vmin=-vmax, vmax=vmax)
        plt.text(
            0.01,
            0.98,
            f"({n_idx}, {t_idx})",
            transform=axs[ii, 0].transAxes,
            va="top",
            ha="left",
        )

        plt.sca(axs[ii, 1])
        _imshow(pred_masked[ii], mask=img_mask[ii], vmin=-vmax, vmax=vmax)

        plt.sca(axs[ii, 2])
        _imshow(target[ii], mask=img_mask[ii], vmin=-vmax, vmax=vmax)

    plt.tight_layout(pad=0.25)
    return fig


def _prep_images(imgs: Tensor | None, nrow: int, stride: int) -> np.ndarray | None:
    if imgs is not None:
        # channels last
        if imgs.ndim == 5:
            imgs = imgs.permute((0, 2, 3, 4, 1))
            imgs = imgs.flatten(0, 1)  # flatten time with batch
        else:
            imgs = imgs.permute((0, 2, 3, 1))
        imgs = imgs[: stride * nrow : stride]
        imgs = imgs.detach().cpu().numpy()
    return imgs


def _imshow(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    **kwargs,
):
    H, W, C = image.shape
    assert C == 1
    image = image.squeeze(2)
    kwargs = {"cmap": FC_CMAP, "interpolation": "nearest", **kwargs}
    if mask is not None:
        image = np.where(mask.squeeze(), image, np.nan)
    plt.imshow(image, **kwargs)
    plt.axis("off")


def fig2pil(fig: Figure, format: str = "png") -> Image.Image:
    with io.BytesIO() as f:
        fig.savefig(f, format=format)
        f.seek(0)
        img = Image.open(f)
        img.load()
    return img
