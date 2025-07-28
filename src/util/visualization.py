import io
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PIL import Image

plt.rcParams["figure.dpi"] = 150

FC_COLORS = np.array(
    [
        [ 64,  80, 160],
        [ 64,  96, 176],
        [ 96, 192, 240],
        [144, 208, 224],
        [255, 255, 255],
        [240, 240,  96],
        [240, 208,  64],
        [224, 112,  64],
        [224,  64,  48],
    ],
    dtype=np.uint8
)

FC_CMAP = LinearSegmentedColormap.from_list("fc", FC_COLORS / 255.0)
FC_CMAP.set_bad("gray")


def plot_mask_pred(
    target: torch.Tensor,
    im_masked: torch.Tensor,
    im_paste: torch.Tensor,
    img_mask: Optional[torch.Tensor] = None,
    vmax: float = 3.0,
    nrow: int = 8,
):
    N, T, H, W, C = target.shape

    # N T H W C -> (N T) H W C
    target = target.flatten(0, 1)[:nrow].cpu().numpy()
    im_masked = im_masked.flatten(0, 1)[:nrow].cpu().numpy()
    im_paste = im_paste.flatten(0, 1)[:nrow].cpu().numpy()

    if img_mask is not None:
        assert img_mask.shape == (N, T, H, W)
        img_mask = img_mask.flatten(0, 1)[:nrow].cpu().numpy()

    ploth = 2.0
    plotw = (W / H) * ploth
    nrow = len(target)
    ncol = 3
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(plotw * ncol, ploth * nrow), squeeze=False
    )

    for ii in range(nrow):
        n_idx, t_idx = ii // T, ii % T

        plt.sca(axs[ii, 0])
        _imshow(im_masked[ii], mask=img_mask[ii], vmin=-vmax, vmax=vmax)
        plt.text(
            0.01,
            0.98,
            f"({n_idx}, {t_idx})",
            transform=axs[ii, 0].transAxes,
            va="top",
            ha="left",
        )

        plt.sca(axs[ii, 1])
        _imshow(im_paste[ii], mask=img_mask[ii], vmin=-vmax, vmax=vmax)

        plt.sca(axs[ii, 2])
        _imshow(target[ii], mask=img_mask[ii], vmin=-vmax, vmax=vmax)

    plt.tight_layout(pad=0.25)
    return fig


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
        image = np.where(mask, image, np.nan)
    plt.imshow(image, **kwargs)
    plt.axis("off")


def fig2pil(fig: Figure, format: str = "png") -> Image.Image:
    with io.BytesIO() as f:
        fig.savefig(f, format=format)
        f.seek(0)
        img = Image.open(f)
        img.load()
    return img
