from typing import Literal

import pytest
import torch
from timm.layers import to_2tuple

from models_mae import MaskedAutoencoderViT


def mae_vit_tiny_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=192,
        depth=6,
        num_heads=3,
        decoder_embed_dim=192,
        decoder_depth=2,
        decoder_num_heads=3,
        sep_pos_embed=True,
        cls_embed=True,
        **kwargs,
    )
    return model


@pytest.mark.parametrize(
    "with_img_mask,with_vis_mask",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
)
@pytest.mark.parametrize(
    "img_size,in_chans,num_frames,t_patch_size,t_pred_patch_size",
    [
        # default case
        (224, 3, 16, 2, 1),
        # single channel
        (224, 1, 16, 2, 1),
        # predict all frames
        (224, 1, 16, 2, 2),
        # mae case, t_patch_size = num_frames, predict one frame
        (224, 1, 16, 16, 1),
        # mae case, t_patch_size = num_frames, predict all frames
        (224, 1, 16, 16, 16),
        # non-square image
        ([144, 224], 1, 16, 2, 1),
    ]
)
def test_mae_vit(
    img_size: int | tuple[int, int],
    in_chans: int,
    num_frames: int,
    t_patch_size: int,
    t_pred_patch_size: int,
    with_img_mask: bool,
    with_vis_mask: bool,
):
    img_size = to_2tuple(img_size)
    H, W = img_size

    model = mae_vit_tiny_patch16(
        img_size=img_size,
        in_chans=in_chans,
        num_frames=num_frames,
        t_patch_size=t_patch_size,
        t_pred_patch_size=t_pred_patch_size,
    )

    x = torch.randn(2, in_chans, num_frames, H, W)
    if with_img_mask:
        img_mask = torch.zeros(H, W)
        img_mask[18: H - 18, 18: W - 18] = 1
    else:
        img_mask = None

    if with_vis_mask: 
        visible_mask = torch.ones(H, W)
        visible_mask[:, W // 2:] = 0
    else:
        visible_mask = None
    
    loss, pred, mask = model.forward(x, img_mask=img_mask, visible_mask=visible_mask)
    print(
        f"loss: {loss:.3e}, pred: {pred.shape}, "
        f"mask: {mask.shape}, {mask.sum().item()}."
    )
    assert not torch.isnan(loss)
