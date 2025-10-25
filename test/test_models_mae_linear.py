import pytest
import torch
from timm.layers import to_2tuple

from models_mae_linear import MaskedAutoencoderLinear


def mae_linear_tiny_patch16(**kwargs):
    model = MaskedAutoencoderLinear(patch_size=16, embed_dim=192, **kwargs)
    return model


@pytest.mark.parametrize(
    "with_img_mask,with_vis_mask",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize(
    "img_size,in_chans,num_frames,t_patch_size,t_pred_patch_size",
    [
        # default case
        (64, 3, 8, 2, 1),
        # single channel
        (64, 1, 8, 2, 1),
        # mae case, t_patch_size = num_frames, predict all frames
        (64, 1, 8, 8, 8),
        # non-square image
        ([64, 96], 1, 8, 2, 1),
    ],
)
@pytest.mark.parametrize("framewise", [False, True])
def test_mae_linear(
    framewise: bool,
    img_size: int | tuple[int, int],
    in_chans: int,
    num_frames: int,
    t_patch_size: int,
    t_pred_patch_size: int,
    with_img_mask: bool,
    with_vis_mask: bool,
):
    # check that model runs for all cases.
    img_size = to_2tuple(img_size)
    H, W = img_size

    model = mae_linear_tiny_patch16(
        img_size=img_size,
        in_chans=in_chans,
        num_frames=num_frames,
        t_patch_size=t_patch_size,
        t_pred_patch_size=t_pred_patch_size,
        framewise=framewise,
    )

    x = torch.randn(2, in_chans, num_frames, H, W)
    if with_img_mask:
        img_mask = torch.zeros(H, W)
        img_mask[18 : H - 18, 18 : W - 18] = 1
    else:
        img_mask = None

    if with_vis_mask:
        visible_mask = torch.ones(H, W)
        visible_mask[:, W // 2 :] = 0
    else:
        visible_mask = None

    loss, pred, mask, decoder_mask = model.forward(
        x, img_mask=img_mask, visible_mask=visible_mask
    )
    print(
        f"loss: {loss:.3e}, pred: {pred.shape}, "
        f"mask: {mask.shape}, {mask.sum().item()}."
    )
    assert not torch.isnan(loss)
    assert loss.item() < 4
