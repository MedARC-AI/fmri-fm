import math

import pytest
import torch
import timm

import flat_mae.models_mae as models_mae

_CFGS = {
    "tiny": dict(
        depth=4,
        embed_dim=64,
        num_heads=8,
        decoder_depth=2,
        decoder_embed_dim=64,
        decoder_num_heads=8,
    )
}


@pytest.mark.parametrize(
    "class_token,reg_tokens",
    [
        (False, 0),
        (True, 0),
        (False, 4),
        (True, 4),
    ],
)
def test_timm_vit_equivalence(class_token: bool, reg_tokens: int):
    model_timm = timm.create_model(
        "vit_small_patch16_224",
        num_classes=0,
        class_token=class_token,
        reg_tokens=reg_tokens,
        global_pool="",
    )
    model_ours = models_mae.vit_small(
        img_size=224,
        patch_size=16,
        qkv_bias=True,
        proj_bias=True,
        class_token=class_token,
        reg_tokens=reg_tokens,
    )

    # load converted state dict from timm model
    state_dict = models_mae._convert_from_timm(model_timm.state_dict())
    model_ours.load_state_dict(state_dict)

    x = torch.randn(1, 3, 224, 224)
    x_timm = x.clone().requires_grad_()
    x_ours = x.clone().requires_grad_()

    embeds_timm = model_timm.forward_features(x_timm)
    num_prefix_tokens = int(class_token) + reg_tokens
    patch_embeds_timm = embeds_timm[:, num_prefix_tokens:]
    cls_embeds_timm = embeds_timm[:, :1] if class_token else None
    reg_embeds_timm = embeds_timm[:, int(class_token) : num_prefix_tokens] if reg_tokens else None

    cls_embeds_ours, reg_embeds_ours, patch_embeds_ours, *_ = model_ours.forward(x_ours)

    # check embeddings are close
    assert torch.allclose(patch_embeds_timm, patch_embeds_ours, atol=1e-5)
    if class_token:
        assert torch.allclose(cls_embeds_timm, cls_embeds_ours, atol=1e-5)
    if reg_tokens:
        assert torch.allclose(reg_embeds_timm, reg_embeds_ours, atol=1e-5)

    # check grads are close
    loss_timm = patch_embeds_timm.sum()
    loss_ours = patch_embeds_ours.sum()
    loss_timm.backward()
    loss_ours.backward()

    assert torch.allclose(x_timm.grad, x_ours.grad, atol=1e-5)


@pytest.mark.parametrize(
    "img_size,patch_size,in_chans",
    [
        [(64, 64), (8, 8), 3],  # rgb images
        [(64, 64), (8, 8), 1],  # gray images
        [(32, 64), (8, 8), 1],  # gray images, not square
        [(4, 64, 64), (2, 8, 8), 3],  # rgb video
        [(4, 64, 64), (2, 8, 8), 1],  # gray video
        [(4, 64, 64), (4, 8, 8), 1],  # gray video full t patch size
        [(4, 64, 1), (4, 1, 1), 1],  # gray vector
    ],
)
def test_mae_vit_input_size(
    img_size: tuple[int, ...],
    patch_size: tuple[int, ...],
    in_chans: int,
):
    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    loss, state = model.forward(x, mask_ratio=0.75)
    assert not torch.isnan(loss)


@pytest.mark.parametrize("pos_embed", ["abs", "sincos"])
def test_mae_vit_pos_embed_2d(pos_embed: str):
    img_size = (32, 64)
    patch_size = (8, 8)
    in_chans = 3
    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pos_embed=pos_embed,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    loss, state = model.forward(x, mask_ratio=0.75)
    assert not torch.isnan(loss)


@pytest.mark.parametrize("pos_embed", ["abs", "sep", "sincos"])
def test_mae_vit_pos_embed_3d(pos_embed: str):
    img_size = (8, 32, 64)
    patch_size = (4, 8, 8)
    in_chans = 3
    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pos_embed=pos_embed,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    loss, state = model.forward(x, mask_ratio=0.75)
    assert not torch.isnan(loss)


@pytest.mark.parametrize("decoding", ["attn", "cross", "crossreg"])
def test_mae_vit_decoding(decoding: str):
    img_size = (32, 64)
    patch_size = (8, 8)
    in_chans = 3
    reg_tokens = 4 if decoding == "crossreg" else 0
    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        decoding=decoding,
        reg_tokens=reg_tokens,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    loss, state = model.forward(x, mask_ratio=0.75)
    assert not torch.isnan(loss)


@pytest.mark.parametrize(
    "with_img_mask,with_visible_mask,with_pred_mask,mask_ratio,pred_mask_ratio",
    [
        (False, False, False, 0.75, None),  # standard mae
        (False, False, False, 0.75, 0.75),  # sparse decoding
        (True, False, False, 0.75, 0.75),  # sparse decoding with data mask
        (False, True, False, None, None),  # custom visible mask
        (False, True, True, None, None),  # custom visible and pred mask
        (True, True, True, None, None),  # custom masks intersected with data mask
    ],
)
def test_mae_vit_masking(
    with_img_mask: bool,
    with_visible_mask: bool,
    with_pred_mask: bool,
    mask_ratio: float | None,
    pred_mask_ratio: float | None,
):
    img_size = (32, 32)
    patch_size = (8, 8)
    in_chans = 3

    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    if with_img_mask:
        img_mask = torch.ones(*img_size)
        img_mask[:12, :12] = 0.0
    else:
        img_mask = None
    if with_visible_mask:
        visible_mask = torch.ones(*img_size)
        visible_mask[:, :16] = 0.0
    else:
        visible_mask = None
    if with_pred_mask:
        pred_mask = torch.ones(*img_size)
        pred_mask[:, 16:] = 0.0
    else:
        pred_mask = None

    loss, state = model.forward(
        x,
        mask_ratio=mask_ratio,
        pred_mask_ratio=pred_mask_ratio,
        img_mask=img_mask,
        visible_mask=visible_mask,
        pred_mask=pred_mask,
    )
    assert not torch.isnan(loss)


@pytest.mark.parametrize("target_norm", ["global", "frame", "patch"])
@pytest.mark.parametrize("with_mask", [False, True])
def test_mae_vit_target_norm(target_norm: str, with_mask: bool):
    img_size = (8, 32, 64)
    patch_size = (4, 8, 8)
    in_chans = 3
    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        target_norm=target_norm,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    if with_mask:
        img_mask = torch.zeros_like(x)
        img_mask[:, :, :, 12:, 12:] = 1.0
    else:
        img_mask = None
    loss, state = model.forward(x, mask_ratio=0.75, img_mask=img_mask)
    assert not torch.isnan(loss)


@pytest.mark.parametrize("t_pred_stride", [2, 4])
def test_mae_vit_t_pred_stride(t_pred_stride: int):
    img_size = (4, 32, 64)
    patch_size = (4, 8, 8)
    in_chans = 1
    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        t_pred_stride=t_pred_stride,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    loss, state = model.forward(x, mask_ratio=0.75)
    assert not torch.isnan(loss)


@pytest.mark.parametrize("pred_edge_pad", [1, 2, 4])
def test_mae_vit_pred_edge_pad(pred_edge_pad: int):
    img_size = (4, 32, 64)
    patch_size = (4, 8, 8)
    in_chans = 3
    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pred_edge_pad=pred_edge_pad,
        **_CFGS["tiny"],
    )

    x = torch.randn(2, in_chans, *img_size)
    loss, state = model.forward(x, mask_ratio=0.75)
    assert not torch.isnan(loss)


def test_mae_vit_expected_loss():
    torch.manual_seed(42)

    img_size = (32, 64)
    num_frames = 8
    patch_size = 8
    t_patch_size = 4
    in_chans = 3

    model = models_mae.MaskedAutoencoderViT(
        img_size=img_size,
        num_frames=num_frames,
        in_chans=in_chans,
        patch_size=patch_size,
        t_patch_size=t_patch_size,
        **_CFGS["tiny"],
    )
    x = torch.randn((1, in_chans, num_frames, *img_size))
    loss, _ = model.forward(x)

    loss_value = loss.item()
    expected_loss_value = 1.1327531337738037
    assert math.isclose(loss_value, expected_loss_value)


@pytest.mark.parametrize("depth", [0, 1, 4])
def test_vit_depth(depth: int):
    img_size = (32, 64)
    num_frames = 8
    patch_size = 8
    t_patch_size = 4
    in_chans = 3

    model = models_mae.MaskedViT(
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        t_patch_size=t_patch_size,
        in_chans=in_chans,
        depth=depth,
        embed_dim=64,
        num_heads=8,
    )

    x = torch.randn((1, in_chans, num_frames, *img_size))
    cls_embeds, reg_embeds, patch_embeds = model.forward_embedding(x)
    assert not cls_embeds.isnan().any()
