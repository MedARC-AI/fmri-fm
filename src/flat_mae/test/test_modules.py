from typing import Type

import pytest
import torch

from flat_mae.modules import (
    Patchify2D,
    Patchify3D,
    AbsolutePosEmbed,
    SeparablePosEmbed,
    SinCosPosEmbed2D,
    SinCosPosEmbed3D,
    masked_normalize,
    normalize,
)


def test_patchify2d_roundtrip():
    """Test 2D patchify and unpatchify are inverses"""
    B, C, H, W = 2, 3, 64, 64
    patch_size = 16

    patchify = Patchify2D(
        img_size=(H, W),
        patch_size=patch_size,
        in_chans=C,
    )

    x = torch.randn(B, C, H, W)
    patches = patchify(x)
    x_recon = patchify.unpatchify(patches)

    p_h, p_w = patch_size, patch_size
    expected_num_patches = (H // p_h) * (W // p_w)
    expected_patch_dim = p_h * p_w * C

    assert patches.shape == (B, expected_num_patches, expected_patch_dim)
    assert x_recon.shape == x.shape
    assert torch.allclose(x, x_recon)


def test_patchify3d_roundtrip():
    """Test 3D patchify and unpatchify are inverses"""
    B, C, T, H, W = 2, 3, 16, 64, 64
    patch_size = (2, 16, 16)

    patchify = Patchify3D(
        img_size=(T, H, W),
        patch_size=patch_size,
        in_chans=C,
    )

    x = torch.randn(B, C, T, H, W)
    patches = patchify(x)
    x_recon = patchify.unpatchify(patches)

    p_t, p_h, p_w = patch_size
    expected_num_patches = (T // p_t) * (H // p_h) * (W // p_w)
    expected_patch_dim = p_t * p_h * p_w * C

    assert patches.shape == (B, expected_num_patches, expected_patch_dim)
    assert x_recon.shape == x.shape
    assert torch.allclose(x, x_recon)


@pytest.mark.parametrize(
    "ndim,cls",
    [
        (2, AbsolutePosEmbed),
        (3, AbsolutePosEmbed),
        (2, SeparablePosEmbed),
        (3, SeparablePosEmbed),
        (2, SinCosPosEmbed2D),
        (3, SinCosPosEmbed3D),
    ],
)
def test_pos_embed(ndim: int, cls: Type[AbsolutePosEmbed]):
    """Test position embeddings"""
    if ndim == 3:
        B, L, D = 2, 24, 64
        grid_size = (3, 4, 2)
    else:
        B, L, D = 2, 16, 64
        grid_size = (4, 4)

    pos_embed = cls(embed_dim=D, grid_size=grid_size)
    is_learned = not isinstance(pos_embed, (SinCosPosEmbed2D, SinCosPosEmbed3D))

    assert pos_embed.grid_size == grid_size
    assert pos_embed.num_patches == L

    # full position embed
    x = torch.randn(B, L, D)
    x_out = pos_embed(x)
    assert x_out.shape == (B, L, D)

    # check sincos pos embed don't have grad
    assert x_out.requires_grad == is_learned

    # partial position embed with ids
    Q = L // 2
    pos_ids = torch.randint(0, L, (B, Q))
    x_sub = x.gather(1, pos_ids.unsqueeze(-1).expand(-1, -1, D))
    x_sub_out = pos_embed(x_sub, pos_ids=pos_ids)
    assert x_sub_out.shape == (B, Q, D)

    x_out_sub = x_out.gather(1, pos_ids.unsqueeze(-1).expand(-1, -1, D))
    assert torch.allclose(x_sub_out, x_out_sub)


def test_masked_normalize():
    """Test masked normalization"""
    B, N, D = 2, 10, 32

    x = torch.randn(B, N, D)
    mask = torch.rand(B, N, D) > 0.5

    x_norm, mean, std = masked_normalize(x, mask, dim=-1)

    assert x_norm.shape == (B, N, D)
    assert mean.shape == (B, N, 1)
    assert std.shape == (B, N, 1)

    # Masked positions should be zero
    assert torch.allclose(x_norm[~mask], torch.zeros_like(x_norm[~mask]))

    # Check masked mean and std are correct
    assert torch.allclose(x[0, 0, mask[0, 0]].mean(), mean[0, 0, 0])
    assert torch.allclose(x[0, 0, mask[0, 0]].std(unbiased=False), std[0, 0, 0])


def test_masked_normalize_equivalence():
    """Test that masked_normalize with all-ones mask equals normalize"""
    B, N, D = 2, 10, 8

    x = torch.randn(B, N, D)
    mask = torch.ones(B, N, D)

    x_masked, mean_masked, std_masked = masked_normalize(x, mask, dim=-1)
    x_norm, mean_norm, std_norm = normalize(x, dim=-1)

    assert torch.allclose(x_masked, x_norm, atol=1e-5)
