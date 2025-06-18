import pytest
import torch
import torch.nn.functional as F

import flat_data


@pytest.fixture
def dummy_mask() -> torch.Tensor:
    mask = torch.ones(16, 16)
    mask = F.pad(mask, (4, 4, 4, 4))
    mask = torch.cat([mask, mask], dim=-1)
    return mask


def test_hemi_masking(dummy_mask: torch.Tensor):
    visible_mask = flat_data.hemi_masking(dummy_mask)
    H, W = visible_mask.shape
    assert not (visible_mask[:, :W // 2].any() and visible_mask[:, W // 2:].any())


def test_inverse_block_masking(dummy_mask: torch.Tensor):
    visible_mask = flat_data.inverse_block_masking(dummy_mask, block_size=8)
    H, W = visible_mask.shape
    indices = visible_mask.nonzero()
    xmin, ymin = indices.amin(dim=0)
    xmax, ymax = indices.amax(dim=0)
    assert torch.maximum(xmax - xmin, ymax - ymin) <= 8


def test_hemi_inverse_block_masking(dummy_mask: torch.Tensor):
    visible_mask = flat_data.hemi_inverse_block_masking(dummy_mask, block_size=8)
    H, W = visible_mask.shape
    indices = visible_mask.nonzero()
    xmin, ymin = indices.amin(dim=0)
    xmax, ymax = indices.amax(dim=0)
    assert torch.maximum(xmax - xmin, ymax - ymin) <= 8
    assert not (visible_mask[:, :W // 2].any() and visible_mask[:, W // 2:].any())


def test_pad_to_multiple():
    img = torch.ones(3, 1, 12, 14)
    img = flat_data.pad_to_multiple(img, 16)
    H, W = img.shape[-2:]
    assert H, W == (16, 16)


@pytest.mark.parametrize("masking", flat_data.MASKING_REGISTRY)
def test_make_masking(masking: str):
    flat_data.make_masking(masking, block_size=8)
