from __future__ import annotations

from typing import Mapping, Sequence

import torch


def _grid_from_masks(mask_2d: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Downsample a pixel mask to a patch-level boolean grid.

    Converts a 2D pixel-space mask into a boolean grid indicating which patches contain
    at least one masked pixel.

    Args:
        mask_2d: [H, W] mask tensor.
        patch_size: Size of each square patch in pixels.

    Returns:
        Bool tensor of shape [H//P, W//P] where True marks a masked patch.
    """
    H, W = mask_2d.shape
    h, w = H // patch_size, W // patch_size
    mask_2d = mask_2d[: h * patch_size, : w * patch_size]
    patches = mask_2d.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(h, w, patch_size * patch_size)
    return (patches.sum(dim=-1) > 0).to(torch.bool)


def collate_data_capi(
    samples_list: Sequence[Mapping[str, torch.Tensor]],
    *,
    img_h: int,
    img_w: int,
    patch_size: int,
    mask_ratio: float | int,
    prediction_subsampling: float | int,
    dtype: torch.dtype,
    time_as_channels: bool = False,
    select_frame_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Collate fMRI flat-map samples into a CAPI training batch.

    Builds visible/masked token indices and stacks images, supporting both
    [C, T, H, W] and [C, H, W] inputs. Optionally merges time into channels or
    selects a specific temporal frame.

    Args:
        samples_list: Sequence of sample dicts with keys: 'image' and 'mask' or 'img_mask';
            optional 'visible_mask'.
        img_h: Target image height.
        img_w: Target image width.
        patch_size: Patch size used by the model.
        mask_ratio: Fraction of eligible patches to mask.
        prediction_subsampling: Fraction of masked-eligible patches to predict.
        dtype: Torch dtype of returned tensors.
        time_as_channels: If True, merge time into the channel dimension.
        select_frame_index: Temporal index used when not merging time.

    Returns:
        Dict with:
          - 'image': [B, C or T, H, W]
          - 'predict_indices': 1D long indices into the flattened [B*N] patch grid
          - 'visible_indices': 1D long indices of visible patches in the [B*N] grid
    """
    batch_size = len(samples_list)

    images = []
    for s in samples_list:
        img = s["image"]  # [C, T, H, W]
        if img.ndim == 4:
            if time_as_channels:
                img = img.squeeze(0)
            else:
                t = img.shape[1]
                idx = min(max(select_frame_index, 0), t - 1)
                img = img[:, idx]
        images.append(img)
    images = torch.stack(images).to(dtype)

    grid_h, grid_w = img_h // patch_size, img_w // patch_size
    n_tokens = grid_h * grid_w

    per_sample = []
    for s in samples_list:
        # support both keys: new repo uses "img_mask" in some paths; flat_data transform returns
        # either "img_mask" (legacy) or "mask" (new). Prefer "mask" if present.
        mask_key = "mask" if "mask" in s else "img_mask"
        img_mask = s[mask_key].squeeze(0).squeeze(0).to(images.dtype)
        valid_patches = _grid_from_masks(img_mask, patch_size)
        eligible = valid_patches.flatten()
        if "visible_mask" in s and s["visible_mask"] is not None:
            vis_mask = s["visible_mask"].squeeze(0).squeeze(0).to(images.dtype)
            vis_patches = _grid_from_masks(vis_mask, patch_size)
            visible_candidates = (eligible & vis_patches.flatten())
        else:
            visible_candidates = eligible
        per_sample.append((eligible, visible_candidates))

    n_eligible_list = [int(e.sum().item()) for e, _ in per_sample]
    keep_target_list = [max(int(n - int(mask_ratio * n)), 0) for n in n_eligible_list]
    keepable_list = [int(vc.sum().item()) for _, vc in per_sample]
    if len(keep_target_list) > 0:
        n_keep_uniform = max(0, min(min(keep_target_list), min(keepable_list)))
    else:
        n_keep_uniform = 0

    visible_token_masks = []
    masked_token_masks = []
    for (eligible, visible_candidates) in per_sample:
        visible_tokens = torch.zeros(n_tokens, dtype=torch.bool)
        if n_keep_uniform > 0 and int(visible_candidates.sum().item()) > 0:
            keepable_idx = visible_candidates.nonzero(as_tuple=False).flatten()
            if len(keepable_idx) >= n_keep_uniform:
                sel = torch.randperm(len(keepable_idx))[: n_keep_uniform]
                keep_idx = keepable_idx[sel]
            else:
                sel = torch.randint(0, len(keepable_idx), (n_keep_uniform,))
                keep_idx = keepable_idx[sel]
            visible_tokens[keep_idx] = True
        masked_tokens = (~visible_tokens) | (~eligible)
        visible_token_masks.append(visible_tokens)
        masked_token_masks.append(masked_tokens)

    visible_token_masks = torch.stack(visible_token_masks)
    masked_token_masks = torch.stack(masked_token_masks)

    masked_eligible_idx_list = []
    masked_eligible_counts = []
    for b in range(batch_size):
        eligible = per_sample[b][0]
        masked_eligible = masked_token_masks[b] & eligible
        idx = masked_eligible.nonzero(as_tuple=False).flatten()
        masked_eligible_idx_list.append(idx)
        masked_eligible_counts.append(len(idx))

    min_masked_eligible = min(masked_eligible_counts) if masked_eligible_counts else 0
    n_predict_uniform = int(float(prediction_subsampling) * min_masked_eligible)
    if n_predict_uniform == 0 and min_masked_eligible > 0 and float(prediction_subsampling) > 0:
        n_predict_uniform = 1

    predict_indices_abs = []
    for b in range(batch_size):
        idx = masked_eligible_idx_list[b]
        if n_predict_uniform > 0 and len(idx) > 0:
            if len(idx) >= n_predict_uniform:
                perm = torch.randperm(len(idx))[: n_predict_uniform]
                sel = idx[perm]
            else:
                perm = torch.randint(0, len(idx), (n_predict_uniform,))
                sel = idx[perm]
            predict_indices_abs.append(sel + b * n_tokens)
    predict_indices_abs = (
        torch.cat(predict_indices_abs) if predict_indices_abs else torch.empty(0, dtype=torch.long)
    )

    visible_indices_abs = visible_token_masks.flatten().nonzero(as_tuple=False).flatten()

    return {
        "image": images,
        "predict_indices": predict_indices_abs,
        "visible_indices": visible_indices_abs,
    }


