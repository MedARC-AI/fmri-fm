import inspect
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
import scipy.sparse
import webdataset as wds
from torch.utils.data import Dataset


def make_flat_wds_dataset(
    url: str | list[str],
    num_frames: int = 16,
    clipping: str = "random",
    clipping_kwargs: dict[str, Any] | None = None,
    shuffle: bool = True,
) -> wds.WebDataset:
    """Make fMRI flat map dataset."""
    # resampling creates an infinite stream of shards sampled with replacement,
    # guaranteeing that no process runs out of data early in distributed training.
    # see webdataset FAQ: https://github.com/webdataset/webdataset/blob/main/FAQ.md
    dataset = wds.WebDataset(
        url,
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
    )
    dataset = dataset.decode().map(extract_flat_sample)

    # generate clips before shuffling for slightly better mixing.
    clipping_kwargs = clipping_kwargs or {}
    clip_fn = make_clipping(clipping, num_frames=num_frames, **clipping_kwargs)
    dataset = dataset.compose(clip_fn)

    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset


class FlatClipsDataset(Dataset):
    """
    Standard folder dataset of pre-extracted fmri flat clips.
    """
    def __init__(
        self,
        root: str | Path,
        transform: Callable[[dict[str, Any]], dict[str, Any]] = None,
    ):
        self.root = Path(root)
        self.files = sorted(p.name for p in self.root.glob("*.npy"))
        self.transform = transform

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.root / self.files[idx]
        key = path.stem
        image = np.load(path)
        sample = {"__key__": key, "image": image}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.files)


def extract_flat_sample(sample: dict[str, Any]):
    # sample metadata
    meta = sample["meta.json"]

    # task trial events in BIDS events format.
    events = sample["events.json"]

    # sparse data mask.
    mask = sample["mask.npz"]
    mask = scipy.sparse.coo_array(
        (mask["data"], (mask["row"], mask["col"])), shape=mask["shape"]
    ).toarray()

    # fMRI bold data, shape (T, D)
    image_values = sample["bold.npy"]

    # unmask to image, shape (T, H, W). mask encoded as zeros.
    image = np.zeros((len(image_values), *mask.shape), dtype=image_values.dtype)
    image[:, mask] = image_values
    return {"meta": meta, "events": events, "image": image}


def random_clips(num_frames: int = 16, oversample: float = 1.0):
    """Webdataset filter to generate random clips.

    The number of clips is `oversample * T / num_frames`.
    """
    def _filter(dataset: Iterable[dict[str, Any]]):
        for sample in dataset:
            image = sample["image"]
            n_clips = int(oversample * len(image) / num_frames)
            indices = np.sort(
                np.random.randint(0, len(image) - num_frames + 1, size=n_clips)
            )
            for start in indices:
                # copy to avoid a memory leak when used with a shuffle buffer.
                clip = image[start : start + num_frames].copy()
                yield {**sample, "image": clip, "start": start}

    return _filter


def sequential_clips(num_frames: int = 16, stride: int | None = None):
    """Webdataset filter to generate sequential clips.

    By default, stride = num_frames.
    """
    stride = stride or num_frames

    def _filter(dataset: Iterable[dict[str, Any]]):
        for sample in dataset:
            image = sample["image"]
            for start in range(0, len(image) - num_frames + 1, stride):
                clip = image[start : start + num_frames].copy()
                yield {**sample, "image": clip, "start": start}

    return _filter


def event_clips(num_frames: int = 16, tr: float = 1.0, hrf_delay: float = 0.0):
    """Webdataset filter to generate event-locked clips.

    tr and hrf_delay are in seconds. A 1s tr is the default for flat datasets. Setting
    hrf_delay > 0, e.g. to 3 or 4 seconds can concentrate the clip more on the
    activation peak.
    """
    def _filter(dataset: Iterable[dict[str, Any]]):
        for sample in dataset:
            image = sample["image"]
            events = sample["events"]
            for event in events:
                start = int((event["onset"] + hrf_delay) / tr)
                if start + num_frames > len(image):
                    continue
                clip = image[start : start + num_frames].copy()
                yield {**sample, "image": clip, "start": start, **event}

    return _filter


CLIPPING_REGISTRY = {
    "random": random_clips,
    "sequential": sequential_clips,
    "event": event_clips,
}


def make_clipping(clipping: str, **kwargs) -> Callable:
    clip_fn = CLIPPING_REGISTRY[clipping]
    kwargs = filter_kwargs(clip_fn, kwargs)
    return clip_fn(**kwargs)


def make_flat_transform(
    clip_vmax: float | None = 3.0,
    normalize: Literal["global", "frame"] | None = None,
    masking: str | None = None,
    masking_kwargs: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Make sample transform for flat map data.
    
    If `normalize='global'`, globally normalizes the clip to mean zero unit variance. If
    `normalize='frame'`, each temporal is independently normalized. fMRI time series
    have slow global variation. By normalizing, we can remove some of this.
    """
    if masking:
        masking_kwargs = masking_kwargs or {}
        mask_fn = make_masking(masking, **masking_kwargs)
    else:
        mask_fn = None
    
    if normalize:
        norm_dim = {"global": None, "frame": -1}[normalize]
        norm_fn = partial(apply_normalize, dim=norm_dim)
    else:
        norm_fn = None

    def transform(sample: dict[str, Any]):
        # (T, H, W)
        image = sample["image"]
        image = torch.from_numpy(image).float()

        # assume mask coded as zeros.
        mask = image[0] != 0

        if norm_fn is not None:
            image = norm_fn(image, mask)

        # clip extreme values.
        if clip_vmax and clip_vmax > 0:
            image = torch.clamp(image, min=-clip_vmax, max=clip_vmax)

        if mask_fn is not None:
            visible_mask = mask_fn(mask)
            visible_mask = visible_mask * mask
        else:
            visible_mask = None

        # TODO: we may soon want to do cropping. to support this, could stack image with
        # mask channel wise to apply crops and then chunk back after.

        # (C, T, H, W)
        image = image[None]
        mask = mask[None, None]
        if visible_mask is not None:
            visible_mask = visible_mask[None, None]

        # drop other keys bc they may be hard to collate.
        # later we may want to keep some of the other data, or include targets.
        sample = {"image": image, "mask": mask}
        if visible_mask is not None:
            sample["visible_mask"] = visible_mask
        return sample

    return transform


def hemi_masking(mask: torch.Tensor) -> torch.Tensor:
    """Mask out left or right hemisphere.

    Assumes the hemispheres are in the left and right image halves.
    """
    H, W = mask.shape
    visible_mask = torch.ones_like(mask)
    if np.random.rand() < 0.5:
        # lh visible
        visible_mask[:, W // 2:] = 0
    else:
        # rh visible
        visible_mask[:, :W // 2] = 0
    return visible_mask


def inverse_block_masking(mask: torch.Tensor, block_size: int = 160) -> torch.Tensor:
    """Sample a block visible mask."""
    H, W = mask.shape
    h_idx = np.random.randint(0, H - block_size)
    w_idx = np.random.randint(0, W - block_size)

    visible_mask = torch.zeros_like(mask)
    visible_mask[h_idx: h_idx + block_size, w_idx: w_idx + block_size] = 1
    return visible_mask


def hemi_inverse_block_masking(
    mask: torch.Tensor, block_size: int = 160
) -> torch.Tensor:
    """Sample a block visible mask constrained to one hemisphere."""
    H, W = mask.shape
    if np.random.rand() < 0.5:
        # lh block
        w_start, w_stop = 0, W // 2
    else:
        # rh block
        w_start, w_stop = W // 2, W
    h_idx = np.random.randint(0, H - block_size)
    w_idx = np.random.randint(w_start, w_stop - block_size)

    visible_mask = torch.zeros_like(mask)
    visible_mask[h_idx: h_idx + block_size, w_idx: w_idx + block_size] = 1
    return visible_mask


# TODO: Other masking strategies to try:
#   - Network masking. Get Schaefer 7 network label map. Mask out networks with some
#     probability p.
#   - Surface neighborhood masking. Get XYZ coordinates of flat mask, eg for inflated
#     surface. Get euclidean neighborhood around random point for visible area.
#     Generalization of inverse block masking with roll to closed cortical surface.
#   - Block constrained to fit mostly in mask. Sample a corner contained in the mask,
#     then decide which of the four box corners it is to maximize overlap.

MASKING_REGISTRY = {
    "hemi": hemi_masking,
    "inverse_block": inverse_block_masking,
    "hemi_inverse_block": hemi_inverse_block_masking,
}


def make_masking(masking: str, **kwargs) -> Callable:
    mask_fn = MASKING_REGISTRY[masking]
    kwargs = filter_kwargs(mask_fn, kwargs)
    mask_fn = partial(mask_fn, **kwargs)
    return mask_fn


def apply_normalize(
    image: torch.Tensor, mask: torch.Tensor, dim: int | None = None, eps: float = 1e-6
) -> torch.Tensor:
    image_values = image[..., mask > 0]
    mean = image_values.mean(dim=dim, keepdim=True).unsqueeze(-1)
    std = image_values.std(dim=dim, keepdim=True).unsqueeze(-1)
    image = (image - mean) / (std + eps)
    image = image * mask
    return image


def pad_to_multiple(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    H, W = img.shape[-2:]
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    img = TF.pad(img, padding)
    return img


def filter_kwargs(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs
