# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import fnmatch
import inspect
import json
from glob import glob
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import braceexpand
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import torchvision.transforms.v2.functional as TF
import torchvision.tv_tensors as tvt
import scipy.sparse
import webdataset as wds
from torch.utils.data import Dataset
from huggingface_hub.utils import get_token

def make_flat_wds_dataset(
    url: str | list[str],
    num_frames: int = 16,
    clipping: str = "random",
    clipping_kwargs: dict[str, Any] | None = None,
    target_id_map: dict[str, int] | str | Path | None = None,
    target_key: str = "trial_type",
    select_files_pattern: str | None = None,
    shuffle: bool = True,
    buffer_size: int = 1000,
    stream_hf: bool = False,
) -> wds.WebDataset:
    """Make fMRI flat map dataset."""
    if select_files_pattern:
        select_files = make_select_files(select_files_pattern)
    else:
        select_files = None

    # resampling creates an infinite stream of shards sampled with replacement,
    # guaranteeing that no process runs out of data early in distributed training.
    # see webdataset FAQ: https://github.com/webdataset/webdataset/blob/main/FAQ.md
    if stream_hf:
        hf_token = get_token()
        assert hf_token, "No HF token found; run hf auth login"
        url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}' --retry 10 --retry-delay 1 --retry-max-time 60 --retry-all-errors"
    dataset = wds.WebDataset(
        expand_urls(url),
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
        select_files=select_files,
    )
    dataset = dataset.decode().map(extract_flat_sample)

    # generate clips before shuffling for slightly better mixing.
    clipping_kwargs = clipping_kwargs or {}
    clip_fn = make_clipping(clipping, num_frames=num_frames, **clipping_kwargs)
    dataset = dataset.compose(clip_fn)

    # add targets
    if target_id_map is not None:
        dataset = dataset.compose(with_targets(target_id_map, target_key=target_key))

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset


def expand_urls(urls: str | list[str]) -> list[str]:
    """
    Expand wds urls:

    - expand glob patterns
    - expand brace expressions
    - filter files that don't exist

    Adapted from `webdataset.shardlists.expand_urls`.
    """
    if isinstance(urls, str):
        urls = [urls]
    results = []
    for url in urls:
        chars = set(url)
        if chars.intersection("[*?"):
            result = sorted(glob(url))
        elif "{" in chars:
            result = braceexpand.braceexpand(url)
        else:
            result = [url]
        results.extend(result)
    return results


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
        self.files = sorted(p.name for p in self.root.glob("*.pt"))
        self.transform = transform

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.root / self.files[idx]
        sample = torch.load(path, weights_only=True)
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
            indices = np.sort(np.random.randint(0, len(image) - num_frames + 1, size=n_clips))
            for start in indices:
                # copy to avoid a memory leak when used with a shuffle buffer.
                clip = image[start : start + num_frames].copy()

                yield {
                    "__key__": sample["__key__"],
                    **sample["meta"],
                    "image": clip,
                    "start": start,
                }

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

                yield {
                    "__key__": sample["__key__"],
                    **sample["meta"],
                    "image": clip,
                    "start": start,
                }

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

                yield {
                    "__key__": sample["__key__"],
                    **sample["meta"],
                    "image": clip,
                    "start": start,
                    **event,
                }

    return _filter


CLIPPING_DICT = {
    "random": random_clips,
    "sequential": sequential_clips,
    "event": event_clips,
}


def make_clipping(clipping: str, **kwargs) -> Callable:
    clip_fn = CLIPPING_DICT[clipping]
    kwargs = filter_kwargs(clip_fn, kwargs)
    return clip_fn(**kwargs)


def with_targets(
    target_id_map: dict[str, int] | str | Path | None = None,
    target_key: str = "trial_type",
):
    """Webdataset filter to augment samples with targets."""

    if isinstance(target_id_map, (str, Path)):
        target_id_map = load_target_id_map(target_id_map)

    def _filter(dataset: Iterable[dict[str, Any]]):
        for sample in dataset:
            label = sample.get(target_key)
            if label not in target_id_map:
                continue
            target = target_id_map[label]
            yield {**sample, "target": target}

    return _filter


def load_target_id_map(target_id_map: Path) -> dict[Any, int]:
    target_id_map = Path(target_id_map)
    if target_id_map.suffix == ".json":
        with open(target_id_map) as f:
            target_id_map = json.load(f)
    elif target_id_map.suffix == ".npy":
        target_id_map = np.load(target_id_map)
    else:
        raise ValueError(f"Unsupported target_id_map {target_id_map}.")
    return target_id_map


def make_select_files(select_files_pattern: str) -> Callable[[str], bool]:
    def _filter(fname: str):
        return fnmatch.fnmatch(fname, select_files_pattern)

    return _filter


def make_flat_transform(
    img_size: tuple[int, int] | None = None,
    clip_vmax: float | None = 3.0,
    normalize: Literal["global", "frame"] | None = None,
    random_crop: bool = False,
    crop_kwargs: dict[str, Any] | None = None,
    target_id_map: dict[str, int] | str | Path | None = None,
    target_key: str | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Make sample transform for flat map data.

    Args:
        img_size: target image size. If input image doesn't match target size, it will
            be padded around the edges.
        clip_vmax: max abs value to clip at
        normalize: If `normalize='global'`, globally normalizes the clip to mean zero
            unit variance. If `normalize='frame'`, each temporal frame is independently
            normalized.
        random_crop: enable random resize crop augmentation
        crop_kwargs: kwargs to pass to RandomResizeCrop
        target_id_map: mapping from sample target key to targets
        target_key: sample key for the prediction target
    """
    if random_crop:
        crop_fn = v2.RandomResizedCrop(size=img_size, **crop_kwargs)
    else:
        crop_fn = None

    if normalize:
        norm_dim = {"global": None, "frame": -1}[normalize]
        norm_fn = partial(apply_normalize, dim=norm_dim)
    else:
        norm_fn = None

    if target_id_map is not None:
        if isinstance(target_id_map, (str, Path)):
            target_id_map = load_target_id_map(target_id_map)

    def transform(sample: dict[str, Any]):
        # (T, H, W)
        image = sample["image"]
        image = torch.as_tensor(image).float()

        # pad to a fixed size (that is divisible by patch size)
        if img_size:
            image = pad_to_size(image, img_size)

        # assume mask coded as zeros, and shared across time.
        mask = (image[0] != 0).float()

        if crop_fn is not None:
            image, mask = crop_fn(image, tvt.Mask(mask))

        if norm_fn is not None:
            image = norm_fn(image, mask)

        # clip extreme values.
        if clip_vmax and clip_vmax > 0:
            image = torch.clamp(image, min=-clip_vmax, max=clip_vmax)

        # (C, T, H, W)
        image = image[None]

        sample_ = {"image": image, "img_mask": mask}

        if target_id_map is not None:
            key = sample[target_key]
            target = torch.as_tensor(target_id_map[key])
            target = target.float() if target.is_floating_point() else target.long()
            sample_["target"] = target
        elif "target" in sample:
            sample_["target"] = sample["target"]
        return sample_

    return transform


def apply_normalize(
    image: torch.Tensor, mask: torch.Tensor, dim: int | None = None, eps: float = 1e-6
) -> torch.Tensor:
    image_values = image[..., mask > 0]
    mean = image_values.mean(dim=dim, keepdim=True).unsqueeze(-1)
    std = image_values.std(dim=dim, keepdim=True).unsqueeze(-1)
    image = (image - mean) / (std + eps)
    image = image * mask
    return image


def pad_to_size(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    H, W = img.shape[-2:]
    H_new, W_new = size
    pad_h = max(H_new - H, 0)
    pad_w = max(W_new - W, 0)
    if pad_h == pad_w == 0:
        return img
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    img = TF.pad(img, padding)
    return img


def filter_kwargs(func: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    sigature = inspect.signature(func)
    kwargs = {k: v for k, v in kwargs.items() if k in sigature.parameters}
    return kwargs
