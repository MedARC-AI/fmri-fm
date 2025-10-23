import glob
import os
import random
import tarfile
from io import BytesIO
from logging import getLogger
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


logger = getLogger()


def _load_npy_from_tar(tf: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    f = tf.extractfile(member)
    assert f is not None, f"Failed to extract {member.name}"
    try:
        return np.load(f, allow_pickle=False)
    except Exception:
        try:
            f.seek(0)
        except Exception:
            pass
        data = f.read()
        return np.load(BytesIO(data), allow_pickle=False)
    finally:
        f.close()


def _uniform_sample_indices(length: int, num_samples: int) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if num_samples > length:
        # If shorter than target, repeat last index
        idx = np.linspace(0, length - 1, num_samples)
    else:
        idx = np.linspace(0, length - 1, num_samples)
    return np.clip(idx.round().astype(int), 0, length - 1)


class HCPDataset(Dataset):
    def __init__(
        self,
        tar_glob: str,
        roi_count_total: int = 450,
        roi_count_keep: int = 400,
        drop_first_rois: int = 50, #drop to keep the cortical ROIs only
        num_frames: int = 160,
        downsample: bool = True,
        sampling_rate: int = 3,
        seq_length: int = 490,
        params_file: str = "normalization_params_hcp_train.npz",
        use_standatdization: bool = False,
    ) -> None:
        super().__init__()
        self.tar_paths: List[str] = sorted(glob.glob(tar_glob))
        if not self.tar_paths:
            raise FileNotFoundError(f"No shards matched: {tar_glob}")
        self.roi_count_total = roi_count_total
        self.roi_count_keep = roi_count_keep
        self.drop_first_rois = drop_first_rois
        self.num_frames = num_frames
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.seq_length = seq_length
        self.use_standatdization = use_standatdization
        # normalization params colocated with shards by default unless absolute path provided
        base_dir = os.path.dirname(self.tar_paths[0]) if self.tar_paths else os.getcwd()
        self.params_file = params_file if os.path.isabs(params_file) else os.path.join(base_dir, params_file)
        os.makedirs(os.path.dirname(self.params_file), exist_ok=True)

        # Build index of (tar_path, member_name) for all .bold.npy
        index: List[Tuple[str, str]] = []
        for tar_path in self.tar_paths:
            try:
                with tarfile.open(tar_path) as tf:
                    for m in tf.getmembers():
                        if m.name.endswith(".bold.npy"):
                            index.append((tar_path, m.name))
            except Exception as e:
                logger.error(f"Failed to read shard {tar_path}: {e}")
        if not index:
            raise RuntimeError("No .bold.npy files found in provided shards")
        self.index = index
        logger.info(f"HCPDataset indexed {len(self.index)} samples across {len(self.tar_paths)} shards")

        # Load or compute dataset-level robust scaling parameters
        self.normalization_params = self._load_or_compute_normalization_params()

    def _get_start_end_idx(self, fmri_size: int, clip_size: int) -> Tuple[int, int]:
        "Reference: https://github.com/facebookresearch/mae_st"
        """
        Sample a clip of size clip_size from a video of size video_size and
        return the indices of the first and last frame of the clip. If clip_idx is
        -1, the clip is randomly sampled, otherwise uniformly split the video to
        num_clips clips, and select the start and end index of clip_idx-th video
        clip.
        Args:
            video_size (int): number of overall frames.
            clip_size (int): size of the clip to sample from the frames.
            clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
                clip_idx is larger than -1, uniformly split the video to num_clips
                clips, and select the start and end index of the clip_idx-th video
                clip.
            num_clips (int): overall number of clips to uniformly sample from the
                given video for testing.
        Returns:
            start_idx (int): the start frame index.
            end_idx (int): the end frame index.
        """
        delta = max(fmri_size - clip_size, 0)
        start_idx = random.uniform(0, delta)
        end_idx = start_idx + clip_size - 1
        return int(start_idx), int(end_idx)

    def _temporal_sampling(self, frames: torch.Tensor, start_idx: int, end_idx: int, num_samples: int) -> torch.Tensor:
        """
        Given the start and end frame index, sample num_samples frames between
        the start and end with equal interval.
        Args:
            frames (tensor): a tensor of video frames, dimension is
                `num video frames` x `channel` x `height` x `width`.
            start_idx (int): the index of the start frame.
            end_idx (int): the index of the end frame.
            num_samples (int): number of frames to sample.
        Returns:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `num clip frames` x `channel` x `height` x `width`.
        """
        index = torch.linspace(float(start_idx), float(end_idx), int(num_samples))
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        return torch.index_select(frames, 1, index)

    def save_normalization_params(self, medians: np.ndarray, iqrs: np.ndarray, filename: str | None = None) -> None:
        """Optionally save normalization params to CSV (for inspection).

        If filename is None, writes next to params_file with .csv extension.
        """
        if filename is None:
            base, _ = os.path.splitext(self.params_file)
            filename = base + ".csv"
        import pandas as pd
        df = pd.DataFrame({
            'roi_index': np.arange(len(medians), dtype=int),
            'median': medians.astype(float),
            'iqr': iqrs.astype(float),
        })
        df.to_csv(filename, index=False)
        logger.info(f"Normalization parameters saved to {filename}")

    def _load_or_compute_normalization_params(self):
        if os.path.exists(self.params_file):
            params_df = np.load(self.params_file)
            medians = params_df['medians']
            iqrs = params_df['iqrs']
            logger.info("Normalization parameters loaded from file.")
            return {'medians': medians, 'iqrs': iqrs}
        else:
            return self._compute_normalization_params()

    def _compute_normalization_params(self):
        """Compute dataset-level per-ROI median/IQR over subject means.

        Pipeline mirrors UKB:
          - For each sample: load (T,450) or (450,T), transpose to ROI-first,
            drop first 50 to keep 400 cortical, optionally truncate to seq_length,
            then take per-ROI mean over time -> shape (400,).
          - Across samples: median and IQR (p75 - p25) per ROI.
        """
        all_data_mean = []
        for tar_path, member_name in self.index:
            try:
                with tarfile.open(tar_path) as tf:
                    member = tf.getmember(member_name)
                    arr = _load_npy_from_tar(tf, member)
                if arr.ndim != 2:
                    continue
                h, w = arr.shape
                if h == self.roi_count_total:
                    arr_roi_t = arr
                elif w == self.roi_count_total:
                    arr_roi_t = arr.T
                else:
                    continue
                arr_roi_t = arr_roi_t[self.drop_first_rois : self.drop_first_rois + self.roi_count_keep, :]
                if arr_roi_t.shape[0] != self.roi_count_keep:
                    continue
                if self.seq_length is not None and arr_roi_t.shape[1] > self.seq_length:
                    arr_roi_t = arr_roi_t[:, : self.seq_length]
                temp_mean = np.mean(arr_roi_t, axis=1).astype(np.float32)  # (400,)
                all_data_mean.append(temp_mean)
            except Exception as e:
                logger.error(f"Failed to process {tar_path}:{member_name} for normalization: {e}")
                continue

        if not all_data_mean:
            # Fallback to zeros/ones to avoid crash, though training will be unnormalized.
            medians = np.zeros((self.roi_count_keep,), dtype=np.float32)
            iqrs = np.ones((self.roi_count_keep,), dtype=np.float32)
        else:
            all_data_mean = np.stack(all_data_mean)
            medians = np.median(all_data_mean, axis=0).astype(np.float32)
            iqrs = (np.percentile(all_data_mean, 75, axis=0) - np.percentile(all_data_mean, 25, axis=0)).astype(np.float32)
            iqrs = np.clip(iqrs, 1e-6, None)

        np.savez(self.params_file, medians=medians, iqrs=iqrs)
        logger.info(f"Saved normalization params to {self.params_file}")
        return {'medians': medians, 'iqrs': iqrs}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        tar_path, member_name = self.index[idx]
        with tarfile.open(tar_path) as tf:
            member = tf.getmember(member_name)
            arr = _load_npy_from_tar(tf, member)

        # Expect time-first or roi-first with total ROI count
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.shape} in {member_name}")

        h, w = arr.shape
        if h == self.roi_count_total:
            # already ROI-first (450, T)
            arr_roi_t = arr
        elif w == self.roi_count_total:
            # time-first (T, 450) -> transpose
            arr_roi_t = arr.T
        else:
            raise ValueError(
                f"File {member_name} has shape {arr.shape} not matching expected ROI count {self.roi_count_total}"
            )

        # Slice cortical only: drop first 50, keep 400
        arr_roi_t = arr_roi_t[self.drop_first_rois : self.drop_first_rois + self.roi_count_keep, :]
        if arr_roi_t.shape[0] != self.roi_count_keep:
            raise ValueError(
                f"After slicing, got {arr_roi_t.shape}; expected ({self.roi_count_keep}, T)"
            )

        # Optional truncate to seq_length before normalization/sampling
        if self.seq_length is not None and arr_roi_t.shape[1] > self.seq_length:
            arr_roi_t = arr_roi_t[:, : self.seq_length]

        # Normalize (dataset-level robust scaling, like UKB). Optionally per-sample standardization.
        ts_array = arr_roi_t.astype(np.float32)
        if not self.use_standatdization:
            med = self.normalization_params['medians']
            iqr = self.normalization_params['iqrs']
            ts_array = (ts_array - med[:, None]) / iqr[:, None]
        ts = torch.from_numpy(ts_array).to(torch.float32)  # (400, T)
        if self.use_standatdization:
            mean = ts.mean()
            std = ts.std()
            if float(std) == 0.0:
                std = torch.tensor(1.0, dtype=ts.dtype)
            ts = (ts - mean) / std

        # Temporal sampling to fixed num_frames
        T = ts.size(1)
        if self.downsample:
            clip_size = self.sampling_rate * self.num_frames
            start_idx, end_idx = self._get_start_end_idx(T, clip_size)
            ts = self._temporal_sampling(ts, start_idx, end_idx, self.num_frames)
        else:
            idxs = _uniform_sample_indices(T, self.num_frames)
            ts = ts[:, idxs]

        # Add channel dim -> (1, 400, num_frames)
        ts = ts.unsqueeze(0)
        return {"fmri": ts}


def make_hcp(
    tar_glob: str,
    batch_size: int,
    collator=None,
    pin_mem: bool = True,
    num_workers: int = 8,
    world_size: int = 1,
    rank: int = 0,
    drop_last: bool = True,
    downsample: bool = True,
    sampling_rate: int = 3,
    seq_length: int = 490,
    params_file: str = "normalization_params_hcp_train.npz",
    use_standatdization: bool = False,
):
    dataset = HCPDataset(
        tar_glob=tar_glob,
        roi_count_total=450,
        roi_count_keep=400,
        drop_first_rois=50,# drop subcortical ROIs
        num_frames=160 if downsample else 160,
        downsample=downsample,
        sampling_rate=sampling_rate,
        seq_length=seq_length,
        params_file=params_file,
        use_standatdization=use_standatdization,
    )
    logger.info("HCP dataset created")

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    logger.info("HCP unsupervised data loader created")
    return dataset, data_loader, dist_sampler