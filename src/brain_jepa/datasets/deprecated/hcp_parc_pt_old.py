"""
HCP Parcellated Dataset Loader - for pre-extracted .pt files.

Expects a directory of .pt files, each containing:
    {
        'fmri': torch.Tensor (shape: 400 x T, already normalized),
        'subject_id': str (optional metadata),
        'run': str (optional metadata),
        ...
    }

This loader applies temporal sampling to get fixed-length clips for training.
"""

import glob
import os
import random
from logging import getLogger
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

logger = getLogger()


class HCPParcPTDataset(Dataset):
    """
    Dataset for HCP parcellated timeseries stored as pre-extracted .pt files.
    
    Each .pt file should contain a dict with key 'fmri' -> tensor of shape (H, T).
    The data is already normalized (robust scaling applied during extraction).
    
    This loader only handles temporal sampling to get fixed-length clips.
    """
    
    def __init__(
        self,
        pt_dir: str,
        num_frames: int = 100, # take first 100 frames so we dont need to interpolate
        downsample: bool = False, # Changed to False - we want simple random clip sampling
        sampling_rate: int = 3, # Keep for backward compatibility but not used when downsample=False
        roi_count: int = 400,
        use_standardization: bool = False, # per sample standardization. Left for comparison with ukbiobank_scale.py. DONT USE THIS.
    ) -> None:
        """
        Args:
            pt_dir: Directory containing .pt files
            num_frames: Target number of frames to sample (default: 160)
            downsample: If True, use temporal jitter + uniform sampling (MAE_ST style). 
                       If False, simple random clip sampling (default: False)
            sampling_rate: Sampling rate for temporal jitter (only used when downsample=True)
            roi_count: Expected number of ROIs (for validation)
            use_standardization: If True, apply per-sample standardization after temporal sampling
        """
        super().__init__()
        self.pt_dir = Path(pt_dir)
        if not self.pt_dir.exists():
            raise FileNotFoundError(f"PT directory not found: {pt_dir}")
        
        self.num_frames = num_frames
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.roi_count = roi_count
        self.use_standardization = use_standardization
        
        # Find all .pt files
        self.pt_files: List[Path] = sorted(self.pt_dir.glob("*.pt"))
        if not self.pt_files:
            raise RuntimeError(f"No .pt files found in {pt_dir}")
        
        logger.info(f"HCPParcPTDataset: found {len(self.pt_files)} samples in {pt_dir}")
    
    def __len__(self) -> int:
        return len(self.pt_files)
    
    def _get_start_end_idx(self, fmri_size: int, clip_size: int) -> tuple[int, int]:
        """
        Sample a clip with temporal jitter (MAE_ST style).
        Temporal jitter: Data augmentation technique to randomly shift the temporal window we sample from
        Reference: https://github.com/facebookresearch/mae_st
        fmri_size: total length of the original time series. # of time points
        clip_size: size of the clip we want to extract in # of time points
        """
        delta = max(fmri_size - clip_size, 0) #max(total length - clip size, 0)
        start_idx = random.uniform(0, delta) #choose a random start index(timepoint)
        end_idx = start_idx + clip_size - 1
        return int(start_idx), int(end_idx)
    
    def _temporal_sampling(
        self, frames: torch.Tensor, start_idx: int, end_idx: int, num_samples: int
    ) -> torch.Tensor:
        """
        Uniformly sample num_samples frames between start and end (inclusive) we get from _get_start_end_idx.
        Reference: https://github.com/facebookresearch/mae_st
        """
        index = torch.linspace(float(start_idx), float(end_idx), int(num_samples))
        index = torch.clamp(index, 0, frames.shape[1] - 1).long()
        return torch.index_select(frames, 1, index)
    
    def __getitem__(self, idx: int):
        pt_file = self.pt_files[idx]
        
        # Load .pt file
        try:
            data = torch.load(pt_file, map_location='cpu', weights_only=True)
        except Exception:
            # Fallback for older torch.save format
            data = torch.load(pt_file, map_location='cpu')
        
        ts = data['fmri']  # shape: (H, T)
        
        # Validate shape
        if ts.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {ts.shape} in {pt_file}")
        if ts.shape[0] != self.roi_count:
            raise ValueError(
                f"Expected {self.roi_count} ROIs, got {ts.shape[0]} in {pt_file}"
            )
        
        T = ts.shape[1]
        
        
        if self.downsample:
            # MAE_ST style: random jitter + uniform sampling
            # Randomly shift the temporal window we sample from (jitter) then uniformly sample to get num_frames
            clip_size = self.sampling_rate * self.num_frames
            start_idx, end_idx = self._get_start_end_idx(T, clip_size)
            ts = self._temporal_sampling(ts, start_idx, end_idx, self.num_frames)
        else:
            # Take first num_frames timeframes
            if self.num_frames >= T:
                # If shorter than target, interpolate
                idxs = torch.linspace(0, T - 1, self.num_frames).long()
            else:
                # Take first num_frames timeframes
                idxs = torch.arange(self.num_frames)
            ts = ts[:, idxs]
        
        # Optional per-sample standardization (usually not used with robust scaling)
        if self.use_standardization:
            mean = ts.mean()
            std = ts.std()
            if float(std) == 0.0:
                std = torch.tensor(1.0, dtype=ts.dtype)
            ts = (ts - mean) / std
        
        # Add channel dimension: (H, T) -> (1, H, T)
        ts = ts.unsqueeze(0)
        
        return {"fmri": ts}


def make_hcp_pt(
    pt_dir: str,
    batch_size: int,
    collator=None,
    pin_mem: bool = True,
    num_workers: int = 8,
    world_size: int = 1,
    rank: int = 0,
    drop_last: bool = True,
    downsample: bool = False, # Changed to False - simple random clip sampling
    sampling_rate: int = 3,
    use_standardization: bool = False,
):
    """
    Create HCP parcellated PT dataset and dataloader.
    
    Args:
        pt_dir: Directory containing pre-extracted .pt files
        batch_size: Batch size per GPU
        collator: Optional collate function
        pin_mem: Pin memory for faster GPU transfer
        num_workers: Number of dataloader workers
        world_size: Number of distributed processes
        rank: Rank of current process
        drop_last: Drop incomplete batches
        downsample: If True, use temporal jitter + uniform sampling (MAE_ST style).
                   If False, simple random clip sampling (default: False)
        sampling_rate: Sampling rate for temporal jitter (only used when downsample=True)
        use_standardization: Apply per-sample standardization
    """
    dataset = HCPParcPTDataset(
        pt_dir=pt_dir,
        num_frames=160,  # Changed to 160 for pretraining
        downsample=downsample,
        sampling_rate=sampling_rate,
        roi_count=400,
        use_standardization=use_standardization,
    )
    logger.info("HCP PT dataset created")
    
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
    logger.info("HCP PT data loader created")
    
    return dataset, data_loader, dist_sampler


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hcp_parc_pt.py <pt_dir>")
        sys.exit(1) 
    
    pt_dir = sys.argv[1]
    dataset = HCPParcPTDataset(pt_dir=pt_dir, num_frames=160, downsample=True)
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample['fmri'].shape}")
    print(f"Sample dtype: {sample['fmri'].dtype}")
    print(f"Sample range: [{sample['fmri'].min():.4f}, {sample['fmri'].max():.4f}]")

