"""
HCP Sex Classification Dataset for Parcellated fMRI Data

This dataset loader is designed to work with Brain-JEPA pre-extracted .pt files
and integrate with the flat-map evaluation framework for linear probe evaluation.

Each .pt file contains:
    {
        'fmri': torch.Tensor,           # Shape: (400, T) - cortical ROIs only
        'subject_id': int,              # e.g., 349244
        'task': str,                    # e.g., 'RELATIONAL'
        'direction': str,               # e.g., 'RL' or 'LR'
        'run': str,                     # e.g., 'RELATIONAL_RL'
        'original_shape': tuple,        # e.g., (167, 450) - BEFORE processing
        'tar_source': str               # e.g., 'hcp-parc_0000.tar'
    }

The sex labels are stored in hcp_sex_target_id_map.json mapping subject_id -> sex (0/1).
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HCPSexParcDataset(Dataset):
    """
    HCP Sex Classification Dataset using parcellated fMRI data (.pt files).
    
    This dataset is compatible with the flat-map evaluation framework and provides
    data in the format expected by the linear probe evaluation code.
    
    Args:
        pt_dir: Directory containing .pt files with parcellated fMRI data
        sex_labels_path: Path to hcp_sex_target_id_map.json
        split: One of 'train', 'val', 'test'
        split_seed: Random seed for train/val/test split
        split_ratios: Tuple of (train, val, test) ratios, default (0.7, 0.15, 0.15)
        num_frames: Number of timepoints to sample per clip
        transform: Optional transform function
        clip_strategy: 'random' or 'first' - how to sample the temporal clip
    """
    
    def __init__(
        self,
        pt_dir: str | Path,
        sex_labels_path: str | Path = 's3://medarc/fmri-fm/datasets/hcp_sex_target_id_map.json',
        split: str = 'train',
        split_seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        num_frames: int = 160,
        transform: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
        clip_strategy: str = 'random',
    ):
        self.pt_dir = Path(pt_dir)
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.clip_strategy = clip_strategy
        
        # Load sex labels
        logger.info(f"Loading sex labels from {sex_labels_path}")
        self.sex_labels = self._load_sex_labels(sex_labels_path)
        
        # Get all .pt files
        all_pt_files = sorted(self.pt_dir.glob("*.pt"))
        logger.info(f"Found {len(all_pt_files)} total .pt files")
        
        # Load metadata and filter by subject_id with sex labels
        logger.info("Loading metadata and filtering by sex labels...")
        self.samples: List[Dict[str, Any]] = []
        subjects_with_labels = set(self.sex_labels.keys())
        
        for pt_file in all_pt_files:
            try:
                # Load only metadata (fast, doesn't load full tensor into memory)
                data = torch.load(pt_file, map_location='cpu', weights_only=True)
                subject_id = str(data['subject_id'])  # Convert to string for JSON key matching
                
                # Check if subject has sex label
                if subject_id in subjects_with_labels:
                    n_timeframes = data['fmri'].shape[1]
                    
                    # Only include samples with enough timeframes
                    if n_timeframes >= self.num_frames:
                        self.samples.append({
                            'pt_file': pt_file,
                            'subject_id': subject_id,
                            'n_timeframes': n_timeframes,
                            'task': data.get('task', 'unknown'),
                            'sex': self.sex_labels[subject_id]
                        })
            except Exception as e:
                logger.warning(f"Error loading {pt_file.name}: {e}")
        
        logger.info(f"Found {len(self.samples)} samples with sex labels and >= {num_frames} frames")
        
        # Create train/val/test split by subject (not by sample)
        self.samples = self._create_split(self.samples, split, split_seed, split_ratios)
        logger.info(f"{split.upper()} split: {len(self.samples)} samples")
        
    def _load_sex_labels(self, sex_labels_path: str | Path) -> Dict[str, int]:
        """Load sex labels from JSON file (local or S3)."""
        sex_labels_path = str(sex_labels_path)
        
        if sex_labels_path.startswith('s3://'):
            # Load from S3
            try:
                import boto3
                s3 = boto3.client('s3')
                
                # Parse S3 path
                parts = sex_labels_path.replace('s3://', '').split('/', 1)
                bucket = parts[0]
                key = parts[1]
                
                # Download and parse
                response = s3.get_object(Bucket=bucket, Key=key)
                sex_labels = json.loads(response['Body'].read().decode('utf-8'))
            except ImportError:
                raise ImportError(
                    "boto3 is required to load sex labels from S3. "
                    "Install it with: pip install boto3\n"
                    "Or download the file locally and provide a local path."
                )
        else:
            # Load from local file
            with open(sex_labels_path, 'r') as f:
                sex_labels = json.load(f)
        
        logger.info(f"Loaded sex labels for {len(sex_labels)} subjects")
        return sex_labels
    
    def _create_split(
        self,
        samples: List[Dict[str, Any]],
        split: str,
        seed: int,
        ratios: tuple[float, float, float]
    ) -> List[Dict[str, Any]]:
        """
        Create train/val/test split by subject (to avoid data leakage).
        All runs from the same subject go into the same split.
        """
        # Group samples by subject
        subject_to_samples = {}
        for sample in samples:
            subject_id = sample['subject_id']
            if subject_id not in subject_to_samples:
                subject_to_samples[subject_id] = []
            subject_to_samples[subject_id].append(sample)
        
        # Get unique subjects and shuffle
        subjects = sorted(subject_to_samples.keys())
        rng = np.random.RandomState(seed)
        rng.shuffle(subjects)
        
        # Split subjects
        n_subjects = len(subjects)
        train_ratio, val_ratio, test_ratio = ratios
        assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train:n_train + n_val])
        test_subjects = set(subjects[n_train + n_val:])
        
        logger.info(f"Split subjects: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test")
        
        # Select samples based on split
        split_map = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }
        
        selected_subjects = split_map[split]
        split_samples = []
        for subject_id in selected_subjects:
            split_samples.extend(subject_to_samples[subject_id])
        
        return split_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample dict compatible with Brain-JEPA parcellated model:
        {
            'image': torch.Tensor,  # Shape: (1, 400, 160) - Brain-JEPA format (C, H, W)
            'mask': torch.Tensor,   # Shape: (1, 400, 160) - all ones (no masking)
            'target': int,          # Sex label (0 or 1)
            'subject_id': str,      # For reference
            'task': str,            # For reference
        }
        
        NOTE: Brain-JEPA expects (C, H, W) = (1, 400 ROIs, 160 timepoints)
        This is different from MAE flat-maps which expect (C, T, H, W) image format!
        """
        sample_info = self.samples[idx]
        
        # Load fMRI data
        data = torch.load(sample_info['pt_file'], map_location='cpu', weights_only=True)
        fmri = data['fmri']  # Shape: (400, T)
        
        # Sample temporal clip
        T = fmri.shape[1]
        if self.clip_strategy == 'random':
            if T > self.num_frames:
                start_idx = torch.randint(0, T - self.num_frames + 1, (1,)).item()
                fmri = fmri[:, start_idx:start_idx + self.num_frames]
        elif self.clip_strategy == 'first':
            fmri = fmri[:, :self.num_frames]
        else:
            raise ValueError(f"Unknown clip_strategy: {self.clip_strategy}")
        
        # Ensure exactly num_frames timepoints
        if fmri.shape[1] < self.num_frames:
            # Pad if necessary (shouldn't happen if filtered correctly)
            padding = self.num_frames - fmri.shape[1]
            fmri = torch.nn.functional.pad(fmri, (0, padding), mode='replicate')
        
        # Brain-JEPA format: (C, H, W) = (1, 400, 160)
        # where C=1 channel, H=400 ROIs, W=160 timepoints
        image = fmri.unsqueeze(0)  # Shape: (1, 400, 160)
        
        # Create mask (all ones, no masking for parcellated data)
        mask = torch.ones_like(image)  # Shape: (1, 400, 160)
        
        # Get sex label
        target = torch.tensor(sample_info['sex'], dtype=torch.long)
        
        sample_dict = {
            'image': image,
            'mask': mask,
            'target': target,
            'subject_id': sample_info['subject_id'],
            'task': sample_info['task'],
        }
        
        # Apply transform if provided
        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict


def make_hcp_sex_parc_datasets(
    pt_dir: str | Path,
    sex_labels_path: str | Path = 's3://medarc/fmri-fm/datasets/hcp_sex_target_id_map.json',
    split_seed: int = 42,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    num_frames: int = 160,
    transform: Callable[[Dict[str, Any]], Dict[str, Any]] = None,
    clip_strategy: str = 'random',
) -> tuple[HCPSexParcDataset, HCPSexParcDataset, HCPSexParcDataset]:
    """
    Convenience function to create train/val/test datasets.
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = HCPSexParcDataset(
        pt_dir=pt_dir,
        sex_labels_path=sex_labels_path,
        split='train',
        split_seed=split_seed,
        split_ratios=split_ratios,
        num_frames=num_frames,
        transform=transform,
        clip_strategy=clip_strategy,
    )
    
    val_dataset = HCPSexParcDataset(
        pt_dir=pt_dir,
        sex_labels_path=sex_labels_path,
        split='val',
        split_seed=split_seed,
        split_ratios=split_ratios,
        num_frames=num_frames,
        transform=transform,
        clip_strategy=clip_strategy,
    )
    
    test_dataset = HCPSexParcDataset(
        pt_dir=pt_dir,
        sex_labels_path=sex_labels_path,
        split='test',
        split_seed=split_seed,
        split_ratios=split_ratios,
        num_frames=num_frames,
        transform=transform,
        clip_strategy=clip_strategy,
    )
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hcp_sex_parc.py <pt_dir>")
        sys.exit(1)
    
    pt_dir = sys.argv[1]
    
    print("Creating datasets...")
    train_ds, val_ds, test_ds = make_hcp_sex_parc_datasets(pt_dir)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    
    print(f"\nLoading first training sample...")
    sample = train_ds[0]
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Target: {sample['target'].item()}")
    print(f"  Subject ID: {sample['subject_id']}")
    print(f"  Task: {sample['task']}")
    
    print("\n✅ Dataset test passed!")

