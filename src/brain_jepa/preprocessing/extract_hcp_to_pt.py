#!/usr/bin/env python3
"""
Extract HCP parcellated timeseries to individual .pt files withor without normalization.

Loads dataset-level robust scaling parameters and applies them during extraction.

Output format per .pt file:
    {
        'fmri': torch.Tensor,      # shape (400, T), float32, normalized
        'subject_id': str,          # extracted from filename
        'run': str,                 # extracted from filename  
        'original_shape': tuple,    # before slicing
        'tar_source': str,          # source tar file
    }

Usage:
    python extract_hcp_to_pt.py \
        --tar-glob "/path/to/hcp-parc/hcp-parc_*.tar" \
        --output-dir "/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-train" \
        --params-file "/path/to/normalization_params_hcp_train.npz" \
        --roi-count-total 450 \
        --roi-count-keep 400 \
        --drop-first-rois 50
"""

import argparse
import glob
import logging
import os
import re
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_npy_from_tar(tf: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    """Load .npy from tar member, handling compressed/uncompressed."""
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


def parse_metadata(member_name: str) -> dict:
    """
    Extract subject_id, task, and direction from filename.
    
    Example: 'sub-349244_mod-tfMRI_task-RELATIONAL_mag-3T_dir-RL.bold.npy'
    -> subject_id='349244', task='RELATIONAL', direction='RL', run='RELATIONAL_RL'
    """
    # Extract subject ID (just the numeric part, drop '_mod' suffix if present)
    subject_match = re.search(r'sub-(\d+)', member_name)
    subject_id = subject_match.group(1) if subject_match else "unknown"
    
    # Extract task
    task_match = re.search(r'task-([A-Z]+)', member_name)
    task = task_match.group(1) if task_match else ""
    
    # Extract direction (LR or RL)
    dir_match = re.search(r'dir-([A-Z]+)', member_name)
    direction = dir_match.group(1) if dir_match else ""
    
    # Combine task and direction for 'run' field
    run_str = f"{task}_{direction}" if task and direction else (task or direction or "unknown")
    
    return {
        "subject_id": subject_id,
        "task": task,
        "direction": direction,
        "run": run_str,
    }


def extract_hcp_to_pt(
    tar_glob: str,
    output_dir: Path,
    params_file: str,
    roi_count_total: int = 450,
    roi_count_keep: int = 400,
    drop_first_rois: int = 50,
    save_metadata: bool = True,
    skip_normalization: bool = False,
) -> None:
    """
    Extract and optionally normalize HCP timeseries to .pt files.
    
    Args:
        tar_glob: Glob pattern for tar shards
        output_dir: Directory to save .pt files
        params_file: Path to normalization_params_hcp_train.npz (ignored if skip_normalization=True)
        roi_count_total: Total ROIs in raw data (450)
        roi_count_keep: Cortical ROIs to keep (400)
        drop_first_rois: Subcortical ROIs to drop (50)
        save_metadata: Whether to include metadata in .pt files
        skip_normalization: If True, skip normalization and save raw data
    """
    # Load normalization parameters (only if not skipping)
    if not skip_normalization:
        if params_file is None:
            raise ValueError("params_file must be specified when not using --skip-normalization")
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Normalization params not found: {params_file}")
        
        params = np.load(params_file)
        medians = params['medians']  # shape (400,)
        iqrs = params['iqrs']  # shape (400,)
        logger.info(f"Loaded normalization params from {params_file}")
        logger.info(f"  Medians: shape={medians.shape}, range=[{medians.min():.4f}, {medians.max():.4f}]")
        logger.info(f"  IQRs: shape={iqrs.shape}, range=[{iqrs.min():.4f}, {iqrs.max():.4f}]")
    else:
        logger.info("Skipping normalization - will save raw data")
    
    # Find tar shards
    tar_paths = sorted(glob.glob(tar_glob))
    if not tar_paths:
        raise FileNotFoundError(f"No shards matched: {tar_glob}")
    logger.info(f"Found {len(tar_paths)} tar shards")
    
    # Build index
    index = []
    for tar_path in tqdm(tar_paths, desc="Indexing shards"):
        try:
            with tarfile.open(tar_path) as tf:
                for m in tf.getmembers():
                    if m.name.endswith(".bold.npy"):
                        index.append((tar_path, m.name))
        except Exception as e:
            logger.error(f"Failed to read shard {tar_path}: {e}")
    
    if not index:
        raise RuntimeError("No .bold.npy files found in provided shards")
    logger.info(f"Indexed {len(index)} samples")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and save
    success_count = 0
    failed_count = 0
    
    for sample_idx, (tar_path, member_name) in enumerate(tqdm(index, desc="Extracting samples")):
        try:
            # Load array
            with tarfile.open(tar_path) as tf:
                member = tf.getmember(member_name)
                arr = load_npy_from_tar(tf, member)
            
            original_shape = arr.shape
            
            if arr.ndim != 2:
                logger.warning(f"Skipping {member_name}: expected 2D, got shape {arr.shape}")
                failed_count += 1
                continue
            
            h, w = arr.shape
            # Detect orientation
            if h == roi_count_total:
                arr_roi_t = arr  # (450, T)
            elif w == roi_count_total:
                arr_roi_t = arr.T  # (T, 450) -> (450, T)
            else:
                logger.warning(f"Skipping {member_name}: shape {arr.shape} doesn't match ROI count {roi_count_total}")
                failed_count += 1
                continue
            
            # Slice cortical only
            arr_roi_t = arr_roi_t[drop_first_rois : drop_first_rois + roi_count_keep, :]  # (400, T)
            
            if arr_roi_t.shape[0] != roi_count_keep:
                logger.warning(f"Skipping {member_name}: after slicing got {arr_roi_t.shape[0]} ROIs, expected {roi_count_keep}")
                failed_count += 1
                continue
            
            # Apply normalization (or skip)
            if skip_normalization:
                arr_final = arr_roi_t  # Keep raw data
            else:
                arr_final = (arr_roi_t - medians[:, None]) / iqrs[:, None]  # Normalize
            
            # Convert to torch tensor
            ts_tensor = torch.from_numpy(arr_final.astype(np.float32))
            
            # Build output dict
            output_dict = {"fmri": ts_tensor}
            
            if save_metadata:
                metadata = parse_metadata(member_name)
                output_dict.update({
                    "subject_id": metadata["subject_id"],
                    "task": metadata["task"],
                    "direction": metadata["direction"],
                    "run": metadata["run"],
                    "original_shape": original_shape,
                    "tar_source": os.path.basename(tar_path),
                })
            
            # Save to .pt file
            output_path = output_dir / f"{sample_idx:07d}.pt"
            torch.save(output_dict, output_path)
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to extract {tar_path}:{member_name}: {e}")
            failed_count += 1
            continue
    
    logger.info("=" * 80)
    logger.info(f"Extraction complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Extract HCP timeseries to .pt files")
    parser.add_argument(
        "--tar-glob",
        type=str,
        required=True,
        help="Glob pattern for tar shards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save .pt files",
    )
    parser.add_argument(
        "--params-file",
        type=str,
        required=False,
        help="Path to normalization_params_hcp_train.npz (required unless --skip-normalization)",
    )
    parser.add_argument("--roi-count-total", type=int, default=450)
    parser.add_argument("--roi-count-keep", type=int, default=400)
    parser.add_argument("--drop-first-rois", type=int, default=50)
    parser.add_argument("--no-metadata", action="store_true", help="Don't save metadata in .pt files")
    parser.add_argument("--skip-normalization", action="store_true", help="Skip normalization, save raw data")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Extracting HCP Parcellated Timeseries to .pt Files")
    logger.info("=" * 80)
    
    extract_hcp_to_pt(
        tar_glob=args.tar_glob,
        output_dir=Path(args.output_dir),
        params_file=args.params_file,
        roi_count_total=args.roi_count_total,
        roi_count_keep=args.roi_count_keep,
        drop_first_rois=args.drop_first_rois,
        save_metadata=not args.no_metadata,
        skip_normalization=args.skip_normalization,
    )


if __name__ == "__main__":
    main()

