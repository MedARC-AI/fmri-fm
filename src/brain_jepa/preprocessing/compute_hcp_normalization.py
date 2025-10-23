#!/usr/bin/env python3
"""
Compute dataset-level rormalisaion parameters for HCP parcellated data.
For more details see the attached README.md

Usage:
    python compute_hcp_normalization.py \
        --tar-glob "/path/to/hcp-parc/hcp-parc_*.tar" \
        --output-dir "/path/to/output" \
        --roi-count-total 450 \
        --roi-count-keep 400 \
        --drop-first-rois 50 \
        --seq-length 1200
"""

import argparse
import glob
import logging
import os
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
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


def compute_normalization_params(
    tar_glob: str,
    roi_count_total: int = 450,
    roi_count_keep: int = 400,
    drop_first_rois: int = 50,
    seq_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-ROI median and IQR over subject temporal means.

    Args:
        tar_glob: Glob pattern for tar shards
        roi_count_total: Total ROIs in raw data (450 = 50 subcortical + 400 cortical)
        roi_count_keep: Number of cortical ROIs to keep (400)
        drop_first_rois: Number of subcortical ROIs to drop (50)
        seq_length: Optional max timepoints to use (e.g., 1200)

    Returns:
        medians: shape (roi_count_keep,)
        iqrs: shape (roi_count_keep,)
    """
    tar_paths = sorted(glob.glob(tar_glob))
    if not tar_paths:
        raise FileNotFoundError(f"No shards matched: {tar_glob}")
    
    logger.info(f"Found {len(tar_paths)} tar shards")
    logger.info(f"ROI config: total={roi_count_total}, keep={roi_count_keep}, drop_first={drop_first_rois}")

    # Build index of all .bold.npy files
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

    # Compute per-subject temporal means
    all_data_mean = []
    failed_count = 0
    
    for tar_path, member_name in tqdm(index, desc="Computing per-subject means"):
        try:
            with tarfile.open(tar_path) as tf:
                member = tf.getmember(member_name)
                arr = load_npy_from_tar(tf, member)
            
            if arr.ndim != 2:
                logger.warning(f"Skipping {member_name}: expected 2D, got shape {arr.shape}")
                failed_count += 1
                continue
            
            h, w = arr.shape
            # Detect orientation: (450, T) or (T, 450)
            if h == roi_count_total:
                arr_roi_t = arr  # already ROI-first
            elif w == roi_count_total:
                arr_roi_t = arr.T  # transpose to ROI-first
            else:
                logger.warning(f"Skipping {member_name}: shape {arr.shape} doesn't match ROI count {roi_count_total}")
                failed_count += 1
                continue
            
            # Slice cortical only: drop first N ROIs (subcortical)
            arr_roi_t = arr_roi_t[drop_first_rois : drop_first_rois + roi_count_keep, :]
            
            if arr_roi_t.shape[0] != roi_count_keep:
                logger.warning(f"Skipping {member_name}: after slicing got {arr_roi_t.shape[0]} ROIs, expected {roi_count_keep}")
                failed_count += 1
                continue
            
            # Optional truncate to seq_length
            if seq_length is not None and arr_roi_t.shape[1] > seq_length:
                arr_roi_t = arr_roi_t[:, :seq_length]
            
            # Compute temporal mean per ROI
            temp_mean = np.mean(arr_roi_t, axis=1).astype(np.float32)  # shape (roi_count_keep,)
            all_data_mean.append(temp_mean)
            
        except Exception as e:
            logger.error(f"Failed to process {tar_path}:{member_name}: {e}")
            failed_count += 1
            continue
    
    if not all_data_mean:
        raise RuntimeError("No valid samples found. Cannot compute normalization parameters.")
    
    logger.info(f"Successfully processed {len(all_data_mean)} samples ({failed_count} failed)")
    
    # Stack and compute dataset-level statistics
    all_data_mean = np.stack(all_data_mean)  # shape (num_subjects, roi_count_keep)
    logger.info(f"Stacked shape: {all_data_mean.shape}")
    
    medians = np.median(all_data_mean, axis=0).astype(np.float32)  # (roi_count_keep,)
    p25 = np.percentile(all_data_mean, 25, axis=0).astype(np.float32)
    p75 = np.percentile(all_data_mean, 75, axis=0).astype(np.float32)
    iqrs = (p75 - p25).astype(np.float32)
    
    # Clip IQR to avoid division by zero
    iqrs = np.clip(iqrs, 1e-6, None)
    
    logger.info(f"Median range: [{medians.min():.4f}, {medians.max():.4f}]")
    logger.info(f"IQR range: [{iqrs.min():.4f}, {iqrs.max():.4f}]")
    
    return medians, iqrs


def main():
    parser = argparse.ArgumentParser(description="Compute HCP normalization parameters")
    parser.add_argument(
        "--tar-glob",
        type=str,
        required=True,
        help="Glob pattern for tar shards (e.g., /path/to/hcp-parc_*.tar)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save normalization_params_hcp_train.npz",
    )
    parser.add_argument("--roi-count-total", type=int, default=450)
    parser.add_argument("--roi-count-keep", type=int, default=400)
    parser.add_argument("--drop-first-rois", type=int, default=50)
    parser.add_argument("--seq-length", type=int, default=None, help="Max timepoints to use (optional)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Computing HCP Normalization Parameters")
    logger.info("=" * 80)
    
    medians, iqrs = compute_normalization_params(
        tar_glob=args.tar_glob,
        roi_count_total=args.roi_count_total,
        roi_count_keep=args.roi_count_keep,
        drop_first_rois=args.drop_first_rois,
        seq_length=args.seq_length,
    )
    
    # Save to .npz
    output_file = output_dir / "normalization_params_hcp_train.npz"
    np.savez(output_file, medians=medians, iqrs=iqrs)
    logger.info(f"Saved normalization parameters to {output_file}")
    
    # Also save human-readable CSV
    import pandas as pd
    csv_file = output_dir / "normalization_params_hcp_train.csv"
    df = pd.DataFrame({
        'roi_index': np.arange(len(medians)),
        'median': medians,
        'iqr': iqrs,
    })
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved CSV for inspection: {csv_file}")
    
    logger.info("✓ Done!")


if __name__ == "__main__":
    main()

