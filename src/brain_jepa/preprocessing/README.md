# HCP Parcellated Data Extraction Pipeline

Scripts to extract HCP parcellated fMRI data from tar shards to individual `.pt` files with/without normalization applied.

## Overview

The extraction pipeline consists of two steps:

1. **Compute Normalization Parameters** - Compute dataset-level robust scaling (median/IQR per ROI). Follows Brain-JEPA
2. **Extract to .pt Files** - Extract each timeseries, normalize it, and save as a `.pt` file

## Why Extract to .pt Files?

- Loading from tar archives is slow due to sequential access

## Usage details
### Option 1: Run the full pipeline

```bash
cd /teamspace/studios/this_studio/shamus/fmri-fm
bash src/brain_jepa/preprocessing/run_hcp_extraction.sh
```

This will:
1. Compute normalization parameters → `normalization_params_hcp_train.npz`
2. Extract all timeseries → `.pt` files in `/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-train/`
3. Test the dataset loader

### Option 2: Run steps manually

#### Step 1: Compute normalization parameters

```bash
python src/brain_jepa/preprocessing/compute_hcp_normalization.py \
    --tar-glob "/teamspace/filestore_folders/shared/fmri-fm/datasets/hcp-parc/hcp-parc_*.tar" \
    --output-dir "/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-train" \
    --roi-count-total 450 \
    --roi-count-keep 400 \
    --drop-first-rois 50
```
Comment: Keep only cortical data- 400 ROIs

**Output:**
- `normalization_params_hcp_train.npz` - NumPy archive with `medians` and `iqrs` arrays
- `normalization_params_hcp_train.csv` - CSV for inspection

#### Step 2: Extract to .pt files

```bash
python src/brain_jepa/preprocessing/extract_hcp_to_pt.py \
    --tar-glob "/teamspace/filestore_folders/shared/fmri-fm/datasets/hcp-parc/hcp-parc_*.tar" \
    --output-dir "/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-train" \
    --params-file "/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-train/normalization_params_hcp_train.npz" \
    --roi-count-total 450 \
    --roi-count-keep 400 \
    --drop-first-rois 50
```

**Output:**
- `0000000.pt`, `0000001.pt`, ... - One `.pt` file per run
- Each file contains:
  ```python
    {
        'fmri': torch.Tensor,           # Shape: (400, T) - the actual data
        'subject_id': int,              # e.g., 349244
        'task': str,                    # e.g., 'RELATIONAL'
        'direction': str,               # e.g., 'RL' or 'LR'
        'run': str,                     # e.g., 'RELATIONAL_RL'
        'original_shape': tuple,        # e.g., (167, 450) - BEFORE processing
        'tar_source': str               # e.g., 'hcp-parc_0000.tar'
    }
  ```

#### Step 3: Test the loader

```bash
python src/brain_jepa/datasets/hcp_parc_pt.py /teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-train
```

## Normalization Details

The pipeline follows Brain-JEPA's UKB normalization:

1. **Compute per-subject temporal means**: For each subject, load full timeseries (400 ROIs × T timepoints) and compute mean per ROI → shape (400,)
2. **Stack across subjects**: All per-subject means → shape (num_subjects, 400)
3. **Compute dataset statistics**:
   - `medians[i]` = median across all subjects for ROI `i`
   - `iqrs[i]` = IQR (p75 - p25) across all subjects for ROI `i`
4. **Apply normalization**: For each sample, `normalized = (timeseries - median) / iqr`

uses median/IQR instead of mean/std and is computed per-ROI at the full dataset level

