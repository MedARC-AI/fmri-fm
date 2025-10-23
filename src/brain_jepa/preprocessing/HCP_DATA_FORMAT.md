# HCP Parcellated Data Format

## Source Location

```
/teamspace/gcs_folders/share/fmri-fm/datasets/hcp-parc/
├── config.yaml
├── hcp-parc_0000.tar
├── hcp-parc_0001.tar
├── ...
└── hcp-parc_1503.tar
```

Total: **1504 tar shards**

## Filename Pattern

Each tar contains multiple `.bold.npy` files with the pattern:

```
sub-{SUBJECT}_mod-tfMRI_task-{TASK}_mag-3T_dir-{DIRECTION}.bold.npy
```

**Example:**
```
sub-349244_mod-tfMRI_task-RELATIONAL_mag-3T_dir-RL.bold.npy
```

### Metadata Fields

| Field | Description | Example Values |
|-------|-------------|----------------|
| `SUBJECT` | Subject ID (numeric) | `349244`, `389357`, `686969` |
| `TASK` | fMRI task | `RELATIONAL`, `LANGUAGE`, `MOTOR`, `GAMBLING`, `WM`, `SOCIAL`, `EMOTION` |
| `DIRECTION` | Phase encoding direction | `LR` (left-right), `RL` (right-left) |

**Note:** Subject IDs in filenames include `_mod` suffix (e.g., `349244_mod`), but we extract just the numeric part (`349244`).

## Data Format

### Array Shape
```python
arr.shape = (T, 450)
```

- **T** = Number of timepoints (varies by task)
  - RELATIONAL: ~167 timepoints
  - LANGUAGE: ~227 timepoints  
  - MOTOR: ~204 timepoints
  - GAMBLING: ~182 timepoints
  - Varies between ~150-250 timepoints

- **450** = Total ROIs
  - First 50: Subcortical (Tian Scale III)
  - Next 400: Cortical (Schaefer 400-parcel)

### Data Type
- **Original**: `float16` (half precision)
- **Extracted**: Converted to `float32` for better precision during normalization

### Value Range
- Raw values: typically [-5, +5] range
- After robust scaling: approximately [-3, +3] (standardized by median/IQR)

## Processing Pipeline

### Step 1: Load from tar
```python
arr = np.load(tar_member)  # Shape: (T, 450), dtype: float16
```

### Step 2: Transpose to ROI-first
```python
arr_roi_t = arr.T  # Shape: (450, T)
```

### Step 3: Slice cortical ROIs only
```python
arr_cortical = arr_roi_t[50:450, :]  # Shape: (400, T) - drop first 50 subcortical
```

### Step 4: Normalize (robust scaling)
```python
# Using dataset-level medians and IQRs computed from all training samples
arr_normalized = (arr_cortical - medians[:, None]) / iqrs[:, None]
```

### Step 5: Save as .pt
```python
torch.save({
    'fmri': torch.from_numpy(arr_normalized).float(),  # (400, T), float32
    'subject_id': '349244',
    'task': 'RELATIONAL',
    'direction': 'RL',
    'run': 'RELATIONAL_RL',
    'original_shape': (167, 450),
    'tar_source': 'hcp-parc_0000.tar',
}, f'{idx:07d}.pt')
```

## Expected Output Format

After extraction, each `.pt` file contains:

```python
{
    'fmri': Tensor(400, T),      # Cortical ROIs, normalized, float32
    'subject_id': str,            # e.g., "349244" (numeric only)
    'task': str,                  # e.g., "RELATIONAL"
    'direction': str,             # e.g., "RL"
    'run': str,                   # e.g., "RELATIONAL_RL" (task + direction)
    'original_shape': tuple,      # e.g., (167, 450) before processing
    'tar_source': str,            # e.g., "hcp-parc_0000.tar"
}
```

## Validation Checks

The extraction script validates:

1. ✅ **Shape**: Must be 2D array
2. ✅ **ROI count**: Must have 450 in one dimension (transposed if needed)
3. ✅ **After slicing**: Must have exactly 400 cortical ROIs
4. ✅ **Metadata parsing**: Subject, task, direction extracted from filename

Failed samples are logged but extraction continues.

## Comparison: Original vs Extracted

| Aspect | Original (tar) | Extracted (.pt) |
|--------|---------------|-----------------|
| **Location** | `/teamspace/gcs_folders/share/fmri-fm/datasets/hcp-parc/` | `/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-train/` |
| **Format** | `.bold.npy` in `.tar` archives | Individual `.pt` files |
| **Shape** | `(T, 450)` time-first | `(400, T)` ROI-first |
| **ROIs** | 450 (50 subC + 400 cortical) | 400 cortical only |
| **Dtype** | `float16` | `float32` |
| **Normalized** | No | Yes (robust scaling) |
| **Access** | Sequential (slow) | Random access (fast) |
| **Metadata** | Filename only | Embedded in .pt file |

## Statistics (from hcp-parc_0000.tar)

Sample of first 5 files:

| Subject | Task | Dir | Shape | Dtype | Range |
|---------|------|-----|-------|-------|-------|
| 349244 | RELATIONAL | RL | (167, 450) | float16 | [-2.75, 3.56] |
| 389357 | LANGUAGE | LR | (227, 450) | float16 | [-2.31, 1.85] |
| 686969 | MOTOR | LR | (204, 450) | float16 | [-2.66, 1.96] |
| 349244 | LANGUAGE | LR | (227, 450) | float16 | [-5.14, 2.02] |
| 196346 | GAMBLING | LR | (182, 450) | float16 | [-2.45, 2.59] |

**Key observations:**
- Variable time lengths (167-227 timepoints)
- Consistent 450 ROIs across all samples
- Values already somewhat normalized (narrow range)
- Same subject can have multiple tasks/directions

## Gradient Mapping Alignment

The gradient CSV (`gradient_mapping_400.csv`) must align with the cortical ROI order:

1. **Original data**: 50 subcortical (Tian) + 400 cortical (Schaefer)
2. **After slicing**: ROIs 50-449 from original → ROIs 0-399 in extracted data
3. **Gradient CSV**: Row 0 → ROI 0 (first cortical parcel from Schaefer atlas)

This is **critical** for the gradient positional encoding in Brain-JEPA.

## How to Inspect Manually

```bash
# List contents of a tar
tar -tf /teamspace/gcs_folders/share/fmri-fm/datasets/hcp-parc/hcp-parc_0000.tar | head

# Count .bold.npy files in a tar
tar -tf /teamspace/gcs_folders/share/fmri-fm/datasets/hcp-parc/hcp-parc_0000.tar | grep '.bold.npy' | wc -l

# Inspect a .pt file
python -c "
import torch
data = torch.load('0000000.pt', map_location='cpu')
print(data.keys())
print(f'Shape: {data[\"fmri\"].shape}')
print(f'Subject: {data[\"subject_id\"]}')
print(f'Task: {data[\"task\"]}')
"
```

