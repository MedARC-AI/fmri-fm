# Brain-JEPA

Code for Brain-JEPA copied with minor changes from the [official repo](https://github.com/Eric-LRL/Brain-JEPA/tree/ff89964da588a08203a6bc15fc9ab404b56c0fe0).

## Installation
Uses same env as that specified in root directory

## Pretrain
pretrain on parcellated  400 cortical ROIs from the Schaefer
```bash
uv run python -m brain_jepa.train --cfg-path src/brain_jepa/config/hcp_train_pt.yaml 
```

## Dataset Loader Options
datasets/hcp_parc_pt.py
Dataset class for parcellated data extracted as .pt files

datasets/hcp_parc.py [DEPRECATED]
Dataset class for  webdataset format. Compatible only with 
```bash
tar_glob: /teamspace/filestore_folders/shared/fmri-fm/datasets/hcp-parc/hcp-parc_*.tar
```
## Setup data 
1) Generate the 400-ROI gradient CSV (drop first 50 rows)

```bash
cd /teamspace/studios/this_studio/shamus/fmri-fm
uv run python - <<'PY'
import pandas as pd
g450 = pd.read_csv("src/brain_jepa/gradient_mapping_450.csv", header=None)
g450.iloc[50:450].to_csv("src/brain_jepa/gradient_mapping_400.csv", index=False, header=False)
print("Wrote src/brain_jepa/gradient_mapping_400.csv")
PY
```

1) Pre-extract .tar dataset as .pt files
For piprlines to do this see brain_jepa/preprocessing 

To use ready datasets, choose either one below
```bash
/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc
```
Extracted .pt files from the full HCP parcellated dataset

```bash
/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-lite
```
Small datasets, kept just the first 2000 .pt files from hcp-parc

```bash
/teamspace/gcs_folders/share/fmri-fm/brain-jepa/hcp-parc-normalized
```
Followed Brain-JEPA normalisation method and extracted as .pt files from the full HCP parcellated dataset

## Pretrain standardisation nuances
Mirror UKB temporal controls: add seq_length and sampling_rate options, keeping the same downsample flag semantics, while still targeting W=160.
