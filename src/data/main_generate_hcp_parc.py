# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.utils import check_random_state

import data.surface_utils as ut

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.WARNING,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)


EXCLUDE_ACQS = {
    # Exclude merged retinotopic localizer scan.
    "tfMRI_7T_RETCCW_AP_RETCW_PA_RETEXP_AP_RETCON_PA_RETBAR1_AP_RETBAR2_PA",
}
EXCLUDE_CONDS = {
    # The sync time is not needed, it's already subtracted
    # https://www.mail-archive.com/hcp-users@humanconnectome.org/msg00616.html
    "Sync",
}

# Total number of HCP subjects released in HCP-1200
HCP_NUM_SUBJECTS = 1098

# https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging
# https://www.humanconnectome.org/hcp-protocols-ya-7t-imaging
HCP_TR = {"3T": 0.72, "7T": 1.0}

DEFAULT_CONFIG = Path(__file__).parent / "config/default_hcp_parc.yaml"


def main(
    shard_id: int = 0,
    cfg_path: str | None = None,
    overrides: list[str] | None = None,
):
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(cfg_path))
    if overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(overrides))

    assert cfg.num_shards % cfg.num_batches == 0
    assert 0 <= shard_id < cfg.num_shards

    rng = check_random_state(cfg.seed)

    shards_per_batch = cfg.num_shards // cfg.num_batches
    batch_id = shard_id // shards_per_batch
    batch_shard_id = shard_id % shards_per_batch

    _logger.setLevel(cfg.log_level)
    _logger.info(f"Generating HCP-Parc ({shard_id=:04d}, {batch_id=:02d}, {batch_shard_id=:02d})")
    _logger.info("Config:\n%s", yaml.safe_dump(OmegaConf.to_object(cfg), sort_keys=False))

    out_dir = Path(cfg.out_dir)
    out_cfg_path = out_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert cfg.overwrite or prev_cfg == cfg, "Current config doesn't match previous config"
    if shard_id == 0:
        out_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, out_cfg_path)

    outpath = out_dir / f"hcp-parc_{shard_id:04d}"
    if outpath.exists() and not cfg.overwrite:
        _logger.info(f"Output path exists: {outpath}; skipping")
        return

    all_subjects = np.array(sorted(p.name for p in Path(cfg.hcp_1200_dir).glob("[0-9]*")))
    assert len(all_subjects) == HCP_NUM_SUBJECTS, "Unexpected number of subjects"

    # Shuffle subjects into batches, keeping related individuals together.
    groups = load_hcp_family_groups(cfg.hcp_restricted_csv)
    groups = groups.loc[all_subjects].values
    splitter = GroupKFold(n_splits=cfg.num_batches, shuffle=True, random_state=rng)
    all_batch_subjects = [
        all_subjects[ind] for _, ind in splitter.split(all_subjects, groups=groups)
    ]

    # Get subjects for current batch.
    batch_subjects = all_batch_subjects[batch_id]
    _logger.info(
        "Subject batch %02d/%d (n=%d)\n\tbatch_subjects[:5] = %s",
        batch_id,
        cfg.num_batches,
        len(batch_subjects),
        batch_subjects[:5],
    )

    # Get timeseries paths for current subjects.
    batch_series_paths = sorted(
        path
        for sub in batch_subjects
        for path in (Path(cfg.hcp_1200_dir) / sub / "MNINonLinear/Results").rglob(
            "*_Atlas_MSMAll.dtseries.nii"
        )
        if path.parent.name not in EXCLUDE_ACQS
    )

    # Shuffle series paths.
    rng.shuffle(batch_series_paths)

    # Split series paths into shards.
    path_offsets = np.linspace(0, len(batch_series_paths), shards_per_batch + 1)
    path_offsets = np.round(path_offsets).astype(int)
    path_start, path_stop = path_offsets[batch_shard_id : batch_shard_id + 2]
    shard_series_paths = batch_series_paths[path_start:path_stop]
    _logger.info(
        "Batch shard %02d/%d (%04d/%d) (n=%d)\n\tshard_series_paths[:5] = %s",
        batch_shard_id,
        shards_per_batch,
        shard_id,
        cfg.num_shards,
        len(shard_series_paths),
        "\n\t" + "\n\t".join(map(str, shard_series_paths[:5])),
    )

    # Load cifti space parcellation
    _logger.info("Loading parcellation: %s", cfg.parcellation_path)
    parc_img = nib.load(cfg.parcellation_path)
    parc = np.squeeze(parc_img.get_fdata())
    parc_one_hot = ut.parc_to_one_hot(parc, sparse=True)
    parc_counts = parc_one_hot.sum(axis=0)
    _logger.info(
        "Num rois: %d, counts: %.0f, %.0f, %.0f",
        len(parc_counts),
        parc_counts.min(),
        np.median(parc_counts),
        parc_counts.max(),
    )

    # Temp output path, in case of incomplete processing.
    tmp_outpath = outpath.parent / f".tmp-{outpath.name}"
    tmp_outpath.mkdir(parents=True, exist_ok=True)

    # Generate wds samples.
    for ii, path in enumerate(tqdm(shard_series_paths)):
        sample = create_sample(path, parc_one_hot=parc_one_hot)
        key = sample["__key__"]
        sample_path = tmp_outpath / f"{ii:03d}__{key}.pt"
        torch.save(sample, sample_path)

    tmp_outpath.rename(outpath)
    _logger.info(f"Done: {outpath}")


def create_sample(path: Path, parc_one_hot: np.ndarray) -> dict[str, Any]:
    metadata = parse_hcp_metadata(path)
    key = "sub-{sub}_mod-{mod}_task-{task}_mag-{mag}_dir-{dir}".format(**metadata)
    metadata["tr"] = HCP_TR[metadata["mag"]]

    # load task events if available
    events = load_hcp_events(path.parent)

    # load series and preprocess
    series_img = nib.load(path)
    series = series_img.get_fdata()
    series = preprocess_series(series, parc_one_hot=parc_one_hot)
    series = torch.from_numpy(series)

    metadata["n_frames"] = len(series)

    sample = {
        "__key__": key,
        "meta": metadata,
        "events": events,
        "bold": series,
    }
    return sample


def load_hcp_family_groups(hcp_restricted_csv: str | Path) -> pd.Series:
    df = pd.read_csv(hcp_restricted_csv, dtype={"Subject": str})
    df.set_index("Subject", inplace=True)
    hcp_family_id = df.loc[:, "Pedigree_ID"]

    # Relabel to [0, N)
    _, hcp_family_groups = np.unique(hcp_family_id.values, return_inverse=True)
    hcp_family_groups = pd.Series(
        hcp_family_groups,
        index=hcp_family_id.index,
        name="Family_Group",
    )
    return hcp_family_groups


def parse_hcp_metadata(path: Path) -> dict[str, str]:
    sub = path.parents[3].name
    acq = path.parent.name
    if "7T" in acq:
        mod, task, mag, dir = acq.split("_")
    else:
        mod, task, dir = acq.split("_")
        mag = "3T"
    metadata = {"sub": sub, "mod": mod, "task": task, "mag": mag, "dir": dir}
    return metadata


def preprocess_series(
    series: np.ndarray,
    *,
    parc_one_hot: np.ndarray,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    # parcellate [n_samples, n_verts] -> [n_samples, n_rois]
    series = ut.parcellate_timeseries(series, parc_one_hot)

    # Cast dtype. Raise on any overflows.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        series = series.astype(dtype)

    return series


def load_hcp_events(run_dir: Path) -> list[dict[str, Any]]:
    """Read all events from a run directory.

    Returns a list of records following the BIDS events specification.

    EV files have the format `'{cond}.txt'` and look like:

    ```
    30.471  10.443  1.0
    41.16   10.238  1.0
    81.57   10.664  1.0
    92.513  10.114  1.0
    131.537 10.485  1.0
    142.293 10.852  1.0
    153.395 11.93   1.0
    165.577 12.213  1.0
    210.319 10.513  1.0
    ```
    """
    ev_dir = Path(run_dir) / "EVs"
    if not ev_dir.exists():
        return []

    events = []
    ev_paths = ev_dir.glob("*.txt")
    for path in ev_paths:
        name = path.stem
        if name in EXCLUDE_CONDS:
            continue

        cond_events = pd.read_csv(path, sep="\t", names=["onset", "duration", "value"])
        if len(cond_events) == 0:
            continue

        cond_events.drop("value", inplace=True, axis=1)
        cond_events = cond_events.astype({"duration": float})
        cond_events["trial_type"] = name
        events.append(cond_events)

    events = pd.concat(events, axis=0, ignore_index=True)
    events = events.sort_values("onset")
    events = events.to_dict(orient="records")
    return events


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    main(**vars(args))
