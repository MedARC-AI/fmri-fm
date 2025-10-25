# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse
import webdataset as wds
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import scale
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

DEFAULT_CONFIG = Path(__file__).parent / "config/default_hcp_flat.yaml"


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
    _logger.info(f"Generating HCP-Flat ({shard_id=:04d}, {batch_id=:02d}, {batch_shard_id=:02d})")
    _logger.info("Config:\n%s", yaml.safe_dump(OmegaConf.to_object(cfg), sort_keys=False))

    out_dir = Path(cfg.out_dir)
    out_cfg_path = out_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert cfg.overwrite or prev_cfg == cfg, "Current config doesn't match previous config"
    if shard_id == 0:
        out_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, out_cfg_path)

    outpath = out_dir / f"hcp-flat_{shard_id:04d}.tar"
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

    # Load flat map surface and cortex mask.
    surf = ut.load_flat("32k_fs_LR", hemi_padding=cfg.hemi_padding)

    roi_img = nib.load(args.roi_path)
    roi_mask = ut.get_cifti_surf_data(roi_img)
    roi_mask = roi_mask.flatten().astype(int) > 0
    surf, mask = ut.extract_valid_flat(surf, roi_mask)

    # Create flat map resampler.
    resampler = ut.FlatResampler(pixel_size=cfg.pixel_size, rect=cfg.bbox)
    resampler.fit(surf)
    _logger.info(
        "Flat map bbox: %s, shape: %s, nnz: %d",
        resampler.bbox_,
        resampler.mask_.shape,
        resampler.mask_.sum(),
    )

    # Temp output path, in case of incomplete processing.
    tmp_outpath = outpath.parent / f".tmp-{outpath.name}"
    outpath.parent.mkdir(exist_ok=True)

    # Generate wds samples.
    with wds.TarWriter(str(tmp_outpath), encoder=False) as sink:
        for path in tqdm(shard_series_paths):
            sample = create_sample(path, mask=mask, resampler=resampler, new_tr=cfg.target_tr)
            sink.write(sample)

    tmp_outpath.rename(outpath)
    _logger.info(f"Done: {outpath}")


def create_sample(
    path: Path,
    mask: np.ndarray,
    resampler: ut.FlatResampler,
    new_tr: float = 1.0,
) -> dict[str, Any]:
    metadata = parse_hcp_metadata(path)
    key = "sub-{sub}_mod-{mod}_task-{task}_mag-{mag}_dir-{dir}".format(**metadata)
    tr = HCP_TR[metadata["mag"]]

    # load task events if available
    events = load_hcp_events(path.parent)

    # load series and preprocess
    series = load_hcp_series(path)
    series = preprocess_series(
        series,
        mask=mask,
        resampler=resampler,
        tr=tr,
        new_tr=new_tr,
    )

    metadata["n_frames"] = len(series)

    # data mask in flat map raster space
    flat_mask = scipy.sparse.coo_array(resampler.mask_)

    meta_json = json.dumps(metadata).encode("utf-8")
    events_json = json.dumps(events).encode("utf-8")
    bold_npy = encode_npy(series)
    flat_mask_npz = encode_sparse_npz(flat_mask)

    # write sample with serialized binary data
    sample = {
        "__key__": key,
        "meta.json": meta_json,
        "events.json": events_json,
        "bold.npy": bold_npy,
        "mask.npz": flat_mask_npz,
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


def load_hcp_series(path: Path) -> np.ndarray:
    """Load HCP surface time series.

    Returns a series array, shape (n_samples, n_vertices)
    """
    series = nib.load(path)
    series = ut.get_cifti_surf_data(series)
    series = np.ascontiguousarray(series.T)
    return series


def preprocess_series(
    series: np.ndarray,
    *,
    mask: np.ndarray,
    resampler: ut.FlatResampler,
    tr: float,
    new_tr: float = 1.0,
    dtype: np.dtype = np.float16,
) -> np.ndarray:
    """Preprocess time series and project to flat map space."""
    # Apply vertex mask.
    series = series[:, mask]

    # Standard scale.
    series = scale(series)

    # Temporal resample.
    series = ut.resample_timeseries(series, tr=tr, new_tr=new_tr)

    # Transform to flat map space.
    series = resampler.transform(series, interpolation="linear")

    # Data checks.
    assert not np.any(np.isnan(series)), "series contains nan"
    assert np.all((series == 0) == ~resampler.mask_), "unexpected sparsity pattern"
    vmax = np.max(np.abs(series))
    assert vmax < 100, f"series contains large values {vmax=:.3f}"

    # Apply mask in flat map space.
    series = series[:, resampler.mask_]

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


def encode_npy(data: np.ndarray) -> bytes:
    with io.BytesIO() as f:
        np.save(f, data)
        buf = f.getvalue()
    return buf


def encode_sparse_npz(data: scipy.sparse.coo_array) -> bytes:
    with io.BytesIO() as f:
        scipy.sparse.save_npz(f, data, compressed=False)
        buf = f.getvalue()
    return buf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    main(**vars(args))
