# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import scipy.sparse
import webdataset as wds
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.preprocessing import scale

import data.surface_utils as ut

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.WARNING,
    datefmt="%y-%m-%d %H:%M:%S",
)
logging.getLogger("nibabel").setLevel(logging.ERROR)

_logger = logging.getLogger(__name__)

# Upsampled 1.0s TR for high-res func1mm data.
# https://cvnlab.slite.page/p/vjWTghPTb3/Time-series-data
NSD_TR = 1.0

# Images were presented for 3 seconds with a 1 second gap.
# https://cvnlab.slite.page/p/9gFSd5MubN#3834b73a
NSD_IMAGE_PRESENTATION_TIME = 3.0

# https://cvnlab.slite.page/p/h_T_2Djeid/Technical-notes
NSD_SES_PER_SUBJECT = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}
NSD_SUBJECTS = list(NSD_SES_PER_SUBJECT)

# Each shard will contain one session.
# 12 or 14 runs, ~500MB.
NUM_SHARDS = sum(NSD_SES_PER_SUBJECT.values())

DEFAULT_CONFIG = Path(__file__).parent / "config/default_nsd_flat.yaml"


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

    assert 0 <= shard_id < NUM_SHARDS, f"invalid {shard_id=}"

    _logger.setLevel(cfg.log_level)
    _logger.info(f"Generating NSD-Flat ({shard_id=:04d})")
    _logger.info("Config:\n%s", yaml.safe_dump(OmegaConf.to_object(cfg), sort_keys=False))

    out_dir = Path(cfg.out_dir)
    out_cfg_path = out_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert cfg.overwrite or prev_cfg == cfg, "Current config doesn't match previous config"
    if shard_id == 0:
        out_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, out_cfg_path)

    outpath = out_dir / f"nsd-flat_{shard_id:04d}.tar"
    if outpath.exists() and not cfg.overwrite:
        _logger.info(f"Output path exists: {outpath}; skipping")
        return

    # Each shard is one session.
    # Figure out the subject and session for the given shard.
    nsd_cum_sessions = np.cumsum(list(NSD_SES_PER_SUBJECT.values()))
    nsd_cum_sessions = np.concatenate([[0], nsd_cum_sessions])
    subidx = np.searchsorted(nsd_cum_sessions, shard_id, side="right") - 1
    sub = NSD_SUBJECTS[subidx]
    session = shard_id - nsd_cum_sessions[subidx] + 1

    ts_dir = Path(cfg.nsd_dir) / f"nsddata_timeseries/ppdata/{sub}/32k_fs_LR/timeseries"
    shard_lh_paths = sorted(ts_dir.glob(f"timeseries_session{session:02d}_run*.lh.func.gii"))
    num_paths = len(shard_lh_paths)
    assert num_paths in {12, 14}, f"unexpected number of files {num_paths}"

    _logger.info(
        "Shard %03d/%d (sub=%s, ses=%02d, n=%d)\n\tshard_lh_paths[:5] = %s",
        shard_id,
        NUM_SHARDS,
        sub,
        session,
        num_paths,
        "\n\t" + "\n\t".join(map(str, shard_lh_paths[:5])),
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
        for path_lh in tqdm(shard_lh_paths):
            sample = create_sample(
                path_lh,
                mask=mask,
                resampler=resampler,
                nsd_dir=cfg.nsd_dir,
                new_tr=cfg.target_tr,
            )
            sink.write(sample)

    tmp_outpath.rename(outpath)
    _logger.info(f"Done: {outpath}")


def create_sample(
    path_lh: Path,
    mask: np.ndarray,
    resampler: ut.FlatResampler,
    nsd_dir: str | Path,
    new_tr: float = 1.0,
) -> dict[str, Any]:
    metadata = parse_nsd_metadata(path_lh)
    key = "sub-{sub}_ses-{ses}_run-{run}".format(**metadata)

    # load task events if available
    events = load_nsd_events(nsd_dir, **metadata)

    # load series and preprocess
    series = load_nsd_series(path_lh)
    series = preprocess_series(
        series,
        mask=mask,
        resampler=resampler,
        tr=NSD_TR,
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


def parse_nsd_metadata(path: Path) -> dict[str, Any]:
    match = re.search(
        r"(subj[0-9]+)/.*/timeseries_session([0-9]+)_run([0-9]+)\.[lr]h.func.gii",
        Path(path).as_posix(),
    )
    metadata = {
        "sub": match.group(1),
        "ses": int(match.group(2)),
        "run": int(match.group(3)),
    }
    return metadata


def load_nsd_series(path_lh: Path) -> np.ndarray:
    """Load NSD surface time series.

    Returns a series array, shape (n_samples, n_vertices)
    """
    path_rh = str(path_lh).replace(".lh", ".rh")

    img_lh = nib.load(path_lh)
    series_lh = np.stack([da.data for da in img_lh.darrays])

    img_rh = nib.load(path_rh)
    series_rh = np.stack([da.data for da in img_rh.darrays])

    series = np.concatenate([series_lh, series_rh], axis=1)

    # Upcast for preprocessing.
    series = series.astype(np.float64)
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


def load_nsd_events(nsd_dir: str | Path, sub: str, ses: int, run: int) -> list[dict[str, Any]]:
    """Load NSD image presentation events.

    Returns a list of records following the BIDS events specification. The nsd_id field
    is the 0-based NSD image index.

    Reference:
        https://cvnlab.slite.page/p/vjWTghPTb3#bb24a15b
    """
    design_path = (
        Path(nsd_dir)
        / f"nsddata_timeseries/ppdata/{sub}/func1mm/design"
        / f"design_session{ses:02d}_run{run:02d}.tsv"
    )

    design = np.loadtxt(design_path, dtype=np.int64)
    trial_indices = design.nonzero()[0]

    # Make 0-based to match the nsd_stim_info_merged.csv table.
    # https://cvnlab.slite.page/p/NKalgWd__F#bf18f984
    nsd_ids = design[trial_indices] - 1

    events = [
        {
            "onset": int(idx) * NSD_TR,
            "duration": NSD_IMAGE_PRESENTATION_TIME,
            "trial_type": "nsd",
            "nsd_id": int(nsd_id),
        }
        for idx, nsd_id in zip(trial_indices, nsd_ids)
    ]
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
