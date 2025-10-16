# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import io
import json
import logging
import tempfile
import warnings
import zipfile
from pathlib import Path

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

# https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/brain_mri.pdf#page=9.22
UKBB_TR = 0.735

# Total number of fMRI sessions.
UKBB_NUM_SESSIONS = 63232

# UKBB data field for cifti surface data.
# https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=32136
UKBB_DATA_FIELD = 32136

# List of filtered fMRI runs
UKBB_BULK_NAME = "ukb680217.bulk"

DEFAULT_CONFIG = Path(__file__).parent / "config/default_ukbb_flat.yaml"

EPS = 1e-6


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

    _logger.setLevel(cfg.log_level)
    _logger.info(f"Generating UKBB-Flat ({shard_id=:04d})")
    _logger.info("Config:\n%s", yaml.safe_dump(OmegaConf.to_object(cfg), sort_keys=False))

    num_shards = UKBB_NUM_SESSIONS // cfg.num_ses_per_shard
    if shard_id >= num_shards:
        _logger.info(f"Shard {shard_id} greater than total shards {num_shards}; exiting.")
        return

    out_dir = Path(cfg.out_dir)
    out_cfg_path = out_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert cfg.overwrite or prev_cfg == cfg, "Current config doesn't match previous config"
    if shard_id == 0:
        out_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, out_cfg_path)

    outpath = out_dir / f"ukbb-flat_{shard_id:05d}.tar"
    if outpath.exists() and not cfg.overwrite:
        _logger.info(f"Output path exists: {outpath}; skipping")
        return

    ukbb_dir = Path(cfg.ukbb_dir)
    bulk_ses_list = np.loadtxt(ukbb_dir / UKBB_BULK_NAME, dtype=str)
    assert len(bulk_ses_list) == UKBB_NUM_SESSIONS, "Unexpected number of sessions"

    ses_start = shard_id * cfg.num_ses_per_shard
    ses_stop = ses_start + cfg.num_ses_per_shard
    shard_ses_list = bulk_ses_list[ses_start:ses_stop]
    _logger.info(
        "UKBB sessions for shard %05d (n=%d):\n%s",
        shard_id,
        len(shard_ses_list),
        shard_ses_list,
    )

    shard_ses_paths = [ukbb_dir / f"{sub}_{ses}.zip" for sub, ses in shard_ses_list]
    if not all(p.exists() for p in shard_ses_paths):
        _logger.warning("Some session zip files missing for shard %05d; exiting", shard_id)
        return

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
        for path in tqdm(shard_ses_paths):
            for sample in create_samples(
                path, mask=mask, resampler=resampler, new_tr=cfg.target_tr
            ):
                sink.write(sample)

    tmp_outpath.rename(outpath)
    _logger.info(f"Done: {outpath}")


def create_samples(
    path: Path,
    mask: np.ndarray,
    resampler: ut.FlatResampler,
    new_tr: float = 1.0,
):
    for filename, series in load_ukbb_series(path):
        metadata = parse_ukbb_metadata(path, filename)
        key = "sub-{sub}_ses-{ses}_mod-{mod}".format(**metadata)

        series = preprocess_series(
            series,
            mask=mask,
            resampler=resampler,
            tr=UKBB_TR,
            new_tr=new_tr,
        )

        metadata["n_frames"] = len(series)

        # data mask in flat map raster space
        flat_mask = scipy.sparse.coo_array(resampler.mask_)

        meta_json = json.dumps(metadata).encode("utf-8")
        bold_npy = encode_npy(series)
        flat_mask_npz = encode_sparse_npz(flat_mask)

        # write sample with serialized binary data
        sample = {
            "__key__": key,
            "meta.json": meta_json,
            "bold.npy": bold_npy,
            "mask.npz": flat_mask_npz,
        }
        yield sample


def load_ukbb_series(archive: Path):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(archive) as zip:
            members = [info for info in zip.filelist if info.filename.endswith(".dtseries.nii")]
            # rfMRI and possibly tfMRI
            assert len(members) in {1, 2}, "expected 1 or 2 series per run zip"

            for info in members:
                zip.extract(info, path=tmpdir)
                img = nib.load(Path(tmpdir) / info.filename)
                series = ut.get_cifti_surf_data(img)
                series = np.ascontiguousarray(series.T)
                yield info.filename, series


def parse_ukbb_metadata(path: Path, filename: str) -> dict[str, str]:
    # https://dnanexus.gitbook.io/uk-biobank-rap/getting-started/data-structure/data-release-versions
    # https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=32136
    sub, field, instance, run = path.stem.split("_")

    assert int(field) == UKBB_DATA_FIELD, "unexpected data field"
    assert int(instance) in {2, 3}, "unexpected instance"
    assert int(run) == 0, "unexpected run"

    assert filename in {
        "surf_fMRI/CIFTIs/bb.rfMRI.MNI.MSMAll.dtseries.nii",
        "surf_fMRI/CIFTIs/bb.tfMRI.MNI.MSMAll.dtseries.nii",
    }, "unexpected filename"
    mod = Path(filename).name.split(".")[1]

    metadata = {"sub": sub, "ses": int(instance), "mod": mod}
    return metadata


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

    # Find vertices with missing data.
    valid_mask = np.std(series, axis=0) > EPS
    invalid_count = np.sum(~valid_mask)
    if invalid_count > 0:
        _logger.warning(f"Data contains {invalid_count} vertices with missing data.")

    # Standard scale.
    series = scale(series)

    # Temporal resample.
    series = ut.resample_timeseries(series, tr=tr, new_tr=new_tr)

    # Transform to flat map space.
    series = resampler.transform(series, interpolation="linear")

    # Mask out invalid values in flat map space.
    valid_mask = resampler.transform(valid_mask, interpolation="nearest")
    series = valid_mask * series

    # Data checks.
    assert not np.any(np.isnan(series)), "series contains nan"
    assert np.all((series != 0) == valid_mask), "unexpected sparsity pattern"
    assert valid_mask[~resampler.mask_].sum() == 0, "unexpected sparsity pattern"
    assert valid_mask[resampler.mask_].mean() > 0.99, "too many missing values"
    vmax = np.max(np.abs(series))
    assert vmax < 100, f"series contains large values {vmax=:.3f}"

    # Apply mask in flat map space.
    series = series[:, resampler.mask_]

    # Cast dtype. Raise on any overflows.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        series = series.astype(dtype)

    return series


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
