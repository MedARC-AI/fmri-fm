# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
from contextlib import contextmanager
from functools import partialmethod
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from nsdcode import NSDmapdata
from tqdm import tqdm

from data.surface_utils import fsaverage_to_32k_fs_LR

# Disable tqdm by default
# https://stackoverflow.com/a/67238486
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s",
    level=logging.WARNING,
    datefmt="%y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

RESAMPLE_FSAVERAGE_DIR = os.getenv(
    "RESAMPLE_FSAVERAGE_DIR",
    "HCPpipelines/global/templates/standard_mesh_atlases/resample_fsaverage",
)
RESAMPLE_FSAVERAGE_DIR = Path(RESAMPLE_FSAVERAGE_DIR)

if not RESAMPLE_FSAVERAGE_DIR.exists():
    raise FileNotFoundError(
        f"resample_fsaverage directory {RESAMPLE_FSAVERAGE_DIR} doesn't exist; clone HCPpipelines?"
    )

BATCH_SIZE = 16


def main(
    path: str | Path,
    out_dir: str | Path | None = None,
    out_base: str | Path | None = None,
    nsd_dir: str | Path | None = None,
    subid: int | None = None,
    func_res: str | None = None,
    overwrite: bool = False,
):
    path = Path(path)

    # Parse info from the path.
    # Example paths:
    #     natural-scenes-dataset/nsddata_timeseries/ppdata/subj01/func1mm/timeseries/timeseries_session01_run01.nii.gz
    #     natural-scenes-dataset/nsddata_betas/ppdata/subj01/func1mm/betas_fithrf/betas_session30.nii.gz
    nsd_dir = path.parents[5] if nsd_dir is None else Path(nsd_dir)
    if subid is None:
        subj_dir = path.parents[2]
        subid = int(subj_dir.name[-2:])
    if func_res is None:
        func_dir = path.parents[1]
        func_res = func_dir.name
    assert func_res == "func1mm", "Only func1mm data supported"
    sourcespace = {"func1mm": "func1pt0", "func1pt8mm": "func1pt8"}[func_res]

    # Prepare output base path.
    out_dir = nsd_dir if out_dir is None else Path(out_dir)
    if out_base is None:
        out_base = out_dir / path.relative_to(path.parents[5])
        out_base = Path(str(out_base).replace(func_res, "32k_fs_LR"))
        out_base = out_base.with_name(path.name.split(".")[0])
    else:
        out_base = Path(out_base)
    out_base.parent.mkdir(exist_ok=True, parents=True)

    # Load memory mapped volume time series.
    img = nib.load(path)
    nvols = img.shape[-1]

    for hemi in ["lh", "rh"]:
        out_path = out_base.with_suffix(f".{hemi}.func.gii")
        if out_path.exists() and not overwrite:
            _logger.info("Output %s exists; skipping.", out_path)
            continue

        # We do the mapping in batches to avoid loading the full volume time series in
        # memory (~10GB).
        frames = []
        for ii in tqdm(range(0, nvols, BATCH_SIZE), disable=False):
            batch = img.slicer[..., ii : ii + BATCH_SIZE].get_fdata()

            # NSD native volume to fsaverage.
            # Suppress NSD print output.
            with suppress_print():
                batch = nsd_func_to_fsaverage(
                    batch,
                    subid=subid,
                    hemi=hemi,
                    nsd_dir=nsd_dir,
                    sourcespace=sourcespace,
                )

            # fsaverage to 32k_fs_LR.
            batch = fsaverage_to_32k_fs_LR(
                batch,
                hemi=hemi,
                resample_fsaverage_dir=RESAMPLE_FSAVERAGE_DIR,
            )

            frames.extend(list(batch))

        # Save data as gifti.
        out_img = nib.gifti.GiftiImage(
            darrays=[nib.gifti.GiftiDataArray(frame) for frame in frames]
        )
        nib.save(out_img, out_path)
        _logger.info("Done: %s %s", out_path, (len(frames), *frames[0].shape))


def nsd_func_to_fsaverage(
    data: np.ndarray,
    subid: int,
    hemi: Literal["lh", "rh"],
    nsd_dir: str | Path,
    sourcespace: Literal["func1pt0", "func1pt8"] = "func1pt0",
) -> np.ndarray:
    """Map native functional volume data to fsaverage following the official NSD pipeline.

    Args:
        data: data array, shape X x Y x Z x D
        subid: subject ID
        hemi: lh or rh
        nsd_dir: Path to NSD root dir.
        sourcespace: input source space, func1pt0 or func1pt8

    Returns:
        Mapped data array, shape (n_samples, n_vertices).

    Reference:
        https://cvnlab.slite.page/p/6CusMRYfk0/Functional-data-NSD#58ba0424
        https://github.com/cvnlab/nsdcode/blob/bfd36a503cc90a3eb3ebfd69b269403f7e186924/examples/examples_nsdmapdata.py#L348
    """
    nsd = NSDmapdata(nsd_dir)

    # Sampling at 3 cortical depths from volume to native surface with cubic resampling.
    data = sum(
        nsd.fit(
            subid,
            sourcespace,
            f"{hemi}.layerB{depth}",
            data,
            "cubic",
            badval=0,
        )
        for depth in range(1, 4)
    )
    data = data / 3.0

    # Nearest neighbor interpolation from native surface to fsaverage.
    data = nsd.fit(
        subid,
        f"{hemi}.white",
        "fsaverage",
        data,
        interptype=None,
        badval=0,
    )

    # Transpose to (n_samples, n_vertices) and cast dtype.
    data = np.ascontiguousarray(data.T)
    data = data.astype(np.float32)
    return data


@contextmanager
def suppress_print():
    devnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    try:
        sys.stdout = devnull
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--out-base", type=str, default=None)
    parser.add_argument("--nsd-dir", type=str, default=None)
    parser.add_argument("--subid", type=int, default=None)
    parser.add_argument("--func-res", type=str, default=None)
    parser.add_argument("--overwrite", "-x", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
