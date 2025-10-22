# Copyright (c) Sophont, Inc
#
# This source code is licensed under the CC-BY-NC license
# found in the LICENSE file in the root directory of this source tree.

"""Utilities for processing surface vertex data to flat maps.

Includes utils for:
- loading pycortex flat maps
- resampling surface vertex data to flat raster grid using pycortex flat surfaces
- resampling time series to fixed TR
- loading cifti surface data
- resampling data from fsaverage to 32k_fs_LR
- computing parcellated time series
"""

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, NamedTuple

import cortex
import nibabel as nib
import numpy as np
import scipy.signal
import scipy.interpolate
from matplotlib.tri import Triangulation
from nibabel.cifti2 import BrainModelAxis
from nibabel.cifti2.cifti2 import Cifti2Image
from scipy.spatial import Delaunay
from scipy.sparse import csr_array
from sklearn.neighbors import NearestNeighbors


class Surface(NamedTuple):
    points: np.ndarray
    polys: np.ndarray


def extract_valid_flat(
    flat: Surface,
    roi_mask: np.ndarray | None = None,
    edge_threshold: float | None = 16.0,
) -> tuple[Surface, np.ndarray]:
    """Extract the valid flat patch from a full flat surface.

    Args:
        flat: flat map surface
        roi_mask: optional roi mask
        edge_threshold: distance threshold to remove stretched out triangles.

    Returns:
        tuple of surface patch and vertex mask.
    """
    points, polys = flat

    # Drop points with non-zero z component (known issue of pycortex flat maps).
    # Cf https://github.com/gallantlab/pycortex/issues/497
    mask = np.abs(points[:, 2]) < 1e-5

    # Intersect with a general cortex mask, e.g. Schaefer parcellation.
    if roi_mask is not None:
        mask = mask & (roi_mask.flatten() > 0)

    # Extract masked surface patch.
    points, polys = extract_patch((points, polys), mask)

    # Drop z coordinate.
    points = points[:, :2]

    # Drop overly stretched out triangles at edges.
    if edge_threshold is not None and edge_threshold > 0:
        lengths = triangle_longest_side((points, polys))
        polys = polys[lengths < edge_threshold]
    return Surface(points, polys), mask


def load_flat(subject: str = "fsaverage", hemi_padding: float = 8.0) -> Surface:
    """Load merged flat surface from pycortex."""
    surf_lh = cortex.db.get_surf(subject, "flat", hemisphere="lh")
    surf_rh = cortex.db.get_surf(subject, "flat", hemisphere="rh")
    surf = stack_surfaces(surf_lh, surf_rh, hemi_padding=hemi_padding)
    return surf


def stack_surfaces(surf_lh: Surface, surf_rh: Surface, hemi_padding: float = 8.0) -> Surface:
    """Stack left and right surfaces along the x (left-right) axis."""
    points_lh, polys_lh = surf_lh
    points_rh, polys_rh = surf_rh

    points_lh = points_lh.copy()
    points_lh[:, 0] = points_lh[:, 0] - points_lh[:, 0].max() - hemi_padding

    points_rh = points_rh.copy()
    points_rh[:, 0] = points_rh[:, 0] - points_rh[:, 0].min() + hemi_padding

    points = np.concatenate([points_lh, points_rh])
    polys = np.concatenate([polys_lh, len(points_lh) + polys_rh])
    return Surface(points, polys)


def extract_patch(surf: Surface, mask: np.ndarray) -> Surface:
    """Extract the surface patch for a given mask."""
    points, polys = surf
    mask = mask.astype(bool)

    mask_points = points[mask]
    mask_indices = np.cumsum(mask) - 1
    poly_mask = mask[polys]
    poly_mask = np.all(poly_mask, axis=1)
    mask_polys = polys[poly_mask]
    mask_polys = mask_indices[mask_polys]
    return Surface(mask_points, mask_polys)


def triangle_area(surf: Surface) -> np.ndarray:
    """Calculate the area of each triangle."""
    points, polys = surf
    assert points.shape[1] == 2, "triangle area only implemented for 2D surfaces."
    A = points[polys[:, 0]]
    B = points[polys[:, 1]]
    C = points[polys[:, 2]]
    AB = B - A
    AC = C - A
    cross = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    return 0.5 * np.abs(cross)


def triangle_longest_side(surf: Surface) -> np.ndarray:
    """Calculate the longest side length of each triangle."""
    points, polys = surf
    A = points[polys[:, 0]]
    B = points[polys[:, 1]]
    C = points[polys[:, 2]]
    ab = np.linalg.norm(B - A, axis=1)
    bc = np.linalg.norm(C - B, axis=1)
    ca = np.linalg.norm(A - C, axis=1)
    return np.maximum.reduce([ab, bc, ca])


def get_cifti_surf_data(
    cifti: Cifti2Image,
    hemi: Literal["lh", "rh", "both"] = "both",
) -> np.ndarray:
    """Get cifti scalar/series data for a given surface."""
    match hemi:
        case "lh":
            data = get_cifti_struct_data(cifti, "CIFTI_STRUCTURE_CORTEX_LEFT")
        case "rh":
            data = get_cifti_struct_data(cifti, "CIFTI_STRUCTURE_CORTEX_RIGHT")
        case "both":
            lh_data = get_cifti_surf_data(cifti, "lh")
            rh_data = get_cifti_surf_data(cifti, "rh")
            data = np.concatenate([lh_data, rh_data], axis=0)
        case _:
            raise ValueError(f"Invalid hemi {hemi}")
    return data


def get_cifti_struct_data(cifti: Cifti2Image, struct: str) -> np.ndarray:
    """Get cifti scalar/series data for a given brain structure."""
    axis = get_brain_model_axis(cifti)
    data = cifti.get_fdata().T
    for name, indices, model in axis.iter_structures():
        if name == struct:
            num_verts = model.vertex.max() + 1
            struct_data = np.zeros((num_verts,) + data.shape[1:], dtype=data.dtype)
            struct_data[model.vertex] = data[indices]
            return struct_data
    raise ValueError(f"Invalid cifti struct {struct}")


def get_brain_model_axis(cifti: Cifti2Image) -> BrainModelAxis:
    for ii in range(cifti.ndim):
        axis = cifti.header.get_axis(ii)
        if isinstance(axis, BrainModelAxis):
            return axis
    raise ValueError("No brain model axis found in cifti")


class Bbox(NamedTuple):
    """Bounding box with format (xmin, xmax, ymin, ymax)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float


class FlatResampler:
    """Resample data from a surface mesh to a raster grid using a flat map.

    Args:
        pixel_size: size of desired pixels in original units.
        rect: Initial data bounding box (left, right, bottom, top), before any padding.
            Defaults to the bounding box of the points.
        pad_width: Width of padding in pixels. If an integer, the same padding is
            applied to all sides. If a list of tuples, the padding is applied
            independently to each side.
        pad_to_multiple: Pad the grid width and height to be a multiple of this value.
    """

    bbox_: Bbox
    grid_: np.ndarray
    mask_: np.ndarray
    n_points_: int

    def __init__(
        self,
        pixel_size: float,
        rect: Bbox | None = None,
        pad_width: int | None = None,
        pad_to_multiple: int | None = None,
    ):
        self.pixel_size = pixel_size
        self.rect = rect
        self.pad_width = pad_width
        self.pad_to_multiple = pad_to_multiple

    def fit(self, surf: Surface) -> "FlatResampler":
        points, polys = surf

        # Fit raster grid to the scattered points.
        grid, bbox = fit_grid(
            points,
            pixel_size=self.pixel_size,
            rect=self.rect,
            pad_width=self.pad_width,
            pad_to_multiple=self.pad_to_multiple,
        )

        # Get mask of pixels contained in the surface interior.
        tri = Triangulation(points[:, 0], points[:, 1], polys)
        trifinder = tri.get_trifinder()
        tri_indices = trifinder(grid[0], grid[1])
        mask = tri_indices >= 0

        # Pre-compute delaunay triangulation for linear interpolation.
        delaunay = Delaunay(points)

        # Pre-compute vertex to grid neighbor mapping for nearest interpolation.
        mask_points = grid[:, mask].T
        nbrs = NearestNeighbors().fit(points)
        neigh_ind = nbrs.kneighbors(mask_points, n_neighbors=1, return_distance=False)
        neigh_ind = neigh_ind.squeeze(1)

        self.bbox_ = bbox
        self.grid_ = grid
        self.mask_ = mask
        self.n_points_ = len(points)

        self._delaunay = delaunay
        self._neigh_ind = neigh_ind
        return self

    def transform(
        self,
        data: np.ndarray,
        fill_value: Any = 0,
        interpolation: Literal["nearest", "linear"] = "nearest",
    ) -> np.ndarray:
        """Transform scattered data onto regular grid.

        Args:
            data: scattered data, shape (..., n_points,).
            fill_value: value to fill pixels outside mask area.
            interpolation: interpolation method (nearest or linear)

        Returns:
            Transformed data, shape (..., height, width).
        """
        assert hasattr(self, "bbox_"), "Resampler not fit; call fit() first."
        assert data.shape[-1] == self.n_points_, "Data does not match resampler."
        assert interpolation in ("nearest", "linear"), "Invalid interpolation"

        if interpolation == "linear":
            data = self._linear_interpolate(data)
        else:
            data = self._nearest_interpolate(data)
        data = np.where(self.mask_, data, fill_value)
        return data

    def _linear_interpolate(self, data: np.ndarray) -> np.ndarray:
        """Resample vertex data to flat map grid using linear interpolation."""
        is_nd = data.ndim > 1
        if is_nd:
            leading_dims = data.shape[:-1]
            data = data.reshape(-1, data.shape[-1]).T
        interp = scipy.interpolate.LinearNDInterpolator(self._delaunay, data)
        xx, yy = self.grid_
        flat_data = interp(xx, yy)
        if is_nd:
            flat_data = np.transpose(flat_data, (2, 0, 1))
            flat_data = flat_data.reshape(leading_dims + self.mask_.shape)
        return flat_data

    def _nearest_interpolate(self, data: np.ndarray) -> np.ndarray:
        """Resample vertex data to flat map grid using nearest neighbor mapping."""
        leading_dims = data.shape[:-1]
        flat_data = np.zeros(leading_dims + self.mask_.shape, dtype=data.dtype)
        flat_data[..., self.mask_] = data[..., self._neigh_ind]
        return flat_data


def fit_grid(
    points: np.ndarray,
    pixel_size: float,
    rect: Bbox | None = None,
    pad_width: int | list[tuple[int, int]] | None = None,
    pad_to_multiple: int | None = None,
) -> tuple[np.ndarray, Bbox]:
    """Fit a regular grid to scattered points with desired padding and pixel size.

    Args:
        points: array of (x, y) points, shape (num_points, 2).
        pixel_size: pixel size in data units.
        rect: Initial data bounding box (left, right, bottom, top), before any padding.
            Defaults to the bounding box of the points.
        pad_width: Width of padding in pixels. If an integer, the same padding is
            applied to all sides. If a list of tuples, the padding is applied
            independently to each side.
        pad_to_multiple: Pad the grid width and height to be a multiple of this value.

    Returns:
        A tuple of the grid, shape (2, height, width), and bounding box.
    """
    if rect is None:
        xmin, ymin = np.floor(points.min(axis=0))
        xmax, ymax = np.ceil(points.max(axis=0))
    else:
        xmin, xmax, ymin, ymax = rect

    w = round((xmax - xmin) / pixel_size)
    h = round((ymax - ymin) / pixel_size)
    xmax = xmin + pixel_size * w
    ymax = ymin + pixel_size * h

    if pad_width:
        xmin, xmax, ymin, ymax = _pad_bbox((xmin, xmax, ymin, ymax), pad_width, pixel_size)

    if pad_to_multiple:
        w = round((xmax - xmin) / pixel_size)
        h = round((ymax - ymin) / pixel_size)
        padw = math.ceil(w / pad_to_multiple) * pad_to_multiple - w
        padh = math.ceil(h / pad_to_multiple) * pad_to_multiple - h
        pad_width_multiple = [
            (padh // 2, padh - padh // 2),
            (padw // 2, padw - padw // 2),
        ]
        xmin, xmax, ymin, ymax = _pad_bbox(
            (xmin, xmax, ymin, ymax),
            pad_width_multiple,
            pixel_size,
        )

    # Nb, this is more reliable than e.g. arange due to floating point errors.
    w = round((xmax - xmin) / pixel_size)
    h = round((ymax - ymin) / pixel_size)
    x = xmin + pixel_size * np.arange(w)
    y = ymax - pixel_size * np.arange(h)  # Nb, upper origin
    grid = np.stack(np.meshgrid(x, y))
    return grid, Bbox(xmin, xmax, ymin, ymax)


def _pad_bbox(
    rect: Bbox,
    pad_width: int | list[tuple[int, int]],
    pixel_size: float,
) -> Bbox:
    xmin, xmax, ymin, ymax = rect
    if isinstance(pad_width, int):
        pad_width = [(pad_width, pad_width)] * 2
    ymin -= pixel_size * pad_width[0][0]
    ymax += pixel_size * pad_width[0][1]
    xmin -= pixel_size * pad_width[1][0]
    xmax += pixel_size * pad_width[1][1]
    return Bbox(xmin, xmax, ymin, ymax)


def resample_timeseries(
    series: np.ndarray,
    tr: float,
    new_tr: float = 1.0,
    antialias: bool = True,
) -> np.ndarray:
    """Resample a time series to a target TR.

    Args:
        series: time series, shape (n_samples, dim).
        tr: repetition time, i.e. 1 / fs.
        new_tr: target repetition time.
        antialias: apply an antialising filter before downsampling.

    Returns:
        Resampled time series, shape (n_new_samples, dim).
    """
    if tr == new_tr:
        return series

    fs = 1.0 / tr
    new_fs = 1.0 / new_tr

    # Anti-aliasing low-pass filter
    # Copied from scipy.signal.decimate
    if antialias and new_fs < fs:
        q = fs / new_fs
        sos = scipy.signal.cheby1(8, 0.05, 0.8 / q, output="sos")
        series = scipy.signal.sosfiltfilt(sos, series, axis=0, padtype="even")

    # Nb, this is more reliable than arange(0, duration, tr) due to floating point
    # errors.
    x = tr * np.arange(len(series))
    new_length = int(tr * len(series) / new_tr)
    new_x = new_tr * np.arange(new_length)

    interp = scipy.interpolate.PchipInterpolator(x, series, axis=0)
    series = interp(new_x)
    return series


def fsaverage_to_32k_fs_LR(
    data: np.ndarray,
    hemi: Literal["lh", "rh"],
    resample_fsaverage_dir: str | Path,
) -> np.ndarray:
    """Resample surface metric data from fsaverage to 32k_fs_LR using wb_command.

    Args:
        data: surface metric data, shape (n_vertices,) or (n_samples, n_vertices)
        hemi: lh or rh
        resample_fsaverage_dir: path to hcp template surfaces for resampling.

    Returns:
        Resampled data array, shape (n_vertices,) or (n_samples, n_vertices)

    Reference:
        https://wiki.humanconnectome.org/docs/assets/Resampling-FreeSurfer-HCP_5_8.pdf
        https://github.com/Washington-University/HCPpipelines/tree/master/global/templates/standard_mesh_atlases/resample_fsaverage
    """
    assert data.ndim in (1, 2), f"Invalid data shape {data.shape}."

    is_1d = data.ndim == 1
    if is_1d:
        data = data[None, :]

    resample_fsaverage_dir = Path(resample_fsaverage_dir)
    hcp_hemi = {"lh": "L", "rh": "R"}[hemi]

    # Template paths needed for resampling.
    fsaverage_sphere = str(
        resample_fsaverage_dir / f"fsaverage_std_sphere.{hcp_hemi}.164k_fsavg_{hcp_hemi}.surf.gii"
    )
    fslr_sphere = str(
        resample_fsaverage_dir / f"fs_LR-deformed_to-fsaverage.{hcp_hemi}.sphere.32k_fs_LR.surf.gii"
    )
    fsaverage_area = str(
        resample_fsaverage_dir
        / f"fsaverage.{hcp_hemi}.midthickness_va_avg.164k_fsavg_{hcp_hemi}.shape.gii"
    )
    fslr_area = str(
        resample_fsaverage_dir / f"fs_LR.{hcp_hemi}.midthickness_va_avg.32k_fs_LR.shape.gii"
    )

    with tempfile.TemporaryDirectory(prefix="wb_command-") as tmpdir:
        input_path = str(Path(tmpdir) / f"input.fsaverage.{hemi}.func.gii")
        output_path = str(Path(tmpdir) / f"output.32k_fs_LR.{hemi}.func.gii")

        # Save data as gifti metric.
        img = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(row) for row in data])
        nib.save(img, input_path)

        # Run `wb_command -metric-resample``.
        cmd = [
            "wb_command",
            "-metric-resample",
            input_path,
            fsaverage_sphere,
            fslr_sphere,
            "ADAP_BARY_AREA",
            output_path,
            "-area-metrics",
            fsaverage_area,
            fslr_area,
        ]
        subprocess.run(cmd, check=True)

        # Load gifti metric back as numpy array.
        img = nib.load(output_path)
        data = np.stack([darray.data for darray in img.darrays])
        if is_1d:
            data = np.squeeze(data, 0)

    return data


def parc_to_one_hot(parc: np.ndarray, sparse: bool = True) -> np.ndarray:
    """Get one hot encoding of the parcellation.

    Args:
        parc: parcellation of shape (num_vertices,) with values in [0, num_rois] where 0
            is background.

    Returns:
        parc_one_hot: one hot encoding of parcellation, shape (num_vertices, num_rois).
    """
    (num_verts,) = parc.shape
    parc = parc.astype(np.int32)
    num_rois = parc.max()

    # one hot parcellation matrix, shape (num_vertices, num_rois)
    if sparse:
        mask = parc > 0
        (row_ind,) = mask.nonzero()
        col_ind = parc[mask] - 1
        values = np.ones(len(row_ind), dtype=np.float32)
        parc_one_hot = csr_array((values, (row_ind, col_ind)), shape=(num_verts, num_rois))
    else:
        parc_one_hot = parc[:, None] == np.arange(1, num_rois + 1)
        parc_one_hot = parc_one_hot.astype(np.float32)
    return parc_one_hot


def parcellate_timeseries(
    series: np.ndarray, parc_one_hot: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Extract parcellated time series.

    Args:
        series: full time series (num_samples, num_vertices)
        parc_one_hot: one hot encoding of parcellation (num_vertices, num_rois)

    Returns:
        parc_series: parcellated time series (num_samples, num_rois)
    """
    parc_one_hot = parc_one_hot.astype(series.dtype)

    # don't include verts with missing data
    data_mask = np.var(series, axis=0) > eps
    parc_one_hot = parc_one_hot * data_mask[:, None]

    # normalize weights to sum to 1
    # Nb, empty parcels will be all zero
    parc_counts = np.asarray(parc_one_hot.sum(axis=0))
    parc_one_hot = parc_one_hot / np.maximum(parc_counts, 1)

    # per roi averaging
    parc_series = series @ parc_one_hot
    return parc_series
