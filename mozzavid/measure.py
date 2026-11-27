"""Per-object 3D measurement in Fiji with conversion to physical units (Requirement 11).

`extract` labels a cleaned mask and measures every component with MorphoLibJ Analyze Regions 3D
on an uncalibrated ImagePlus, then converts the voxel-unit table to micrometres once in Python.
This keeps a single source of truth for the voxel size and avoids double scaling.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mozzavid import fiji

log = logging.getLogger(__name__)

# Output column order (Requirement 11.2). The empty-mask frame uses the same order.
_SCHEMA: tuple[str, ...] = (
    "label", "volume_um3", "equiv_diameter_um", "surface_area_um2", "sphericity",
    "feret_diameter_max_um", "centroid_z_um", "centroid_y_um", "centroid_x_um", "touches_border",
)
_SPHERICITY_MAX: float = 1.05  # digital surface estimates overshoot slightly on small objects


def extract(mask: np.ndarray, voxel_size_um: float) -> pd.DataFrame:
    """Label `mask` and measure each 3D component in Fiji; return metrics in µm (Requirement 11).

    An all-False mask returns an empty frame with the full schema. Otherwise the mask is
    labelled and measured on an uncalibrated image, the Fiji columns are renamed through
    `fiji.COLUMN_MAP`, and scaling to physical units happens once in `_to_physical`.
    """
    if not mask.any():
        fiji.collect_garbage()
        return _empty_object_frame()
    imp = fiji.mask_to_imageplus(mask)
    labels = None
    try:
        labels = fiji.label_components(imp)
        raw = fiji.table_to_dataframe(fiji.analyze_regions_3d(labels))
    finally:
        imp.close()
        if labels is not None:
            labels.close()
    raw = raw.rename(columns=fiji.COLUMN_MAP)
    physical = _to_physical(raw, float(voxel_size_um), mask.shape)
    fiji.collect_garbage()  # free this volume's Java images before the next volume
    return physical


def _empty_object_frame() -> pd.DataFrame:
    """Return an empty DataFrame carrying the full per-object schema (Requirement 11.11)."""
    frame = pd.DataFrame({name: pd.Series(dtype="float64") for name in _SCHEMA})
    frame["label"] = frame["label"].astype("int64")
    frame["touches_border"] = frame["touches_border"].astype("bool")
    return frame


def _to_physical(raw: pd.DataFrame, s: float, shape: tuple[int, int, int]) -> pd.DataFrame:
    """Scale voxel-unit Fiji measurements to µm and add derived shape metrics (Requirement 11.3-11.6)."""
    volume = raw["voxel_count"] * s**3
    surface = raw["surface_area_px2"] * s**2
    out = pd.DataFrame({
        "label": raw["label"].astype("int64"),
        "volume_um3": volume,
        "equiv_diameter_um": (6.0 * volume / np.pi) ** (1 / 3),
        "surface_area_um2": surface,
        "sphericity": np.pi ** (1 / 3) * (6.0 * volume) ** (2 / 3) / surface,
        "feret_diameter_max_um": _feret_um(raw, s),
        "centroid_z_um": raw["centroid_z_px"] * s,
        "centroid_y_um": raw["centroid_y_px"] * s,
        "centroid_x_um": raw["centroid_x_px"] * s,
        "touches_border": _touches_border(raw, shape),
    })
    _warn_high_sphericity(out)
    return out[list(_SCHEMA)]


def _feret_um(raw: pd.DataFrame, s: float) -> pd.Series:
    """Return the max Feret diameter in µm as 2 × the largest equivalent-ellipsoid semi-axis (H7).

    MorphoLibJ's headless `MaxFeretDiameter3D` is accurate but scales as O(points²) per object
    and does not finish on real volumes with ~1800 components. The equivalent-ellipsoid proxy
    is already in the region table, costs nothing extra, and is actually closer to the analytic
    max Feret on test spheres and ellipsoids (within ~1.3% vs MorphoLibJ's ~14%). See
    docs/reproducibility.md under H7.
    """
    radii = raw[["ellipsoid_radius1_px", "ellipsoid_radius2_px", "ellipsoid_radius3_px"]]
    return 2.0 * radii.max(axis=1) * s


def _touches_border(raw: pd.DataFrame, shape: tuple[int, int, int]) -> pd.Series:
    """True where a component's bounding box reaches a volume face (Requirement 11.6).

    MorphoLibJ's `Box.*.Min` is the minimum voxel index and `Box.*.Max` is the maximum index
    plus one, so a face at index 0 has `Min == 0` and a face at the last index has `Max == L`
    (see docs/reproducibility.md). `shape` is (Z,Y,X), so X maps to length `shape[2]`.
    """
    z, y, x = shape
    touches = (
        (raw["bbox_x_min_px"] == 0) | (raw["bbox_x_max_px"] == x)
        | (raw["bbox_y_min_px"] == 0) | (raw["bbox_y_max_px"] == y)
        | (raw["bbox_z_min_px"] == 0) | (raw["bbox_z_max_px"] == z)
    )
    return touches.astype(bool)


def _warn_high_sphericity(out: pd.DataFrame) -> None:
    """Log a warning for every row whose sphericity exceeds 1.05, keeping the row (Requirement 11.12)."""
    for label in out.loc[out["sphericity"] > _SPHERICITY_MAX, "label"]:
        log.warning("object label %s has sphericity > %.2f; keeping the row", label, _SPHERICITY_MAX)
