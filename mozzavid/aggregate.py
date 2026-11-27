"""Per-sample and per-volume aggregation of object measurements (Requirement 14).

Pure Python (numpy/pandas only). `summarize` pools the objects of one sample into a single
row; `summarize_all` and `summarize_per_volume` reuse it so the per-sample and per-volume
metric definitions cannot drift apart. The Sample is the statistical unit (design H10).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOXELS_PER_VOLUME: int = 192**3  # MozzaVID Small volumes are 192^3; used for analyzed volume


def summarize(objects: pd.DataFrame, sample_id: str, cheese_type: str,
              analyzed_volume_um3: float) -> dict:
    """Pool the objects of one sample (or volume) into one summary row (Requirement 14.2-14.6).

    Size statistics use interior objects only (number-weighted mean/median/std; volume-weighted
    d10/d50/d90). `foreground_fraction` and `n_objects` include border-touching objects. Results
    do not depend on the row order of `objects`.
    """
    interior = objects.loc[~objects["touches_border"]]
    diameters = interior["equiv_diameter_um"].to_numpy(dtype=float)
    weights = interior["volume_um3"].to_numpy(dtype=float)
    d10, d50, d90 = _weighted_percentiles(diameters, weights, (10.0, 50.0, 90.0))
    return {
        "sample_id": sample_id, "cheese_type": cheese_type,
        "foreground_fraction": float(np.sort(objects["volume_um3"].to_numpy(dtype=float)).sum()) / analyzed_volume_um3,
        "n_objects": int(len(objects)), "n_interior_objects": int(len(interior)),
        "objects_per_mm3": len(objects) / (analyzed_volume_um3 * 1e-9),
        **_number_stats(np.sort(diameters)), "d10_um": d10, "d50_um": d50, "d90_um": d90,
    }


def _number_stats(diameters: np.ndarray) -> dict:
    """Return number-weighted mean/median/std of equivalent diameter, or NaN when empty (Req 14.5, 14.7)."""
    if diameters.size == 0:
        return {"mean_equiv_diameter_um": np.nan, "median_equiv_diameter_um": np.nan,
                "std_equiv_diameter_um": np.nan}
    return {"mean_equiv_diameter_um": float(np.mean(diameters)),
            "median_equiv_diameter_um": float(np.median(diameters)),
            "std_equiv_diameter_um": float(np.std(diameters, ddof=0))}


def _weighted_percentiles(values: np.ndarray, weights: np.ndarray,
                          percentiles: tuple[float, ...]) -> tuple[float, ...]:
    """Volume-weighted percentiles via cumulative weight fraction and linear interpolation (Req 14.6).

    Returns NaN for each requested percentile when there are no values. The cumulative fraction
    is centred on each value's weight interval, the standard definition for D_v10/50/90.
    """
    if values.size == 0:
        return tuple(np.nan for _ in percentiles)
    order = np.argsort(values, kind="stable")
    sorted_values, sorted_weights = values[order], weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    fraction = cumulative / sorted_weights.sum()
    return tuple(float(np.interp(p / 100.0, fraction, sorted_values)) for p in percentiles)


def summarize_all(objects: pd.DataFrame, processed_meta: pd.DataFrame,
                  voxel_size_um: float) -> pd.DataFrame:
    """One summary row per Sample, iterating over processed metadata (Requirement 14.1).

    Iterating over `processed_meta` (not `objects`) means a sample whose volumes produced no
    objects still gets a row. `analyzed_volume_um3 = n_volumes × 192³ × s³`.
    """
    rows = []
    for sample_id, group in processed_meta.groupby("sample_id", sort=True):
        n_volumes = int(len(group))
        analyzed = n_volumes * VOXELS_PER_VOLUME * voxel_size_um**3
        sample_objects = objects.loc[objects["sample_id"] == sample_id] if "sample_id" in objects else objects.iloc[0:0]
        row = summarize(sample_objects, sample_id, str(group["cheese_type"].iloc[0]), analyzed)
        rows.append({**row, "n_volumes": n_volumes})
    return _order_summary(pd.DataFrame(rows), per_volume=False)


def summarize_per_volume(objects: pd.DataFrame, processed_meta: pd.DataFrame,
                         voxel_size_um: float) -> pd.DataFrame:
    """One summary row per processed Volume, for the ICC (Requirement 14.10).

    Uses `analyzed_volume_um3 = 192³ × s³`, so a zero-object volume still gets a row.
    """
    analyzed = VOXELS_PER_VOLUME * voxel_size_um**3
    rows = []
    for _, meta_row in processed_meta.iterrows():
        volume_id = meta_row["volume_id"]
        volume_objects = objects.loc[objects["volume_id"] == volume_id] if "volume_id" in objects else objects.iloc[0:0]
        row = summarize(volume_objects, meta_row["sample_id"], str(meta_row["cheese_type"]), analyzed)
        rows.append({**row, "volume_id": volume_id})
    return _order_summary(pd.DataFrame(rows), per_volume=True)


def _order_summary(frame: pd.DataFrame, per_volume: bool) -> pd.DataFrame:
    """Place identifier columns first in a stable order and sort rows deterministically."""
    lead = ["volume_id"] if per_volume else []
    lead += ["sample_id", "cheese_type"] + ([] if per_volume else ["n_volumes"])
    rest = [c for c in frame.columns if c not in lead]
    sort_keys = ["sample_id", "volume_id"] if per_volume else ["cheese_type", "sample_id"]
    return frame[lead + rest].sort_values(sort_keys, ignore_index=True)
