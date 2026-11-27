"""TIFF read/write, metadata CSV parsing and data-root resolution for the MozzaVID Small split.

This module is pure Python (no Fiji/JVM). It is the single source of `VOXEL_SIZE_UM`
and of the dataset layout rules (Requirements 1-4).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# Physical edge length of one voxel, in micrometres: 0.55 um raw pixel size x 8 (the Small
# downscale factor), from Pieta et al., arXiv:2412.04880 (paper S3.1 states 0.55 um; S3.2
# states the 8x downscale). Verified consistent with the stored geometry: Small volumes are
# 192^3 voxels, and 192 x 8 x 0.55 um == 192 x 4.4 um == 0.8448 mm per edge. The cubic 192^3
# shape confirms isotropic voxels; the TIFFs carry no physical resolution tags, so the paper
# remains the source for the absolute scale.
VOXEL_SIZE_UM: float = 4.4

DATA_ROOT_ENV: str = "MOZZAVID_DATA_ROOT"
DEFAULT_DATA_ROOT: Path = Path("Small")
COARSE_CSV: str = "dataset_coarse.csv"
FINE_CSV: str = "dataset_fine.csv"

# Validation errors list at most this many offending items, to keep messages readable.
_MAX_LISTED = 20


def resolve_data_root(cli_value: str | None) -> Path:
    """Return the Data_Root as an absolute path, without checking that it exists.

    Precedence: the CLI value, then the `MOZZAVID_DATA_ROOT` environment variable, then
    `./Small` (Requirement 1).
    """
    chosen = cli_value or os.environ.get(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT
    return Path(chosen).resolve()


def load_tiff(path: Path) -> np.ndarray:
    """Read a 3D TIFF as float32 (Z,Y,X) in [0,1]; float16 is cast because ImageJ has no half-float.

    Floating-point input is returned with identical values; integer input is divided by the
    dtype maximum. Raises on wrong rank, non-finite / out-of-range values or unsupported dtype
    (Requirement 3).
    """
    arr = tifffile.imread(path)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3D volume, got shape {arr.shape}")
    out = _to_float32_unit(arr, path)
    if not np.isfinite(out).all() or out.min() < 0.0 or out.max() > 1.0:
        raise ValueError(f"{path}: values must be finite and within [0,1]")
    return out


def _to_float32_unit(arr: np.ndarray, path: Path) -> np.ndarray:
    """Convert a loaded TIFF array to float32 in [0,1] by dtype branch (Requirement 3.1, 3.2, 3.5)."""
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):  # guarded, cheap; unused for MozzaVID Small
        return (arr / np.iinfo(arr.dtype).max).astype(np.float32)
    raise TypeError(f"{path}: unsupported dtype {arr.dtype}")


def save_tiff_float32(arr: np.ndarray, path: Path) -> None:
    """Write `arr` as a float32 TIFF that `load_tiff` reads back with identical values (Requirement 3.7)."""
    tifffile.imwrite(path, np.asarray(arr, dtype=np.float32))


def scan_volume_files(data_root: Path) -> list[str]:
    """Return relative `/`-separated paths of every `*.tiff` under `data_root` (Requirement 2.6)."""
    root = Path(data_root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.tiff"))


def load_metadata(data_root: Path, csv_name: str = COARSE_CSV) -> pd.DataFrame:
    """Parse the label CSV, derive cheese/sample keys from paths, and cross-check against disk.

    Returns one row per Volume sorted by `(cheese_index, sample_id, volume_id)` with a stable
    order on repeated calls. Raises `ValueError` when any validation rule fails (Requirement 2).
    """
    data_root = Path(data_root)
    df = pd.read_csv(data_root / csv_name, usecols=["file", "label_cheese", "split"])
    _check_no_escaping_paths(df)  # before key derivation, which assumes `cheese_<N>/...` layout
    df = _derive_keys(df, data_root)
    _check_files_match_disk(df, scan_volume_files(data_root))
    _check_cheese_labels(df)
    _check_group_sizes(df)
    _check_fine_csv_if_present(df, data_root)
    return df.drop(columns="file").sort_values(
        ["cheese_index", "sample_id", "volume_id"], ignore_index=True
    )


def _derive_keys(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Add `volume_id`, `cheese_type`, `cheese_index`, `sample_id` and `path` from the CSV `file` column."""
    parts = df["file"].str.split("/", expand=True)
    return df.assign(
        volume_id=df["file"],
        cheese_type=parts[0],
        sample_id=parts[0] + "/" + parts[1],
        cheese_index=parts[0].str.removeprefix("cheese_").astype(int),
        path=[data_root / f for f in df["file"]],
    )


def _check_files_match_disk(df: pd.DataFrame, disk_files: list[str]) -> None:
    """Raise if the CSV `file` set and the on-disk `*.tiff` set differ (Requirement 2.6)."""
    csv_files = set(df["file"])
    disk = set(disk_files)
    missing = sorted(csv_files - disk)  # listed in CSV but not on disk
    extra = sorted(disk - csv_files)  # on disk but not in CSV
    if missing or extra:
        listed = (missing + extra)[:_MAX_LISTED]
        raise ValueError(
            f"CSV/disk file sets differ: {len(missing)} missing, {len(extra)} extra. "
            f"Examples: {listed}"
        )


def _check_no_escaping_paths(df: pd.DataFrame) -> None:
    """Raise `ValueError` naming any `file` entry that resolves outside Data_Root (Requirement 2.10)."""
    bad = [f for f in df["file"] if ".." in Path(f).parts or Path(f).is_absolute()]
    if bad:
        raise ValueError(f"file entries resolve outside Data_Root: {bad[:_MAX_LISTED]}")


def _check_cheese_labels(df: pd.DataFrame) -> None:
    """Raise if any row has `label_cheese != cheese_index - 1` (Requirement 2.7)."""
    wrong = df[df["label_cheese"] != df["cheese_index"] - 1]
    if not wrong.empty:
        rows = wrong["volume_id"].tolist()[:_MAX_LISTED]
        raise ValueError(f"label_cheese must equal cheese_index - 1; offending volumes: {rows}")


def _check_group_sizes(df: pd.DataFrame) -> None:
    """Raise if any Sample has not 1-4 Volumes or any Cheese_Type has < 2 Samples (Requirement 2.8)."""
    per_sample = df.groupby("sample_id").size()
    bad_samples = per_sample[(per_sample < 1) | (per_sample > 4)].index.tolist()
    if bad_samples:
        raise ValueError(f"samples with volume count outside 1-4: {bad_samples[:_MAX_LISTED]}")
    samples_per_type = df.groupby("cheese_type")["sample_id"].nunique()
    bad_types = samples_per_type[samples_per_type < 2].index.tolist()
    if bad_types:
        raise ValueError(f"cheese types with fewer than 2 samples: {bad_types[:_MAX_LISTED]}")


def _check_fine_csv_if_present(df: pd.DataFrame, data_root: Path) -> None:
    """Verify `dataset_fine.csv` `label_sample` maps one-to-one onto `sample_id` (Requirement 2.9)."""
    fine_path = Path(data_root) / FINE_CSV
    if not fine_path.exists():
        return
    fine = pd.read_csv(fine_path, usecols=["file", "label_sample"])
    merged = fine.merge(df[["volume_id", "sample_id"]], left_on="file", right_on="volume_id")
    conflicts = _one_to_one_conflicts(merged)
    if conflicts:
        raise ValueError(f"fine label_sample is not one-to-one with sample_id: {conflicts[:_MAX_LISTED]}")


def _one_to_one_conflicts(merged: pd.DataFrame) -> list:
    """Return label/sample pairs that break a one-to-one mapping in either direction."""
    label_to_sample = merged.groupby("label_sample")["sample_id"].nunique()
    sample_to_label = merged.groupby("sample_id")["label_sample"].nunique()
    bad_labels = label_to_sample[label_to_sample > 1].index.tolist()
    bad_samples = sample_to_label[sample_to_label > 1].index.tolist()
    return bad_labels + bad_samples
