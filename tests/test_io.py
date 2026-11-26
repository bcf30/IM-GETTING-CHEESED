"""Unit and property tests for `mozzavid.io` (Requirements 1, 2, 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from mozzavid import io


# --- Property 2: Load dtype (Requirements 3.1, 3.6, 3.7) ---


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_load_tiff_float_dtype_identity(tmp_path: Path, seed: int, dtype: np.dtype) -> None:
    """Float TIFFs load as float32 with identical values and shape, and round-trip through save."""
    rng = np.random.default_rng(seed)
    arr = rng.random((7, 5, 9)).astype(dtype)
    path = tmp_path / "vol.tiff"
    tifffile.imwrite(path, arr)

    loaded = io.load_tiff(path)

    assert loaded.dtype == np.float32
    assert loaded.shape == arr.shape
    np.testing.assert_array_equal(loaded, arr.astype(np.float32))

    out = tmp_path / "roundtrip.tiff"
    io.save_tiff_float32(loaded, out)
    reloaded = io.load_tiff(out)
    np.testing.assert_array_equal(reloaded, loaded)
    assert reloaded.shape == loaded.shape


# --- load_tiff branches and errors (Requirements 3.2, 3.3, 3.4, 3.5) ---


def test_load_tiff_integer_branch(tmp_path: Path) -> None:
    """Integer TIFFs are divided by the dtype maximum to land in [0,1]."""
    arr = np.array([[[0, 255], [128, 64]]], dtype=np.uint8)
    path = tmp_path / "int.tiff"
    tifffile.imwrite(path, arr)

    loaded = io.load_tiff(path)

    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, arr / 255.0)


def test_load_tiff_rejects_non_3d(tmp_path: Path) -> None:
    """A 2D TIFF raises ValueError naming the shape."""
    path = tmp_path / "flat.tiff"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="3D volume"):
        io.load_tiff(path)


def test_load_tiff_rejects_nan(tmp_path: Path) -> None:
    """Non-finite values raise ValueError."""
    arr = np.zeros((2, 2, 2), dtype=np.float32)
    arr[0, 0, 0] = np.nan
    path = tmp_path / "nan.tiff"
    tifffile.imwrite(path, arr)
    with pytest.raises(ValueError, match="finite"):
        io.load_tiff(path)


def test_load_tiff_rejects_out_of_range(tmp_path: Path) -> None:
    """Float values above 1 raise ValueError."""
    arr = np.full((2, 2, 2), 2.0, dtype=np.float32)
    path = tmp_path / "big.tiff"
    tifffile.imwrite(path, arr)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        io.load_tiff(path)


def test_load_tiff_rejects_unsupported_dtype(tmp_path: Path) -> None:
    """A complex/boolean dtype raises TypeError naming the dtype."""
    arr = np.zeros((2, 2, 2), dtype=bool)
    path = tmp_path / "bool.tiff"
    tifffile.imwrite(path, arr)
    with pytest.raises(TypeError, match="unsupported dtype"):
        io.load_tiff(path)


# --- resolve_data_root precedence (Requirement 1) ---


def test_resolve_data_root_cli_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """A passed value takes precedence over the environment variable."""
    monkeypatch.setenv(io.DATA_ROOT_ENV, "from_env")
    assert io.resolve_data_root("from_cli") == Path("from_cli").resolve()


def test_resolve_data_root_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no CLI value, the environment variable is used."""
    monkeypatch.setenv(io.DATA_ROOT_ENV, "from_env")
    assert io.resolve_data_root(None) == Path("from_env").resolve()


def test_resolve_data_root_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither a CLI value nor the variable set, `./Small` is used, as an absolute path."""
    monkeypatch.delenv(io.DATA_ROOT_ENV, raising=False)
    resolved = io.resolve_data_root(None)
    assert resolved.is_absolute()
    assert resolved == io.DEFAULT_DATA_ROOT.resolve()


# --- load_metadata on tmp_data (Requirement 2) ---


def test_load_metadata_basic(tmp_data: Path) -> None:
    """Metadata has one row per volume, derived keys and a stable sorted order."""
    df = io.load_metadata(tmp_data)
    assert len(df) == 6
    assert set(df["cheese_type"]) == {"cheese_1", "cheese_2"}
    assert df["sample_id"].iloc[0] == "cheese_1/1"
    assert (df["cheese_index"] == df["label_cheese"] + 1).all()
    # Stable order across calls.
    again = io.load_metadata(tmp_data)
    pd.testing.assert_frame_equal(df, again)


def test_load_metadata_handles_non_contiguous_folders(tmp_data: Path) -> None:
    """A gap in sample-folder numbering (folder 2 missing) is accepted (Requirement 2.4)."""
    df = io.load_metadata(tmp_data)
    assert "cheese_1/3" in set(df["sample_id"])


def test_load_metadata_raises_on_missing_file(tmp_data: Path, tmp_path: Path) -> None:
    """A CSV entry with no TIFF on disk raises ValueError."""
    root = _copy_dataset(tmp_data, tmp_path)
    (root / "cheese_1/1/0.tiff").unlink()
    with pytest.raises(ValueError, match="CSV/disk"):
        io.load_metadata(root)


def test_load_metadata_raises_on_extra_file(tmp_data: Path, tmp_path: Path) -> None:
    """A TIFF on disk absent from the CSV raises ValueError."""
    root = _copy_dataset(tmp_data, tmp_path)
    tifffile.imwrite(root / "cheese_2/2/2.tiff", np.zeros((4, 4, 4), dtype=np.float32), photometric="minisblack")
    with pytest.raises(ValueError, match="CSV/disk"):
        io.load_metadata(root)


def test_load_metadata_raises_on_wrong_label(tmp_data: Path, tmp_path: Path) -> None:
    """A row where `label_cheese != cheese_index - 1` raises ValueError."""
    root = _copy_dataset(tmp_data, tmp_path)
    csv = pd.read_csv(root / "dataset_coarse.csv")
    csv.loc[0, "label_cheese"] = 9
    csv.to_csv(root / "dataset_coarse.csv", index=False)
    with pytest.raises(ValueError, match="label_cheese"):
        io.load_metadata(root)


def test_load_metadata_raises_on_undersized_type(tmp_data: Path, tmp_path: Path) -> None:
    """A cheese type with only one sample raises ValueError (Requirement 2.8)."""
    root = _copy_dataset(tmp_data, tmp_path)
    # Merge cheese_2's two samples into one so the type keeps only a single sample.
    _relabel_sample(root, "cheese_2/2/0.tiff", "cheese_2/1/9.tiff", new_dir="cheese_2/1")
    _relabel_sample(root, "cheese_2/2/1.tiff", "cheese_2/1/8.tiff", new_dir="cheese_2/1")
    with pytest.raises(ValueError, match="fewer than 2 samples"):
        io.load_metadata(root)


def test_load_metadata_raises_on_escaping_path(tmp_data: Path, tmp_path: Path) -> None:
    """A `..` path entry raises ValueError (Requirement 2.10)."""
    root = _copy_dataset(tmp_data, tmp_path)
    csv = pd.read_csv(root / "dataset_coarse.csv")
    csv.loc[0, "file"] = "../escape.tiff"
    csv.to_csv(root / "dataset_coarse.csv", index=False)
    with pytest.raises(ValueError, match="outside Data_Root"):
        io.load_metadata(root)


# --- helpers ---


def _copy_dataset(src: Path, tmp_path: Path) -> Path:
    """Copy the `tmp_data` dataset into a fresh directory so mutations never touch the fixture."""
    import shutil

    dst = tmp_path / "dataset_copy"
    shutil.copytree(src, dst)
    return dst


def _relabel_sample(root: Path, old_rel: str, new_rel: str, new_dir: str) -> None:
    """Move a TIFF on disk and update its CSV `file` entry so the layout stays consistent."""
    (root / new_dir).mkdir(parents=True, exist_ok=True)
    (root / old_rel).rename(root / new_rel)
    csv = pd.read_csv(root / "dataset_coarse.csv")
    csv.loc[csv["file"] == old_rel, "file"] = new_rel
    csv.to_csv(root / "dataset_coarse.csv", index=False)
