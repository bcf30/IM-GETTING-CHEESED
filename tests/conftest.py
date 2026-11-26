"""Shared pytest fixtures for the mozzavid Test_Suite (Requirement 25.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile


@pytest.fixture(scope="session")
def ij():
    """Return the headless ImageJ gateway shared by the whole test session.

    Session scope pays the 15-20 s JVM start once. Errors from `fiji.get_ij()` are
    deliberately not caught: a broken Fiji install must show up as test errors, not as
    skips that hide it (Requirement 25.7). `mozzavid.fiji` is imported here rather than at
    module level so tests that do not use this fixture are collected even if the JVM
    stack (imagej/scyjava) cannot be imported.
    """
    from mozzavid import fiji

    return fiji.get_ij()


@pytest.fixture
def rng():
    """Return a numpy Generator seeded with 0.

    Function scope gives every test a fresh generator, so a test's random draws do not
    depend on which other tests ran before it.
    """
    return np.random.default_rng(0)


# Miniature dataset for `tmp_data`: two cheese types, two samples each, a non-contiguous
# sample folder (`cheese_1` folders 1 and 3), and both float dtypes. `label_cheese` is
# `cheese_index - 1`. Volume counts stay within 1-4 and each type keeps >= 2 samples, so
# `load_metadata` validation passes.
_TMP_LAYOUT: tuple[tuple[str, int, str], ...] = (
    ("cheese_1/1/0.tiff", 0, "train"),
    ("cheese_1/1/1.tiff", 0, "train"),
    ("cheese_1/3/0.tiff", 0, "train"),  # non-contiguous: folder 2 is skipped
    ("cheese_2/1/0.tiff", 1, "val"),
    ("cheese_2/2/0.tiff", 1, "val"),
    ("cheese_2/2/1.tiff", 1, "val"),
)


@pytest.fixture(scope="session")
def tmp_data(tmp_path_factory) -> Path:
    """Write a miniature on-disk dataset (32^3 TIFFs + `dataset_coarse.csv`) and return its root.

    Volumes alternate float16 and float32 so loader tests cover both float branches. The layout
    has two cheese types with two samples each and a non-contiguous sample folder (Requirement 26.6).
    """
    root = tmp_path_factory.mktemp("tmp_data")
    rng = np.random.default_rng(0)
    rows = []
    for idx, (rel, label_cheese, split) in enumerate(_TMP_LAYOUT):
        _write_tmp_volume(root / rel, rng, dtype=np.float16 if idx % 2 == 0 else np.float32)
        rows.append({"file": rel, "label_cheese": label_cheese, "label_sample": idx, "split": split})
    pd.DataFrame(rows).to_csv(root / "dataset_coarse.csv", index=False)
    return root


def _write_tmp_volume(path: Path, rng: np.random.Generator, dtype: np.dtype) -> None:
    """Write one 32^3 random volume in [0,1] of the given float dtype, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = rng.random((32, 32, 32)).astype(dtype)
    tifffile.imwrite(path, arr)


@pytest.fixture
def real_data() -> Path:
    """Return the resolved MozzaVID Small Data_Root, skipping the test when it is absent.

    The skip message contains the resolved path so a missing dataset is easy to diagnose
    (Requirement 25.2).
    """
    from mozzavid import io

    root = io.resolve_data_root(None)
    if not root.exists():
        pytest.skip(f"MozzaVID Small data root not found at {root}")
    return root
