"""Tests for mozzavid.aggregate (Requirement 14; design Property 10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mozzavid import aggregate


def _objects(rng: np.random.Generator, n: int, volume_id="cheese_1/1/0.tiff") -> pd.DataFrame:
    """Build a random object table with the columns aggregate needs."""
    volumes = rng.uniform(1.0, 1000.0, size=n)
    return pd.DataFrame({
        "label": np.arange(1, n + 1),
        "volume_um3": volumes,
        "equiv_diameter_um": (6.0 * volumes / np.pi) ** (1 / 3),
        "touches_border": rng.random(n) < 0.3,
        "volume_id": volume_id,
        "sample_id": "cheese_1/1",
        "cheese_type": "cheese_1",
    })


@pytest.mark.parametrize("seed", range(8))
def test_summarize_row_permutation_invariant(seed):
    """Property 10: summaries are identical under row permutation (Requirement 14.8)."""
    rng = np.random.default_rng(seed)
    objects = _objects(rng, 40)
    base = aggregate.summarize(objects, "cheese_1/1", "cheese_1", 1e9)
    shuffled = aggregate.summarize(objects.sample(frac=1.0, random_state=seed), "cheese_1/1",
                                   "cheese_1", 1e9)
    for key, value in base.items():
        assert shuffled[key] == value or (value != value and shuffled[key] != shuffled[key])


@pytest.mark.parametrize("seed", range(8))
def test_summarize_bounds_and_percentile_order(seed):
    """Property 10: fraction in [0,1] and d10 <= d50 <= d90 (Requirement 14.9)."""
    rng = np.random.default_rng(seed)
    total = rng.uniform(1e5, 1e7)
    row = aggregate.summarize(_objects(rng, 50), "cheese_1/1", "cheese_1", total)
    assert 0.0 <= row["foreground_fraction"] <= 1.0
    if row["n_interior_objects"] > 0:
        assert row["d10_um"] <= row["d50_um"] <= row["d90_um"]


def test_weighted_percentiles_hand_computed():
    """Volume-weighted percentiles match a hand-computed example (Requirement 14.6)."""
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    d10, d50, d90 = aggregate._weighted_percentiles(values, weights, (10.0, 50.0, 90.0))
    assert d50 == pytest.approx(2.0)
    assert d10 == pytest.approx(1.0)  # below the first centre, clamped to the smallest value
    assert d90 == pytest.approx(3.0)


def test_no_interior_objects_gives_nan_sizes():
    """All size metrics are NaN and n_interior_objects is 0 without interior objects (Req 14.7)."""
    objects = pd.DataFrame({
        "label": [1, 2], "volume_um3": [10.0, 20.0], "equiv_diameter_um": [2.0, 3.0],
        "touches_border": [True, True],
    })
    row = aggregate.summarize(objects, "s", "c", 1e6)
    assert row["n_interior_objects"] == 0
    for key in ("mean_equiv_diameter_um", "median_equiv_diameter_um", "std_equiv_diameter_um",
                "d10_um", "d50_um", "d90_um"):
        assert np.isnan(row[key])


def test_summarize_per_volume_zero_object_volume_gets_row():
    """A processed volume with no objects still gets a per-volume row (Requirement 14.10)."""
    objects = pd.DataFrame(columns=["volume_id", "volume_um3", "equiv_diameter_um", "touches_border"])
    meta = pd.DataFrame({"volume_id": ["cheese_1/1/0.tiff"], "sample_id": ["cheese_1/1"],
                         "cheese_type": ["cheese_1"]})
    per_volume = aggregate.summarize_per_volume(objects, meta, 1.0)
    assert len(per_volume) == 1
    assert per_volume.loc[0, "n_objects"] == 0


def test_foreground_fraction_matches_definition():
    """foreground_fraction equals total object volume over analyzed volume (Requirement 14.3)."""
    objects = pd.DataFrame({"volume_um3": [100.0, 200.0], "equiv_diameter_um": [5.0, 6.0],
                            "touches_border": [False, False]})
    row = aggregate.summarize(objects, "s", "c", 1000.0)
    assert row["foreground_fraction"] == pytest.approx(0.3)
    assert row["objects_per_mm3"] == pytest.approx(2 / (1000.0 * 1e-9))
