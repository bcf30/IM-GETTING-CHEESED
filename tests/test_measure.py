"""Tests for mozzavid.measure (Requirement 11; design Properties 7, 8, 9; H7)."""

from __future__ import annotations

import numpy as np
import pytest

from mozzavid import measure
from tests import _shapes

_SCHEMA = ("label", "volume_um3", "equiv_diameter_um", "surface_area_um2", "sphericity",
           "feret_diameter_max_um", "centroid_z_um", "centroid_y_um", "centroid_x_um",
           "touches_border")


@pytest.mark.parametrize("seed", range(6))
def test_volume_conservation_and_one_row_per_component(ij, seed):
    """Property 7: total volume equals voxels * s^3 and one row per component (Req 11.7, 11.8)."""
    rng = np.random.default_rng(seed)
    centres = _non_overlapping_centres(rng, n=rng.integers(2, 4))
    mask = _shapes.separated_spheres((40, 40, 40), centres)
    s = float(rng.uniform(0.5, 5.0))
    objects = measure.extract(mask, s)
    assert len(objects) == len(centres)
    assert objects["volume_um3"].sum() == pytest.approx(mask.sum() * s**3, rel=1e-9)


def _non_overlapping_centres(rng, n):
    """Return `n` (centre, radius) pairs for spheres of radius 5-7 that do not overlap."""
    picks = []
    while len(picks) < n:
        centre = tuple(int(rng.integers(9, 31)) for _ in range(3))
        radius = float(rng.integers(5, 8))
        if all(np.linalg.norm(np.subtract(centre, c)) > radius + r + 3 for c, r in picks):
            picks.append((centre, radius))
    return picks


@pytest.mark.parametrize("seed", range(5))
def test_unit_scaling(ij, seed):
    """Property 8: lengths scale by k, areas by k^2, volumes by k^3, sphericity unchanged (Req 11.9)."""
    rng = np.random.default_rng(seed)
    mask = _shapes.sphere_mask((40, 40, 40), (20, 20, 20), float(rng.integers(5, 9)))
    s = float(rng.uniform(0.5, 2.0))
    k = float(rng.uniform(1.5, 3.0))
    base = measure.extract(mask, s).iloc[0]
    scaled = measure.extract(mask, k * s).iloc[0]
    assert scaled["volume_um3"] == pytest.approx(base["volume_um3"] * k**3, rel=1e-9)
    assert scaled["surface_area_um2"] == pytest.approx(base["surface_area_um2"] * k**2, rel=1e-9)
    assert scaled["equiv_diameter_um"] == pytest.approx(base["equiv_diameter_um"] * k, rel=1e-9)
    assert scaled["sphericity"] == pytest.approx(base["sphericity"], rel=1e-9)


@pytest.mark.parametrize("radius", [4, 6, 8, 10, 12])
def test_sphere_recovery(ij, radius):
    """Property 9: equiv diameter within 20% of 2*r*s and sphericity in [0.7, 1.05] (Req 11.10)."""
    mask = _shapes.sphere_mask((48, 48, 48), (24, 24, 24), float(radius))
    s = 1.3
    row = measure.extract(mask, s).iloc[0]
    assert abs(row["equiv_diameter_um"] - 2 * radius * s) <= 0.2 * (2 * radius * s)
    assert 0.7 <= row["sphericity"] <= 1.05


@pytest.mark.parametrize("radius", [6, 8, 10])
def test_feret_within_15_percent_of_analytic(ij, radius):
    """H7: the 3D max Feret is within 15% of a sphere's analytic diameter (Requirement 11.15)."""
    mask = _shapes.sphere_mask((48, 48, 48), (24, 24, 24), float(radius))
    row = measure.extract(mask, 1.0).iloc[0]
    assert abs(row["feret_diameter_max_um"] - 2 * radius) <= 0.15 * (2 * radius)


def test_border_flag_on_edge_sphere(ij):
    """A sphere cut by the volume edge is flagged touches_border=True (Requirement 11.6)."""
    mask = _shapes.sphere_mask((32, 32, 32), (0, 16, 16), 6.0)
    row = measure.extract(mask, 1.0).iloc[0]
    assert bool(row["touches_border"]) is True


def test_interior_sphere_not_border(ij):
    """An interior sphere is flagged touches_border=False (Requirement 11.6)."""
    mask = _shapes.sphere_mask((40, 40, 40), (20, 20, 20), 6.0)
    row = measure.extract(mask, 1.0).iloc[0]
    assert bool(row["touches_border"]) is False


def test_empty_mask_full_schema(ij):
    """An all-False mask returns an empty frame with the exact schema (Requirement 11.11)."""
    frame = measure.extract(np.zeros((16, 16, 16), dtype=bool), 1.0)
    assert len(frame) == 0
    assert tuple(frame.columns) == _SCHEMA


def test_exact_column_order(ij):
    """A non-empty result carries the exact column order of the schema (Requirement 11.2)."""
    mask = _shapes.sphere_mask((24, 24, 24), (12, 12, 12), 5.0)
    frame = measure.extract(mask, 1.0)
    assert tuple(frame.columns) == _SCHEMA
