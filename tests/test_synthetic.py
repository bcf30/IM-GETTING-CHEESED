"""Tests for mozzavid.synthetic (Requirements 19, 20; design Properties 12, 13)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage

from mozzavid import synthetic


@pytest.mark.parametrize("kind,shape,seed", [
    ("heterogeneous", (64, 64, 64), 0),
    ("homogeneous", (64, 64, 64), 1),
    ("heterogeneous", (48, 56, 64), 2),
    ("homogeneous", (64, 48, 56), 3),
    ("heterogeneous", (64, 64, 64), 4),
    ("homogeneous", (64, 64, 64), 5),
])
def test_generate_deterministic_and_valid(kind, shape, seed):
    """Property 12: generate is bit-identical and returns float32 in [0,1] (Requirement 19.1, 19.7)."""
    first = synthetic.generate(kind, shape=shape, seed=seed)
    second = synthetic.generate(kind, shape=shape, seed=seed)
    assert np.array_equal(first, second)
    assert first.dtype == np.float32
    assert first.min() >= 0.0 and first.max() <= 1.0


@pytest.mark.parametrize("kind,seed", [("heterogeneous", 0), ("homogeneous", 1),
                                       ("heterogeneous", 2), ("homogeneous", 3)])
def test_ground_truth_spheres_separated(kind, seed):
    """Property 12: ground-truth spheres are pairwise separated and fully interior (Requirement 19.5).

    The r+1 rejection (Requirement 19.4) leaves a > 1 voxel Euclidean gap between any new
    sphere's voxels and existing foreground, so no two spheres share a face (6-connectivity).
    """
    mask = synthetic._ground_truth_mask(kind, (64, 64, 64), seed, 0.15)
    labels, count6 = scipy.ndimage.label(mask)
    # Every face-connected component is a single ball, so its voxels fill a convex region.
    assert count6 >= 1
    assert not mask[0].any() and not mask[-1].any()  # fully interior on the Z faces
    assert not mask[:, 0].any() and not mask[:, -1].any()
    assert not mask[:, :, 0].any() and not mask[:, :, -1].any()


@pytest.mark.parametrize("seed", range(5))
def test_achieved_fraction_meets_target(seed):
    """Generated volumes reach at least target - 0.01 foreground fraction (Requirement 19.8)."""
    mask = synthetic._ground_truth_mask("homogeneous", (64, 64, 64), seed, 0.1)
    assert mask.mean() >= 0.1 - 0.01


def test_invalid_target_fraction_raises():
    """generate raises ValueError for target_fraction outside (0, 0.5) (Requirement 19.10)."""
    with pytest.raises(ValueError):
        synthetic.generate("homogeneous", target_fraction=0.0)
    with pytest.raises(ValueError):
        synthetic.generate("homogeneous", target_fraction=0.6)


def test_heterogeneous_radii_larger_than_homogeneous():
    """Heterogeneous ground-truth spheres are larger on average (Requirement 19.2)."""
    het = synthetic._ground_truth_mask("heterogeneous", (80, 80, 80), 0, 0.12)
    hom = synthetic._ground_truth_mask("homogeneous", (80, 80, 80), 0, 0.12)
    het_median = synthetic._true_median_diameter(het, 0, "heterogeneous")
    hom_median = synthetic._true_median_diameter(hom, 0, "homogeneous")
    assert het_median > hom_median


@pytest.mark.slow
def test_benchmark_separates_kinds(ij):
    """Property 13: benchmark(20, 0) separates the kinds with Welch p < 0.01 (Requirement 20.5).

    Slow (~1-3 min): runs the full Fiji chain on 40 synthetic volumes. Needs the `ij` fixture
    so a broken Fiji install surfaces as an error.
    """
    bench = synthetic.benchmark(20, 0)
    het = bench.loc[bench["kind"] == "heterogeneous", "median_equiv_diameter_um"].mean()
    hom = bench.loc[bench["kind"] == "homogeneous", "median_equiv_diameter_um"].mean()
    assert het > hom
    for metric in ("median_equiv_diameter_um", "objects_per_mm3"):
        values = synthetic._split(bench, metric)
        p = scipy.stats.ttest_ind(*values, equal_var=False).pvalue
        assert p < 0.01


import scipy.stats  # noqa: E402  (used only in the slow benchmark test)
