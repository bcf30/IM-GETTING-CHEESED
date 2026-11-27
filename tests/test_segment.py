"""Tests for mozzavid.segment (Requirements 9, 10; design Properties 4, 5, 6)."""

from __future__ import annotations

import numpy as np
import pytest

from mozzavid import segment
from tests import _shapes


def _bimodal(rng: np.random.Generator, shape, fraction: float, noise_sd: float) -> np.ndarray:
    """Return a two-level volume with the given foreground fraction and Gaussian noise."""
    foreground = rng.random(shape) < fraction
    values = np.where(foreground, 0.8, 0.2) + rng.normal(0, noise_sd, shape)
    return np.clip(values, 0.0, 1.0).astype(np.float32), foreground


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("method", ["otsu", "li"])
def test_threshold_complement(ij, seed, method):
    """Property 4: invert=True yields the complement of invert=False (Requirement 9.4)."""
    rng = np.random.default_rng(seed)
    arr = rng.random((24, 24, 24)).astype(np.float32)
    normal = segment.segment(arr, method, invert=False)
    inverted = segment.segment(arr, method, invert=True)
    assert np.array_equal(inverted, ~normal)


def test_segment_rejects_unknown_method(ij):
    """segment raises ValueError listing the allowed methods (Requirement 9.5)."""
    arr = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="otsu"):
        segment.segment(arr, "triangle")


@pytest.mark.parametrize("seed", range(6))
def test_bimodal_recovery(ij, seed):
    """Property 5: Otsu foreground fraction is within 0.05 of the true fraction (Requirement 9.6)."""
    rng = np.random.default_rng(seed)
    fraction = rng.uniform(0.1, 0.5)
    arr, _ = _bimodal(rng, (32, 32, 32), fraction, noise_sd=0.05)
    mask = segment.segment(arr, "otsu", invert=False)
    assert abs(mask.mean() - fraction) < 0.05


@pytest.mark.parametrize("seed", range(6))
def test_cleanup_subset_and_idempotent(ij, seed):
    """Property 6: cleanup is a subset of the input and idempotent (Requirement 10.1, 10.5)."""
    rng = np.random.default_rng(seed)
    centres = [((rng.integers(8, 40), rng.integers(8, 40), rng.integers(8, 40)),
                float(rng.integers(2, 6))) for _ in range(6)]
    mask = _shapes.separated_spheres((48, 48, 48), centres)
    cleaned = segment.cleanup(mask, min_size=50)
    assert np.all(cleaned <= mask)  # no voxel added
    assert np.array_equal(segment.cleanup(cleaned, 50), cleaned)  # idempotent


def test_cleanup_removes_small_keeps_large(ij):
    """A 3-voxel blob is removed and a 100-voxel blob kept (Requirement 10.3, 10.4, 25.6)."""
    mask = np.zeros((32, 32, 32), dtype=bool)
    mask[1, 1, 1:4] = True  # 3-voxel line, below min_size
    big = _shapes.sphere_mask((32, 32, 32), (20, 20, 20), 4.0)  # ~250 voxels
    mask |= big
    cleaned = segment.cleanup(mask, min_size=50)
    assert not cleaned[1, 1, 1:4].any()
    assert cleaned[big].all()


def test_cleanup_zero_min_size_returns_copy(ij):
    """cleanup with min_size=0 returns an unchanged copy (Requirement 10.6)."""
    mask = _shapes.sphere_mask((24, 24, 24), (12, 12, 12), 5.0)
    cleaned = segment.cleanup(mask, min_size=0)
    assert np.array_equal(cleaned, mask)
    assert cleaned is not mask


def test_cleanup_diagonal_cubes_follow_connectivity(ij):
    """Two corner-touching cubes merge at 26-connectivity, so both survive together (Req 10.2)."""
    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[2:5, 2:5, 2:5] = True      # cube A (27 voxels)
    mask[5:8, 5:8, 5:8] = True      # cube B sharing only the corner at (5,5,5)
    # As one 26-connected component of 54 voxels, both survive a min_size=40 opening.
    cleaned = segment.cleanup(mask, min_size=40)
    assert cleaned.sum() == 54
