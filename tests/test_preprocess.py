"""Tests for mozzavid.preprocess (Requirement 8; design Property 3)."""

from __future__ import annotations

import numpy as np
import pytest

from mozzavid import preprocess


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.0])
def test_smoothing_reduces_noise_and_preserves_mean(ij, seed, sigma):
    """Property 3: Gaussian smoothing lowers variance and keeps the mean (Requirement 8.3, 8.4)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.5, 0.1, size=(24, 24, 24)).astype(np.float32)
    smoothed = preprocess.denoise(noise, sigma)
    assert smoothed.shape == noise.shape
    assert smoothed.dtype == np.float32
    assert smoothed.var() < noise.var()
    assert abs(smoothed.mean() - noise.mean()) < 1e-3


def test_denoise_rejects_non_positive_sigma(ij):
    """denoise raises ValueError for sigma <= 0 (Requirement 8.5)."""
    arr = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        preprocess.denoise(arr, 0.0)
    with pytest.raises(ValueError):
        preprocess.denoise(arr, -1.0)
