"""Gaussian denoising in Fiji (ImageJ Ops). Fiji-bound layer (Requirement 8).

The only image operation here is an isotropic 3D Gaussian smoothing run inside Fiji through
`mozzavid.fiji`; no thresholding or measurement happens in this module.
"""

from __future__ import annotations

import numpy as np

from mozzavid import fiji


def denoise(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Smooth a 3D volume with an isotropic Gaussian in Fiji; `sigma` is in voxels (Requirement 8).

    The Gaussian is normalised, so the mean is preserved to within 1e-3 and the variance is
    reduced. The output keeps the input shape and is returned as float32.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0 voxels, got {sigma}")
    ij = fiji.get_ij()
    smoothed = ij.op().filter().gauss(fiji.to_java_img(arr), float(sigma))
    return fiji.from_java_array(smoothed).astype(np.float32, copy=False)
