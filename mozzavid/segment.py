"""Thresholding and small-object removal in Fiji. Fiji-bound layer (Requirements 9, 10).

`segment` runs a global automatic threshold (Otsu or Li) through ImageJ Ops. `cleanup` removes
connected components smaller than `min_size` voxels through MorphoLibJ, reached only via the
dispatch helpers in `mozzavid.fiji`. No measurement or unit scaling happens here.
"""

from __future__ import annotations

import numpy as np

from mozzavid import fiji


def segment(arr: np.ndarray, method: str = "otsu", invert: bool = False) -> np.ndarray:
    """Return a boolean mask from a global automatic threshold computed in Fiji (Requirement 9).

    With `invert=False` the mask is True above the Fiji-computed threshold; `invert=True`
    returns its complement, which is the phase of interest when the inclusions are darker than
    the matrix (see H2). An unknown `method` raises `ValueError` listing the allowed methods.
    """
    ij = fiji.get_ij()
    threshold_ops = {"otsu": ij.op().threshold().otsu, "li": ij.op().threshold().li}
    if method not in threshold_ops:
        raise ValueError(f"method must be one of {sorted(threshold_ops)}, got {method!r}")
    bright = fiji.from_java_array(threshold_ops[method](fiji.to_java_img(arr))).astype(bool)
    return ~bright if invert else bright


def cleanup(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove components smaller than `min_size` voxels (MorphoLibJ, CONNECTIVITY_3D) (Requirement 10).

    `min_size < 1` returns a copy unchanged. Otherwise the mask is labelled and size-opened in
    Fiji; the result is a subset of the input in which every surviving component has at least
    `min_size` voxels. The input ImagePlus and both intermediate 32-bit label images are closed
    so a long sequential run does not leak ~28 MB per volume (Requirement 13.10).
    """
    if min_size < 1:
        return mask.copy()
    imp = fiji.mask_to_imageplus(mask)
    labels = opened = None
    try:
        labels = fiji.label_components(imp)
        opened = fiji.size_opening(labels, min_size)
        return fiji.imageplus_to_mask(opened)
    finally:
        _close_all(imp, labels, opened)


def _close_all(*images) -> None:
    """Close each non-None ImagePlus, ignoring images that were never created (Requirement 13.10)."""
    for image in images:
        if image is not None:
            image.close()
