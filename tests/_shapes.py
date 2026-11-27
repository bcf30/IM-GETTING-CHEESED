"""numpy-only analytic shape builders for tests (Requirement 25.6; no scikit-image)."""

from __future__ import annotations

import numpy as np


def sphere_mask(shape: tuple[int, int, int], centre: tuple[float, float, float],
                radius: float) -> np.ndarray:
    """Return a boolean (Z,Y,X) mask of a solid sphere of `radius` voxels centred at `centre`."""
    grids = np.ogrid[tuple(slice(0, dim) for dim in shape)]
    distance_sq = sum((grid - centre[axis]) ** 2 for axis, grid in enumerate(grids))
    return distance_sq <= radius**2


def separated_spheres(shape: tuple[int, int, int],
                      centres_radii: list[tuple[tuple, float]]) -> np.ndarray:
    """Return the union of several spheres given as (centre, radius) pairs."""
    mask = np.zeros(shape, dtype=bool)
    for centre, radius in centres_radii:
        mask |= sphere_mask(shape, centre, radius)
    return mask


def cube_mask(shape: tuple[int, int, int], corner: tuple[int, int, int], size: int) -> np.ndarray:
    """Return a boolean mask with a `size`^3 cube whose minimum corner is `corner`."""
    mask = np.zeros(shape, dtype=bool)
    z, y, x = corner
    mask[z:z + size, y:y + size, x:x + size] = True
    return mask
