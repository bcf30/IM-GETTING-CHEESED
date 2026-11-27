"""Synthetic sphere-packing volumes and the measurement benchmark (Requirements 19, 20).

`generate` is pure numpy and never imports `mozzavid.fiji`. `benchmark` reaches Fiji only
through `pipeline.process_volume`, so the measurement chain is identical to the dataset path.
The benchmark needs no dataset, so a reviewer can validate the chain against ground truth.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scipy.stats

from mozzavid import aggregate, pipeline, visualize

log = logging.getLogger(__name__)

SYNTHETIC_VOXEL_SIZE_UM: float = 1.0
SYNTHETIC_MIN_SIZE_VOXELS: int = 20
# (mean radius µm, coefficient of variation) per kind; see design "Resolved inconsistencies" #1.
SYNTHETIC_RADIUS_PARAMS: dict = {"heterogeneous": (15.0, 0.6), "homogeneous": (3.0, 0.15)}
FG_LEVEL: float = 0.8
BG_LEVEL: float = 0.3
NOISE_SD: float = 0.05
MAX_PLACEMENT_TRIES: int = 2000
_KINDS: tuple[str, ...] = ("heterogeneous", "homogeneous")
_BENCH_METRICS: tuple[str, ...] = ("median_equiv_diameter_um", "objects_per_mm3")


def generate(kind: str, shape=(96, 96, 96), seed: int = 0,
             target_fraction: float = 0.15) -> np.ndarray:
    """Return a float32 volume of non-overlapping bright spheres with lognormal radii (Req 19).

    Spheres lie fully inside the volume and never touch (r+1 rejection), so the ground-truth
    count is known. Foreground is rendered at 0.8, background at 0.3, with added Gaussian noise.
    `(kind, shape, seed)` gives a bit-identical array.
    """
    if not 0.0 < target_fraction < 0.5:
        raise ValueError(f"target_fraction must be in (0, 0.5), got {target_fraction}")
    mask = _ground_truth_mask(kind, tuple(shape), seed, target_fraction)
    rng = np.random.default_rng(seed + 1)  # separate stream for the pixel noise
    noisy = np.where(mask, FG_LEVEL, BG_LEVEL) + rng.normal(0.0, NOISE_SD, mask.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def _ground_truth_mask(kind: str, shape: tuple, seed: int, target_fraction: float) -> np.ndarray:
    """Return the boolean union of pairwise non-touching, fully interior spheres (Req 19.5).

    Used by `generate` for the image and by `_bench_row` for the ground-truth size moments.
    Deterministic in `seed`; stops at `target_fraction` or on placement exhaustion (logged).
    """
    rng = np.random.default_rng(seed)
    mean_um, cv = SYNTHETIC_RADIUS_PARAMS[kind]
    mask = np.zeros(shape, dtype=bool)
    while mask.mean() < target_fraction:
        radius = _draw_radius(rng, mean_um, cv, shape)
        if not _place_sphere_if_free(mask, radius, rng):
            log.warning("sphere placement exhausted at fraction %.4f (target %.4f)",
                        mask.mean(), target_fraction)
            break
    return mask


def _draw_radius(rng: np.random.Generator, mean_um: float, cv: float, shape: tuple) -> float:
    """Draw one lognormal radius in voxels, clipped to [1.5, min(shape)/4] (Requirement 19.2, 19.3)."""
    sigma_ln = np.sqrt(np.log(1.0 + cv**2))
    mu_ln = np.log(mean_um) - sigma_ln**2 / 2.0
    radius_um = rng.lognormal(mu_ln, sigma_ln)
    radius_vox = radius_um / SYNTHETIC_VOXEL_SIZE_UM
    return float(np.clip(radius_vox, 1.5, min(shape) / 4.0))


def _place_sphere_if_free(mask: np.ndarray, radius: float, rng: np.random.Generator) -> bool:
    """Try to add one fully-interior, non-touching sphere; return True on success (Req 19.4).

    A candidate centre is drawn so the sphere fits inside the volume, and the sphere is
    rejected if any voxel within `radius + 1` of the centre is already foreground.
    """
    low = int(np.ceil(radius))
    highs = [dim - low - 1 for dim in mask.shape]
    if any(high < low for high in highs):
        return False
    for _ in range(MAX_PLACEMENT_TRIES):
        centre = np.array([rng.integers(low, high + 1) for high in highs])
        if _sphere_is_free(mask, centre, radius):
            _paint_sphere(mask, centre, radius)
            return True
    return False


def _sphere_is_free(mask: np.ndarray, centre: np.ndarray, radius: float) -> bool:
    """True if no existing foreground voxel lies within `radius + 1` of `centre`."""
    region, offset = _local_region(mask, centre, radius + 1.0)
    if not region.any():
        return True
    distances = _distance_field(region.shape, centre - offset)
    return not region[distances <= radius + 1.0].any()


def _paint_sphere(mask: np.ndarray, centre: np.ndarray, radius: float) -> None:
    """Set to True every voxel within `radius` of `centre`."""
    region, offset = _local_region(mask, centre, radius)
    distances = _distance_field(region.shape, centre - offset)
    region[distances <= radius] = True


def _local_region(mask: np.ndarray, centre: np.ndarray, reach: float):
    """Return a writable view of the box around `centre` with half-width `reach` and its origin."""
    reach_int = int(np.ceil(reach))
    low = np.maximum(centre - reach_int, 0)
    high = np.minimum(centre + reach_int + 1, mask.shape)
    view = mask[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    return view, low


def _distance_field(shape: tuple, centre: np.ndarray) -> np.ndarray:
    """Return the Euclidean distance from `centre` to each voxel of a box of the given shape."""
    grids = np.ogrid[tuple(slice(0, dim) for dim in shape)]
    return np.sqrt(sum((grid - centre[axis]) ** 2 for axis, grid in enumerate(grids)))


def benchmark(n_samples: int = 20, seed: int = 0, output_dir=None) -> pd.DataFrame:
    """Run the Fiji chain on `n_samples` volumes per kind and Welch-test the two kinds (Req 20).

    Per-volume seeds are drawn deterministically from `seed`. Each volume goes through
    `pipeline.process_volume` with the synthetic parameters. Writes the discrimination figure
    when `output_dir` is given. No Data_Root access.
    """
    seeds = np.random.default_rng(seed).integers(0, 2**31, size=2 * n_samples)
    kinds = [kind for kind in _KINDS for _ in range(n_samples)]
    bench = pd.DataFrame([_bench_row(kind, int(s)) for kind, s in zip(kinds, seeds)])
    pvalues = {m: float(scipy.stats.ttest_ind(*_split(bench, m), equal_var=False).pvalue)
               for m in _BENCH_METRICS}
    if output_dir is not None:
        from pathlib import Path
        visualize.synthetic_discrimination(bench, pvalues, Path(output_dir) / visualize.FIG_SYNTHETIC)
    return bench


def _bench_row(kind: str, seed: int) -> dict:
    """Process one synthetic volume and return its measured and ground-truth summary row (Req 20.3)."""
    arr = generate(kind, seed=seed)
    truth = _ground_truth_mask(kind, arr.shape, seed, 0.15)
    result = pipeline.process_volume(arr, SYNTHETIC_VOXEL_SIZE_UM, "otsu", 1.0,
                                     SYNTHETIC_MIN_SIZE_VOXELS, invert=False)
    analyzed = float(np.prod(arr.shape)) * SYNTHETIC_VOXEL_SIZE_UM**3
    summary = aggregate.summarize(result.objects, kind, kind, analyzed)
    return {"kind": kind, "seed": seed,
            "median_equiv_diameter_um": summary["median_equiv_diameter_um"],
            "objects_per_mm3": summary["objects_per_mm3"],
            "true_median_um": _true_median_diameter(truth, seed, kind),
            "achieved_fraction": float(truth.mean())}


def _true_median_diameter(mask: np.ndarray, seed: int, kind: str) -> float:
    """Return the ground-truth median equivalent diameter from the known sphere radii.

    Recomputing per-sphere radii is avoided by labelling the mask: each non-touching sphere is
    one component, and its equivalent diameter follows from its voxel count.
    """
    import scipy.ndimage
    labels, count = scipy.ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if count == 0:
        return float("nan")
    voxel_counts = np.bincount(labels.ravel())[1:]
    volumes = voxel_counts * SYNTHETIC_VOXEL_SIZE_UM**3
    diameters = (6.0 * volumes / np.pi) ** (1 / 3)
    return float(np.median(diameters))


def _split(bench: pd.DataFrame, metric: str) -> tuple:
    """Return the metric values of the two kinds as a (heterogeneous, homogeneous) tuple."""
    return tuple(bench.loc[bench["kind"] == kind, metric].dropna().to_numpy() for kind in _KINDS)
