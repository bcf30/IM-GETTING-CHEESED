"""Matplotlib figures for the mozzavid pipeline.

Figure file names live here so that `pipeline`, `synthetic` and `validate_segmentation`
agree on them. Every function plots only the values it is given and computes no metrics; it
uses the non-interactive Agg backend so it runs headless (Requirement 18).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; selected before pyplot import

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection above)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG_MONTAGE: str = "montage_representative_volumes.png"
FIG_SEGMENTATION: str = "segmentation_examples.png"
FIG_SIZE_DISTRIBUTION: str = "size_distribution_by_type.png"
FIG_FRACTION: str = "porosity_comparison.png"
FIG_SYNTHETIC: str = "synthetic_discrimination.png"


def montage_central_slices(volumes: dict[str, np.ndarray], path: Path) -> Path:
    """Plot the central Z slice of each passed volume, titled with its `volume_id`.

    `volumes` maps `volume_id` to a 3D array in (Z,Y,X) order. The figure is written to `path`
    and the written path is returned. No metrics are computed (Requirement 18.3, 18.5, 18.6).
    """
    items = list(volumes.items())
    fig, axes = _grid(len(items))
    for ax, (volume_id, arr) in zip(axes, items):
        central = arr[arr.shape[0] // 2]
        ax.imshow(central, cmap="gray")
        ax.set_title(volume_id, fontsize=8)
        ax.axis("off")
    for ax in axes[len(items):]:  # hide unused cells of the grid
        ax.axis("off")
    return _save(fig, path)


def segmentation_overlays(pairs: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    """Plot each volume's central Z slice with its cleaned-mask outline (Requirement 18.3).

    `pairs` maps `volume_id` to `(volume, mask)`, both 3D (Z,Y,X) arrays. The mask boundary is
    drawn as a contour over the grey slice. The written path is returned; no metrics are computed.
    """
    items = list(pairs.items())
    fig, axes = _grid(len(items))
    for ax, (volume_id, (arr, mask)) in zip(axes, items):
        z = arr.shape[0] // 2
        ax.imshow(arr[z], cmap="gray")
        ax.contour(mask[z].astype(float), levels=[0.5], colors="red", linewidths=0.6)
        ax.set_title(volume_id, fontsize=8)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    return _save(fig, path)


def size_distribution_by_type(per_sample: pd.DataFrame, path: Path) -> Path:
    """Plot per-cheese-type size distributions (volume-weighted d50 and number-weighted median).

    One box per cheese type over its samples' `d50_um` and a marker for the median of
    `median_equiv_diameter_um`. Plots only passed-in values (Requirement 18.3, 18.5).
    """
    types = sorted(per_sample["cheese_type"].unique())
    d50_by_type = [per_sample.loc[per_sample["cheese_type"] == t, "d50_um"].dropna().to_numpy()
                   for t in types]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(types)), 4))
    _boxplot(ax, d50_by_type, types)
    ax.set_ylabel("Equivalent diameter (µm)")
    ax.set_xlabel("Cheese type")
    ax.set_title("Volume-weighted d50 by cheese type")
    return _save(fig, path)


def fraction_comparison(per_sample: pd.DataFrame, path: Path) -> Path:
    """Plot the segmented-phase volume fraction by cheese type (Requirement 18.4).

    The file keeps the name `porosity_comparison.png`, but the y-axis reads
    "Segmented-phase volume fraction". Plots only passed-in values.
    """
    types = sorted(per_sample["cheese_type"].unique())
    fraction_by_type = [per_sample.loc[per_sample["cheese_type"] == t,
                                       "foreground_fraction"].dropna().to_numpy() for t in types]
    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(types)), 4))
    _boxplot(ax, fraction_by_type, types)
    ax.set_ylabel("Segmented-phase volume fraction")
    ax.set_xlabel("Cheese type")
    ax.set_title("Segmented-phase volume fraction by cheese type")
    return _save(fig, path)


def synthetic_discrimination(bench: pd.DataFrame, pvalues: dict[str, float], path: Path) -> Path:
    """Plot measured vs ground-truth median diameter per kind with the Welch p-values (Req 17.6).

    `bench` has columns `kind`, `median_equiv_diameter_um` and `true_median_um`. The figure
    shows the measured distributions and the ground-truth markers; Welch p-values go in the
    title. Plots only passed-in values.
    """
    kinds = sorted(bench["kind"].unique())
    measured = [bench.loc[bench["kind"] == k, "median_equiv_diameter_um"].dropna().to_numpy()
                for k in kinds]
    fig, ax = plt.subplots(figsize=(6, 4))
    _boxplot(ax, measured, kinds)
    for i, k in enumerate(kinds, start=1):
        truth = bench.loc[bench["kind"] == k, "true_median_um"].dropna()
        if not truth.empty:
            ax.plot(i, truth.mean(), marker="D", color="black", label="ground truth" if i == 1 else None)
    ax.set_ylabel("Median equivalent diameter (µm)")
    ax.set_title("Synthetic discrimination\n"
                 + ", ".join(f"{m}: p={p:.2g}" for m, p in pvalues.items()))
    ax.legend(loc="best", fontsize=8)
    return _save(fig, path)


def _boxplot(ax, data: list[np.ndarray], labels: list[str]) -> None:
    """Draw a box plot of `data` with `labels`, skipping the box for any empty group."""
    positions = [i + 1 for i, d in enumerate(data) if len(d)]
    non_empty = [d for d in data if len(d)]
    if non_empty:
        ax.boxplot(non_empty, positions=positions)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)


def _grid(n: int) -> tuple[plt.Figure, list]:
    """Return a figure and a flat list of axes sized to hold `n` panels."""
    cols = min(n, 5) if n else 1
    rows = (n + cols - 1) // cols if n else 1
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    return fig, list(axes.ravel())


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write `fig` to `path` with a tight layout and close it; return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path