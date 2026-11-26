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
