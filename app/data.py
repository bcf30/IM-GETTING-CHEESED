"""Cached data access for the dashboard: result tables, metadata and parsed ANOVA stats.

This module is the single read path for `results/*` and dataset metadata. It performs no
image work and never touches the JVM (that lives in `app/fiji_runtime.py`), keeping data
loading separate from Fiji execution. All loaders are cached so the heavy CSV is read once
per session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from mozzavid import io

# Repository root is two levels up from this file (app/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = _REPO_ROOT / "results"

# The four metrics the pipeline tests across cheese types, in report order.
STAT_METRICS: tuple[str, ...] = (
    "foreground_fraction",
    "objects_per_mm3",
    "median_equiv_diameter_um",
    "d50_um",
)

# Short label, one-line "what it is", and a longer "how to read it" for each metric. Text is
# written to match the actual definitions in mozzavid/measure.py and mozzavid/aggregate.py so
# the dashboard explanations cannot drift from the computation.
METRIC_INFO: dict[str, dict[str, str]] = {
    "foreground_fraction": {
        "label": "Foreground fraction",
        "what": "Share of the scanned volume that is segmented structure (mostly the fat phase).",
        "how": (
            "The total volume of all detected objects divided by the scanned volume. A higher "
            "value means more of the cheese is dense fat/protein structure and less is the "
            "surrounding matrix. This is a volume fraction of the segmented phase, **not** "
            "porosity (void space), so read it as 'how much structure', not 'how many holes'. "
            "It is the most robust discriminator between cheese types because it does not depend "
            "on how objects are split or merged."
        ),
    },
    "objects_per_mm3": {
        "label": "Objects per mm³",
        "what": "Number of separate structures (connected components) per cubic millimetre.",
        "how": (
            "Counts distinct connected objects and normalises by the scanned volume. A high "
            "count means many small, separated structures; a low count means fewer, larger, or "
            "merged ones. At 4.4 µm/voxel, adjacent fat domains can percolate into one large "
            "connected object, so this is partly a count of 'connected domains', not strictly "
            "individual globules. Together with foreground fraction it gives the strongest "
            "separation between cheese types."
        ),
    },
    "median_equiv_diameter_um": {
        "label": "Median equivalent diameter (µm)",
        "what": "Typical object size: the median sphere-equivalent diameter across objects.",
        "how": (
            "Each object's equivalent diameter is the diameter of a sphere with the same volume "
            "(6V/π)^(1/3). The median is taken over interior objects only (those not touching a "
            "volume face), number-weighted so every object counts once. Read it as the typical "
            "structure size: larger values mean coarser microstructure. Border-touching objects "
            "are excluded here because their true size is unknown (they are clipped by the scan edge)."
        ),
    },
    "d50_um": {
        "label": "D50 (µm)",
        "what": "Volume-weighted median diameter: half the structure volume sits in objects smaller than this.",
        "how": (
            "Like a sieve measure. Objects are ranked by equivalent diameter and weighted by "
            "their volume; D50 is the diameter below which half the total structure volume lies "
            "(D10 and D90 mark the 10% and 90% points). Because it is volume-weighted, a few "
            "large objects pull it up more than they do the number-weighted median, so comparing "
            "D50 against the median diameter hints at how skewed the size distribution is."
        ),
    },
}

# Dashboard-level orientation shown above the views.
OVERVIEW: str = (
    "This dashboard summarises the 3D microstructure of 25 mozzarella cheese types from the "
    "MozzaVID Small CT dataset (149 samples). Each CT volume was segmented into discrete "
    "structures (mostly fat), and every structure was measured for size and shape. The views "
    "below compare those measurements across cheese types and let you drill into individual "
    "samples. All sizes use a 4.4 µm voxel edge."
)

# How the group-comparison statistics on each chart should be read.
STATS_NOTE: str = (
    "**Kruskal-Wallis** tests whether a metric differs across the cheese types without assuming "
    "a normal distribution. A small *p* means the types are not all the same. **epsilon²** is "
    "the effect size (0-1): how much of the variation is explained by cheese type, so it tells "
    "you whether a significant difference is also a large one."
)


@dataclass(frozen=True)
class MetricStats:
    """Parsed Kruskal-Wallis result for one metric, as written to `anova_results.txt`."""

    statistic: float
    p_value: float
    epsilon_squared: float
    levene_p: float
    n_groups: int


@st.cache_data(show_spinner=False)
def load_per_sample() -> pd.DataFrame:
    """Return the 149-row per-sample summary table (one row per cheese sample)."""
    return pd.read_csv(RESULTS_DIR / "per_sample_summary.csv")


@st.cache_data(show_spinner="Loading per-object metrics (first time only)...")
def load_per_object(columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Return the per-object metrics table, optionally restricted to `columns`.

    The gzip holds ~708k rows; reading only the needed columns keeps the first render fast.
    Passing `columns=None` reads the full table (used by the 3D view, which needs centroids).
    """
    usecols = list(columns) if columns is not None else None
    return pd.read_csv(RESULTS_DIR / "per_object_metrics.csv.gz", usecols=usecols)


@st.cache_data(show_spinner=False)
def load_metric_stats() -> dict[str, MetricStats]:
    """Parse `anova_results.txt` into one `MetricStats` per metric.

    The file is machine-written with a stable `=== metric ===` / `Kruskal-Wallis: key=value`
    layout, so a line-oriented regex is sufficient and avoids re-running any statistics.
    """
    text = (RESULTS_DIR / "anova_results.txt").read_text(encoding="utf-8")
    stats: dict[str, MetricStats] = {}
    current: str | None = None
    kw: dict[str, str] | None = None
    for line in text.splitlines():
        header = re.match(r"^===\s*(\S+)\s*===", line)
        if header:
            current = header.group(1)
            kw = None
            continue
        if current and line.startswith("Kruskal-Wallis:"):
            # The KW line holds statistic/p/epsilon_squared/levene_p; `groups=` is on the
            # following line (next to a `sizes={...}` dict), so it is read separately below.
            kw = dict(re.findall(r"(statistic|p|epsilon_squared|levene_p)=([-\d.eE+]+)", line))
            continue
        if current and kw is not None and line.lstrip().startswith("groups="):
            groups = re.match(r"\s*groups=(\d+)", line)
            if groups:
                stats[current] = MetricStats(
                    statistic=float(kw["statistic"]),
                    p_value=float(kw["p"]),
                    epsilon_squared=float(kw["epsilon_squared"]),
                    levene_p=float(kw["levene_p"]),
                    n_groups=int(groups.group(1)),
                )
            kw = None
    return stats


@st.cache_data(show_spinner=False)
def load_metadata(data_root: str) -> pd.DataFrame:
    """Return the per-volume metadata for a data root, via the pipeline's own loader.

    `data_root` is a string (not `Path`) so Streamlit can hash it for the cache key.
    """
    return io.load_metadata(Path(data_root))


def cheese_types() -> list[str]:
    """Return the sorted unique cheese types present in the per-sample summary."""
    return sorted(load_per_sample()["cheese_type"].unique().tolist())
