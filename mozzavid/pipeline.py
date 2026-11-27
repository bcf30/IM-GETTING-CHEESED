"""Pipeline orchestration: process volumes, aggregate, test and write outputs.

Orchestration layer. It calls the Fiji-bound steps through `preprocess`, `segment` and
`measure`, aggregates per sample and volume, runs the statistics, and writes the CSV, text and
figure outputs. It contains no image algorithms and never imports `mozzavid.synthetic` (that
would create an import cycle) or `mozzavid.fiji` directly (Requirements 4, 12, 13,
architecture Requirements 13.11, 27.9).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from mozzavid import aggregate, io, measure, preprocess, segment, stats, visualize

log = logging.getLogger(__name__)

# Segmented phase (fat + voids) is the darker phase, so Otsu/Li "True above threshold" must be
# complemented. Recorded under H2 in docs/reproducibility.md from figures/polarity_check.png.
DEFAULT_INVERT: bool = True
# Metrics compared across cheese types and checked for within-sample agreement.
STAT_METRICS: tuple[str, ...] = (
    "foreground_fraction", "objects_per_mm3", "median_equiv_diameter_um", "d50_um",
)
_VOLUME_EDGE_VOXELS: int = 192
_MAX_MONTAGE_VOLUMES: int = 10
_CSV_GZIP_THRESHOLD_BYTES: int = 50 * 1024 * 1024
_FLOAT_FORMAT: str = "%.8g"


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for one pipeline run."""

    data_root: Path
    output_dir: Path
    figures_dir: Path
    limit: int | None = None
    method: Literal["otsu", "li"] = "otsu"
    sigma: float = 1.5
    min_size: int = 50
    invert: bool = DEFAULT_INVERT
    voxel_size_um: float = io.VOXEL_SIZE_UM
    seed: int = 0


@dataclass(frozen=True)
class VolumeResult:
    """Per-volume output of `process_volume`."""

    objects: pd.DataFrame
    foreground_fraction: float
    mask: np.ndarray
    seconds: float


def process_volume(arr: np.ndarray, voxel_size_um: float, method: str, sigma: float,
                   min_size: int, invert: bool) -> VolumeResult:
    """Run denoise -> segment -> cleanup -> measure for one volume (all image ops in Fiji)."""
    start = time.perf_counter()
    smoothed = preprocess.denoise(arr, sigma)
    mask = segment.cleanup(segment.segment(smoothed, method, invert), min_size)
    objects = measure.extract(mask, voxel_size_um)
    return VolumeResult(objects=objects, foreground_fraction=float(mask.mean()),
                        mask=mask, seconds=time.perf_counter() - start)


def select_volumes(meta: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Return all volumes (limit=None) or a stratified round-robin subset across cheese types.

    Volumes are ordered by `(cheese_type, sample_id, volume_id)` and taken round-robin across
    cheese types, so a small `limit` still covers every type (Requirement 13).
    """
    if limit is None:
        return meta.reset_index(drop=True)
    ordered = meta.sort_values(["cheese_type", "sample_id", "volume_id"])
    ranked = ordered.groupby("cheese_type").cumcount()
    chosen = ordered.assign(_rank=ranked).sort_values(["_rank", "cheese_type"]).head(limit)
    return chosen.drop(columns="_rank").reset_index(drop=True)


def run(config: PipelineConfig) -> dict[str, Path]:
    """Process selected volumes, aggregate per sample, test across cheese types, write outputs."""
    _log_parameters(config)
    meta = select_volumes(io.load_metadata(config.data_root), config.limit)
    objects, processed_meta, skipped, montage = _process_all(meta, config)
    per_sample = aggregate.summarize_all(objects, processed_meta, config.voxel_size_um)
    per_volume = aggregate.summarize_per_volume(objects, processed_meta, config.voxel_size_um)
    results = {m: _safe(stats.compare, per_sample, m) for m in STAT_METRICS}
    posthocs = {m: stats.posthoc(per_sample, m, r) for m, r in results.items() if r is not None}
    iccs = {m: _safe(stats.icc1, per_volume, m) for m in STAT_METRICS}
    return _write_outputs(config, objects, per_sample, results, posthocs, iccs, skipped, montage)


def _log_parameters(config: PipelineConfig) -> None:
    """Log sigma and min_size in voxels, micrometres and cubic micrometres at run start (Req 13.9)."""
    s = config.voxel_size_um
    log.info("sigma=%.3g voxels (%.3g um); min_size=%d voxels (%.4g um3); method=%s; invert=%s",
             config.sigma, config.sigma * s, config.min_size, config.min_size * s**3,
             config.method, config.invert)


def _process_all(meta: pd.DataFrame, config: PipelineConfig):
    """Process each selected volume sequentially; collect objects, processed rows, skips, montage.

    Failing volumes are logged at ERROR, skipped, and their `volume_id` collected. Raw arrays and
    masks are kept only for up to 10 montage volumes spread over `cheese_index` (Requirement 13).
    """
    montage_ids = _montage_volume_ids(meta)
    object_frames, processed_rows, skipped, montage = [], [], [], {}
    for _, row in meta.iterrows():
        outcome = _process_row(row, config, montage_ids, montage)
        if outcome is None:
            skipped.append(row["volume_id"])
            continue
        object_frames.append(outcome)
        processed_rows.append(row)
    objects = pd.concat(object_frames, ignore_index=True) if object_frames else _empty_objects()
    processed_meta = pd.DataFrame(processed_rows) if processed_rows else meta.iloc[:0]
    return objects, processed_meta, skipped, montage


def _process_row(row: pd.Series, config: PipelineConfig, montage_ids: set[str],
                 montage: dict) -> pd.DataFrame | None:
    """Process one volume; return its labelled object frame, or None if it was skipped."""
    try:
        arr = io.load_tiff(row["path"])
        result = process_volume(arr, config.voxel_size_um, config.method, config.sigma,
                                config.min_size, config.invert)
    except Exception as error:  # a bad volume is logged and skipped, not fatal (Requirement 13.6)
        log.error("skipping volume %s: %s", row["volume_id"], error)
        return None
    log.info("volume %s: %.1fs, %d objects, foreground fraction %.4f", row["volume_id"],
             result.seconds, len(result.objects), result.foreground_fraction)
    if row["volume_id"] in montage_ids:
        montage[row["volume_id"]] = (arr, result.mask)
    return result.objects.assign(volume_id=row["volume_id"], sample_id=row["sample_id"],
                                 cheese_type=row["cheese_type"])


def _montage_volume_ids(meta: pd.DataFrame) -> set[str]:
    """Pick up to 10 volume ids, one per cheese type evenly spaced over `cheese_index`."""
    first_per_type = meta.sort_values(["cheese_index", "sample_id", "volume_id"]).groupby(
        "cheese_type", sort=True).first().reset_index()
    if len(first_per_type) <= _MAX_MONTAGE_VOLUMES:
        chosen = first_per_type
    else:
        idx = np.linspace(0, len(first_per_type) - 1, _MAX_MONTAGE_VOLUMES).round().astype(int)
        chosen = first_per_type.iloc[np.unique(idx)]
    return set(chosen["volume_id"])


def _empty_objects() -> pd.DataFrame:
    """Return an empty object frame with the measure schema plus the pipeline id columns."""
    frame = measure._empty_object_frame()
    for column in ("volume_id", "sample_id", "cheese_type"):
        frame[column] = pd.Series(dtype="object")
    return frame


def _safe(func, *args):
    """Call `func(*args)`, returning None when it raises `ValueError` for insufficient data."""
    try:
        return func(*args)
    except ValueError as error:
        log.warning("%s unavailable: %s", getattr(func, "__name__", func), error)
        return None


def _write_outputs(config: PipelineConfig, objects: pd.DataFrame, per_sample: pd.DataFrame,
                   results: dict, posthocs: dict, iccs: dict, skipped: list[str],
                   montage: dict) -> dict[str, Path]:
    """Write all CSV, text, markdown and figure outputs deterministically; return their paths."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_object_metrics": _write_objects_csv(config.output_dir, objects),
        "per_sample_summary": _write_csv(config.output_dir / "per_sample_summary.csv",
                                         _sorted_per_sample(per_sample)),
        "anova_results": _write_text(config.output_dir / "anova_results.txt",
                                     stats.format_report(results, posthocs)),
        "summary": _write_text(config.output_dir / "summary.md",
                                _summary_markdown(config, per_sample, results, iccs, skipped)),
        "skipped": _write_text(config.output_dir / "skipped_volumes.txt", "\n".join(skipped)),
    }
    paths.update(_write_figures(config, per_sample, montage))
    return paths


def _sorted_per_sample(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Return the per-sample frame in a deterministic row order."""
    if per_sample.empty:
        return per_sample
    return per_sample.sort_values(["cheese_type", "sample_id"], ignore_index=True)


def _write_objects_csv(output_dir: Path, objects: pd.DataFrame) -> Path:
    """Write the per-object CSV deterministically, gzipping and logging if it exceeds 50 MB."""
    ordered = objects.sort_values(list(objects.columns), ignore_index=True) if not objects.empty else objects
    path = output_dir / "per_object_metrics.csv"
    estimate = int(ordered.memory_usage(deep=True).sum())
    if estimate > _CSV_GZIP_THRESHOLD_BYTES:
        path = output_dir / "per_object_metrics.csv.gz"
        log.info("per-object table ~%d bytes > 50 MB; writing %s", estimate, path.name)
        ordered.to_csv(path, index=False, float_format=_FLOAT_FORMAT, compression="gzip")
        return path
    return _write_csv(path, ordered)


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Write a DataFrame to CSV with a fixed float format and no index; return the path."""
    frame.to_csv(path, index=False, float_format=_FLOAT_FORMAT)
    return path


def _write_text(path: Path, text: str) -> Path:
    """Write text to `path` with a trailing newline and `\\n` line endings; return the path."""
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def _write_figures(config: PipelineConfig, per_sample: pd.DataFrame, montage: dict) -> dict:
    """Write the four dataset figures and return their paths keyed by figure constant."""
    figs = config.figures_dir
    volumes = {vid: arr for vid, (arr, _mask) in montage.items()}
    overlays = {vid: (arr, mask) for vid, (arr, mask) in montage.items()}
    return {
        visualize.FIG_MONTAGE: visualize.montage_central_slices(volumes, figs / visualize.FIG_MONTAGE),
        visualize.FIG_SEGMENTATION: visualize.segmentation_overlays(
            overlays, figs / visualize.FIG_SEGMENTATION),
        visualize.FIG_SIZE_DISTRIBUTION: visualize.size_distribution_by_type(
            per_sample, figs / visualize.FIG_SIZE_DISTRIBUTION),
        visualize.FIG_FRACTION: visualize.fraction_comparison(per_sample, figs / visualize.FIG_FRACTION),
    }


def _summary_markdown(config: PipelineConfig, per_sample: pd.DataFrame, results: dict,
                      iccs: dict, skipped: list[str]) -> str:
    """Render `results/summary.md` from the computed numbers, with no free-form claims."""
    s = config.voxel_size_um
    lines = [
        "# MozzaVID microstructure summary",
        "",
        f"- Data root: `{config.data_root}`",
        f"- Samples summarised: {len(per_sample)}",
        f"- Voxel size: {s} µm (derived, TODO-VERIFY)",
        f"- Gaussian sigma: {config.sigma} voxels ({config.sigma * s:.3g} µm)",
        f"- Min object size: {config.min_size} voxels ({config.min_size * s**3:.4g} µm³)",
        f"- Threshold: {config.method}, invert={config.invert}",
        "",
        "## Group comparison per metric",
        "",
    ]
    lines.extend(_metric_markdown(metric, results.get(metric), iccs.get(metric))
                 for metric in STAT_METRICS)
    lines.extend(["", "## Skipped volumes", "", _skipped_markdown(skipped), "",
                  "## Limitations", "", _limitations_markdown()])
    return "\n".join(lines)


def _metric_markdown(metric: str, result, icc) -> str:
    """Return one markdown block for a metric: test result and ICC(1), or "insufficient data"."""
    icc_text = "insufficient data" if icc is None else (
        f"{icc:.4f}" if icc == icc else "NaN (zero variance)")
    if result is None:
        return f"### {metric}\n\n- Comparison: insufficient data\n- ICC(1): {icc_text}\n"
    return (f"### {metric}\n\n"
            f"- Test: {result.test_name}\n"
            f"- Statistic: {result.statistic:.4g}, p = {result.p_value:.4g}\n"
            f"- {result.effect_size_name}: {result.effect_size:.4g}\n"
            f"- Levene p: {result.levene_p:.4g}, groups: {result.n_groups}\n"
            f"- ICC(1) of volumes within samples: {icc_text}\n")


def _skipped_markdown(skipped: list[str]) -> str:
    """Return the skipped-volume list as markdown, or a note that none were skipped."""
    if not skipped:
        return "None."
    return "\n".join(f"- {vid}" for vid in skipped)


def _limitations_markdown() -> str:
    """Return the fixed limitations paragraph (power, resolution, terminology)."""
    return (
        "- With 25 cheese types, the post-hoc stage tests 300 pairs; power per pair is limited "
        "at 5–6 samples per type, so only large differences are detectable.\n"
        "- The Small split resolves structures down to ~3 µm; globules below this are not "
        "separable at the derived 4.4 µm voxel size.\n"
        "- `foreground_fraction` is the volume fraction of the segmented (mostly fat) phase, "
        "not porosity."
    )
