"""Tests for mozzavid.pipeline (Requirements 12, 13; design Property 12 pipeline part)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from mozzavid import pipeline
from tests import _shapes


def _metadata_25_types() -> pd.DataFrame:
    """Build a synthetic 25-type metadata frame with several samples and volumes each."""
    rows = []
    for c in range(1, 26):
        for sample in range(1, 4):
            for v in range(2):
                vid = f"cheese_{c}/{sample}/{v}.tiff"
                rows.append({"volume_id": vid, "cheese_type": f"cheese_{c}",
                             "cheese_index": c, "sample_id": f"cheese_{c}/{sample}"})
    return pd.DataFrame(rows)


def test_select_volumes_round_robin_two_per_type():
    """limit=50 selects exactly 2 volumes from each of the 25 cheese types (Requirement 13.4)."""
    selected = pipeline.select_volumes(_metadata_25_types(), 50)
    assert len(selected) == 50
    assert (selected.groupby("cheese_type").size() == 2).all()


def test_select_volumes_none_returns_all():
    """limit=None selects every volume (Requirement 13.2)."""
    meta = _metadata_25_types()
    assert len(pipeline.select_volumes(meta, None)) == len(meta)


def test_process_volume_on_synthetic(ij):
    """process_volume returns a populated VolumeResult on a synthetic volume (Requirement 12.2)."""
    mask = _shapes.sphere_mask((48, 48, 48), (24, 24, 24), 8.0)
    arr = np.where(mask, 0.2, 0.8).astype(np.float32)  # dark sphere on bright matrix
    result = pipeline.process_volume(arr, 1.0, "otsu", 1.5, 50, invert=True)
    assert result.mask.shape == arr.shape
    assert 0.0 < result.foreground_fraction < 1.0
    assert len(result.objects) >= 1
    assert result.seconds > 0.0


def _write_small_dataset(root, rng) -> None:
    """Write a tiny 2-type dataset of segmentable volumes with a coarse CSV."""
    layout = [("cheese_1/1/0.tiff", 0), ("cheese_1/1/1.tiff", 0), ("cheese_1/2/0.tiff", 0),
              ("cheese_2/1/0.tiff", 1), ("cheese_2/2/0.tiff", 1), ("cheese_2/2/1.tiff", 1)]
    rows = []
    for rel, label in layout:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = _shapes.sphere_mask((32, 32, 32), tuple(rng.integers(10, 22, 3)), 6.0)
        tifffile.imwrite(path, np.where(mask, 0.2, 0.8).astype(np.float32))
        rows.append({"file": rel, "label_cheese": label, "label_sample": len(rows), "split": "t"})
    pd.DataFrame(rows).to_csv(root / "dataset_coarse.csv", index=False)


def test_run_is_byte_deterministic(ij, tmp_path):
    """Property 12: two runs on the same data write byte-identical CSVs (Requirement 13.5)."""
    data = tmp_path / "data"
    data.mkdir()
    _write_small_dataset(data, np.random.default_rng(0))
    first = _run_once(data, tmp_path / "out1", tmp_path / "fig1")
    second = _run_once(data, tmp_path / "out2", tmp_path / "fig2")
    for name in ("per_object_metrics.csv", "per_sample_summary.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _run_once(data, out, fig):
    """Run the pipeline once and return the output directory."""
    config = pipeline.PipelineConfig(data_root=data, output_dir=out, figures_dir=fig, invert=True)
    pipeline.run(config)
    return out


def test_corrupt_tiff_is_skipped_and_listed(ij, tmp_path):
    """A corrupt TIFF is skipped, listed, and the run still completes (Requirement 13.6, 13.12)."""
    data = tmp_path / "data"
    data.mkdir()
    _write_small_dataset(data, np.random.default_rng(1))
    (data / "cheese_1/1/0.tiff").write_bytes(b"not a tiff")
    config = pipeline.PipelineConfig(data_root=data, output_dir=tmp_path / "out",
                                     figures_dir=tmp_path / "fig", invert=True)
    paths = pipeline.run(config)
    skipped = (tmp_path / "out" / "skipped_volumes.txt").read_text()
    assert "cheese_1/1/0.tiff" in skipped


def test_insufficient_data_written_for_undersized_groups(ij, tmp_path):
    """"insufficient data" is written when a metric has no usable groups (Requirement 13.8).

    Every sphere here touches the volume edge, so there are no interior objects and the
    interior-only size metrics (median_equiv_diameter_um, d50_um) are NaN for all samples,
    which makes `compare` raise and the pipeline write "insufficient data" for them.
    """
    data = tmp_path / "data"
    data.mkdir()
    _write_border_dataset(data)
    config = pipeline.PipelineConfig(data_root=data, output_dir=tmp_path / "out",
                                     figures_dir=tmp_path / "fig", invert=True)
    pipeline.run(config)
    anova = (tmp_path / "out" / "anova_results.txt").read_text()
    assert "insufficient data" in anova


def _write_border_dataset(root) -> None:
    """Write a 2-type dataset whose only object in each volume touches the volume edge."""
    layout = [("cheese_1/1/0.tiff", 0), ("cheese_1/2/0.tiff", 0),
              ("cheese_2/1/0.tiff", 1), ("cheese_2/2/0.tiff", 1)]
    rows = []
    for rel, label in layout:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = _shapes.sphere_mask((32, 32, 32), (0, 16, 16), 7.0)  # cut by the z=0 face
        tifffile.imwrite(path, np.where(mask, 0.2, 0.8).astype(np.float32))
        rows.append({"file": rel, "label_cheese": label, "label_sample": len(rows), "split": "t"})
    pd.DataFrame(rows).to_csv(root / "dataset_coarse.csv", index=False)
