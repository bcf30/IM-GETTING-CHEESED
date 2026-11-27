"""End-to-end smoke test: run the CLI as a subprocess on real data (Requirement 26.4, 26.5)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mozzavid import visualize

pytestmark = pytest.mark.smoke

_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py")
_FIGURES = (visualize.FIG_MONTAGE, visualize.FIG_SEGMENTATION, visualize.FIG_SIZE_DISTRIBUTION,
            visualize.FIG_FRACTION, visualize.FIG_SYNTHETIC)


def test_cli_end_to_end(real_data, tmp_path):
    """Run the CLI with --limit 2 --synthetic-samples 2 and check every output (Req 26.4, 26.5)."""
    output = tmp_path / "results"
    figures = tmp_path / "figures"
    completed = _run_cli(real_data, output, figures)
    assert completed.returncode == 0, completed.stderr
    _assert_text_outputs(output)
    _assert_figures(figures)


def _run_cli(data_root: Path, output: Path, figures: Path) -> subprocess.CompletedProcess:
    """Invoke scripts/run_pipeline.py as a subprocess with a tiny stratified subset."""
    return subprocess.run(
        [sys.executable, _SCRIPT, "--data-root", str(data_root), "--output", str(output),
         "--figures", str(figures), "--limit", "2", "--synthetic-samples", "2"],
        capture_output=True, text=True, timeout=1800)


def _assert_text_outputs(output: Path) -> None:
    """Assert the CSV, txt and md outputs exist, are non-empty and mark insufficient data."""
    for name in ("per_object_metrics.csv", "per_sample_summary.csv", "anova_results.txt",
                 "summary.md", "skipped_volumes.txt"):
        assert (output / name).exists(), name
    assert (output / "per_sample_summary.csv").stat().st_size > 0
    assert "insufficient data" in (output / "anova_results.txt").read_text()
    assert "insufficient data" in (output / "summary.md").read_text()


def _assert_figures(figures: Path) -> None:
    """Assert all five figures were written and are non-empty."""
    for name in _FIGURES:
        path = figures / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
