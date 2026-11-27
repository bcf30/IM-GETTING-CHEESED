"""Command-line entry point: run the dataset pipeline, then the synthetic benchmark (Req 21).

Parses flags, builds a `PipelineConfig`, calls `pipeline.run`, then `synthetic.benchmark`.
Exits 1 if any volume was skipped (after still writing the benchmark figure), else 0. This is
the only module that orchestrates both the dataset run and the synthetic extension.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow `python scripts/run_pipeline.py` without an install
    sys.path.insert(0, str(_REPO_ROOT))

from mozzavid import io, pipeline, synthetic  # noqa: E402  (import after the sys.path fix above)


def _positive_float(value: str) -> float:
    """argparse type for a strictly positive float, used by --sigma (Requirement 21.4)."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _synthetic_samples(value: str) -> int:
    """argparse type for --synthetic-samples: an integer >= 2 (Welch needs two per group)."""
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be >= 2")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the run CLI (Requirement 21.1)."""
    parser = argparse.ArgumentParser(description="Run the mozzavid microstructure pipeline.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="results")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--method", choices=["otsu", "li"], default="otsu")
    parser.add_argument("--sigma", type=_positive_float, default=1.5)
    parser.add_argument("--min-size", type=int, default=50)
    parser.add_argument("--invert", dest="invert", action="store_true", default=pipeline.DEFAULT_INVERT)
    parser.add_argument("--no-invert", dest="invert", action="store_false")
    parser.add_argument("--synthetic-samples", type=_synthetic_samples, default=20)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _config_from_args(args: argparse.Namespace) -> pipeline.PipelineConfig:
    """Resolve the data root and build a frozen PipelineConfig from parsed arguments."""
    return pipeline.PipelineConfig(
        data_root=io.resolve_data_root(args.data_root), output_dir=Path(args.output),
        figures_dir=Path(args.figures), limit=args.limit, method=args.method,
        sigma=args.sigma, min_size=args.min_size, invert=args.invert)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the pipeline and the benchmark, and return the exit code (Req 21.2, 21.3)."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = _config_from_args(args)
    paths = pipeline.run(config)
    synthetic.benchmark(n_samples=args.synthetic_samples, seed=0, output_dir=config.figures_dir)
    skipped = Path(paths["skipped"]).read_text(encoding="utf-8").strip()
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
