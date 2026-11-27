"""Validation CLI: segment N volumes from distinct cheese types and write overlays (Req 21.5, 21.6).

Processes N volumes through `pipeline.process_volume`, writes central-slice segmentation
overlays to `<figures>/validation/segmentation_examples.png`, and optionally exports float32
copies for the Fiji macro to read.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # allow `python scripts/validate_segmentation.py` without install
    sys.path.insert(0, str(_REPO_ROOT))

from mozzavid import io, pipeline, visualize  # noqa: E402  (import after the sys.path fix above)

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the validation CLI (Requirement 21.5, 21.6)."""
    parser = argparse.ArgumentParser(description="Validate segmentation on a few real volumes.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--export-float32", default=None)
    return parser


def _select_distinct_types(meta, n: int):
    """Return the first volume of each of up to `n` distinct cheese types (Requirement 21.5)."""
    first_per_type = meta.sort_values(["cheese_index", "sample_id", "volume_id"]).groupby(
        "cheese_type", sort=False).head(1)
    return first_per_type.sort_values("cheese_index").head(n)


def _process_and_collect(rows, pairs: dict, export_dir: Path | None) -> None:
    """Process each selected volume, store its (arr, mask) pair and optionally export float32."""
    for _, row in rows.iterrows():
        arr = io.load_tiff(row["path"])
        result = pipeline.process_volume(arr, io.VOXEL_SIZE_UM, "otsu", 1.5, 50,
                                         pipeline.DEFAULT_INVERT)
        pairs[row["volume_id"]] = (arr, result.mask)
        log.info("validated %s: foreground_fraction=%.4f", row["volume_id"],
                 result.foreground_fraction)
        if export_dir is not None:
            _export_float32(arr, row["volume_id"], export_dir)


def _export_float32(arr, volume_id: str, export_dir: Path) -> None:
    """Write a float32 copy of a volume under `export_dir`, mirroring its relative path."""
    out = export_dir / volume_id.replace("/", "_")
    out.parent.mkdir(parents=True, exist_ok=True)
    io.save_tiff_float32(arr, out)


def main(argv: list[str] | None = None) -> int:
    """Run the validation CLI and write the overlay figure (Requirement 21.5)."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    meta = io.load_metadata(io.resolve_data_root(args.data_root))
    rows = _select_distinct_types(meta, args.n)
    export_dir = Path(args.export_float32) if args.export_float32 else None
    pairs: dict = {}
    _process_and_collect(rows, pairs, export_dir)
    out_path = Path(args.figures) / "validation" / visualize.FIG_SEGMENTATION
    visualize.segmentation_overlays(pairs, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
