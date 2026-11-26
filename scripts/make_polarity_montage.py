"""One-off: write figures/polarity_check.png for the H2 polarity checkpoint (task 4.7).

Loads the first volume of five cheese types spread across the cheese_index range and plots
their central Z slices via `visualize.montage_central_slices`. No metrics are computed.
"""

from __future__ import annotations

from pathlib import Path

from mozzavid import io, visualize

CHEESE_INDICES = (1, 7, 13, 19, 25)


def main() -> None:
    """Select one volume per chosen cheese type and write the polarity montage."""
    root = io.resolve_data_root(None)
    meta = io.load_metadata(root)
    volumes = {}
    for idx in CHEESE_INDICES:
        row = meta[meta["cheese_index"] == idx].iloc[0]
        volumes[row["volume_id"]] = io.load_tiff(row["path"])
    out = visualize.montage_central_slices(volumes, Path("figures") / "polarity_check.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
