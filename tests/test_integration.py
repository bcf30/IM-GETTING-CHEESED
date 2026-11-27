"""Data-dependent integration tests against ./Small (Requirement 26; design Properties 7, 14)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mozzavid import io, measure, pipeline

pytestmark = pytest.mark.integration

_SCHEMA = ("label", "volume_um3", "equiv_diameter_um", "surface_area_um2", "sphericity",
           "feret_diameter_max_um", "centroid_z_um", "centroid_y_um", "centroid_x_um",
           "touches_border")


def test_metadata_counts(real_data):
    """Property 14: the full metadata has 591 rows, 149 samples and 25 cheese types (Req 26.3)."""
    meta = io.load_metadata(real_data)
    assert len(meta) == 591
    assert meta["sample_id"].nunique() == 149
    assert meta["cheese_type"].nunique() == 25


def test_metadata_raises_when_file_added(real_data, monkeypatch):
    """Property 14: load_metadata raises if a TIFF is added to the disk listing (Req 26.3).

    The on-disk `./Small` is never modified; `scan_volume_files` is monkeypatched to return a
    listing with one extra path, mimicking an added file.
    """
    real_listing = io.scan_volume_files(real_data)
    monkeypatch.setattr(io, "scan_volume_files", lambda root: real_listing + ["cheese_1/1/999.tiff"])
    with pytest.raises(ValueError):
        io.load_metadata(real_data)


def test_metadata_raises_when_file_removed(real_data, monkeypatch):
    """Property 14: load_metadata raises if a TIFF is removed from the disk listing (Req 26.3)."""
    real_listing = io.scan_volume_files(real_data)
    monkeypatch.setattr(io, "scan_volume_files", lambda root: real_listing[:-1])
    with pytest.raises(ValueError):
        io.load_metadata(real_data)


def _five_volumes_distinct_types(meta):
    """Return the first volume of each of 5 cheese types spread over cheese_index."""
    first_per_type = meta.sort_values(["cheese_index", "sample_id", "volume_id"]).groupby(
        "cheese_type", sort=False).head(1).sort_values("cheese_index")
    picks = np.linspace(0, len(first_per_type) - 1, 5).round().astype(int)
    return first_per_type.iloc[np.unique(picks)]


def test_five_real_volumes(ij, real_data, caplog):
    """Process 5 real volumes: schema, fraction in (0,1), volume conservation, logged seconds (Req 26.2)."""
    meta = io.load_metadata(real_data)
    rows = _five_volumes_distinct_types(meta)
    with caplog.at_level(logging.INFO, logger="mozzavid.pipeline"):
        for _, row in rows.iterrows():
            _check_one_volume(row)
    assert any("objects" in message for message in caplog.messages)


def _check_one_volume(row):
    """Run one real volume through process_volume and assert its invariants (Property 7)."""
    log = logging.getLogger("mozzavid.pipeline")
    arr = io.load_tiff(row["path"])
    result = pipeline.process_volume(arr, io.VOXEL_SIZE_UM, "otsu", 1.5, 50, pipeline.DEFAULT_INVERT)
    log.info("volume %s: %.1fs, %d objects", row["volume_id"], result.seconds, len(result.objects))
    assert result.seconds > 0.0
    assert len(result.objects) > 0
    assert tuple(result.objects.columns) == _SCHEMA
    assert 0.0 < result.foreground_fraction < 1.0
    expected = result.mask.sum() * io.VOXEL_SIZE_UM**3
    assert result.objects["volume_um3"].sum() == pytest.approx(expected, rel=1e-9)
