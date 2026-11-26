"""Tests for the Fiji gateway, including Property 1 round-trip fidelity (Requirement 7).

Every test here needs a running JVM, so they all take the session-scoped `ij` fixture
(conftest.py). Property 1 is checked over a seeded family of non-cubic shapes rather than
with a property-based library (Requirement 27.10, 25.5). The remaining tests cover singleton
identity, that every pinned command/class constant resolves in the running Fiji
(Requirement 5.2), and `table_to_dataframe` on both supported Java table types
(Requirements 7.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scyjava

from mozzavid import fiji

# Seeded, non-cubic (Z,Y,X) shapes: distinct extents on every axis so an accidental axis swap
# would change a dimension and fail the slice/height/width assertions below.
ROUND_TRIP_SHAPES: tuple[tuple[int, int, int], ...] = (
    (3, 5, 7),
    (4, 2, 6),
    (2, 7, 3),
    (6, 4, 5),
    (5, 3, 8),
    (7, 6, 2),
    (1, 4, 9),
)


@pytest.mark.parametrize("shape", ROUND_TRIP_SHAPES)
def test_float_round_trip_is_exact(ij, rng, shape):
    """Property 1: a float32 (Z,Y,X) array survives to_java_img -> from_java_array unchanged.

    Validates Requirements 7.1, 7.2. float32 values are exact across the numpy<->Java
    boundary, so equality is asserted exactly, not within a tolerance.
    """
    arr = rng.random(shape, dtype=np.float64).astype(np.float32)

    restored = fiji.from_java_array(fiji.to_java_img(arr))

    assert restored.shape == shape
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, arr)


@pytest.mark.parametrize("shape", ROUND_TRIP_SHAPES)
def test_mask_round_trip_is_exact(ij, rng, shape):
    """Property 1: a boolean (Z,Y,X) mask survives mask_to_imageplus -> imageplus_to_mask.

    Validates Requirements 7.3, 7.4. The ImagePlus must read back as Z slices of YxX with a
    single channel and frame (Requirement 7.3, hypothesis H4), which is where an axis
    mix-up would surface.
    """
    z, y, x = shape
    mask = rng.random(shape) < 0.5

    imp = fiji.mask_to_imageplus(mask)

    assert int(imp.getNSlices()) == z
    assert int(imp.getHeight()) == y
    assert int(imp.getWidth()) == x
    assert int(imp.getNChannels()) == 1
    assert int(imp.getNFrames()) == 1

    restored = fiji.imageplus_to_mask(imp)
    assert restored.shape == shape
    assert restored.dtype == bool
    np.testing.assert_array_equal(restored, mask)


def test_get_ij_returns_the_same_singleton(ij):
    """get_ij() returns the one process-level gateway on every call (Requirement 5.2).

    The `ij` fixture already holds the first instance; a second call must be the same object,
    not a second JVM start.
    """
    assert fiji.get_ij() is ij


def test_every_pinned_constant_resolves(ij):
    """Every CMD_* command is registered and every CLASS_* class imports (Requirement 5.2, 6.12).

    verify_constants() raises naming anything that does not resolve, so a Fiji or MorphoLibJ
    change that renames a command or class fails here rather than mid-pipeline.
    """
    fiji.verify_constants()


def _build_results_table(rows):
    """Build an ij.measure.ResultsTable with a text Label and two numeric columns.

    Mirrors the shape of the Analyze Regions 3D output: a string key column and float
    measurements, so table_to_dataframe is exercised on both column kinds.
    """
    results_table_class = scyjava.jimport("ij.measure.ResultsTable")
    table = results_table_class()
    for index, (label, volume, sphericity) in enumerate(rows):
        table.setValue("Label", index, label)
        table.setValue("Volume", index, volume)
        table.setValue("Sphericity", index, sphericity)
    return table


def test_table_to_dataframe_on_results_table(ij):
    """table_to_dataframe converts an ij.measure.ResultsTable with numeric dtypes (Req 7.5).

    One row per table row, Fiji headings unchanged, numeric columns typed as floats.
    """
    rows = [("1", 120.0, 0.91), ("2", 64.0, 0.85), ("3", 300.0, 0.97)]

    frame = fiji.table_to_dataframe(_build_results_table(rows))

    assert list(frame.columns) == ["Label", "Volume", "Sphericity"]
    assert len(frame) == len(rows)
    assert pd.api.types.is_numeric_dtype(frame["Volume"])
    assert pd.api.types.is_numeric_dtype(frame["Sphericity"])
    np.testing.assert_array_equal(frame["Volume"].to_numpy(), [120.0, 64.0, 300.0])
    np.testing.assert_allclose(frame["Sphericity"].to_numpy(), [0.91, 0.85, 0.97])


def _build_scijava_table(rows):
    """Build an org.scijava.table.DefaultGenericTable with one string and one numeric column."""
    table_class = scyjava.jimport("org.scijava.table.DefaultGenericTable")
    table = table_class()
    table.appendColumn("name")
    table.appendColumn("count")
    for name, count in rows:
        table.appendRow()
        last = int(table.getRowCount()) - 1
        table.set("name", last, scyjava.to_java(name))
        table.set("count", last, scyjava.to_java(count))
    return table


def test_table_to_dataframe_on_scijava_table(ij):
    """table_to_dataframe converts an org.scijava.table.Table with numeric dtypes (Req 7.5).

    The SciJava table path is separate from the ResultsTable path, so it is covered on its
    own table instance built in-test.
    """
    rows = [("a", 1), ("b", 2), ("c", 3)]

    frame = fiji.table_to_dataframe(_build_scijava_table(rows))

    assert list(frame.columns) == ["name", "count"]
    assert len(frame) == len(rows)
    assert pd.api.types.is_numeric_dtype(frame["count"])
    np.testing.assert_array_equal(frame["count"].to_numpy(), [1, 2, 3])


def test_table_to_dataframe_rejects_unsupported_type(ij):
    """A non-table object raises TypeError, so callers cannot pass the wrong Java object."""
    with pytest.raises(TypeError):
        fiji.table_to_dataframe(object())
