"""Tests for displacement_tracker/evaluation/annotation_reference.py."""

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from _helpers import CRS_UTM, CRS_WGS84, utm_to_lonlat, write_annotation_csv
from displacement_tracker.evaluation.annotation_reference import (
    ManualAnnotationReferenceSource,
    _select_date,
)

# 0.01-degree grid anchored at (34.0 E, 32.0 N): cell (row r, col c) covers
# lon [34.0 + 0.01c, 34.0 + 0.01(c+1)) and lat (32.0 - 0.01(r+1), 32.0 - 0.01r].
GRID_TRANSFORM = from_origin(34.0, 32.0, 0.01, 0.01)
GRID_SHAPE = (10, 10)


# ==========================================================
# Date selection
# ==========================================================


def test_multi_date_csv_without_date_raises_listing_available(tmp_path):
    # Given: a CSV with annotations on 2024-10-14 and 2024-11-01
    csv = write_annotation_csv(
        tmp_path / "ann.csv",
        ["2024-10-14,31.99,34.005,5\n", "2024-11-01,31.99,34.005,7\n"],
    )

    # When: the source is constructed without a date
    # Then: it raises ValueError whose message lists both available dates
    with pytest.raises(ValueError, match=r"2024-10-14.*2024-11-01"):
        ManualAnnotationReferenceSource(csv)


def test_single_date_csv_works_without_date(tmp_path):
    # Given: a CSV whose rows all carry the same date, holding counts 5 and 4
    csv = write_annotation_csv(
        tmp_path / "ann.csv",
        ["2024-10-14,31.995,34.005,5\n", "2024-10-14,31.965,34.025,4\n"],
    )

    # When: the source is constructed with date=None and rasterized
    source = ManualAnnotationReferenceSource(csv)
    counts = source.counts_on_grid(GRID_SHAPE, GRID_TRANSFORM, CRS_WGS84)

    # Then: no error is raised and both tiles are loaded (5 + 4 = 9)
    assert counts.sum() == pytest.approx(9.0)


def test_date_selection_accepts_dashed_and_compact_forms(tmp_path):
    # Given: a two-date CSV (counts 5 on 2024-10-14, 7 on 2024-11-01)
    csv = write_annotation_csv(
        tmp_path / "ann.csv",
        ["2024-10-14,31.995,34.005,5\n", "2024-11-01,31.995,34.005,7\n"],
    )

    # When: the source is built with date "2024-10-14" and with "20241014"
    # Then: both forms select only the October rows (total count 5)
    for date in ("2024-10-14", "20241014"):
        source = ManualAnnotationReferenceSource(csv, date=date)
        counts = source.counts_on_grid(GRID_SHAPE, GRID_TRANSFORM, CRS_WGS84)
        assert counts.sum() == pytest.approx(5.0)


def test_unknown_date_raises_listing_available(tmp_path):
    # Given: a CSV with only 2024-10-14 annotations
    csv = write_annotation_csv(tmp_path / "ann.csv", ["2024-10-14,31.995,34.005,5\n"])

    # When: the source is built for the absent date 2023-01-01
    # Then: ValueError names the requested date and the available one
    with pytest.raises(ValueError, match=r"2023-01-01.*2024-10-14"):
        ManualAnnotationReferenceSource(csv, date="2023-01-01")


def test_select_date_ignores_nan_dates_when_counting_ambiguity():
    import pandas as pd

    # Given: a frame with one real date plus a NaN date entry
    df = pd.DataFrame({"date": ["2024-10-14", np.nan], "manual_tent_count": [1, 2]})

    # When: _select_date runs with date=None
    out = _select_date(df, None, "date", "fake.csv")

    # Then: NaN does not count as a second date, so the frame passes through
    #       with both rows intact
    assert len(out) == 2


def test_missing_csv_raises_file_not_found(tmp_path):
    # Given: a path to a CSV that does not exist
    missing = str(tmp_path / "nope.csv")

    # When: the source is constructed on that path
    # Then: FileNotFoundError is raised (not a pandas read error)
    with pytest.raises(FileNotFoundError):
        ManualAnnotationReferenceSource(missing)


def test_missing_count_column_raises(tmp_path):
    # Given: a CSV lacking the manual_tent_count column
    csv = tmp_path / "ann.csv"
    csv.write_text("date,latitude,longitude\n2024-10-14,31.99,34.005\n")

    # When: the source is constructed
    # Then: the column validation raises ValueError naming the column
    with pytest.raises(ValueError, match="manual_tent_count"):
        ManualAnnotationReferenceSource(str(csv))


# ==========================================================
# counts_on_grid
# ==========================================================


def test_counts_on_grid_additive_merge_and_cell_placement(tmp_path):
    # Given: two tiles at (34.005, 31.995) with counts 5 and 7 — both in grid
    #        cell (row 0, col 0) — and one tile at (34.025, 31.965) with count 4
    #        in cell (row 3, col 2)
    csv = write_annotation_csv(
        tmp_path / "ann.csv",
        [
            "2024-10-14,31.995,34.005,5\n",
            "2024-10-14,31.995,34.005,7\n",
            "2024-10-14,31.965,34.025,4\n",
        ],
    )
    source = ManualAnnotationReferenceSource(csv)

    # When: counts_on_grid rasterizes onto the 10x10 0.01-degree grid
    counts = source.counts_on_grid(GRID_SHAPE, GRID_TRANSFORM, CRS_WGS84)

    # Then: cell (0,0) holds the SUM 12 (additive merge), cell (3,2) holds 4,
    #       everything else is 0, and the dtype is float32
    assert counts.dtype == np.float32
    assert counts[0, 0] == pytest.approx(12.0)
    assert counts[3, 2] == pytest.approx(4.0)
    assert counts.sum() == pytest.approx(16.0)


def test_counts_on_grid_reprojects_tiles_into_grid_crs(tmp_path):
    # Given: a tile whose WGS84 centroid corresponds to UTM 36N (500000+50,
    #        3474000+50) and a 100 m UTM grid anchored at (500000, 3474100)
    lon, lat = utm_to_lonlat(500050.0, 3474050.0)
    csv = write_annotation_csv(tmp_path / "ann.csv", [f"2024-10-14,{lat},{lon},9\n"])
    source = ManualAnnotationReferenceSource(csv)
    utm_transform = from_origin(500000.0, 3474100.0, 100.0, 100.0)

    # When: counts_on_grid runs with crs EPSG:32636
    counts = source.counts_on_grid((3, 3), utm_transform, CRS_UTM)

    # Then: the tile's count 9 lands in cell (row 0, col 0) of the UTM grid
    assert counts[0, 0] == pytest.approx(9.0)
    assert counts.sum() == pytest.approx(9.0)


def test_counts_on_grid_clip_geom_is_applied_in_grid_crs(tmp_path):
    # Given: a tile whose WGS84 centroid reprojects to UTM 36N (500050,
    #        3474050), the 3x3 100 m UTM grid anchored at (500000, 3474100),
    #        and clip boxes expressed in the GRID CRS (as documented)
    lon, lat = utm_to_lonlat(500050.0, 3474050.0)
    csv = write_annotation_csv(tmp_path / "ann.csv", [f"2024-10-14,{lat},{lon},9\n"])
    source = ManualAnnotationReferenceSource(csv)
    utm_transform = from_origin(500000.0, 3474100.0, 100.0, 100.0)

    # When: counts_on_grid runs with crs EPSG:32636 and the clip box
    #       box(500000, 3474000, 500100, 3474100), which contains the
    #       REPROJECTED tile
    containing = box(500000.0, 3474000.0, 500100.0, 3474100.0)
    counts = source.counts_on_grid((3, 3), utm_transform, CRS_UTM, containing)

    # Then: the box keeps the tile — count 9 in cell (0, 0), since x 500050
    #       falls in column 0's [500000, 500100) and y 3474050 in row 0's
    #       (3474000, 3474100] — which requires reprojecting BEFORE clipping
    #       (in WGS84 the tile sits near lon 34, nowhere near x=500000)
    assert counts[0, 0] == pytest.approx(9.0)
    assert counts.sum() == pytest.approx(9.0)

    # When: the same call runs with box(500200, 3474000, 500300, 3474100),
    #       which is disjoint from the reprojected tile
    disjoint = box(500200.0, 3474000.0, 500300.0, 3474100.0)
    counts = source.counts_on_grid((3, 3), utm_transform, CRS_UTM, disjoint)

    # Then: nothing survives the clip and the grid is all zeros
    assert counts.sum() == 0.0


def test_counts_on_grid_clip_geom_excludes_outside_tiles(tmp_path):
    # Given: tiles at (34.005, 31.995) count 5 and (34.025, 31.965) count 4,
    #        and a clip box covering only the first tile
    csv = write_annotation_csv(
        tmp_path / "ann.csv",
        ["2024-10-14,31.995,34.005,5\n", "2024-10-14,31.965,34.025,4\n"],
    )
    source = ManualAnnotationReferenceSource(csv)
    clip = box(34.0, 31.98, 34.02, 32.0)

    # When: counts_on_grid runs with that clip_geom
    counts = source.counts_on_grid(GRID_SHAPE, GRID_TRANSFORM, CRS_WGS84, clip)

    # Then: only the first tile contributes; total is 5 and cell (3,2) is 0
    assert counts[0, 0] == pytest.approx(5.0)
    assert counts[3, 2] == pytest.approx(0.0)
    assert counts.sum() == pytest.approx(5.0)


def test_counts_on_grid_all_clipped_returns_zeros(tmp_path):
    # Given: one tile and a clip geometry that contains no tiles
    csv = write_annotation_csv(tmp_path / "ann.csv", ["2024-10-14,31.995,34.005,5\n"])
    source = ManualAnnotationReferenceSource(csv)
    clip = box(35.0, 30.0, 35.1, 30.1)

    # When: counts_on_grid runs
    counts = source.counts_on_grid(GRID_SHAPE, GRID_TRANSFORM, CRS_WGS84, clip)

    # Then: a float32 all-zero array of the grid shape is returned
    assert counts.shape == GRID_SHAPE
    assert counts.dtype == np.float32
    assert counts.sum() == 0.0


def test_nan_counts_are_dropped_not_rasterized(tmp_path):
    # Given: two tiles in the same cell, one with count 5 and one with an
    #        empty (NaN) count
    csv = write_annotation_csv(
        tmp_path / "ann.csv",
        ["2024-10-14,31.995,34.005,5\n", "2024-10-14,31.995,34.005,\n"],
    )

    # When: the source loads and rasterizes them
    source = ManualAnnotationReferenceSource(csv)
    counts = source.counts_on_grid(GRID_SHAPE, GRID_TRANSFORM, CRS_WGS84)

    # Then: the NaN tile is dropped, leaving 5 in the cell (not NaN, not 5+0)
    assert counts[0, 0] == pytest.approx(5.0)
    assert np.isfinite(counts).all()


def test_custom_count_column(tmp_path):
    # Given: a CSV whose count column is named other_count (value 11)
    csv = tmp_path / "ann.csv"
    csv.write_text("date,latitude,longitude,other_count\n2024-10-14,31.995,34.005,11\n")

    # When: the source is built with count_column="other_count" and rasterized
    source = ManualAnnotationReferenceSource(str(csv), count_column="other_count")
    counts = source.counts_on_grid(GRID_SHAPE, GRID_TRANSFORM, CRS_WGS84)

    # Then: that column drives the rasterized totals
    assert counts[0, 0] == pytest.approx(11.0)
