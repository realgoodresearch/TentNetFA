"""Unit tests for displacement_tracker/util/reference_data.py.

Covers date parsing/selection, point rasterization onto the master grid,
vector/unosat/raster reference sources, and the build_reference_source
config dispatch — plus one integration test through
validation_core.prepare_grouped_cell_inputs.
"""

from datetime import datetime

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from _helpers import CRS_UTM, CRS_WGS84, write_geotiff
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from displacement_tracker.util.reference_data import (
    PointsSource,
    RasterReferenceSource,
    UnosatReferenceSource,
    VectorReferenceSource,
    build_reference_source,
    extract_date_from_filename,
    infer_target_date,
    rasterize_point_counts,
)
from displacement_tracker.util.validation_core import prepare_grouped_cell_inputs


def _points_gdf(coords, crs=CRS_UTM, **columns):
    return gpd.GeoDataFrame(columns, geometry=[Point(x, y) for x, y in coords], crs=crs)


def _write_export(path, coords):
    """A readable single-point export at `path` (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _points_gdf([coords], crs=CRS_WGS84).to_file(path, driver="GeoJSON")
    return path


# The window most expectations are derived against: 4x4 one-degree cells with
# origin (0, 4), so cell (row, col) covers lon [col, col+1), lat (4-row-1, 4-row].
DEGREE_GRID = ((4, 4), from_origin(0, 4, 1, 1), CRS_WGS84)


# ---------------------------------------------------------------------------
# Date parsing on file names
# ---------------------------------------------------------------------------


def test_extract_date_dashed_format():
    # Given: a file name carrying a YYYY-MM-DD date
    name = "pred_2024-05-17.geojson"

    # When: extract_date_from_filename parses it
    parsed = extract_date_from_filename(name)

    # Then: the corresponding datetime is returned
    assert parsed == datetime(2024, 5, 17)


def test_extract_date_compact_format():
    # Given: a file name carrying a compact YYYYMMDD date
    name = "/data/exports/UNOSAT_20240103_shelters.gpkg"

    # When: extract_date_from_filename parses it
    parsed = extract_date_from_filename(name)

    # Then: the corresponding datetime is returned
    assert parsed == datetime(2024, 1, 3)


def test_extract_date_invalid_or_absent_returns_none():
    # Given: names with no date, an impossible calendar date, and a date
    #        only present in a parent directory (not in the file name)
    no_date = "notes.geojson"
    impossible_date = "export_20241340.geojson"
    date_in_parent_dir = "/data/2024-01-01/export.geojson"

    # When: extract_date_from_filename parses each
    from_no_date = extract_date_from_filename(no_date)
    from_impossible_date = extract_date_from_filename(impossible_date)
    from_parent_dir = extract_date_from_filename(date_in_parent_dir)

    # Then: all three return None
    assert from_no_date is None
    assert from_impossible_date is None
    assert from_parent_dir is None


def test_infer_target_date_median_odd_count():
    # Given: three dated prediction files (one outlier) plus one undated file
    paths = [
        "a_2024-01-01.geojson",
        "b_2024-01-05.geojson",
        "c_2024-03-01.geojson",
        "undated.geojson",
    ]

    # When: infer_target_date computes the median of the parseable dates
    target = infer_target_date(paths)

    # Then: the middle date wins and the undated file is ignored
    assert target == datetime(2024, 1, 5)


def test_infer_target_date_sorts_dates_before_taking_median():
    # Given: three dated files listed out of chronological order (the
    #        latest date first)
    paths = [
        "c_2024-03-01.geojson",
        "a_2024-01-01.geojson",
        "b_2024-01-05.geojson",
    ]

    # When: infer_target_date computes the median
    target = infer_target_date(paths)

    # Then: the chronological middle date wins — the dates are sorted
    #       before indexing, not taken in listing order (the unsorted
    #       middle entry would be 2024-01-01)
    assert target == datetime(2024, 1, 5)


def test_infer_target_date_even_count_takes_upper_median():
    # Given: four dated files
    paths = [
        "p_2024-01-11.geojson",
        "p_2024-01-01.geojson",
        "p_2024-01-10.geojson",
        "p_2024-01-02.geojson",
    ]

    # When: infer_target_date runs
    target = infer_target_date(paths)

    # Then: the upper of the two middle dates is returned
    assert target == datetime(2024, 1, 10)


def test_infer_target_date_no_parseable_dates_returns_none():
    # Given: file names with no parseable date at all
    paths = ["a.geojson", "b.gpkg"]

    # When: infer_target_date runs
    target = infer_target_date(paths)

    # Then: it returns None rather than guessing
    assert target is None


# ---------------------------------------------------------------------------
# Rasterization of point counts
# ---------------------------------------------------------------------------


def test_rasterize_point_counts_cells_and_accumulation():
    # Given: a 4x4 grid of 100 m cells with origin (0, 400) and three points:
    #        (150, 350) and (155, 345) share cell (row 0, col 1);
    #        (250, 50) falls in cell (row 3, col 2)
    transform = from_origin(0, 400, 100, 100)
    gdf = _points_gdf([(150, 350), (155, 345), (250, 50)])

    # When: rasterize_point_counts burns them with additive merging
    counts = rasterize_point_counts(gdf, (4, 4), transform)

    # Then: the shared cell counts 2, the other cell counts 1, the rest are
    #       0, and the array comes back as float32
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0, 1] = 2.0
    expected[3, 2] = 1.0
    np.testing.assert_array_equal(counts, expected)
    assert counts.dtype == np.float32


def test_rasterize_point_counts_empty_returns_zeros():
    # Given: an empty GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[], crs=CRS_UTM)

    # When: rasterize_point_counts runs on a 3x2 grid
    counts = rasterize_point_counts(gdf, (3, 2), from_origin(0, 300, 100, 100))

    # Then: an all-zero float32 array of the requested shape is returned
    assert counts.shape == (3, 2)
    assert counts.dtype == np.float32
    assert not counts.any()


# ---------------------------------------------------------------------------
# PointsSource: CRS handling and clipping
# ---------------------------------------------------------------------------


def test_points_source_missing_crs_raises():
    # Given: reference points with no CRS and a grid that does have one
    source = PointsSource(gpd.GeoDataFrame(geometry=[Point(1, 1)], crs=None))

    # When: counts_on_grid is asked to resolve them
    # Then: a ValueError is raised instead of silently rasterizing
    with pytest.raises(ValueError, match="no CRS"):
        source.counts_on_grid((2, 2), from_origin(0, 200, 100, 100), CRS_UTM)


def test_points_source_reprojects_to_grid_crs():
    # Given: one WGS84 point at (lon 0.001, lat 0.0005), which is
    #        (~111.32 m, ~55.66 m) in EPSG:3857, and a 2x4 grid of 100 m
    #        cells in EPSG:3857 with origin (0, 200)
    source = PointsSource(_points_gdf([(0.001, 0.0005)], crs=CRS_WGS84))

    # When: counts_on_grid resolves the point onto that grid
    counts = source.counts_on_grid((2, 4), from_origin(0, 200, 100, 100), "EPSG:3857")

    # Then: the count lands in row 1 (y in [0, 100)), col 1 (x in [100, 200))
    expected = np.zeros((2, 4), dtype=np.float32)
    expected[1, 1] = 1.0
    np.testing.assert_array_equal(counts, expected)


def test_points_source_without_a_grid_crs_rasterizes_unprojected():
    # Given: reference points carrying no CRS, at (150, 350) on a 4x4 grid
    #        of 100 m cells with origin (0, 400)
    source = PointsSource(gpd.GeoDataFrame(geometry=[Point(150, 350)], crs=None))

    # When: counts_on_grid resolves them against a window that also has no
    #       CRS
    counts = source.counts_on_grid((4, 4), from_origin(0, 400, 100, 100), None)

    # Then: the missing-CRS guard is skipped rather than raising — with no
    #       grid CRS there is nothing to reproject to, so the coordinates
    #       are burned as-is into cell (0, 1). A CRS-less master grid
    #       therefore accepts unprojected reference points silently; the
    #       guard above only fires once the grid declares a CRS.
    assert counts[0, 1] == 1.0
    assert counts.sum() == pytest.approx(1.0)


def test_points_source_clip_geom_excludes_outside_points():
    # Given: points at (150, 350) and (350, 50) on a 4x4 grid with origin
    #        (0, 400), and a clip box covering only x in [0, 200],
    #        y in [200, 400]
    source = PointsSource(_points_gdf([(150, 350), (350, 50)]))

    # When: counts_on_grid runs with that clip geometry
    counts = source.counts_on_grid(
        (4, 4),
        from_origin(0, 400, 100, 100),
        CRS_UTM,
        clip_geom=box(0, 200, 200, 400),
    )

    # Then: only the inside point is counted; the outside cell stays 0
    assert counts[0, 1] == 1.0
    assert counts[3, 3] == 0.0
    assert counts.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# VectorReferenceSource: real files, centroids, layer/where
# ---------------------------------------------------------------------------


def test_vector_source_reads_file_and_reduces_polygons_to_centroids(tmp_path):
    # Given: a GeoJSON with one point at (1.5, 3.5) and one square polygon
    #        (1, 1)-(2, 2) whose centroid is (1.5, 1.5), on a 4x4 grid of
    #        1-degree cells with origin (0, 4) in EPSG:4326
    path = tmp_path / "annotations.geojson"
    gdf = gpd.GeoDataFrame(geometry=[Point(1.5, 3.5), box(1, 1, 2, 2)], crs=CRS_WGS84)
    gdf.to_file(path, driver="GeoJSON")

    # When: VectorReferenceSource loads the file and resolves counts
    counts = VectorReferenceSource(str(path)).counts_on_grid(*DEGREE_GRID)

    # Then: the point counts in cell (0, 1) and the polygon's centroid in
    #       cell (2, 1)
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0, 1] = 1.0
    expected[2, 1] = 1.0
    np.testing.assert_array_equal(counts, expected)


def test_vector_source_missing_file_raises(tmp_path):
    # Given: a path that does not exist
    missing = tmp_path / "missing.geojson"

    # When: VectorReferenceSource is constructed
    # Then: FileNotFoundError is raised up front
    with pytest.raises(FileNotFoundError):
        VectorReferenceSource(str(missing))


def test_vector_source_layer_and_where_filters(tmp_path):
    # Given: a two-layer GPKG; layer "tents" has a 'tent' point at
    #        (0.5, 3.5) and a 'rubble' point at (1.5, 2.5); layer "other"
    #        has one point at (3.5, 0.5). The second write is mode="a" so
    #        the two-layer intent does not rest on whether the installed
    #        engine treats a default mode="w" write to an existing GPKG as
    #        adding a layer or recreating the datasource.
    path = tmp_path / "multi.gpkg"
    _points_gdf(
        [(0.5, 3.5), (1.5, 2.5)], crs=CRS_WGS84, kind=["tent", "rubble"]
    ).to_file(path, layer="tents")
    _points_gdf([(3.5, 0.5)], crs=CRS_WGS84, kind=["x"]).to_file(
        path, layer="other", mode="a"
    )

    # When: VectorReferenceSource reads layer "tents" with
    #       where="kind = 'tent'"
    source = VectorReferenceSource(str(path), layer="tents", where="kind = 'tent'")
    counts = source.counts_on_grid(*DEGREE_GRID)

    # Then: the filtered read rasterizes only the tent point — cell (0, 0)
    #       counts 1, total exactly 1
    assert counts[0, 0] == 1.0
    assert counts.sum() == pytest.approx(1.0)

    # When: the other layer is read with no filter
    other = VectorReferenceSource(str(path), layer="other").counts_on_grid(*DEGREE_GRID)

    # Then: only cell (3, 3) counts, proving the layer argument really
    #       selects a layer
    assert other[3, 3] == 1.0
    assert other.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# UNOSAT export selection
# ---------------------------------------------------------------------------


def test_unosat_source_reads_a_plain_file_export(tmp_path):
    # Given: a path that is a single export file rather than a directory,
    #        holding one point in cell (0, 0)
    path = _write_export(tmp_path / "export_20240101.geojson", (0.5, 3.5))

    # When: the source resolves counts from it with no date given
    counts = UnosatReferenceSource(str(path)).counts_on_grid(*DEGREE_GRID)

    # Then: the file is read as-is — directory selection never runs, so the
    #       absent date is not an error
    assert counts[0, 0] == 1.0
    assert counts.sum() == pytest.approx(1.0)


def test_unosat_source_matches_an_explicit_date_ignoring_dashes(tmp_path):
    # Given: a dashed-date export in a child directory (point in cell (0, 0))
    #        and a compact-date export at the top level (point in cell (2, 2))
    _write_export(tmp_path / "child" / "exp_2024-01-15.geojson", (0.5, 3.5))
    _write_export(tmp_path / "exp_20240220.geojson", (2.5, 1.5))

    # When: each is requested with an explicit date in the opposite formatting
    by_compact_request = UnosatReferenceSource(
        str(tmp_path), date="20240115"
    ).counts_on_grid(*DEGREE_GRID)
    by_dashed_request = UnosatReferenceSource(
        str(tmp_path), date="2024-02-20"
    ).counts_on_grid(*DEGREE_GRID)

    # Then: each request resolves to its own export and nothing else —
    #       dashes are dropped on both sides of the comparison, and the
    #       nested export is found by the recursive walk
    assert by_compact_request[0, 0] == 1.0
    assert by_compact_request.sum() == pytest.approx(1.0)
    assert by_dashed_request[2, 2] == 1.0
    assert by_dashed_request.sum() == pytest.approx(1.0)


def test_unosat_source_unmatched_date_lists_the_available_exports(tmp_path):
    # Given: a directory whose only export is dated 2024-01-15. Selection
    #        raises before any file is opened, so an empty file is a
    #        sufficient fixture for the error cases here and below.
    (tmp_path / "exp_2024-01-15.geojson").touch()

    # When: an explicit date with no match is requested
    # Then: ValueError reports 0 matches and names the available export
    with pytest.raises(ValueError, match=r"found 0.*exp_2024-01-15\.geojson"):
        UnosatReferenceSource(str(tmp_path), date="2024-03-03")


def test_unosat_source_ambiguous_date_refuses_to_choose(tmp_path):
    # Given: two exports carrying the same date stamp
    (tmp_path / "a_20240115.geojson").touch()
    (tmp_path / "b_2024-01-15.gpkg").touch()

    # When: that date is requested explicitly
    # Then: ValueError reports 2 matches instead of picking one arbitrarily
    with pytest.raises(ValueError, match="found 2"):
        UnosatReferenceSource(str(tmp_path), date="2024-01-15")


def test_unosat_source_considers_gdb_directories_as_export_candidates(tmp_path):
    # Given: one date stamp shared by a nested .geojson export, a .GDB
    #        directory (a geodatabase is a directory on disk, its internals
    #        carry no vector suffix, and ArcGIS ships them upper-cased), a
    #        .txt file and a plain directory
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "exp_20240101.geojson").touch()
    gdb = tmp_path / "fake_20240101.GDB"
    gdb.mkdir()
    (gdb / "gdbtable").touch()
    (tmp_path / "notes_20240101.txt").touch()
    (tmp_path / "plaindir_20240101").mkdir()

    # When: that shared date is requested
    with pytest.raises(ValueError, match="found 2") as excinfo:
        UnosatReferenceSource(str(tmp_path), date="2024-01-01")

    # Then: exactly two candidates are reported — the geodatabase directory
    #       and the nested file. The directory qualifies only via the
    #       suffix check, which lower-cases before comparing (a directory
    #       never satisfies the is_file() branch), so a case-sensitive
    #       comparison there would drop it and leave one match. The .txt and
    #       the plain directory never enter the listing even though they
    #       carry the same date stamp, and the geodatabase's own contents
    #       are not walked into.
    message = str(excinfo.value)
    assert "fake_20240101.GDB" in message
    assert "exp_20240101.geojson" in message
    assert "notes_20240101.txt" not in message
    assert "plaindir_20240101" not in message
    assert "gdbtable" not in message


def test_unosat_source_matches_export_suffixes_case_insensitively(tmp_path):
    # Given: an export whose suffix is upper case
    _write_export(tmp_path / "exp_20240301.GEOJSON", (0.5, 3.5))

    # When: auto-discovery resolves the directory against a nearby date
    counts = UnosatReferenceSource(
        str(tmp_path), nearest_to=datetime(2024, 3, 2)
    ).counts_on_grid(*DEGREE_GRID)

    # Then: the upper-case export is both matched and readable — the suffix
    #       is lower-cased before comparison
    assert counts[0, 0] == 1.0
    assert counts.sum() == pytest.approx(1.0)


def test_unosat_source_picks_the_export_closest_to_the_prediction_date(tmp_path):
    # Given: exports dated 2024-01-01 (point in cell (0, 0)) and 2024-02-01
    #        (point in cell (2, 2))
    _write_export(tmp_path / "exp_20240101.geojson", (0.5, 3.5))
    _write_export(tmp_path / "exp_20240201.geojson", (2.5, 1.5))

    # When: auto-discovery runs against prediction date 2024-01-20 — 19 days
    #       after the January export, 12 days before the February one
    counts = UnosatReferenceSource(
        str(tmp_path), nearest_to=datetime(2024, 1, 20)
    ).counts_on_grid(*DEGREE_GRID)

    # Then: the February export wins on absolute distance, so a signed
    #       comparison picking the earlier export would fail here
    assert counts[2, 2] == 1.0
    assert counts.sum() == pytest.approx(1.0)


def test_unosat_source_breaks_a_date_tie_on_file_name(tmp_path):
    # Given: two exports exactly five days either side of the prediction
    #        date, the lexicographically smaller name holding the cell (0, 0)
    #        point and the larger one the cell (2, 2) point
    _write_export(tmp_path / "a_20240120.geojson", (0.5, 3.5))
    _write_export(tmp_path / "b_20240110.geojson", (2.5, 1.5))

    # When: auto-discovery runs against a date equidistant from both
    counts = UnosatReferenceSource(
        str(tmp_path), nearest_to=datetime(2024, 1, 15)
    ).counts_on_grid(*DEGREE_GRID)

    # Then: the lexicographically smaller file name breaks the tie, so the
    #       choice is deterministic rather than filesystem-order dependent
    assert counts[0, 0] == 1.0
    assert counts.sum() == pytest.approx(1.0)


def test_unosat_source_bare_directory_demands_an_explicit_date(tmp_path):
    # Given: a directory of exports, with neither an explicit date nor a
    #        prediction date to auto-discover against
    (tmp_path / "exp_20240101.geojson").touch()

    # When: the source is built
    # Then: it refuses with an actionable ValueError naming reference.date
    with pytest.raises(ValueError, match="reference.date"):
        UnosatReferenceSource(str(tmp_path))


def test_unosat_source_undated_directory_refuses_auto_discovery(tmp_path):
    # Given: a directory whose exports carry no parseable date stamp
    (tmp_path / "shelters.geojson").touch()

    # When: auto-discovery against a prediction date is attempted
    # Then: ValueError says no date-stamped exports were found, rather than
    #       falling back to an arbitrary file
    with pytest.raises(ValueError, match="No date-stamped exports"):
        UnosatReferenceSource(str(tmp_path), nearest_to=datetime(2024, 1, 1))


# ---------------------------------------------------------------------------
# RasterReferenceSource
# ---------------------------------------------------------------------------


def test_raster_source_non_square_grid_preserves_shape_and_values(tmp_path):
    # Given: a 3x5 counts GeoTIFF holding 0..14 (3 rows, 5 columns)
    transform = from_origin(0, 300, 100, 100)
    data = np.arange(15, dtype="float32").reshape(3, 5)
    path = tmp_path / "rect.tif"
    write_geotiff(path, data, transform)

    # When: counts_on_grid reads it back at the same (3, 5) shape,
    #       transform and CRS
    out = RasterReferenceSource(str(path)).counts_on_grid((3, 5), transform, CRS_UTM)

    # Then: the array round-trips with shape (3, 5), identical values and
    #       float32 dtype — grid_shape is (rows, cols), so the VRT's
    #       height/width must not be swapped (a swap is invisible on the
    #       square grids the rest of this file uses)
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out, data)
    assert out.dtype == np.float32


def test_raster_source_missing_file_raises(tmp_path):
    # Given: a raster path that does not exist
    missing = tmp_path / "nope.tif"

    # When: RasterReferenceSource is constructed
    # Then: FileNotFoundError is raised at construction time
    with pytest.raises(FileNotFoundError):
        RasterReferenceSource(str(missing))


def test_raster_source_band_selection_and_sanitization(tmp_path):
    # Given: a two-band raster; band 1 is all 7s, band 2 holds
    #        [[2, -3], [nan, 5]] — and the band requested as the string "2",
    #        which is what a quoted YAML value delivers
    transform = from_origin(0, 200, 100, 100)
    band1 = np.full((2, 2), 7.0, dtype="float32")
    band2 = np.array([[2.0, -3.0], [np.nan, 5.0]], dtype="float32")
    path = tmp_path / "twoband.tif"
    write_geotiff(path, np.stack([band1, band2]), transform)

    # When: counts_on_grid reads it
    out = RasterReferenceSource(str(path), band="2").counts_on_grid(
        (2, 2), transform, CRS_UTM
    )

    # Then: the band is coerced to int at construction, so band 2 is the one
    #       read rather than the string reaching rasterio; NaN becomes 0, the
    #       negative is clipped to 0, and valid counts stay
    np.testing.assert_array_equal(
        out, np.array([[2.0, 0.0], [0.0, 5.0]], dtype="float32")
    )


def test_raster_source_shifted_window_fills_zero_and_clips(tmp_path):
    # Given: a 4x4 counts raster with origin (0, 400) holding value 1 in
    #        every cell
    src_transform = from_origin(0, 400, 100, 100)
    path = tmp_path / "ones.tif"
    write_geotiff(path, np.ones((4, 4), dtype="float32"), src_transform)

    # When: counts_on_grid reads a 4x4 window shifted one cell right and one
    #       cell down (origin (100, 300))
    shifted = from_origin(100, 300, 100, 100)
    out = RasterReferenceSource(str(path)).counts_on_grid((4, 4), shifted, CRS_UTM)

    # Then: the overlapping 3x3 block stays 1, out-of-bounds cells fill 0
    expected = np.zeros((4, 4), dtype="float32")
    expected[:3, :3] = 1.0
    np.testing.assert_array_equal(out, expected)

    # When: the same raster is read on its own aligned grid with a clip box
    #       over the top-left cell only (x in [0, 100], y in [300, 400])
    clipped = RasterReferenceSource(str(path)).counts_on_grid(
        (4, 4), src_transform, CRS_UTM, clip_geom=box(0, 300, 100, 400)
    )

    # Then: every cell outside the box is zeroed — only the clipped cell
    #       keeps its 1
    assert clipped[0, 0] == 1.0
    assert clipped.sum() == pytest.approx(1.0)


def test_raster_source_resolves_grid_in_a_different_crs(tmp_path):
    # Given: a 2x2 counts raster [[1, 2], [3, 4]] in EPSG:32636 with 100 km
    #        cells and origin (400000, 200000) — columns split at easting
    #        500000 (the 33E central meridian), rows split at northing
    #        100000 — and a requested 2x2 EPSG:4326 grid of 1-degree cells
    #        with origin (32E, 2N), whose cell centres (32.5E/33.5E,
    #        1.5N/0.5N) project to eastings ~444.4/555.6 km and northings
    #        ~165.8/55.3 km: each at least 34 km inside a distinct source
    #        cell, so nearest resampling is unambiguous
    src_transform = from_origin(400000, 200000, 100000, 100000)
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    path = tmp_path / "utm_counts.tif"
    write_geotiff(path, data, src_transform, crs=CRS_UTM)

    # When: counts_on_grid resolves the WGS84 window
    out = RasterReferenceSource(str(path)).counts_on_grid(
        (2, 2), from_origin(32, 2, 1, 1), CRS_WGS84
    )

    # Then: each output cell carries the value of the source cell its
    #       centre projects into — [[1, 2], [3, 4]] again
    np.testing.assert_array_equal(out, data)


# ---------------------------------------------------------------------------
# Type inference and build_reference_source dispatch
# ---------------------------------------------------------------------------


def test_build_reference_source_infers_the_vector_type_from_the_suffix(tmp_path):
    # Given: a real export whose suffix is upper case (the raster suffix is
    #        inferred the same way in the band-option test below)
    path = _write_export(tmp_path / "points.GeoJSON", (0.5, 3.5))

    # When: build_reference_source gets it as a bare path carrying no type
    source = build_reference_source(str(path))

    # Then: the suffix is lower-cased before lookup, so it resolves to the
    #       vector type and the file reads
    assert isinstance(source, VectorReferenceSource)
    assert source.counts_on_grid(*DEGREE_GRID)[0, 0] == 1.0


def test_build_reference_source_rejects_an_uninferable_suffix():
    # Given: a path whose suffix matches no registered source type
    unknown_path = "table.csv"

    # When: build_reference_source is given it with no explicit type
    # Then: ValueError enumerates the registered types in sorted order. The
    #       conftest fixture restores the built-in registry around every
    #       test, so this exact message holds regardless of which other test
    #       files pytest collected first.
    with pytest.raises(ValueError, match="one of: raster, unosat, vector"):
        build_reference_source(unknown_path)


def test_build_reference_source_bare_path_and_null_options(tmp_path):
    # Given: a real GeoJSON with one point at (0.5, 3.5)
    path = _write_export(tmp_path / "ref.geojson", (0.5, 3.5))

    # When: build_reference_source gets a bare string path, and separately a
    #       mapping whose optional keys are all None (YAML nulls)
    for cfg in (
        str(path),
        {"path": str(path), "type": None, "layer": None, "where": None},
    ):
        source = build_reference_source(cfg)

        # Then: both produce a vector source resolving the point to cell (0, 0)
        assert isinstance(source, VectorReferenceSource)
        counts = source.counts_on_grid(*DEGREE_GRID)
        assert counts[0, 0] == 1.0
        assert counts.sum() == pytest.approx(1.0)


def test_build_reference_source_demands_a_path():
    # Given: a config naming a type but carrying no path
    no_path = {"type": "vector"}

    # When: build_reference_source validates it
    # Then: ValueError names the missing key
    with pytest.raises(ValueError, match="missing required key: path"):
        build_reference_source(no_path)


def test_build_reference_source_rejects_a_non_mapping_config():
    # Given: a config that is neither a path nor a mapping
    not_a_mapping = 42

    # When: build_reference_source validates it
    # Then: TypeError says what the config is allowed to be — the wrong kind of
    #       thing entirely, unlike the ValueError rejections of a mapping whose
    #       contents are at fault
    with pytest.raises(TypeError, match="path or a mapping"):
        build_reference_source(not_a_mapping)


def test_build_reference_source_rejects_an_unknown_type():
    # Given: a config naming a type that is not registered
    unknown_type = {"path": "x.geojson", "type": "satellite"}

    # When: build_reference_source validates it
    # Then: ValueError quotes the unknown type rather than falling back to
    #       suffix inference, which would have produced a vector source
    with pytest.raises(ValueError, match="Unknown reference type 'satellite'"):
        build_reference_source(unknown_type)


def test_build_reference_source_drops_foreign_options(tmp_path):
    # Given: a vector config that still carries a unosat-only 'date' key
    #        (left over from switching types)
    path = _write_export(tmp_path / "ref.geojson", (0.5, 3.5))
    cfg = {"path": str(path), "type": "vector", "date": "2024-01-01"}

    # When: build_reference_source dispatches to the vector factory
    source = build_reference_source(cfg)

    # Then: the source builds fine — 'date' is dropped, not passed through
    assert isinstance(source, VectorReferenceSource)


def test_build_reference_source_nearest_to_reaches_only_unosat(tmp_path):
    # Given: a directory with exports dated 2024-01-01 and 2024-02-01
    _write_export(tmp_path / "exportA_2024-01-01.geojson", (0.5, 3.5))
    feb_path = _write_export(tmp_path / "exportB_2024-02-01.geojson", (2.5, 1.5))

    # When: build_reference_source runs on a unosat config without a date,
    #       with nearest_to=2024-01-25 (24 days vs 7 days away)
    source = build_reference_source(
        {"path": str(tmp_path), "type": "unosat"},
        nearest_to=datetime(2024, 1, 25),
    )
    counts = source.counts_on_grid(*DEGREE_GRID)

    # Then: the unosat source auto-discovers the February export — its point
    #       is the only one on the grid, in cell (2, 2)
    assert counts[2, 2] == 1.0
    assert counts.sum() == pytest.approx(1.0)

    # When: build_reference_source runs on a plain vector config with the
    #       same nearest_to
    vector = build_reference_source(str(feb_path), nearest_to=datetime(2024, 1, 25))

    # Then: the vector source builds — nearest_to is not passed through to
    #       the vector factory
    assert isinstance(vector, VectorReferenceSource)


def test_build_reference_source_explicit_date_beats_nearest_to(tmp_path):
    # Given: exports dated 2024-01-01 (point in cell (0, 0)) and 2024-02-01
    #        (point in cell (2, 2)), a unosat config pinning date
    #        "2024-01-01", and nearest_to=2024-01-30 — which by timestamp
    #        distance (29 days vs 2 days) favours the February export
    _write_export(tmp_path / "exportA_2024-01-01.geojson", (0.5, 3.5))
    _write_export(tmp_path / "exportB_2024-02-01.geojson", (2.5, 1.5))

    # When: build_reference_source builds the source and counts are resolved
    source = build_reference_source(
        {"type": "unosat", "path": str(tmp_path), "date": "2024-01-01"},
        nearest_to=datetime(2024, 1, 30),
    )
    counts = source.counts_on_grid(*DEGREE_GRID)

    # Then: the explicitly pinned January export wins over nearest_to
    #       auto-discovery — its point is the one rasterized
    assert counts[0, 0] == 1.0
    assert counts[2, 2] == 0.0
    assert counts.sum() == pytest.approx(1.0)


def test_build_reference_source_raster_band_option(tmp_path):
    # Given: a two-band raster (band 1 all 7s, band 2 all 3s) and a config
    #        selecting band 2 that deliberately carries no 'type' key, so
    #        the .tif suffix has to drive the inference (the vector half of
    #        that inference is pinned above)
    transform = from_origin(0, 200, 100, 100)
    path = tmp_path / "twoband.tif"
    write_geotiff(
        path,
        np.stack(
            [np.full((2, 2), 7.0), np.full((2, 2), 3.0)],
        ),
        transform,
    )

    # When: the built source resolves counts on the same grid
    source = build_reference_source({"path": str(path), "band": 2})
    out = source.counts_on_grid((2, 2), transform, CRS_UTM)

    # Then: the band-2 values come back
    np.testing.assert_array_equal(out, np.full((2, 2), 3.0, dtype="float32"))


# ---------------------------------------------------------------------------
# Consumption via validation_core.prepare_grouped_cell_inputs
# ---------------------------------------------------------------------------


def test_reference_source_is_resolved_on_the_prediction_hull_window(tmp_path):
    # Given: a 10x10 master grid of 100 m cells with origin (0, 1000);
    #        predictions at the four corners (60,940) (340,940) (60,660)
    #        (340,660) whose convex hull spans x in [60,340], y in [660,940]
    #        (crop window: rows 0-3, cols 0-3); a reference point at
    #        (150, 850) inside the hull and one at (30, 970) inside the
    #        window but outside the hull
    grid_path = tmp_path / "master_grid.tif"
    write_geotiff(grid_path, np.zeros((10, 10)), from_origin(0, 1000, 100, 100))
    pred_gdf = _points_gdf(
        [(60, 940), (340, 940), (60, 660), (340, 660)],
        peak_value=[1.0, 1.0, 1.0, 1.0],
        adjusted_peak=[1.0, 1.0, 1.0, 1.0],
    )
    reference = PointsSource(_points_gdf([(150, 850), (30, 970)]))

    # When: prepare_grouped_cell_inputs resolves the reference against the
    #       window it crops for those predictions
    with rasterio.open(grid_path) as src_grid:
        grouped = prepare_grouped_cell_inputs(pred_gdf, reference, src_grid)

    # Then: the source is handed the cropped window — a (4, 4) grid at
    #       origin (0, 1000) — with the hull as clip_geom, so the point
    #       inside the hull counts in cell (1, 1) while the one inside the
    #       window but outside the hull is clipped away. The window's own
    #       mask/rowcol arithmetic belongs to validation_core and is left to
    #       be pinned where that function lives.
    assert grouped["grid_shape"] == (4, 4)
    assert grouped["out_transform"].c == pytest.approx(0.0)
    assert grouped["out_transform"].f == pytest.approx(1000.0)
    val = grouped["val_raster"]
    assert val[1, 1] == 1.0
    assert val[0, 0] == 0.0
    assert val.sum() == pytest.approx(1.0)
