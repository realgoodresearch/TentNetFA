"""Unit tests for displacement_tracker/util/reference_data.py.

Covers date parsing/selection, point rasterization onto the master grid,
vector/unosat/raster reference sources, and the build_reference_source
config dispatch — plus one integration test through
validation_core.prepare_grouped_cell_inputs.
"""

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from _helpers import CRS_UTM, CRS_WGS84, write_geotiff
from displacement_tracker.util.reference_data import (
    PointsSource,
    RasterReferenceSource,
    UnosatReferenceSource,
    VectorReferenceSource,
    _infer_type,
    _list_exports,
    _select_export,
    build_reference_source,
    extract_date_from_filename,
    infer_target_date,
    rasterize_point_counts,
)
from displacement_tracker.util.validation_core import prepare_grouped_cell_inputs


def _points_gdf(coords, crs=CRS_UTM, **columns):
    return gpd.GeoDataFrame(columns, geometry=[Point(x, y) for x, y in coords], crs=crs)


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

    # When: infer_target_date runs (index len//2 == 2 of the sorted dates)
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
    counts = VectorReferenceSource(str(path)).counts_on_grid(
        (4, 4), from_origin(0, 4, 1, 1), CRS_WGS84
    )

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
    # Given: a two-layer GPKG; layer "tents" (written first, so it is also
    #        the default layer) has a 'tent' point at (0.5, 3.5) and a
    #        'rubble' point at (1.5, 2.5); layer "other" has one point at
    #        (3.5, 0.5)
    path = tmp_path / "multi.gpkg"
    _points_gdf(
        [(0.5, 3.5), (1.5, 2.5)], crs=CRS_WGS84, kind=["tent", "rubble"]
    ).to_file(path, layer="tents")
    _points_gdf([(3.5, 0.5)], crs=CRS_WGS84, kind=["x"]).to_file(path, layer="other")
    grid = ((4, 4), from_origin(0, 4, 1, 1), CRS_WGS84)

    # When: VectorReferenceSource reads layer "tents" with
    #       where="kind = 'tent'"
    source = VectorReferenceSource(str(path), layer="tents", where="kind = 'tent'")
    counts = source.counts_on_grid(*grid)

    # Then: the filtered read rasterizes only the tent point — cell (0, 0)
    #       counts 1, total exactly 1
    assert counts[0, 0] == 1.0
    assert counts.sum() == pytest.approx(1.0)

    # When: the non-default layer "other" is read with no filter
    other = VectorReferenceSource(str(path), layer="other").counts_on_grid(*grid)

    # Then: only cell (3, 3) counts, proving the layer argument really
    #       selects a layer
    assert other[3, 3] == 1.0
    assert other.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# UNOSAT export selection
# ---------------------------------------------------------------------------


def test_select_export_plain_file_passes_through(tmp_path):
    # Given: a path that is a file, not a directory
    path = tmp_path / "export_20240101.geojson"
    path.touch()

    # When: _select_export resolves it
    selected = _select_export(str(path), date=None)

    # Then: the path is returned unchanged, date logic untouched
    assert selected == str(path)


def test_select_export_explicit_date_is_dash_insensitive(tmp_path):
    # Given: a directory with a dashed-date export in a child directory and
    #        a compact-date export at the top level
    sub = tmp_path / "child"
    sub.mkdir()
    dashed = sub / "exp_2024-01-15.geojson"
    dashed.touch()
    compact = tmp_path / "exp_20240220.geojson"
    compact.touch()

    # When: each is requested with an explicit date in the opposite
    #       formatting
    by_compact_request = _select_export(str(tmp_path), date="20240115")
    by_dashed_request = _select_export(str(tmp_path), date="2024-02-20")

    # Then: the matching export is still found (comparison drops dashes)
    assert by_compact_request == str(dashed)
    assert by_dashed_request == str(compact)


def test_select_export_unmatched_date_lists_available(tmp_path):
    # Given: a directory whose only export is dated 2024-01-15
    (tmp_path / "exp_2024-01-15.geojson").touch()

    # When: an explicit date with no match is requested
    # Then: ValueError reports 0 matches and names the available export
    with pytest.raises(ValueError, match=r"found 0.*exp_2024-01-15\.geojson"):
        _select_export(str(tmp_path), date="2024-03-03")


def test_select_export_ambiguous_date_raises(tmp_path):
    # Given: two exports carrying the same date stamp
    (tmp_path / "a_20240115.geojson").touch()
    (tmp_path / "b_2024-01-15.gpkg").touch()

    # When: that date is requested explicitly
    # Then: ValueError reports 2 matches instead of picking one arbitrarily
    with pytest.raises(ValueError, match="found 2"):
        _select_export(str(tmp_path), date="2024-01-15")


def test_select_export_nearest_to_picks_closest_then_name(tmp_path):
    # Given: exports dated 2024-01-01 and 2024-02-01
    (tmp_path / "exp_20240101.geojson").touch()
    feb = tmp_path / "exp_20240201.geojson"
    feb.touch()

    # When: nearest_to is 2024-01-20 (19 days vs 12 days away)
    nearest = _select_export(str(tmp_path), date=None, nearest_to=datetime(2024, 1, 20))

    # Then: the 2024-02-01 export wins
    assert nearest == str(feb)

    # Given: two exports exactly 5 days either side of nearest_to
    tie_dir = tmp_path / "tie"
    tie_dir.mkdir()
    (tie_dir / "b_20240110.geojson").touch()
    a_file = tie_dir / "a_20240120.geojson"
    a_file.touch()

    # When: the timestamp distance ties
    tied = _select_export(str(tie_dir), date=None, nearest_to=datetime(2024, 1, 15))

    # Then: the lexicographically smaller file name breaks the tie
    assert tied == str(a_file)


def test_select_export_directory_error_cases(tmp_path):
    # Given: a directory of exports but neither an explicit date nor a
    #        nearest_to context
    (tmp_path / "exp_20240101.geojson").touch()

    # When: _select_export runs
    # Then: it refuses with an actionable ValueError naming reference.date
    with pytest.raises(ValueError, match="reference.date"):
        _select_export(str(tmp_path), date=None, nearest_to=None)

    # Given: a directory whose exports carry no parseable date stamp
    undated = tmp_path / "undated"
    undated.mkdir()
    (undated / "shelters.geojson").touch()

    # When: auto-discovery by nearest_to is attempted
    # Then: ValueError says no date-stamped exports were found
    with pytest.raises(ValueError, match="No date-stamped exports"):
        _select_export(str(undated), date=None, nearest_to=datetime(2024, 1, 1))


def test_list_exports_includes_gdb_dirs_and_nested_files_only(tmp_path):
    # Given: a tree with a nested .geojson file, a .gdb directory (whose
    #        internals have no vector suffix), a .txt file and a plain
    #        directory
    sub = tmp_path / "sub"
    sub.mkdir()
    nested = sub / "exp_20240101.geojson"
    nested.touch()
    gdb = tmp_path / "fake.gdb"
    gdb.mkdir()
    (gdb / "gdbtable").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "plaindir").mkdir()

    # When: _list_exports scans it
    found = _list_exports(Path(tmp_path))

    # Then: exactly the .gdb directory and the nested vector file are
    #       returned, sorted by path
    assert found == [gdb, nested]


def test_list_exports_matches_suffixes_case_insensitively(tmp_path):
    # Given: a tree with an uppercase-suffix export file and an
    #        uppercase-suffix .GDB directory
    upper_gdb = tmp_path / "archive.GDB"
    upper_gdb.mkdir()
    upper_file = tmp_path / "exp_20240301.GEOJSON"
    upper_file.touch()

    # When: _list_exports scans it
    found = _list_exports(Path(tmp_path))

    # Then: both are returned, sorted by path — suffix matching is
    #       case-insensitive
    assert found == [upper_gdb, upper_file]


def test_unosat_source_reads_the_pinned_export(tmp_path):
    # Given: a directory with a 2024-01-01 export (point in cell (0, 0))
    #        and a 2024-02-01 export (point in cell (2, 2))
    _points_gdf([(0.5, 3.5)], crs=CRS_WGS84).to_file(
        tmp_path / "exportA_2024-01-01.geojson", driver="GeoJSON"
    )
    _points_gdf([(2.5, 1.5)], crs=CRS_WGS84).to_file(
        tmp_path / "exportB_2024-02-01.geojson", driver="GeoJSON"
    )

    # When: UnosatReferenceSource pins date="2024-01-01" and resolves counts
    source = UnosatReferenceSource(str(tmp_path), date="2024-01-01")
    counts = source.counts_on_grid((4, 4), from_origin(0, 4, 1, 1), CRS_WGS84)

    # Then: only the January export's point shows up on the grid
    assert counts[0, 0] == 1.0
    assert counts[2, 2] == 0.0
    assert counts.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RasterReferenceSource
# ---------------------------------------------------------------------------


def test_raster_source_aligned_roundtrip(tmp_path):
    # Given: a counts GeoTIFF written on the exact requested grid
    transform = from_origin(0, 400, 100, 100)
    data = np.arange(16, dtype="float32").reshape(4, 4)
    path = tmp_path / "counts.tif"
    write_geotiff(path, data, transform)

    # When: counts_on_grid reads it back at the same shape/transform/CRS
    out = RasterReferenceSource(str(path)).counts_on_grid((4, 4), transform, CRS_UTM)

    # Then: the array is returned bit-identical as float32
    np.testing.assert_array_equal(out, data)
    assert out.dtype == np.float32


def test_raster_source_non_square_grid_preserves_shape_and_values(tmp_path):
    # Given: a 3x5 counts GeoTIFF holding 0..14 (3 rows, 5 columns)
    transform = from_origin(0, 300, 100, 100)
    data = np.arange(15, dtype="float32").reshape(3, 5)
    path = tmp_path / "rect.tif"
    write_geotiff(path, data, transform)

    # When: counts_on_grid reads it back at the same (3, 5) shape,
    #       transform and CRS
    out = RasterReferenceSource(str(path)).counts_on_grid((3, 5), transform, CRS_UTM)

    # Then: the array round-trips with shape (3, 5) and identical values —
    #       grid_shape is (rows, cols), so the VRT's height/width must not
    #       be swapped (a swap is invisible on square grids)
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out, data)


def test_raster_source_missing_file_raises(tmp_path):
    # Given: a raster path that does not exist
    missing = tmp_path / "nope.tif"

    # When: RasterReferenceSource is constructed
    # Then: FileNotFoundError is raised at construction time
    with pytest.raises(FileNotFoundError):
        RasterReferenceSource(str(missing))


def test_raster_source_band_selection_and_sanitization(tmp_path):
    # Given: a two-band raster; band 1 is all 7s, band 2 holds
    #        [[2, -3], [nan, 5]]
    transform = from_origin(0, 200, 100, 100)
    band1 = np.full((2, 2), 7.0, dtype="float32")
    band2 = np.array([[2.0, -3.0], [np.nan, 5.0]], dtype="float32")
    path = tmp_path / "twoband.tif"
    write_geotiff(path, np.stack([band1, band2]), transform)

    # When: counts_on_grid reads band 2
    out = RasterReferenceSource(str(path), band=2).counts_on_grid(
        (2, 2), transform, CRS_UTM
    )

    # Then: NaN becomes 0, the negative is clipped to 0, valid counts stay
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


def test_infer_type_from_suffix():
    # Given: paths with raster, vector, and unknown suffixes
    raster_path = "counts.tif"
    vector_path = "points.GeoJSON"
    unknown_path = "table.csv"

    # When: _infer_type inspects the two recognised suffixes
    raster_type = _infer_type(raster_path)
    vector_type = _infer_type(vector_path)

    # Then: .tif -> raster, .GeoJSON -> vector (case-insensitive)
    assert raster_type == "raster"
    assert vector_type == "vector"

    # When: _infer_type inspects the unknown suffix
    # Then: .csv -> ValueError enumerating the registered types in sorted
    #       order. The conftest fixture restores the built-in registry
    #       around every test, so this message is the same regardless of
    #       which other test files pytest collected first.
    with pytest.raises(ValueError, match="one of: raster, unosat, vector"):
        _infer_type(unknown_path)


def test_build_reference_source_bare_path_and_null_options(tmp_path):
    # Given: a real GeoJSON with one point at (0.5, 3.5)
    path = tmp_path / "ref.geojson"
    _points_gdf([(0.5, 3.5)], crs=CRS_WGS84).to_file(path, driver="GeoJSON")
    grid = ((4, 4), from_origin(0, 4, 1, 1), CRS_WGS84)

    # When: build_reference_source gets a bare string path, and separately a
    #       mapping whose optional keys are all None (YAML nulls)
    for cfg in (
        str(path),
        {"path": str(path), "type": None, "layer": None, "where": None},
    ):
        source = build_reference_source(cfg)

        # Then: both produce a vector source resolving the point to cell (0, 0)
        assert isinstance(source, VectorReferenceSource)
        counts = source.counts_on_grid(*grid)
        assert counts[0, 0] == 1.0
        assert counts.sum() == pytest.approx(1.0)


def test_build_reference_source_invalid_configs():
    # Given: a config with no path, a non-mapping config, and an unknown type
    no_path = {"type": "vector"}
    not_a_mapping = 42
    unknown_type = {"path": "x.geojson", "type": "satellite"}

    # When: build_reference_source validates each
    # Then: each raises a ValueError with the specific complaint
    with pytest.raises(ValueError, match="missing required key: path"):
        build_reference_source(no_path)
    with pytest.raises(ValueError, match="path or a mapping"):
        build_reference_source(not_a_mapping)
    with pytest.raises(ValueError, match="Unknown reference type 'satellite'"):
        build_reference_source(unknown_type)


def test_build_reference_source_drops_foreign_options(tmp_path):
    # Given: a vector config that still carries a unosat-only 'date' key
    #        (left over from switching types)
    path = tmp_path / "ref.geojson"
    _points_gdf([(0.5, 3.5)], crs=CRS_WGS84).to_file(path, driver="GeoJSON")
    cfg = {"path": str(path), "type": "vector", "date": "2024-01-01"}

    # When: build_reference_source dispatches to the vector factory
    source = build_reference_source(cfg)

    # Then: the source builds fine — 'date' is dropped, not passed through
    assert isinstance(source, VectorReferenceSource)


def test_build_reference_source_nearest_to_reaches_only_unosat(tmp_path):
    # Given: a directory with exports dated 2024-01-01 and 2024-02-01
    _points_gdf([(0.5, 3.5)], crs=CRS_WGS84).to_file(
        tmp_path / "exportA_2024-01-01.geojson", driver="GeoJSON"
    )
    feb_path = tmp_path / "exportB_2024-02-01.geojson"
    _points_gdf([(2.5, 1.5)], crs=CRS_WGS84).to_file(feb_path, driver="GeoJSON")

    # When: build_reference_source runs on a unosat config without a date,
    #       with nearest_to=2024-01-25 (24 days vs 7 days away)
    source = build_reference_source(
        {"path": str(tmp_path), "type": "unosat"},
        nearest_to=datetime(2024, 1, 25),
    )
    counts = source.counts_on_grid((4, 4), from_origin(0, 4, 1, 1), CRS_WGS84)

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
    _points_gdf([(0.5, 3.5)], crs=CRS_WGS84).to_file(
        tmp_path / "exportA_2024-01-01.geojson", driver="GeoJSON"
    )
    _points_gdf([(2.5, 1.5)], crs=CRS_WGS84).to_file(
        tmp_path / "exportB_2024-02-01.geojson", driver="GeoJSON"
    )

    # When: build_reference_source builds the source and counts are resolved
    source = build_reference_source(
        {"type": "unosat", "path": str(tmp_path), "date": "2024-01-01"},
        nearest_to=datetime(2024, 1, 30),
    )
    counts = source.counts_on_grid((4, 4), from_origin(0, 4, 1, 1), CRS_WGS84)

    # Then: the explicitly pinned January export wins over nearest_to
    #       auto-discovery — its point is the one rasterized
    assert counts[0, 0] == 1.0
    assert counts[2, 2] == 0.0
    assert counts.sum() == pytest.approx(1.0)


def test_build_reference_source_raster_band_option(tmp_path):
    # Given: a two-band raster (band 1 all 7s, band 2 all 3s) and a raster
    #        config selecting band 2
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


def test_prepare_grouped_cell_inputs_resolves_reference_on_hull_window(tmp_path):
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

    # When: prepare_grouped_cell_inputs builds the grid products
    with rasterio.open(grid_path) as src_grid:
        grouped = prepare_grouped_cell_inputs(pred_gdf, reference, src_grid)

    # Then: grid_shape is (4, 4) with origin (0, 1000); the inside reference
    #       point counts in cell (1, 1); the outside one is clipped to 0;
    #       cell (0, 0) (centre (50, 950)) is outside the analysis mask
    #       while cell (1, 1) is inside; prediction (60, 940) maps to
    #       row 0, col 0 and (340, 660) to row 3, col 3
    assert grouped["grid_shape"] == (4, 4)
    assert grouped["out_transform"].c == pytest.approx(0.0)
    assert grouped["out_transform"].f == pytest.approx(1000.0)
    val = grouped["val_raster"]
    assert val[1, 1] == 1.0
    assert val[0, 0] == 0.0
    assert val.sum() == pytest.approx(1.0)
    mask_array = grouped["mask_array"]
    assert not mask_array[0, 0]
    assert mask_array[1, 1]
    prepped = grouped["pred_prepped"].sort_values(["row", "col"])
    assert list(zip(prepped["row"], prepped["col"])) == [
        (0, 0),
        (0, 3),
        (3, 0),
        (3, 3),
    ]
