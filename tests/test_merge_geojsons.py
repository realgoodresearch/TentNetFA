"""Unit tests for displacement_tracker/h_merge_geojsons.py (merge pipeline)."""

import json

import click
import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from _helpers import write_geojson
from displacement_tracker.h_merge_geojsons import (
    MERGE_CONFIG_KEYS,
    filter_points_by_zone,
    iter_date_folders,
    list_geojson_files,
    load_points_from_geojson,
    load_thresholds,
    load_zone_geometry,
    merge_geojsons,
    merge_kwargs_from_config,
    parse_compact_date,
    process_geojson_folder,
    resolve_threshold,
    save_merged_gpkg,
    sort_preds_by_date,
)

# One degree of arc on the sphere used by the code is ~111194.93 m, so
# 1e-5 degrees is ~1.112 m -- comfortably below a 3 m merge radius.
NEAR_DEG = 1e-5


def default_merge_kwargs(**overrides):
    """Baseline kwargs for process_geojson_folder with no filtering active."""
    kwargs = dict(
        min_distance_m=3.0,
        agreement=1,
        min_adj_peak=0.0,
        adjustment_factor=1.0,
        thresholds_data={},
        exclusion_geom=None,
        inclusion_geom=None,
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# parse_compact_date / list_geojson_files / iter_date_folders
# ---------------------------------------------------------------------------


def test_parse_compact_date_real_calendar_validation():
    # Given: compact 8-digit strings, some of which are impossible dates
    candidates = ("20240115", "20240229", "20230229", "20240230", "20241301")

    # When: parse_compact_date checks each against the real calendar
    validated = [parse_compact_date(s) for s in candidates]

    # Then: only genuine YYYYMMDD dates validate (2024 is a leap year,
    #       2023 is not; month 13 and Feb 30 never exist)
    assert validated == [True, True, False, False, False]


def test_list_geojson_files_case_insensitive_and_sorted(tmp_path):
    # Given: a directory with mixed-case .json/.geojson files, a .txt file
    #        and a subdirectory whose name ends in .json
    (tmp_path / "b.json").write_text("{}")
    (tmp_path / "A.GEOJSON").write_text("{}")
    (tmp_path / "z.JSON").write_text("{}")
    (tmp_path / "notes.txt").write_text("nope")
    (tmp_path / "sub.json").mkdir()

    # When: list_geojson_files scans it
    names = [p.name for p in list_geojson_files(tmp_path)]

    # Then: only regular files with .json/.geojson suffixes (any case) are
    #       returned, in sorted path order (ASCII: uppercase before lowercase)
    assert names == ["A.GEOJSON", "b.json", "z.JSON"]


def test_iter_date_folders_only_valid_dates(tmp_path):
    # Given: subfolders 20230101 and 20240115 (valid dates), 20240230
    #        (8 digits but not a real date), "notadate", "2024", and a plain
    #        FILE named 20240116
    for name in ("20240115", "20230101", "20240230", "notadate", "2024"):
        (tmp_path / name).mkdir()
    (tmp_path / "20240116").write_text("i am a file")

    # When: iter_date_folders scans the base directory
    folders = [p.name for p in iter_date_folders(tmp_path)]

    # Then: only the two valid-date directories come back, sorted ascending
    assert folders == ["20230101", "20240115"]


# ---------------------------------------------------------------------------
# load_thresholds / resolve_threshold
# ---------------------------------------------------------------------------


def test_load_thresholds_none_and_empty_yaml(tmp_path):
    # Given: no thresholds path, and a thresholds YAML that is empty
    empty = tmp_path / "empty.yaml"
    empty.write_text("")

    # When: load_thresholds is called for each
    from_none = load_thresholds(None)
    from_empty = load_thresholds(str(empty))

    # Then: both resolve to an empty dict (no thresholds configured)
    assert from_none == {}
    assert from_empty == {}


def test_load_thresholds_missing_file_raises(tmp_path):
    # Given: a thresholds path that does not exist on disk
    missing = tmp_path / "nope.yaml"

    # When: load_thresholds tries to read it
    # Then: a ClickException naming the path is raised
    with pytest.raises(click.ClickException, match="nope.yaml"):
        load_thresholds(str(missing))


def test_resolve_threshold_per_file_beats_default_beats_global():
    # Given: thresholds data with default=0.5 and a per_file entry of "0.2"
    #        (a string) for a.json, and a global threshold of 0.9
    data = {"default": 0.5, "per_file": {"a.json": "0.2"}}

    # When: resolve_threshold is asked for a.json and for b.json
    for_a = resolve_threshold("a.json", data, 0.9)
    for_b = resolve_threshold("b.json", data, 0.9)

    # Then: a.json gets the per-file value coerced to float 0.2, while
    #       b.json falls back to the default 0.5 -- the global never applies
    assert for_a == pytest.approx(0.2)
    assert for_b == pytest.approx(0.5)


def test_resolve_threshold_global_fallback_and_null_per_file():
    # Given: thresholds data whose per_file section is explicitly null, and
    #        thresholds data with no keys at all -- neither has a default key
    null_per_file = {"per_file": None}
    no_keys = {}

    # When: resolve_threshold is asked for a filename with global 0.7
    from_null_per_file = resolve_threshold("a.json", null_per_file, 0.7)
    from_no_keys = resolve_threshold("a.json", no_keys, 0.7)

    # Then: the global threshold 0.7 is returned in both cases (null per_file
    #       is not an error and does not shadow the fallback chain)
    assert from_null_per_file == pytest.approx(0.7)
    assert from_no_keys == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# load_points_from_geojson
# ---------------------------------------------------------------------------


def test_load_points_skips_non_points_and_defaults_adjusted_peak(tmp_path):
    # Given: a GeoJSON with a full Point, a Point missing adjusted_peak, a
    #        Point with adjusted_peak null, a Point with null properties, a
    #        LineString, a null geometry, and Points with empty/1-element
    #        coordinates
    gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
                "properties": {"peak_value": 0.7, "adjusted_peak": 0.6},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                "properties": {"peak_value": 0.5},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
                "properties": {"peak_value": 0.4, "adjusted_peak": None},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [5.0, 6.0]},
                "properties": None,
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [1, 1]],
                },
                "properties": {"peak_value": 0.9, "adjusted_peak": 0.9},
            },
            {"type": "Feature", "geometry": None, "properties": {}},
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": []},
                "properties": {},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [7.0]},
                "properties": {},
            },
        ],
    }
    path = tmp_path / "pts.geojson"
    path.write_text(json.dumps(gj))

    # When: load_points_from_geojson parses the file
    points = [tuple(p) for p in load_points_from_geojson(path)]

    # Then: only the four valid Points survive, as (lat, lon, peak, adj)
    #       with lon/lat swapped from GeoJSON order and missing/None
    #       adjusted_peak (and missing peak_value) defaulting to 0.0
    assert points == [
        (20.0, 10.0, 0.7, 0.6),
        (2.0, 1.0, 0.5, 0.0),
        (4.0, 3.0, 0.4, 0.0),
        (6.0, 5.0, 0.0, 0.0),
    ]


# ---------------------------------------------------------------------------
# filter_points_by_zone
# ---------------------------------------------------------------------------


def test_filter_points_by_zone_inside_outside_semantics():
    # Given: a zone box spanning lon 10..20 / lat 0..5, a point at
    #        (lat=2, lon=15) inside it, and its lat/lon-swapped twin at
    #        (lat=15, lon=2) which is only "inside" if axes are confused
    zone = box(10.0, 0.0, 20.0, 5.0)
    inside = (2.0, 15.0, 0.5, 0.5)
    swapped = (15.0, 2.0, 0.6, 0.6)

    # When: filtering with keep_inside=True and keep_inside=False
    kept_inside = filter_points_by_zone([inside, swapped], zone, True)
    kept_outside = filter_points_by_zone([inside, swapped], zone, False)

    # Then: keep_inside=True keeps only the true inside point and
    #       keep_inside=False keeps only the outside point, proving the
    #       (lat, lon) tuple is mapped to shapely Point(lon, lat)
    assert kept_inside == [inside]
    assert kept_outside == [swapped]


def test_filter_points_by_zone_none_zone_is_identity():
    # Given: two points and no zone geometry
    pts = [(2.0, 15.0, 0.5, 0.5), (15.0, 2.0, 0.6, 0.6)]

    # When: filter_points_by_zone runs with zone_geom=None
    kept_inside = filter_points_by_zone(pts, None, True)
    kept_outside = filter_points_by_zone(pts, None, False)

    # Then: the input list is returned unchanged regardless of keep_inside
    assert kept_inside == pts
    assert kept_outside == pts


def test_load_zone_geometry_missing_file_raises():
    # Given: a zones path that does not exist, labelled "exclusion"
    missing = "/definitely/not/here.gpkg"

    # When: load_zone_geometry tries to load it
    # Then: a ClickException naming the missing exclusion zones file is raised
    with pytest.raises(click.ClickException, match="exclusion zones file not found"):
        load_zone_geometry(missing, "exclusion")


# ---------------------------------------------------------------------------
# merge_kwargs_from_config
# ---------------------------------------------------------------------------


def test_merge_kwargs_from_config_only_set_keys_and_false_preserved():
    # Given: a merge config where some known keys are set (a falsy-yet-set
    #        0.0, a None, and keys outside MERGE_CONFIG_KEYS -- including
    #        process_by_date, which the cli forwards explicitly instead)
    cfg = {
        "min_distance_m": 5.0,
        "agreement": None,
        "min_adj_peak": 0.0,
        "process_by_date": True,
        "input_folder": "somewhere",
        "output": "out.gpkg",
    }
    assert "process_by_date" not in MERGE_CONFIG_KEYS

    # When: merge_kwargs_from_config extracts the kwargs
    kwargs = merge_kwargs_from_config(cfg)

    # Then: None-valued and unknown keys are dropped, but the falsy-yet-set
    #       0.0 value survives (is-not-None semantics, not truthiness)
    assert kwargs == {
        "min_distance_m": 5.0,
        "min_adj_peak": 0.0,
    }


# ---------------------------------------------------------------------------
# save_merged_gpkg
# ---------------------------------------------------------------------------


def test_save_merged_gpkg_empty_points_removes_stale_output(tmp_path):
    # Given: an existing (stale) file at the output path and an empty
    #        points list
    out = tmp_path / "merged.gpkg"
    out.write_text("stale bytes")

    # When: save_merged_gpkg runs
    save_merged_gpkg([], out)

    # Then: the stale file is deleted and no new file is written
    assert not out.exists()


# ---------------------------------------------------------------------------
# process_geojson_folder
# ---------------------------------------------------------------------------


def test_process_folder_merges_cluster_and_drops_low_agreement(tmp_path):
    # Given: three points within ~1.6 m of each other split across two
    #        files (peaks 0.5/0.9/0.3, the 0.9 peak carrying adj 0.7) plus
    #        one isolated point ~1.5 km away
    write_geojson(
        tmp_path / "a.geojson",
        [(0.0, 0.0, 0.5, 0.4), (NEAR_DEG, 0.0, 0.9, 0.7)],
    )
    write_geojson(
        tmp_path / "b.geojson",
        [(0.0, NEAR_DEG, 0.3, 0.99), (0.01, 0.01, 0.8, 0.8)],
    )
    out = tmp_path / "out.gpkg"

    # When: process_geojson_folder runs with min_distance_m=3, agreement=2
    #       and no thresholding -- and reports success
    assert process_geojson_folder(tmp_path, out, **default_merge_kwargs(agreement=2))

    # Then: the trio collapses to one centroid at the mean lat/lon
    #       (1e-5/3 each), carrying max peak 0.9 and THAT point's adjusted
    #       peak 0.7 (not the cluster max 0.99); the singleton is dropped
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row["peak_value"] == pytest.approx(0.9)
    assert row["adjusted_peak"] == pytest.approx(0.7)
    assert row.geometry.x == pytest.approx(NEAR_DEG / 3, rel=1e-9)
    assert row.geometry.y == pytest.approx(NEAR_DEG / 3, rel=1e-9)
    assert row["name"] == "tents"


def test_process_folder_rescales_before_thresholding(tmp_path):
    # Given: two far-apart points -- A(peak=0.5, adj=0.3) and
    #        B(peak=0.2, adj=0.9) -- with adjustment_factor=0 so the
    #        rescaled value collapses to the raw peak
    #        (rescaled = peak + 0*(adj-peak))
    write_geojson(
        tmp_path / "a.geojson",
        [(0.0, 0.0, 0.5, 0.3), (0.01, 0.01, 0.2, 0.9)],
    )
    out = tmp_path / "out.gpkg"

    # When: process_geojson_folder filters at min_adj_peak=0.4
    process_geojson_folder(
        tmp_path,
        out,
        **default_merge_kwargs(min_adj_peak=0.4, adjustment_factor=0.0),
    )

    # Then: A survives (0.5 >= 0.4) with its adjusted_peak REPLACED by the
    #       rescaled 0.5, while B is dropped (0.2 < 0.4) despite its raw
    #       adjusted_peak of 0.9
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf.iloc[0]["peak_value"] == pytest.approx(0.5)
    assert gdf.iloc[0]["adjusted_peak"] == pytest.approx(0.5)


def test_process_folder_applies_per_file_thresholds(tmp_path):
    # Given: a.json and b.json each holding one far-apart point with
    #        adjusted_peak 0.5, thresholds data default=0.6 and a per-file
    #        override of 0.4 for a.json only
    write_geojson(tmp_path / "a.json", [(0.0, 0.0, 0.5, 0.5)])
    write_geojson(tmp_path / "b.json", [(0.01, 0.01, 0.5, 0.5)])
    out = tmp_path / "out.gpkg"

    # When: process_geojson_folder runs with adjustment_factor=1
    process_geojson_folder(
        tmp_path,
        out,
        **default_merge_kwargs(
            thresholds_data={"default": 0.6, "per_file": {"a.json": 0.4}}
        ),
    )

    # Then: a.json's point passes its per-file threshold (0.5 >= 0.4) while
    #       b.json's point fails the default (0.5 < 0.6), leaving one point
    #       at a.json's location
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.x == pytest.approx(0.0)
    assert gdf.iloc[0].geometry.y == pytest.approx(0.0)


def test_process_folder_skips_corrupt_file_keeps_good(tmp_path):
    # Given: one unparseable JSON file next to one valid single-point file
    (tmp_path / "bad.json").write_text("{this is not json")
    write_geojson(tmp_path / "good.json", [(1.0, 2.0, 0.5, 0.5)])
    out = tmp_path / "out.gpkg"

    # When: process_geojson_folder runs -- and reports success
    assert process_geojson_folder(tmp_path, out, **default_merge_kwargs())

    # Then: the output holds exactly the good point
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.x == pytest.approx(1.0)
    assert gdf.iloc[0].geometry.y == pytest.approx(2.0)


def test_process_folder_all_corrupt_raises(tmp_path):
    # Given: a folder where every GeoJSON file fails to parse
    (tmp_path / "bad1.json").write_text("{oops")
    (tmp_path / "bad2.geojson").write_text("also not json {")

    # When: process_geojson_folder runs
    # Then: a ValueError is raised instead of silently writing nothing
    with pytest.raises(ValueError, match="failed to load"):
        process_geojson_folder(
            tmp_path, tmp_path / "out.gpkg", **default_merge_kwargs()
        )


def test_process_folder_all_points_filtered_is_skip_not_failure(tmp_path):
    # Given: a folder with one valid GeoJSON whose two points carry
    #        adjusted peaks 0.5 and 0.8, and min_adj_peak=0.99 rejecting
    #        both (adjustment_factor=1 so rescaled == adjusted)
    write_geojson(
        tmp_path / "a.geojson",
        [(0.0, 0.0, 0.5, 0.5), (0.01, 0.01, 0.8, 0.8)],
    )
    out = tmp_path / "out.gpkg"

    # When: process_geojson_folder runs
    result = process_geojson_folder(
        tmp_path, out, **default_merge_kwargs(min_adj_peak=0.99)
    )

    # Then: the all-filtered folder is a SKIP, not a failure -- it returns
    #       True (readable files were processed) and writes no output GPKG
    assert result is True
    assert not out.exists()


def test_process_folder_empty_returns_false(tmp_path):
    # Given: a folder containing no GeoJSON/JSON files at all
    (tmp_path / "readme.txt").write_text("nothing here")
    out = tmp_path / "out.gpkg"

    # When: process_geojson_folder runs
    result = process_geojson_folder(tmp_path, out, **default_merge_kwargs())

    # Then: it returns False and writes no output file
    assert result is False
    assert not out.exists()


# ---------------------------------------------------------------------------
# sort_preds_by_date
# ---------------------------------------------------------------------------


def test_sort_preds_by_date_moves_only_valid_dated_files(tmp_path):
    # Given: root-level files with a valid _YYYYMMDD_ date, a valid date
    #        directly before the extension, an impossible date (Feb 30), a
    #        date not delimited by _ or . , and no date at all
    (tmp_path / "pred_20240115_a.json").write_text("{}")
    (tmp_path / "pred_20240116.json").write_text("{}")
    (tmp_path / "pred_20240230_b.json").write_text("{}")
    (tmp_path / "pred_20240117x.json").write_text("{}")
    (tmp_path / "nodate.json").write_text("{}")

    # When: sort_preds_by_date runs
    sort_preds_by_date(tmp_path)

    # Then: only the two validly dated files move into their YYYYMMDD
    #       folders; the others stay at the root and no folder is created
    #       for the invalid date
    assert (tmp_path / "20240115" / "pred_20240115_a.json").exists()
    assert (tmp_path / "20240116" / "pred_20240116.json").exists()
    assert (tmp_path / "pred_20240230_b.json").exists()
    assert not (tmp_path / "20240230").exists()
    assert (tmp_path / "pred_20240117x.json").exists()
    assert (tmp_path / "nodate.json").exists()


def test_sort_preds_by_date_collision_leaves_root_copy(tmp_path):
    # Given: a root-level file whose destination already exists in the date
    #        folder with different content
    date_dir = tmp_path / "20240116"
    date_dir.mkdir()
    (date_dir / "pred_20240116_c.json").write_text("existing")
    (tmp_path / "pred_20240116_c.json").write_text("root copy")

    # When: sort_preds_by_date runs
    sort_preds_by_date(tmp_path)

    # Then: the root copy is left in place and the existing destination is
    #       not overwritten
    assert (tmp_path / "pred_20240116_c.json").read_text() == "root copy"
    assert (date_dir / "pred_20240116_c.json").read_text() == "existing"


# ---------------------------------------------------------------------------
# merge_geojsons (top-level orchestration)
# ---------------------------------------------------------------------------


def test_merge_geojsons_requires_output_without_process_by_date(tmp_path):
    # Given: a valid input folder but no output path and process_by_date off
    input_folder = str(tmp_path)

    # When: merge_geojsons runs
    # Then: a ClickException demands merge.output
    with pytest.raises(click.ClickException, match="merge.output is required"):
        merge_geojsons(input_folder, None)


def test_merge_geojsons_missing_input_folder_raises(tmp_path):
    # Given: an input folder path that does not exist
    ghost = tmp_path / "ghost"

    # When: merge_geojsons runs
    # Then: a ClickException reports the missing folder
    with pytest.raises(click.ClickException, match="Input folder not found"):
        merge_geojsons(str(ghost), str(tmp_path / "out.gpkg"))


def test_merge_geojsons_empty_folder_raises(tmp_path):
    # Given: an existing input folder with no GeoJSON files
    out = tmp_path / "out.gpkg"

    # When: merge_geojsons runs in whole-folder mode
    # Then: a ClickException reports that no GeoJSON files were found
    with pytest.raises(click.ClickException, match="No GeoJSON files found"):
        merge_geojsons(str(tmp_path), str(out))


def test_merge_geojsons_exclusion_zone_drops_covered_point(tmp_path):
    # Given: two far-apart points at (lon 0, lat 0) and (lon 0.01, lat 0.01)
    #        and an exclusion GPKG whose polygon covers only the second
    preds = tmp_path / "preds"
    preds.mkdir()
    write_geojson(preds / "a.geojson", [(0.0, 0.0, 0.5, 0.5), (0.01, 0.01, 0.6, 0.6)])
    zone_path = tmp_path / "exclusion.gpkg"
    gpd.GeoDataFrame(
        {"geometry": [box(0.005, 0.005, 0.015, 0.015)]}, crs="EPSG:4326"
    ).to_file(zone_path, driver="GPKG")
    out = tmp_path / "out.gpkg"

    # When: merge_geojsons runs with that exclusion file
    merge_geojsons(str(preds), str(out), exclusion_zones_gpkg=str(zone_path))

    # Then: the output GPKG contains only the uncovered point at the origin
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.equals(Point(0.0, 0.0))


def test_merge_geojsons_projected_exclusion_zone_reprojected_to_4326(tmp_path):
    # Given: two far-apart points at (lon 0, lat 0) and (lon 0.01, lat 0.01)
    #        and an exclusion GPKG written in EPSG:3857 whose box spans
    #        1000..1250 m on both axes -- in Web Mercator, 0.01 degrees is
    #        ~1113.19 m, so after reprojection to EPSG:4326 the box covers
    #        ONLY the second point (in raw metre coordinates it covers
    #        neither)
    preds = tmp_path / "preds"
    preds.mkdir()
    write_geojson(preds / "a.geojson", [(0.0, 0.0, 0.5, 0.5), (0.01, 0.01, 0.6, 0.6)])
    zone_path = tmp_path / "exclusion_3857.gpkg"
    gpd.GeoDataFrame(
        {"geometry": [box(1000.0, 1000.0, 1250.0, 1250.0)]}, crs="EPSG:3857"
    ).to_file(zone_path, driver="GPKG")
    out = tmp_path / "out.gpkg"

    # When: merge_geojsons runs with that projected exclusion file
    merge_geojsons(str(preds), str(out), exclusion_zones_gpkg=str(zone_path))

    # Then: only the uncovered origin point survives, proving the zone was
    #       reprojected to EPSG:4326 before filtering
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.equals(Point(0.0, 0.0))


def test_merge_geojsons_exclusion_unions_all_zone_features(tmp_path):
    # Given: three far-apart points at lon/lat (0, 0), (0.01, 0.01) and
    #        (0.02, 0.02), and an exclusion GPKG holding TWO disjoint
    #        polygons -- one covering only the second point, one covering
    #        only the third
    preds = tmp_path / "preds"
    preds.mkdir()
    write_geojson(
        preds / "a.geojson",
        [(0.0, 0.0, 0.5, 0.5), (0.01, 0.01, 0.6, 0.6), (0.02, 0.02, 0.7, 0.7)],
    )
    zone_path = tmp_path / "exclusion.gpkg"
    gpd.GeoDataFrame(
        {
            "geometry": [
                box(0.005, 0.005, 0.015, 0.015),
                box(0.015, 0.015, 0.025, 0.025),
            ]
        },
        crs="EPSG:4326",
    ).to_file(zone_path, driver="GPKG")
    out = tmp_path / "out.gpkg"

    # When: merge_geojsons runs with that exclusion file
    merge_geojsons(str(preds), str(out), exclusion_zones_gpkg=str(zone_path))

    # Then: both covered points are dropped (the zone is the union of ALL
    #       features, not just the first) and only the origin point survives
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.equals(Point(0.0, 0.0))


def test_merge_geojsons_inclusion_zone_keeps_only_covered_point(tmp_path):
    # Given: two far-apart points at (lon 0, lat 0) and (lon 0.01, lat 0.01)
    #        and an inclusion GPKG whose polygon covers only the second
    preds = tmp_path / "preds"
    preds.mkdir()
    write_geojson(preds / "a.geojson", [(0.0, 0.0, 0.5, 0.5), (0.01, 0.01, 0.6, 0.6)])
    zone_path = tmp_path / "inclusion.gpkg"
    gpd.GeoDataFrame(
        {"geometry": [box(0.005, 0.005, 0.015, 0.015)]}, crs="EPSG:4326"
    ).to_file(zone_path, driver="GPKG")
    out = tmp_path / "out.gpkg"

    # When: merge_geojsons runs with that inclusion file
    merge_geojsons(str(preds), str(out), inclusion_zone=str(zone_path))

    # Then: the output GPKG contains only the covered point -- inclusion
    #       filtering keeps inside points and drops everything else
    gdf = gpd.read_file(out)
    assert len(gdf) == 1
    assert gdf.iloc[0].geometry.equals(Point(0.01, 0.01))


def test_merge_geojsons_default_min_distance_is_3m(tmp_path):
    # Given: one file with two points 1e-4 degrees of latitude apart
    #        (~11.1 m) -- farther than the 3 m default merge radius but
    #        well within a 30 m one
    preds = tmp_path / "preds"
    preds.mkdir()
    write_geojson(
        preds / "a.geojson",
        [(0.0, 0.0, 0.5, 0.5), (0.0, 10 * NEAR_DEG, 0.6, 0.6)],
    )
    out = tmp_path / "out.gpkg"

    # When: merge_geojsons runs WITHOUT passing min_distance_m
    merge_geojsons(str(preds), str(out))

    # Then: both points survive unmerged, pinning the 3 m signature default
    gdf = gpd.read_file(out)
    assert len(gdf) == 2


def test_merge_geojsons_process_by_date_writes_per_date_gpkg(tmp_path):
    # Given: root-level predictions for two dates -- 20240101 has two files
    #        whose points sit ~1.1 m apart, 20240202 has one single-point
    #        file -- and no output path
    write_geojson(tmp_path / "pred_20240101_a.json", [(0.0, 0.0, 0.5, 0.5)])
    write_geojson(tmp_path / "pred_20240101_b.json", [(0.0, NEAR_DEG, 0.9, 0.8)])
    write_geojson(tmp_path / "pred_20240202_a.json", [(0.02, 0.02, 0.4, 0.4)])

    # When: merge_geojsons runs with process_by_date=True and the default
    #       min_distance_m of 3 m
    merge_geojsons(str(tmp_path), None, process_by_date=True)

    # Then: files are sorted into date folders and each date is merged into
    #       <input>/<date>.gpkg: 20240101.gpkg holds ONE centroid at the
    #       pairwise mean, 20240202.gpkg holds its single point
    assert (tmp_path / "20240101" / "pred_20240101_a.json").exists()
    jan = gpd.read_file(tmp_path / "20240101.gpkg")
    assert len(jan) == 1
    assert jan.iloc[0].geometry.y == pytest.approx(NEAR_DEG / 2, rel=1e-9)
    assert jan.iloc[0]["peak_value"] == pytest.approx(0.9)
    feb = gpd.read_file(tmp_path / "20240202.gpkg")
    assert len(feb) == 1
    assert feb.iloc[0].geometry.x == pytest.approx(0.02)


def test_merge_geojsons_process_by_date_isolates_failing_date(tmp_path):
    # Given: a good root-level prediction for 20240101 and a pre-existing
    #        20240202 folder containing only a corrupt file
    write_geojson(tmp_path / "pred_20240101_a.json", [(1.0, 2.0, 0.5, 0.5)])
    bad_dir = tmp_path / "20240202"
    bad_dir.mkdir()
    (bad_dir / "bad.json").write_text("{corrupt")

    # When: merge_geojsons runs with process_by_date=True
    # Then: a ClickException reports 1 of 2 failed dates, naming 20240202
    with pytest.raises(click.ClickException, match=r"1 of 2.*20240202"):
        merge_geojsons(str(tmp_path), None, process_by_date=True)

    # Then: 20240101.gpkg was still produced with its point, and the failing
    #       date wrote no GPKG of its own
    jan = gpd.read_file(tmp_path / "20240101.gpkg")
    assert len(jan) == 1
    assert not (tmp_path / "20240202.gpkg").exists()


def test_merge_geojsons_process_by_date_all_filtered_date_is_skipped(tmp_path):
    # Given: root-level predictions for two dates -- 20240101's only point
    #        carries adjusted peak 0.5, below min_adj_peak=0.99, while
    #        20240202's point (0.995) passes it
    write_geojson(tmp_path / "pred_20240101_a.json", [(0.0, 0.0, 0.5, 0.5)])
    write_geojson(tmp_path / "pred_20240202_a.json", [(0.02, 0.02, 0.995, 0.995)])

    # When: merge_geojsons runs with process_by_date=True and that threshold
    merge_geojsons(str(tmp_path), None, process_by_date=True, min_adj_peak=0.99)

    # Then: no ClickException is raised -- the all-filtered date is skipped,
    #       not failed: 20240202.gpkg exists with its point and no
    #       20240101.gpkg is written
    assert not (tmp_path / "20240101.gpkg").exists()
    feb = gpd.read_file(tmp_path / "20240202.gpkg")
    assert len(feb) == 1
    assert feb.iloc[0].geometry.equals(Point(0.02, 0.02))


def test_merge_geojsons_process_by_date_no_date_folders_raises(tmp_path):
    # Given: an input folder whose only file carries no date pattern
    write_geojson(tmp_path / "nodate.json", [(0.0, 0.0, 0.5, 0.5)])

    # When: merge_geojsons runs with process_by_date=True
    # Then: after sorting finds nothing to bucket, a ClickException reports
    #       that no date folders exist
    with pytest.raises(click.ClickException, match="No date folders found"):
        merge_geojsons(str(tmp_path), None, process_by_date=True)
