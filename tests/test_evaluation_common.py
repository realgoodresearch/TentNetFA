"""Tests for displacement_tracker/evaluation/scripts/common.py and the
config-resolution/preflight logic of displacement_tracker/evaluation/run_all_analyses.py.
"""

import json
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # run_all_analyses imports plot modules

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from shapely.geometry import Point, box

from displacement_tracker.evaluation.run_all_analyses import cli, load_config, require
from displacement_tracker.evaluation.scripts.common import (
    build_hex_grid,
    choose_utm_crs_from_gdf,
    finite_xy,
    group_error_summary,
    hex_error_aggregation,
    load_annotation_points,
    load_annotations,
    load_layer,
    make_hexagon,
    mean_error_ci,
    read_annotations,
)

# ==========================================================
# Annotation loading
# ==========================================================


def test_read_annotations_missing_columns_listed_sorted(tmp_path):
    # Given: a CSV with only columns a and b
    csv = tmp_path / "ann.csv"
    csv.write_text("a,b\n1,2\n")

    # When: read_annotations requires columns b, z, y and a (z and y absent)
    # Then: it raises ValueError listing the missing columns in sorted order
    #       ['y', 'z'], regardless of the order they were requested in
    with pytest.raises(ValueError, match=r"\['y', 'z'\]"):
        read_annotations(str(csv), required_columns=("b", "z", "y", "a"))


def test_load_annotations_tile_error_is_model_minus_manual(tmp_path):
    # Given: a CSV with manual counts [10, 4] and model counts [7, 9]
    csv = tmp_path / "ann.csv"
    csv.write_text("manual,model\n10,7\n4,9\n")

    # When: load_annotations computes tile_error
    df = load_annotations(str(csv), manual_column="manual", model_column="model")

    # Then: tile_error is model - manual, i.e. [-3, 5]
    assert df["tile_error"].tolist() == [-3, 5]


def test_load_annotation_points_requires_lat_lon_and_builds_wgs84_points(tmp_path):
    # Given: a CSV with lat/lon and count columns
    csv = tmp_path / "ann.csv"
    csv.write_text("latitude,longitude,manual,model\n31.5,34.25,3,8\n")

    # When: load_annotation_points loads it
    gdf = load_annotation_points(str(csv), manual_column="manual", model_column="model")

    # Then: the result is EPSG:4326 points at (lon, lat) with tile_error attached
    assert gdf.crs.to_epsg() == 4326
    assert gdf.geometry.iloc[0].x == pytest.approx(34.25)
    assert gdf.geometry.iloc[0].y == pytest.approx(31.5)
    assert gdf["tile_error"].iloc[0] == 5


def test_load_annotation_points_missing_latitude_raises(tmp_path):
    # Given: a CSV that has longitude but no latitude column
    csv = tmp_path / "ann.csv"
    csv.write_text("longitude,manual,model\n34.25,3,8\n")

    # When: load_annotation_points is called
    # Then: the implicit latitude requirement fails with ValueError
    with pytest.raises(ValueError, match="latitude"):
        load_annotation_points(str(csv), manual_column="manual", model_column="model")


def test_load_layer_reprojects_to_target_crs(tmp_path):
    # Given: a GeoJSON point layer at lon 34.3 in EPSG:4326
    path = tmp_path / "layer.geojson"
    gpd.GeoDataFrame(
        {"name": ["p"]}, geometry=[Point(34.3, 31.4)], crs="EPSG:4326"
    ).to_file(path, driver="GeoJSON")

    # When: load_layer is asked for EPSG:3857
    out = load_layer(str(path), "EPSG:3857", required_columns=("name",))

    # Then: the returned layer is in 3857 with x = R * radians(lon) (spherical
    #       Mercator definition, R = 6378137 m)
    assert out.crs.to_epsg() == 3857
    assert out.geometry.iloc[0].x == pytest.approx(6378137 * math.radians(34.3))


def test_load_layer_missing_required_column_raises(tmp_path):
    # Given: a vector layer whose only attribute column is "name"
    path = tmp_path / "layer.geojson"
    gpd.GeoDataFrame({"name": ["p"]}, geometry=[Point(0, 0)], crs="EPSG:4326").to_file(
        path, driver="GeoJSON"
    )

    # When: load_layer requires a column "kind" that is absent
    # Then: it raises ValueError mentioning the missing column
    with pytest.raises(ValueError, match="kind"):
        load_layer(str(path), None, required_columns=("name", "kind"))


def test_finite_xy_drops_pairs_with_any_nonfinite_member():
    # Given: x = [1, nan, 3, inf, 5] and y = [10, 20, nan, 40, 50]
    xs = [1, np.nan, 3, np.inf, 5]
    ys = [10, 20, np.nan, 40, 50]

    # When: finite_xy filters the pair
    x, y = finite_xy(xs, ys)

    # Then: only positions where BOTH are finite survive: (1,10) and (5,50)
    assert x.tolist() == [1.0, 5.0]
    assert y.tolist() == [10.0, 50.0]


# ==========================================================
# Error summaries
# ==========================================================


def test_mean_error_ci_hand_computed_four_values():
    # Given: errors [1, 2, 3, 6] (mean 3; sample var = 14/3; std = sqrt(14/3))
    errors = [1, 2, 3, 6]

    # When: mean_error_ci runs with z = 1.96
    stats = mean_error_ci(errors)

    # Then: margin = 1.96 * sqrt(14/3) / 2 and the CI is mean -/+ margin
    std = math.sqrt(14 / 3)
    margin = 1.96 * std / 2
    assert stats["mean_tile_error"] == pytest.approx(3.0)
    assert stats["std_tile_error"] == pytest.approx(std, rel=1e-12)
    assert stats["ci_lower"] == pytest.approx(3.0 - margin, rel=1e-12)
    assert stats["ci_upper"] == pytest.approx(3.0 + margin, rel=1e-12)
    assert stats["num_tiles"] == 4


def test_mean_error_ci_single_value_has_zero_std_and_degenerate_ci():
    # Given: a single error value 4.5
    errors = [4.5]

    # When: mean_error_ci runs
    stats = mean_error_ci(errors)

    # Then: std is 0.0 and both CI bounds equal the mean
    assert stats == {
        "mean_tile_error": 4.5,
        "std_tile_error": 0.0,
        "ci_lower": 4.5,
        "ci_upper": 4.5,
        "num_tiles": 1,
    }


def test_mean_error_ci_ignores_nonfinite_and_none_when_empty():
    # Given: errors [1, nan, 3] (finite subset has mean 2, std sqrt(2), n=2)
    errors = [1, np.nan, 3]

    # When: mean_error_ci runs
    stats = mean_error_ci(errors)

    # Then: non-finite values are excluded so margin = 1.96 * sqrt(2)/sqrt(2) = 1.96
    assert stats["num_tiles"] == 2
    assert stats["mean_tile_error"] == pytest.approx(2.0)
    assert stats["ci_lower"] == pytest.approx(2.0 - 1.96, rel=1e-12)
    assert stats["ci_upper"] == pytest.approx(2.0 + 1.96, rel=1e-12)

    # When: mean_error_ci runs on inputs with no finite value at all
    # Then: there is nothing to summarize, so the result is None
    assert mean_error_ci([np.nan, np.inf]) is None
    assert mean_error_ci([]) is None


def test_group_error_summary_per_group_stats_and_skips():
    # Given: groups A -> errors [1, 3], B -> [5], C -> [nan], and one row with
    #        a NaN group key
    frame = pd.DataFrame(
        {
            "group": ["A", "A", "B", "C", np.nan],
            "tile_error": [1.0, 3.0, 5.0, np.nan, 99.0],
        }
    )

    # When: group_error_summary aggregates tile_error by group
    out = group_error_summary(frame, "group")

    # Then: A has mean 2, std sqrt(2), CI 2 -/+ 1.96, n=2; B has mean 5 with a
    #       zero-width CI; C and the NaN-key row produce no rows at all
    assert out["group"].tolist() == ["A", "B"]
    a = out[out["group"] == "A"].iloc[0]
    assert a["mean_tile_error"] == pytest.approx(2.0)
    assert a["std_tile_error"] == pytest.approx(math.sqrt(2), rel=1e-12)
    assert a["ci_lower"] == pytest.approx(2.0 - 1.96, rel=1e-12)
    assert a["num_tiles"] == 2
    b = out[out["group"] == "B"].iloc[0]
    assert b["mean_tile_error"] == pytest.approx(5.0)
    assert b["ci_lower"] == pytest.approx(5.0)
    assert b["ci_upper"] == pytest.approx(5.0)


# ==========================================================
# Hex grid
# ==========================================================


def test_choose_utm_crs_northern_and_southern_hemispheres():
    # Given: a point at lon 34.3 / lat 31.4 (zone int(214.3/6)+1 = 36, north)
    #        and one at lon -58.4 / lat -34.6 (zone int(121.6/6)+1 = 21, south)
    north = gpd.GeoDataFrame(geometry=[Point(34.3, 31.4)], crs="EPSG:4326")
    south = gpd.GeoDataFrame(geometry=[Point(-58.4, -34.6)], crs="EPSG:4326")

    # When: choose_utm_crs_from_gdf picks a UTM CRS for each
    north_crs = choose_utm_crs_from_gdf(north)
    south_crs = choose_utm_crs_from_gdf(south)

    # Then: they resolve to EPSG:32636 and EPSG:32721 respectively
    assert north_crs.to_epsg() == 32636
    assert south_crs.to_epsg() == 32721


def test_choose_utm_crs_requires_a_crs():
    # Given: a layer with no CRS set
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])

    # When: choose_utm_crs_from_gdf runs
    # Then: it refuses with ValueError instead of guessing
    with pytest.raises(ValueError, match="no CRS"):
        choose_utm_crs_from_gdf(gdf)


def test_make_hexagon_geometry():
    # Given: side length a=2 centered at (10, 5)
    center_x, center_y, side = 10.0, 5.0, 2.0

    # When: make_hexagon builds the polygon
    hexagon = make_hexagon(center_x, center_y, side)

    # Then: area is 3*sqrt(3)/2 * a^2 = 6*sqrt(3) and bounds span
    #       [center_x - a, center_x + a] x [center_y - a*sqrt(3)/2, ...]
    assert hexagon.area == pytest.approx(6 * math.sqrt(3), rel=1e-12)
    minx, miny, maxx, maxy = hexagon.bounds
    assert (minx, maxx) == pytest.approx((8.0, 12.0))
    assert miny == pytest.approx(5 - math.sqrt(3), rel=1e-12)
    assert maxy == pytest.approx(5 + math.sqrt(3), rel=1e-12)


def test_build_hex_grid_tiles_the_bounds_without_gaps_or_overlaps():
    # Given: bounds (0, 0, 20, 20), side length a=5, and four interior sample
    #        points chosen to sit off any hex edge
    bounds = (0.0, 0.0, 20.0, 20.0)
    samples = [Point(1.1, 2.3), Point(9.7, 4.4), Point(15.2, 18.9), Point(19.3, 0.6)]

    # When: build_hex_grid lays out the flat-topped grid
    hexes = build_hex_grid(bounds, 5.0)

    # Then: every sample point is covered by exactly one hexagon — no gaps, no
    #       double coverage
    for sample in samples:
        covering = sum(1 for h in hexes if h.contains(sample))
        assert covering == 1, f"{sample.wkt} covered by {covering} hexes"


def test_hex_error_aggregation_groups_points_and_computes_stats(tmp_path):
    # Given: a lon/lat boundary box around (34.30, 31.40) (UTM zone 36N), two
    #        annotated tiles at the exact same location with errors 1 and 3,
    #        and a third tile ~1.66 km north with error 7; hex_size_m=1000 so
    #        the hex circumradius is 500 m (max in-hex distance 1000 m)
    boundary_path = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(
        geometry=[box(34.28, 31.38, 34.32, 31.42)], crs="EPSG:4326"
    ).to_file(boundary_path, driver="GeoJSON")

    points = gpd.GeoDataFrame(
        {"tile_error": [1.0, 3.0, 7.0]},
        geometry=[Point(34.30, 31.40), Point(34.30, 31.40), Point(34.30, 31.415)],
        crs="EPSG:4326",
    )

    # When: hex_error_aggregation aggregates tile_error onto the hex grid
    hex_gdf, points_with_hex, boundary_proj = hex_error_aggregation(
        points, str(boundary_path), hex_size_m=1000.0
    )

    # Then: the co-located pair shares one hex (n=2, mean 2, std sqrt(2)), the
    #       far tile sits alone in a DIFFERENT hex (n=1, mean 7, std NaN), and
    #       hexes without points carry n_tiles=0 with NaN mean
    assert boundary_proj.crs.to_epsg() == 32636
    assert len(points_with_hex) == 3

    pair_hex_ids = points_with_hex.loc[points_with_hex["tile_error"] < 5, "hex_id"]
    lone_hex_id = points_with_hex.loc[points_with_hex["tile_error"] == 7.0, "hex_id"]
    assert pair_hex_ids.nunique() == 1
    pair_hex = pair_hex_ids.iloc[0]
    lone_hex = lone_hex_id.iloc[0]
    assert pair_hex != lone_hex  # 1.66 km apart > 1 km max in-hex distance

    pair_row = hex_gdf.set_index("hex_id").loc[pair_hex]
    assert pair_row["n_tiles"] == 2
    assert pair_row["mean_err"] == pytest.approx(2.0)
    assert pair_row["std_err"] == pytest.approx(math.sqrt(2), rel=1e-12)

    lone_row = hex_gdf.set_index("hex_id").loc[lone_hex]
    assert lone_row["n_tiles"] == 1
    assert lone_row["mean_err"] == pytest.approx(7.0)
    assert pd.isna(lone_row["std_err"])

    empty = hex_gdf[~hex_gdf["hex_id"].isin([pair_hex, lone_hex])]
    assert not empty.empty
    assert (empty["n_tiles"] == 0).all()
    assert empty["mean_err"].isna().all()


def test_hex_error_aggregation_hex_geometry_matches_hex_size(tmp_path):
    # Given: a small UTM-zone-36N boundary and hex_size_m=1000, which the
    #        aggregation must convert to a hexagon side (circumradius) of
    #        a = hex_size_m / 2 = 500 m — a side of 1000 m instead would
    #        quadruple each hex's area and silently coarsen the spatial
    #        resolution of the error maps
    boundary_path = tmp_path / "boundary.geojson"
    gpd.GeoDataFrame(
        geometry=[box(34.28, 31.38, 34.32, 31.42)], crs="EPSG:4326"
    ).to_file(boundary_path, driver="GeoJSON")
    points = gpd.GeoDataFrame(
        {"tile_error": [1.0]}, geometry=[Point(34.30, 31.40)], crs="EPSG:4326"
    )

    # When: hex_error_aggregation builds its hex grid
    hex_gdf, _, _ = hex_error_aggregation(points, str(boundary_path), hex_size_m=1000.0)

    # Then: every returned hex has area 3*sqrt(3)/2 * 500^2, and its x-extent
    #       is exactly 1000.0 m (make_hexagon places vertices at 0 and 180
    #       degrees, so the corner-to-corner diameter 2a runs along x while
    #       the y-extent is only sqrt(3)*a)
    expected_area = 3 * math.sqrt(3) / 2 * 500.0**2
    assert hex_gdf.geometry.area.to_numpy() == pytest.approx(expected_area, rel=1e-9)
    minx, miny, maxx, maxy = hex_gdf.geometry.iloc[0].bounds
    assert maxx - minx == pytest.approx(1000.0, abs=1e-6)
    assert maxy - miny == pytest.approx(math.sqrt(3) * 500.0, abs=1e-6)


def test_hex_error_aggregation_empty_boundary_raises(tmp_path):
    # Given: a boundary layer file containing zero features
    boundary_path = tmp_path / "empty.geojson"
    gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_file(
        boundary_path, driver="GeoJSON"
    )
    points = gpd.GeoDataFrame(
        {"tile_error": [1.0]}, geometry=[Point(34.3, 31.4)], crs="EPSG:4326"
    )

    # When: hex_error_aggregation runs
    # Then: it raises ValueError instead of building a grid over nothing
    with pytest.raises(ValueError, match="empty"):
        hex_error_aggregation(points, str(boundary_path), hex_size_m=1000.0)


# ==========================================================
# run_all_analyses config resolution and preflight
# ==========================================================


def test_load_config_resolves_relative_paths_against_config_dir(tmp_path):
    # Given: a config file in tmp/cfgdir with a relative annotation_csv, an
    #        absolute boundary_shp, a null prediction_dir and a non-path key
    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "analysis.json"
    absolute_boundary = str(tmp_path / "elsewhere" / "bounds.shp")
    cfg_file.write_text(
        json.dumps(
            {
                "annotation_csv": "data/ann.csv",
                "boundary_shp": absolute_boundary,
                "prediction_dir": None,
                "model_column": "model_tent_count",
            }
        )
    )

    # When: load_config loads it
    cfg = load_config(cfg_file)

    # Then: the relative path resolves against cfgdir (not the CWD), the
    #       absolute path is preserved, null stays None and non-path keys
    #       are untouched
    assert cfg["annotation_csv"] == str((cfg_dir / "data" / "ann.csv").resolve())
    assert cfg["boundary_shp"] == absolute_boundary
    assert cfg["prediction_dir"] is None
    assert cfg["model_column"] == "model_tent_count"


def test_require_rejects_missing_and_falsy_values():
    # Given: a config where key "a" is present, "b" is empty and "c" is absent
    cfg = {"a": "value", "b": ""}

    # When: require is called for the present key "a"
    value = require(cfg, "a")

    # Then: its value is returned
    assert value == "value"

    # When: require is called for the empty "b" and for the absent "c"
    # Then: both raise ClickException naming the offending key
    with pytest.raises(click.ClickException, match="b"):
        require(cfg, "b")
    with pytest.raises(click.ClickException, match="c"):
        require(cfg, "c")


def test_cli_preflight_lists_exactly_the_missing_inputs(tmp_path):
    # Given: a config whose annotation_csv exists but whose four spatial
    #        layers point at nonexistent relative paths
    ann = tmp_path / "ann.csv"
    ann.write_text("date,latitude,longitude,manual_tent_count,model_tent_count\n")
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "annotation_csv": "ann.csv",
                "output_dir": "results",
                "boundary_shp": "missing/bounds.shp",
                "agriculture_geojson": "missing/agri.json",
                "h3_geojson": "missing/h3.json",
                "destruction_geojson": "missing/destr.json",
            }
        )
    )

    # When: the run-evaluation CLI runs its preflight check
    result = CliRunner().invoke(cli, ["--config", str(cfg_file)])

    # Then: it fails listing each missing resolved path, does not list the
    #       existing annotation CSV, and does not create output_dir
    assert result.exit_code != 0
    for rel in ("bounds.shp", "agri.json", "h3.json", "destr.json"):
        assert str((tmp_path / "missing" / rel).resolve()) in result.output
    assert str(ann.resolve()) not in result.output
    assert not (tmp_path / "results").exists()


def test_cli_partial_new_model_config_fails_naming_missing_key(tmp_path):
    # Given: a config whose preflight input files all exist as tiny dummies
    #        and which sets prediction_dir but omits sample_tif and
    #        new_model_column — a partially specified new-model run; this
    #        exercises the fail-fast misconfiguration guard, not click
    #        argument plumbing
    ann = tmp_path / "ann.csv"
    ann.write_text("date,latitude,longitude,manual_tent_count,model_tent_count\n")
    for dummy in ("bounds.shp", "agri.json", "h3.json", "destr.json"):
        (tmp_path / dummy).write_text("dummy")
    (tmp_path / "preds").mkdir()
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(
        json.dumps(
            {
                "annotation_csv": "ann.csv",
                "output_dir": "results",
                "boundary_shp": "bounds.shp",
                "agriculture_geojson": "agri.json",
                "h3_geojson": "h3.json",
                "destruction_geojson": "destr.json",
                "prediction_dir": "preds",
            }
        )
    )

    # When: the run-evaluation CLI resolves the config
    result = CliRunner().invoke(cli, ["--config", str(cfg_file)])

    # Then: because ANY new-model key being set marks the run as a new-model
    #       run, it aborts with "Missing required config key" naming
    #       sample_tif instead of silently falling back to evaluating the
    #       old model_column
    assert result.exit_code != 0
    assert "Missing required config key: sample_tif" in result.output
