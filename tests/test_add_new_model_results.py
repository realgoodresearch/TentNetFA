"""Tests for displacement_tracker/evaluation/scripts/add_new_model_results.py."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import Point

from _helpers import CRS_UTM as UTM
from _helpers import annotation_header, utm_to_lonlat, write_geotiff
from displacement_tracker.evaluation.scripts.add_new_model_results import (
    _unique_column_name,
    add_new_model_results,
)

# Two tile centroids in UTM 36N, 1 km apart. Their 100 m tiles span
# [x-50, x+50] x [y-50, y+50].
C1 = (500000.0, 3474000.0)
C2 = (501000.0, 3474000.0)


def lonlat(xy):
    return utm_to_lonlat(xy[0], xy[1])


def write_annotation_csv(path, rows):
    """rows: list of (date, utm_xy) — centroid stored as WGS84 lat/lon."""
    lines = [annotation_header()]
    for date, xy in rows:
        lon, lat = lonlat(xy)
        lines.append(f"{date},{lat},{lon},1\n")
    path.write_text("".join(lines))
    return str(path)


def write_sample_tif(path, crs=UTM):
    """A 1x1 placeholder raster; only its CRS and transform are read."""
    return write_geotiff(
        path,
        np.zeros((1, 1), dtype=np.float32),
        rasterio.transform.from_origin(500000.0, 3474000.0, 1.0, 1.0),
        crs=crs,
    )


def write_predictions(path, utm_points, crs=UTM):
    """Write prediction points given as UTM coordinates, in the given CRS."""
    if crs == UTM:
        geoms = [Point(x, y) for x, y in utm_points]
    else:
        geoms = [Point(*lonlat(p)) for p in utm_points]
    gpd.GeoDataFrame(geometry=geoms, crs=crs).to_file(path, driver="GPKG")
    return str(path)


# ==========================================================
# _unique_column_name
# ==========================================================


def test_unique_column_name_collision_fallback():
    # Given: a frame that already has columns "m" and "m_1"
    df = pd.DataFrame(columns=["m", "m_1"])

    # When: a unique name is requested for the taken base "m"
    taken = _unique_column_name(df, "m")

    # Then: it falls back to the first free suffix, "m_2"
    assert taken == "m_2"

    # When: a unique name is requested for the absent base "x"
    free = _unique_column_name(df, "x")

    # Then: "x" is used as-is
    assert free == "x"


# ==========================================================
# add_new_model_results
# ==========================================================


def test_counts_points_within_reconstructed_100m_tiles(tmp_path):
    # Given: two annotated tiles (centroids C1 and C2, 100 m tiles = +/-50 m
    #        boxes); the date's predictions hold 3 points strictly inside C1's
    #        tile (offsets (10,10), (-49,0), (0,49)), one point 60 m east of C1
    #        (outside both tiles), and none near C2
    ann = write_annotation_csv(
        tmp_path / "ann.csv", [("2024-10-14", C1), ("2024-10-14", C2)]
    )
    tif = write_sample_tif(tmp_path / "sample.tif")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    write_predictions(
        pred_dir / "20241014.gpkg",
        [
            (C1[0] + 10, C1[1] + 10),
            (C1[0] - 49, C1[1]),
            (C1[0], C1[1] + 49),
            (C1[0] + 60, C1[1]),
        ],
    )

    # When: add_new_model_results joins the predictions onto the annotations
    out_csv, column = add_new_model_results(
        annotation_csv=ann,
        output_csv=str(tmp_path / "out.csv"),
        prediction_dir=str(pred_dir),
        sample_tif=tif,
        new_model_column="new_model",
    )

    # Then: C1's tile counts 3, C2's tile counts 0, and the manual column is
    #       preserved unchanged in the written CSV
    assert column == "new_model"
    result = pd.read_csv(out_csv)
    assert result["new_model"].tolist() == [3, 0]
    assert result["manual_tent_count"].tolist() == [1, 1]


def test_predictions_are_matched_per_date_file(tmp_path):
    # Given: two annotation rows at the SAME centroid C1 but different dates;
    #        20241014.gpkg holds 2 points inside the tile and 20241101.gpkg
    #        holds 5
    ann = write_annotation_csv(
        tmp_path / "ann.csv", [("2024-10-14", C1), ("2024-11-01", C1)]
    )
    tif = write_sample_tif(tmp_path / "sample.tif")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    inside = (C1[0] + 5, C1[1] - 5)
    write_predictions(pred_dir / "20241014.gpkg", [inside] * 2)
    write_predictions(pred_dir / "20241101.gpkg", [inside] * 5)

    # When: add_new_model_results runs
    out_csv, column = add_new_model_results(
        annotation_csv=ann,
        output_csv=str(tmp_path / "out.csv"),
        prediction_dir=str(pred_dir),
        sample_tif=tif,
        new_model_column="new_model",
    )

    # Then: each row is counted only against its own date's file: [2, 5]
    result = pd.read_csv(out_csv)
    assert result.loc[result["date"] == "2024-10-14", column].tolist() == [2]
    assert result.loc[result["date"] == "2024-11-01", column].tolist() == [5]


def test_lonlat_predictions_reprojected_to_raster_crs(tmp_path):
    # Given: prediction points stored in EPSG:4326 (converted from UTM offsets
    #        (10,10) and (-30,20) inside C1's tile) while the sample tif is UTM
    ann = write_annotation_csv(tmp_path / "ann.csv", [("2024-10-14", C1)])
    tif = write_sample_tif(tmp_path / "sample.tif")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    write_predictions(
        pred_dir / "20241014.gpkg",
        [(C1[0] + 10, C1[1] + 10), (C1[0] - 30, C1[1] + 20)],
        crs="EPSG:4326",
    )

    # When: add_new_model_results runs
    out_csv, column = add_new_model_results(
        annotation_csv=ann,
        output_csv=str(tmp_path / "out.csv"),
        prediction_dir=str(pred_dir),
        sample_tif=tif,
        new_model_column="new_model",
    )

    # Then: predictions are reprojected before the spatial join, so both count
    assert pd.read_csv(out_csv)[column].tolist() == [2]


def test_missing_prediction_file_yields_na_only_for_that_date(tmp_path):
    # Given: annotations on 2024-10-14 (predictions present, 1 point inside)
    #        and 2024-11-01 (no gpkg on disk)
    ann = write_annotation_csv(
        tmp_path / "ann.csv", [("2024-10-14", C1), ("2024-11-01", C1)]
    )
    tif = write_sample_tif(tmp_path / "sample.tif")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    write_predictions(pred_dir / "20241014.gpkg", [(C1[0], C1[1])])

    # When: add_new_model_results runs
    out_csv, column = add_new_model_results(
        annotation_csv=ann,
        output_csv=str(tmp_path / "out.csv"),
        prediction_dir=str(pred_dir),
        sample_tif=tif,
        new_model_column="new_model",
    )

    # Then: the October row gets its count and the November row is NA,
    #       not zero — absence of a file is not "zero tents"
    result = pd.read_csv(out_csv)
    assert result.loc[result["date"] == "2024-10-14", column].tolist() == [1]
    assert result.loc[result["date"] == "2024-11-01", column].isna().all()


def test_column_name_collision_appends_suffix_and_keeps_original(tmp_path):
    # Given: the annotation CSV already carries a column named "new_model"
    #        with value 42
    ann_path = tmp_path / "ann.csv"
    lon, lat = lonlat(C1)
    ann_path.write_text(
        "date,latitude,longitude,manual_tent_count,new_model\n"
        f"2024-10-14,{lat},{lon},1,42\n"
    )
    tif = write_sample_tif(tmp_path / "sample.tif")
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    write_predictions(pred_dir / "20241014.gpkg", [(C1[0], C1[1])])

    # When: add_new_model_results is asked to add "new_model" again
    out_csv, column = add_new_model_results(
        annotation_csv=str(ann_path),
        output_csv=str(tmp_path / "out.csv"),
        prediction_dir=str(pred_dir),
        sample_tif=tif,
        new_model_column="new_model",
    )

    # Then: the counts land in "new_model_1" and the original column's value
    #       survives untouched
    assert column == "new_model_1"
    result = pd.read_csv(out_csv)
    assert result["new_model"].tolist() == [42]
    assert result["new_model_1"].tolist() == [1]


def test_missing_inputs_raise_file_not_found(tmp_path):
    # Given: no annotation CSV at the path handed to the function
    # When: add_new_model_results runs
    # Then: the precondition fails fast with FileNotFoundError naming the
    #       annotation input
    with pytest.raises(FileNotFoundError, match="annotation"):
        add_new_model_results(
            annotation_csv=str(tmp_path / "nope.csv"),
            output_csv=str(tmp_path / "out.csv"),
            prediction_dir=str(tmp_path),
            sample_tif=str(tmp_path / "nope.tif"),
            new_model_column="m",
        )

    # Given: an annotation CSV that now exists, but still no sample tif
    ann = write_annotation_csv(tmp_path / "ann.csv", [("2024-10-14", C1)])

    # When: add_new_model_results runs
    # Then: the next precondition fails fast, naming the tif
    with pytest.raises(FileNotFoundError, match="tif"):
        add_new_model_results(
            annotation_csv=ann,
            output_csv=str(tmp_path / "out.csv"),
            prediction_dir=str(tmp_path),
            sample_tif=str(tmp_path / "nope.tif"),
            new_model_column="m",
        )


def test_sample_tif_without_crs_raises_value_error(tmp_path):
    # Given: a sample tif written with no CRS
    ann = write_annotation_csv(tmp_path / "ann.csv", [("2024-10-14", C1)])
    tif = write_sample_tif(tmp_path / "nocrs.tif", crs=None)

    # When: add_new_model_results tries to read the raster CRS
    # Then: it raises ValueError instead of counting in an undefined CRS
    with pytest.raises(ValueError, match="CRS"):
        add_new_model_results(
            annotation_csv=ann,
            output_csv=str(tmp_path / "out.csv"),
            prediction_dir=str(tmp_path),
            sample_tif=tif,
            new_model_column="m",
        )
