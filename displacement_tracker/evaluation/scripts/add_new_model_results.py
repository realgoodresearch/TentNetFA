#!/usr/bin/env python3

import os
from typing import Optional, Tuple

import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, box


TILE_SIZE_METERS = 100


def _unique_column_name(df: pd.DataFrame, base_name: str) -> str:
    """
    Return a column name that does not already exist in df.
    If base_name exists, append _1, _2, ... until unique.
    """
    if base_name not in df.columns:
        return base_name

    i = 1
    while f"{base_name}_{i}" in df.columns:
        i += 1
    return f"{base_name}_{i}"


def _load_predictions_for_date(
    date_str: str,
    prediction_dir: str,
    raster_crs,
    prediction_cache: dict,
):
    """
    Load prediction geopackage for a given date.
    Expected format: predictions/YYYYMMDD.gpkg
    """
    date_compact = pd.to_datetime(date_str).strftime("%Y%m%d")
    gpkg_path = os.path.join(prediction_dir, f"{date_compact}.gpkg")

    if gpkg_path in prediction_cache:
        return prediction_cache[gpkg_path]

    if not os.path.exists(gpkg_path):
        print(f"WARNING: Missing prediction file: {gpkg_path}")
        prediction_cache[gpkg_path] = None
        return None

    preds = gpd.read_file(gpkg_path)

    if preds.empty:
        print(f"WARNING: Empty prediction file: {gpkg_path}")
        prediction_cache[gpkg_path] = None
        return None

    # Assume WGS84 if CRS is missing
    if preds.crs is None:
        preds = preds.set_crs("EPSG:4326")

    # Reproject once here so downstream counting is simpler
    if raster_crs is not None and preds.crs != raster_crs:
        preds = preds.to_crs(raster_crs)

    prediction_cache[gpkg_path] = preds
    return preds


def _count_new_predictions_for_row(
    row: pd.Series,
    prediction_dir: str,
    raster_crs,
    prediction_cache: dict,
    tile_size_meters: int,
) -> Optional[int]:
    """
    Reconstruct the original tile from stored centroid lat/lon,
    then count points from the new model inside that tile.
    """
    preds = _load_predictions_for_date(
        date_str=row["date"],
        prediction_dir=prediction_dir,
        raster_crs=raster_crs,
        prediction_cache=prediction_cache,
    )
    if preds is None:
        return pd.NA

    # Convert centroid from WGS84 to raster CRS
    centroid_wgs84 = gpd.GeoSeries(
        [Point(row["longitude"], row["latitude"])],
        crs="EPSG:4326",
    )
    centroid_proj = centroid_wgs84.to_crs(raster_crs).iloc[0]

    cx, cy = centroid_proj.x, centroid_proj.y
    half = tile_size_meters / 2

    tile_geom = box(cx - half, cy - half, cx + half, cy + half)

    count = preds.within(tile_geom).sum()
    return int(count)


def add_new_model_results(
    annotation_csv: str = "manual_annotation_results.csv",
    output_csv: str = "manual_annotation_results_with_new_model.csv",
    prediction_dir: str = "predictions",
    sample_tif: str = "sample.tif",
    new_model_column: str = "model_new_tent_count",
    tile_size_meters: int = TILE_SIZE_METERS,
) -> Tuple[str, str]:
    """
    Add a new model count column to the annotation CSV and save the updated CSV.

    Returns:
        (output_csv_path, actual_column_name_used)
    """
    if not os.path.exists(annotation_csv):
        raise FileNotFoundError(f"Missing annotation CSV: {annotation_csv}")

    if not os.path.exists(sample_tif):
        raise FileNotFoundError(f"Missing sample tif: {sample_tif}")

    df = pd.read_csv(annotation_csv)

    actual_column_name = _unique_column_name(df, new_model_column)
    if actual_column_name not in df.columns:
        df[actual_column_name] = pd.NA

    with rasterio.open(sample_tif) as src:
        raster_crs = src.crs

    if raster_crs is None:
        raise ValueError(f"Could not read CRS from {sample_tif}")

    prediction_cache = {}

    for idx, row in df.iterrows():
        print(f"Processing row {idx + 1}/{len(df)}")
        df.loc[idx, actual_column_name] = _count_new_predictions_for_row(
            row=row,
            prediction_dir=prediction_dir,
            raster_crs=raster_crs,
            prediction_cache=prediction_cache,
            tile_size_meters=tile_size_meters,
        )

    df.to_csv(output_csv, index=False)
    print(f"\nSaved updated CSV to: {output_csv}")
    print(f"Added column: {actual_column_name}")

    return output_csv, actual_column_name


def main():
    add_new_model_results()


if __name__ == "__main__":
    main()
