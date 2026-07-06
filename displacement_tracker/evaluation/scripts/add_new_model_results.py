"""Append a new model's tent counts to the manual-annotation CSV.

For every annotated tile (identified by its centroid lat/lon and date), the
matching per-date prediction GeoPackage (predictions/YYYYMMDD.gpkg, as
produced by `merge-geojsons --process-by-date`) is loaded and the predicted
points falling inside the reconstructed 100 m tile are counted.
"""

import os

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

TILE_SIZE_METERS = 100


def _unique_column_name(df: pd.DataFrame, base_name: str) -> str:
    """Return base_name, or base_name_1, _2, ... if it already exists in df."""
    if base_name not in df.columns:
        return base_name

    i = 1
    while f"{base_name}_{i}" in df.columns:
        i += 1
    return f"{base_name}_{i}"


def _count_predictions_for_date(
    group: pd.DataFrame,
    gpkg_path: str,
    raster_crs,
    tile_size_meters: float,
) -> pd.Series:
    """Count predicted points inside each annotated tile of one date."""
    if not os.path.exists(gpkg_path):
        print(f"WARNING: Missing prediction file: {gpkg_path}")
        return pd.Series(pd.NA, index=group.index)

    preds = gpd.read_file(gpkg_path)
    if preds.empty:
        print(f"WARNING: Empty prediction file: {gpkg_path}")
        return pd.Series(pd.NA, index=group.index)

    if preds.crs is None:
        preds = preds.set_crs("EPSG:4326")
    if preds.crs != raster_crs:
        preds = preds.to_crs(raster_crs)

    # Reconstruct the original tiles around the stored centroids.
    centroids = gpd.GeoSeries(
        gpd.points_from_xy(group["longitude"], group["latitude"]),
        index=group.index,
        crs="EPSG:4326",
    ).to_crs(raster_crs)

    half = tile_size_meters / 2
    tiles = gpd.GeoDataFrame(
        geometry=[box(c.x - half, c.y - half, c.x + half, c.y + half) for c in centroids],
        index=group.index,
        crs=raster_crs,
    )

    joined = gpd.sjoin(preds[["geometry"]], tiles, how="inner", predicate="within")
    counts = joined.groupby("index_right").size()
    return counts.reindex(group.index, fill_value=0).astype(int)


def add_new_model_results(
    annotation_csv: str,
    output_csv: str,
    prediction_dir: str,
    sample_tif: str,
    new_model_column: str,
    tile_size_meters: float = TILE_SIZE_METERS,
) -> tuple[str, str]:
    """
    Add a new model count column to the annotation CSV and save the result.

    Returns:
        (output_csv_path, actual_column_name_used)
    """
    if not os.path.exists(annotation_csv):
        raise FileNotFoundError(f"Missing annotation CSV: {annotation_csv}")
    if not os.path.exists(sample_tif):
        raise FileNotFoundError(f"Missing sample tif: {sample_tif}")

    df = pd.read_csv(annotation_csv)

    actual_column_name = _unique_column_name(df, new_model_column)
    df[actual_column_name] = pd.NA

    with rasterio.open(sample_tif) as src:
        raster_crs = src.crs
    if raster_crs is None:
        raise ValueError(f"Could not read CRS from {sample_tif}")

    for date_str, group in df.groupby("date"):
        date_compact = pd.to_datetime(date_str).strftime("%Y%m%d")
        gpkg_path = os.path.join(prediction_dir, f"{date_compact}.gpkg")
        print(f"Counting predictions for {date_compact} ({len(group)} tiles)")
        df.loc[group.index, actual_column_name] = _count_predictions_for_date(
            group, gpkg_path, raster_crs, tile_size_meters
        )

    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV to: {output_csv}")
    print(f"Added column: {actual_column_name}")

    return output_csv, actual_column_name
