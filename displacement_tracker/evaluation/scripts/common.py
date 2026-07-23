"""Shared loading and statistics helpers for the evaluation scripts.

Every analysis in this package follows the same pattern: load the annotated
tiles, compute per-group mean tile error with a 95% CI, write a CSV and a
plot. The loading/statistics halves of that pattern live here; the plotting
halves live in ``plots.py`` so non-plotting consumers (e.g. the reference
data integration in ``annotation_reference.py``) can reuse the loaders
without pulling in matplotlib.
"""

import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("evaluation")

Z_95 = 1.96


# ==========================================================
# ANNOTATION AND LAYER LOADING
# ==========================================================

def read_annotations(annotation_csv: str, required_columns=()) -> pd.DataFrame:
    """Read the annotation CSV and validate that required columns exist."""
    df = pd.read_csv(annotation_csv)
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")
    return df


def as_points(
    df: pd.DataFrame,
    lat_column: str = "latitude",
    lon_column: str = "longitude",
) -> gpd.GeoDataFrame:
    """Tile centroids as an EPSG:4326 point GeoDataFrame."""
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_column], df[lat_column]),
        crs="EPSG:4326",
    )


def load_annotations(
    annotation_csv: str,
    manual_column: str,
    model_column: str,
    extra_columns: tuple = (),
) -> pd.DataFrame:
    """Read the annotation CSV and add a tile_error column."""
    df = read_annotations(
        annotation_csv, {manual_column, model_column, *extra_columns}
    )
    df["tile_error"] = df[model_column] - df[manual_column]
    return df


def load_annotation_points(
    annotation_csv: str,
    manual_column: str,
    model_column: str,
    extra_columns: tuple = (),
) -> gpd.GeoDataFrame:
    """Like load_annotations, but returns tile centroids as points."""
    df = load_annotations(
        annotation_csv,
        manual_column,
        model_column,
        extra_columns=("latitude", "longitude", *extra_columns),
    )
    return as_points(df)


def load_layer(path: str, target_crs, required_columns=()) -> gpd.GeoDataFrame:
    """Read a vector layer, validate required columns, align to target_crs."""
    gdf = gpd.read_file(path)
    missing = set(required_columns) - set(gdf.columns)
    if missing:
        raise ValueError(f"{path} missing required column(s): {sorted(missing)}")
    if target_crs is not None and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def finite_xy(x, y):
    """Remove NaN / inf pairs from x and y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


# ==========================================================
# ERROR SUMMARIES
# ==========================================================

def mean_error_ci(errors, z: float = Z_95) -> dict | None:
    """Mean error with sample std and a normal-approximation CI.

    Returns None when there are no values; std and the CI margin are 0.0
    for a single value.
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    n = len(errors)
    if n == 0:
        return None

    mean = float(np.mean(errors))
    std = float(np.std(errors, ddof=1)) if n > 1 else 0.0
    margin = z * std / math.sqrt(n)
    return {
        "mean_tile_error": mean,
        "std_tile_error": std,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin,
        "num_tiles": n,
    }


def group_error_summary(
    frame: pd.DataFrame,
    group_column: str,
    value_column: str = "tile_error",
) -> pd.DataFrame:
    """Per-group mean_error_ci over value_column, one row per group."""
    rows = []
    for name, group in frame.groupby(group_column, observed=True, dropna=True):
        stats = mean_error_ci(group[value_column].dropna().values)
        if stats is None:
            continue
        rows.append({group_column: name, **stats})
    return pd.DataFrame(rows)


# ==========================================================
# HEX GRID
# ==========================================================

def choose_utm_crs_from_gdf(gdf: gpd.GeoDataFrame) -> CRS:
    """Choose a UTM CRS based on the centroid of the layer."""
    if gdf.crs is None:
        raise ValueError("Boundary layer has no CRS.")

    lonlat = gdf.to_crs("EPSG:4326")
    centroid = lonlat.geometry.union_all().centroid
    zone = int((centroid.x + 180) / 6) + 1
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def make_hexagon(center_x: float, center_y: float, a: float) -> Polygon:
    """Regular hexagon with side length a centered at (center_x, center_y)."""
    pts = []
    for ang in (0, 60, 120, 180, 240, 300):
        rad = math.radians(ang)
        pts.append((center_x + a * math.cos(rad), center_y + a * math.sin(rad)))
    return Polygon(pts)


def build_hex_grid(bounds, a: float) -> list[Polygon]:
    """Flat-topped hex grid covering bounds = (minx, miny, maxx, maxy); a is the side length."""
    minx, miny, maxx, maxy = bounds
    horiz = 3 / 2 * a
    vert = math.sqrt(3) * a

    hexes = []
    x = minx - 2 * a
    row = 0

    while x <= maxx + 2 * a:
        y = miny - vert + (row % 2) * (vert / 2)
        while y <= maxy + vert:
            hexes.append(make_hexagon(x, y, a))
            y += vert
        x += horiz
        row += 1

    return hexes


def hex_error_aggregation(
    points_gdf: gpd.GeoDataFrame,
    boundary_shp: str,
    hex_size_m: float,
):
    """Aggregate tile errors onto a hex grid clipped to a boundary layer.

    Projects everything to a UTM CRS, builds a hex grid over the boundary,
    assigns each annotated tile to its hex and computes n/mean/std of
    tile_error per hex.

    Returns (hex_gdf, points_with_hex, boundary_proj).
    """
    boundary = gpd.read_file(boundary_shp)
    if boundary.empty:
        raise ValueError(f"Boundary layer is empty: {boundary_shp}")

    utm_crs = choose_utm_crs_from_gdf(boundary)
    boundary_proj = boundary.to_crs(utm_crs)
    points_proj = points_gdf.to_crs(utm_crs)

    area_union = boundary_proj.geometry.union_all()

    hexes = build_hex_grid(area_union.bounds, hex_size_m / 2.0)
    hex_gdf = gpd.GeoDataFrame(geometry=hexes, crs=utm_crs)
    hex_gdf = hex_gdf[hex_gdf.intersects(area_union)].reset_index(drop=True)
    hex_gdf["hex_id"] = hex_gdf.index

    points_with_hex = gpd.sjoin(
        points_proj,
        hex_gdf[["hex_id", "geometry"]],
        how="left",
        predicate="within",
    )
    points_with_hex = points_with_hex.dropna(subset=["hex_id"]).copy()
    points_with_hex["hex_id"] = points_with_hex["hex_id"].astype(int)

    agg = points_with_hex.groupby("hex_id").agg(
        n_tiles=("tile_error", "size"),
        mean_err=("tile_error", "mean"),
        std_err=("tile_error", "std"),
    ).reset_index()

    hex_gdf = hex_gdf.merge(agg, on="hex_id", how="left")
    hex_gdf["n_tiles"] = hex_gdf["n_tiles"].fillna(0).astype(int)

    return hex_gdf, points_with_hex, boundary_proj


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
