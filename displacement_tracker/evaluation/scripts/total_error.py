#!/usr/bin/env python3

import os
import math
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from pyproj import CRS


# ==========================================================
# HELPERS
# ==========================================================

def choose_utm_crs_from_gdf(gdf: gpd.GeoDataFrame) -> CRS:
    lonlat = gdf.to_crs("EPSG:4326")
    union = lonlat.geometry.union_all() if hasattr(lonlat.geometry, "union_all") else lonlat.unary_union
    cx, cy = union.centroid.x, union.centroid.y
    zone = int((cx + 180) / 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def make_hexagon(center_x: float, center_y: float, a: float) -> Polygon:
    angles = [0, 60, 120, 180, 240, 300]
    pts = []
    for ang in angles:
        rad = math.radians(ang)
        pts.append((center_x + a * math.cos(rad), center_y + a * math.sin(rad)))
    return Polygon(pts)


def build_hex_grid(bounds, a: float):
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


# ==========================================================
# CORE FUNCTION
# ==========================================================

def evaluate_total_error(
    annotation_csv: str = "manual_annotation_results_with_new_model.csv",
    boundary_shp: str = "gaza_boundaries/GazaStrip_MunicipalBoundaries.shp",
    output_dir: str = "results",
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
    hex_size_m: float = 1000.0,
    z: float = 1.96,
):
    """
    Build a hexagonal aggregation of tile-level error and export:
        - hex_with_cis.shp
        - hex_with_cis.csv
        - hex_mean_error_map.png

    Returns:
        hex_gdf
    """

    os.makedirs(output_dir, exist_ok=True)

    out_hex_shp = os.path.join(output_dir, "hex_with_cis.shp")
    out_hex_csv = os.path.join(output_dir, "hex_with_cis.csv")
    out_map_png = os.path.join(output_dir, "hex_mean_error_map.png")

    # -----------------------
    # LOAD DATA
    # -----------------------
    print("Loading annotation CSV...")
    df = pd.read_csv(annotation_csv)

    required_cols = {"latitude", "longitude", manual_column, model_column}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    df["tile_error"] = df[model_column] - df[manual_column]

    points = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    print("Loading boundary...")
    boundary = gpd.read_file(boundary_shp)

    utm_crs = choose_utm_crs_from_gdf(boundary)
    boundary = boundary.to_crs(utm_crs)
    points = points.to_crs(utm_crs)

    area_union = boundary.geometry.union_all() if hasattr(boundary.geometry, "union_all") else boundary.unary_union

    # -----------------------
    # BUILD HEX GRID
    # -----------------------
    a = hex_size_m / 2.0
    bbox = area_union.bounds
    hexes = build_hex_grid(bbox, a)

    hex_gdf = gpd.GeoDataFrame(geometry=hexes, crs=utm_crs)
    hex_gdf = hex_gdf[hex_gdf.intersects(area_union)].reset_index(drop=True)
    hex_gdf["hex_id"] = hex_gdf.index

    print(f"Total hexes: {len(hex_gdf)}")

    # -----------------------
    # SPATIAL JOIN
    # -----------------------
    points_with_hex = gpd.sjoin(
        points,
        hex_gdf[["hex_id", "geometry"]],
        how="left",
        predicate="within",
    )

    points_with_hex = points_with_hex.dropna(subset=["hex_id"]).copy()
    points_with_hex["hex_id"] = points_with_hex["hex_id"].astype(int)

    # -----------------------
    # AGGREGATE PER HEX
    # -----------------------
    agg = points_with_hex.groupby("hex_id").agg(
        n_tiles=("tile_error", "size"),
        mean_err=("tile_error", "mean"),
        std_err=("tile_error", "std"),
    ).reset_index()

    hex_gdf = hex_gdf.merge(agg, on="hex_id", how="left")

    hex_gdf["n_tiles"] = hex_gdf["n_tiles"].fillna(0).astype(int)

    # -----------------------
    # ANALYTIC CI CALCULATION
    # -----------------------

    TILE_SIZE_M = 100.0
    TILE_AREA_M2 = TILE_SIZE_M * TILE_SIZE_M

    hex_gdf["hex_area_m2"] = hex_gdf.geometry.area
    hex_gdf["N_total_tiles"] = (hex_gdf["hex_area_m2"] / TILE_AREA_M2).round().astype(int)

    hex_gdf["se_mean"] = hex_gdf["std_err"] / np.sqrt(hex_gdf["n_tiles"])
    hex_gdf.loc[hex_gdf["n_tiles"] <= 1, "se_mean"] = np.nan

    hex_gdf["ci_lo_mean"] = hex_gdf["mean_err"] - z * hex_gdf["se_mean"]
    hex_gdf["ci_hi_mean"] = hex_gdf["mean_err"] + z * hex_gdf["se_mean"]

    hex_gdf["total_err_est"] = hex_gdf["mean_err"] * hex_gdf["N_total_tiles"]
    hex_gdf["se_total_est"] = hex_gdf["N_total_tiles"] * hex_gdf["se_mean"]

    hex_gdf["ci_lo_total"] = hex_gdf["total_err_est"] - z * hex_gdf["se_total_est"]
    hex_gdf["ci_hi_total"] = hex_gdf["total_err_est"] + z * hex_gdf["se_total_est"]

    round_cols = [
        "hex_area_m2",
        "mean_err",
        "std_err",
        "se_mean",
        "ci_lo_mean",
        "ci_hi_mean",
        "total_err_est",
        "se_total_est",
        "ci_lo_total",
        "ci_hi_total",
    ]
    for c in round_cols:
        if c in hex_gdf.columns and pd.api.types.is_numeric_dtype(hex_gdf[c]):
            hex_gdf[c] = hex_gdf[c].round(3)

    # -----------------------
    # EXPORT
    # -----------------------
    print("Saving shapefile...")
    hex_gdf.to_file(out_hex_shp)
    hex_gdf.drop(columns="geometry").to_csv(out_hex_csv, index=False)

    # -----------------------
    # PLOT MAP
    # -----------------------
    valid = hex_gdf["n_tiles"] > 0
    if valid.any():
        vmin = np.nanmin(hex_gdf.loc[valid, "mean_err"])
        vmax = np.nanmax(hex_gdf.loc[valid, "mean_err"])
    else:
        vmin, vmax = -1, 1

    fig, ax = plt.subplots(figsize=(12, 12))
    boundary.boundary.plot(ax=ax, color="black", linewidth=0.5)

    hex_gdf.plot(
        column="mean_err",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        edgecolor="k",
        linewidth=0.1,
    )

    plt.title("Hex Mean Tile Error")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_map_png, dpi=200)
    plt.close()

    print("Done.")
    return hex_gdf


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    evaluate_total_error(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        boundary_shp="gaza_boundaries/GazaStrip_MunicipalBoundaries.shp",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
        hex_size_m=1000.0,
        z=1.96,
    )
