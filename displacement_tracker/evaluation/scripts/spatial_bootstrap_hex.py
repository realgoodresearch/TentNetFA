#!/usr/bin/env python3
"""
spatial_bootstrap_hex.py

Create hex grid, compute per-hex tile-error summaries, compute bootstrap CIs
(per-hex and spatial-block for region total), export shapefile and map.
"""

import os
import math
from typing import Tuple, Optional

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
    """
    Choose a UTM CRS (EPSG) based on centroid longitude/latitude of the layer.
    """
    if gdf.crs is None:
        raise ValueError("Boundary layer has no CRS.")

    lonlat = gdf.to_crs("EPSG:4326")
    union = lonlat.geometry.union_all() if hasattr(lonlat.geometry, "union_all") else lonlat.unary_union
    cx, cy = union.centroid.x, union.centroid.y

    zone = int((cx + 180) / 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def make_hexagon(center_x: float, center_y: float, a: float) -> Polygon:
    """
    Create a regular hexagon with side length a centered at (center_x, center_y).
    """
    angles = [0, 60, 120, 180, 240, 300]
    pts = []
    for ang in angles:
        rad = math.radians(ang)
        x = center_x + a * math.cos(rad)
        y = center_y + a * math.sin(rad)
        pts.append((x, y))
    return Polygon(pts)


def build_hex_grid(bounds, a: float):
    """
    Build a flat-topped hex grid covering bbox bounds = (minx, miny, maxx, maxy),
    where a is the hex side length in meters.
    """
    minx, miny, maxx, maxy = bounds
    hex_width = 2 * a
    horiz = 3 / 2 * a
    vert = math.sqrt(3) * a

    hexes = []
    x = minx - hex_width
    row = 0

    while x <= maxx + hex_width:
        y0 = miny - vert
        y = y0 + (row % 2) * (vert / 2)
        while y <= maxy + vert:
            hexes.append(make_hexagon(x, y, a))
            y += vert
        x += horiz
        row += 1

    return hexes


# ==========================================================
# CORE FUNCTION
# ==========================================================

def spatial_bootstrap_hex(
    annotation_csv: str = "manual_annotation_results_with_new_model.csv",
    boundary_shp: str = "gaza_boundaries/GazaStrip_MunicipalBoundaries.shp",
    output_dir: str = "results",
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
    hex_size_m: float = 1000.0,
    min_samples_per_hex: int = 5,
    b_hex: int = 1000,
    b_block: int = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Create hex grid, compute per-hex tile-error summaries, compute bootstrap CIs
    (per-hex and spatial-block for region total), export shapefile and map.

    Returns:
        hex_gdf
    """
    os.makedirs(output_dir, exist_ok=True)

    out_hex_shp = os.path.join(output_dir, "hex_with_cis.shp")
    out_hex_csv = os.path.join(output_dir, "hex_with_cis.csv")
    out_map_png = os.path.join(output_dir, "hex_mean_error_map.png")
    out_summary_csv = os.path.join(output_dir, "spatial_bootstrap_summary.csv")

    np.random.seed(random_seed)

    # -----------------------
    # LOAD DATA
    # -----------------------
    print("Loading annotation CSV...")
    df = pd.read_csv(annotation_csv)

    required = {"latitude", "longitude", manual_column, model_column}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    df["tile_error"] = df[model_column] - df[manual_column]
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )

    print("Loading boundary shapefile...")
    boundary = gpd.read_file(boundary_shp)
    if boundary.empty:
        raise ValueError("Boundary shapefile empty.")

    utm_crs = choose_utm_crs_from_gdf(boundary)
    print("Projecting to UTM CRS:", utm_crs.to_string())

    boundary_proj = boundary.to_crs(utm_crs)
    points_proj = points_gdf.to_crs(utm_crs)

    area_union = boundary_proj.geometry.union_all() if hasattr(boundary_proj.geometry, "union_all") else boundary_proj.unary_union

    # -----------------------
    # MAKE HEX GRID
    # -----------------------
    a = hex_size_m / 2.0

    print("Building hex grid...")
    bbox = area_union.bounds
    hex_polys = build_hex_grid(bbox, a)

    hex_gdf = gpd.GeoDataFrame(geometry=hex_polys, crs=utm_crs)
    hex_gdf = hex_gdf[hex_gdf.intersects(area_union)].reset_index(drop=True)
    hex_gdf["hex_id"] = hex_gdf.index

    print(f"Total hexes covering area: {len(hex_gdf)}")

    # -----------------------
    # SPATIAL JOIN: points -> hex
    # -----------------------
    points_with_hex = gpd.sjoin(
        points_proj,
        hex_gdf[["hex_id", "geometry"]],
        how="left",
        predicate="within",
    )

    points_with_hex = points_with_hex.dropna(subset=["hex_id"]).copy()
    points_with_hex["hex_id"] = points_with_hex["hex_id"].astype(int)

    print("Points assigned to hexes:", len(points_with_hex))

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
    hex_gdf["mean_err"] = hex_gdf["mean_err"].astype(float)
    hex_gdf["std_err"] = hex_gdf["std_err"].astype(float)

    # -----------------------
    # HEX-LEVEL BOOTSTRAP
    # -----------------------
    def hex_bootstrap_ci(points_df: pd.DataFrame, hex_id: int, b: int = b_hex, alpha: float = 0.05):
        """
        Bootstrap CI for mean error within a hex by resampling tiles in the hex.
        Returns (lo, hi). If hex has < min_samples_per_hex returns (nan, nan).
        """
        sub = points_df[points_df["hex_id"] == hex_id]
        n = len(sub)
        if n < min_samples_per_hex:
            return (np.nan, np.nan)

        arr = sub["tile_error"].values
        boot_means = np.empty(b)

        for i in range(b):
            sample = np.random.choice(arr, size=n, replace=True)
            boot_means[i] = sample.mean()

        lo = np.percentile(boot_means, 100 * (alpha / 2.0))
        hi = np.percentile(boot_means, 100 * (1 - alpha / 2.0))
        return (lo, hi)

    print("Computing hex-level bootstrap CIs...")
    ci_lo = []
    ci_hi = []
    for _, row in hex_gdf.iterrows():
        hex_id = row["hex_id"]
        lo, hi = hex_bootstrap_ci(points_with_hex, hex_id, b=b_hex)
        ci_lo.append(lo)
        ci_hi.append(hi)

    hex_gdf["ci_lo"] = ci_lo
    hex_gdf["ci_hi"] = ci_hi

    # -----------------------
    # SPATIAL (BLOCK) BOOTSTRAP FOR REGION TOTAL
    # -----------------------
    blocks = hex_gdf[hex_gdf["n_tiles"] > 0].copy()
    if blocks.empty:
        raise ValueError("No hexes contain annotations; cannot perform block bootstrap.")

    blocks["total_err_hex"] = blocks["mean_err"] * blocks["n_tiles"]
    nblocks = len(blocks)

    print(f"Performing spatial block bootstrap across {nblocks} populated hex blocks (B={b_block}) ...")
    totals_arr = blocks["total_err_hex"].values
    ns_arr = blocks["n_tiles"].values

    block_totals = np.empty(b_block)
    for i in range(b_block):
        sampled_idx = np.random.choice(np.arange(nblocks), size=nblocks, replace=True)
        block_totals[i] = totals_arr[sampled_idx].sum()

    region_total_est = float(totals_arr.sum())
    alpha = 0.05
    region_lo = float(np.percentile(block_totals, 100 * (alpha / 2.0)))
    region_hi = float(np.percentile(block_totals, 100 * (1 - alpha / 2.0)))

    print("Region total error estimate (from observed blocks):", region_total_est)
    print(f"Spatial-block bootstrap 95% CI for regional total error: [{region_lo:.1f}, {region_hi:.1f}]")

    total_tiles_in_blocks = int(ns_arr.sum())
    region_mean_obs = region_total_est / total_tiles_in_blocks
    mean_boots = block_totals / total_tiles_in_blocks
    region_mean_lo = float(np.percentile(mean_boots, 100 * (alpha / 2.0)))
    region_mean_hi = float(np.percentile(mean_boots, 100 * (1 - alpha / 2.0)))

    print("Region mean tile error (observed):", region_mean_obs)
    print(
        f"Spatial-block bootstrap 95% CI for region mean tile error: "
        f"[{region_mean_lo:.3f}, {region_mean_hi:.3f}]"
    )

    # -----------------------
    # EXPORT HEX SHAPE & CSV
    # -----------------------
    hex_gdf["mean_err"] = hex_gdf["mean_err"].round(3)
    hex_gdf["std_err"] = hex_gdf["std_err"].round(3)
    hex_gdf["ci_lo"] = hex_gdf["ci_lo"].round(3)
    hex_gdf["ci_hi"] = hex_gdf["ci_hi"].round(3)

    print("Saving hex shapefile and CSV...")
    hex_gdf.to_file(out_hex_shp)
    hex_gdf.drop(columns="geometry").to_csv(out_hex_csv, index=False)

    print("Saved:", out_hex_shp, out_hex_csv)

    # -----------------------
    # PLOT MAP
    # -----------------------
    print("Creating map...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    boundary_proj.boundary.plot(ax=ax, linewidth=0.6, color="black")

    hex_gdf_plot = hex_gdf.copy()
    hex_gdf_plot["plot_val"] = hex_gdf_plot["mean_err"].fillna(0.0)

    valid = hex_gdf_plot["n_tiles"] > 0
    if valid.any():
        vmin = np.nanmin(hex_gdf_plot.loc[valid, "plot_val"])
        vmax = np.nanmax(hex_gdf_plot.loc[valid, "plot_val"])
    else:
        vmin, vmax = -1.0, 1.0

    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    hex_gdf_plot.plot(
        column="plot_val",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        edgecolor="k",
        linewidth=0.1,
    )

    sig = hex_gdf[
        (hex_gdf["n_tiles"] >= min_samples_per_hex)
        & ((hex_gdf["ci_lo"] > 0) | (hex_gdf["ci_hi"] < 0))
    ]
    if not sig.empty:
        sig.boundary.plot(ax=ax, color="black", linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean tile error")

    ax.set_title("Hex-aggregated Mean Tile Error and Significant Hexes")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_map_png, dpi=200)
    plt.close()
    print("Saved map to", out_map_png)

    # -----------------------
    # SUMMARY OUTPUT
    # -----------------------
    summary = {
        "region_total_error_estimate": region_total_est,
        "region_total_error_ci_lo": region_lo,
        "region_total_error_ci_hi": region_hi,
        "region_mean_error_estimate": region_mean_obs,
        "region_mean_error_ci_lo": region_mean_lo,
        "region_mean_error_ci_hi": region_mean_hi,
        "n_hex_blocks": int(nblocks),
        "n_tiles_used": int(total_tiles_in_blocks),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_summary_csv, index=False)
    print("Saved summary to", out_summary_csv)

    print("Done.")
    return hex_gdf


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    spatial_bootstrap_hex(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        boundary_shp="gaza_boundaries/GazaStrip_MunicipalBoundaries.shp",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
        hex_size_m=1000.0,
        min_samples_per_hex=5,
        b_hex=1000,
        b_block=2000,
        random_seed=42,
    )
