#!/usr/bin/env python3
"""
spatial_bootstrap_hex.py

Create hex grid, compute per-hex tile-error summaries, compute bootstrap CIs
(per-hex and spatial-block for region total), export shapefile and map.

Requires: geopandas, pandas, numpy, matplotlib, shapely, pyproj
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.affinity import translate
from pyproj import CRS

# -----------------------
# CONFIG
# -----------------------
ANNOTATION_CSV = "displacement_tracker/evaluation/manual_annotation_results.csv"
BOUNDARY_SHP = "gaza_boundaries/GazaStrip_MunicipalBoundaries.shp"
OUT_DIR = "displacement_tracker/evaluation/results"
OUT_HEX_SHP = os.path.join(OUT_DIR, "hex_with_cis.shp")
OUT_HEX_CSV = os.path.join(OUT_DIR, "hex_with_cis.csv")
OUT_MAP_PNG = os.path.join(OUT_DIR, "hex_mean_error_map.png")

HEX_SIZE_M = 1000             # distance across flats ~ hex width in meters (adjust)
MIN_SAMPLES_PER_HEX = 5      # min tiles in a hex to compute hex-level bootstrap CI
B_HEX = 1000                 # bootstrap iterations for hex CI (resample tiles within hex)
B_BLOCK = 2000               # bootstrap iterations for spatial-block CI (resample hexes)
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# -----------------------
# HELPERS
# -----------------------
def choose_utm_crs_from_gdf(gdf):
    """
    Choose a UTM CRS (EPSG) based on centroid longitude of the gdf's bounds.
    Returns pyproj CRS object.
    """
    # Ensure gdf in lat/lon
    if gdf.crs is None:
        raise ValueError("Boundary layer has no CRS.")
    lonlat = gdf.to_crs("EPSG:4326")
    cx = lonlat.unary_union.centroid.x
    cy = lonlat.unary_union.centroid.y
    zone = int((cx + 180) / 6) + 1
    if cy >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return CRS.from_epsg(epsg)


def make_hexagon(center_x, center_y, a):
    """
    Create a regular hexagon with side length a (meters) centered at (center_x, center_y).
    Returns shapely Polygon.
    """
    # flat-topped hexagon coordinates relative to center
    # horizontal distance between centers = 3/2 * a
    # vertical distance = sqrt(3) * a
    angles = [0, 60, 120, 180, 240, 300]
    pts = []
    for ang in angles:
        rad = math.radians(ang)
        x = center_x + a * math.cos(rad)
        y = center_y + a * math.sin(rad)
        pts.append((x, y))
    return Polygon(pts)


def build_hex_grid(bounds, a):
    """
    Build a flat-topped hex grid covering the bbox `bounds` = (minx,miny,maxx,maxy)
    where a = side length (meters).
    Returns list of shapely Polygons (hexes).
    """
    minx, miny, maxx, maxy = bounds
    hex_width = 2 * a
    horiz = 3/2 * a
    vert = math.sqrt(3) * a

    hexes = []
    # start positions
    x = minx - hex_width
    row = 0
    while x <= maxx + hex_width:
        # staggered rows
        y0 = miny - vert
        y = y0 + (row % 2) * (vert / 2)
        while y <= maxy + vert:
            hex_poly = make_hexagon(x, y, a)
            hexes.append(hex_poly)
            y += vert
        x += horiz
        row += 1

    return hexes


# -----------------------
# LOAD DATA
# -----------------------
print("Loading annotation CSV...")
df = pd.read_csv(ANNOTATION_CSV)
required = {"latitude", "longitude", "manual_tent_count", "model_tent_count"}
if not required.issubset(df.columns):
    raise ValueError(f"Annotation CSV missing required columns: {required - set(df.columns)}")

df["tile_error"] = df["model_tent_count"] - df["manual_tent_count"]
points_gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])], crs="EPSG:4326")

print("Loading boundary shapefile...")
boundary = gpd.read_file(BOUNDARY_SHP)
if boundary.empty:
    raise ValueError("Boundary shapefile empty.")

# choose projected CRS (UTM) for metric operations
utm_crs = choose_utm_crs_from_gdf(boundary)
print("Projecting to UTM CRS:", utm_crs.to_string())

boundary_proj = boundary.to_crs(utm_crs)
points_proj = points_gdf.to_crs(utm_crs)

# union boundary to single polygon for grid clipping
area_union = boundary_proj.unary_union

# -----------------------
# MAKE HEX GRID
# -----------------------
# side length a: choose from HEX_SIZE_M which we interpret as approximate width across flats.
# For flat-topped hex, side length a relates to width across flats = 2*a
a = HEX_SIZE_M / 2.0

print("Building hex grid (this may take a few seconds)...")
bbox = area_union.bounds  # minx,miny,maxx,maxy
hex_polys = build_hex_grid(bbox, a)

hex_gdf = gpd.GeoDataFrame(geometry=hex_polys, crs=utm_crs)

# keep only hexes that intersect the boundary
hex_gdf = hex_gdf[hex_gdf.intersects(area_union)].reset_index(drop=True)

# clip hexes to boundary optionally (we'll keep full hex shapes but you may clip if desired)
# hex_gdf['geometry'] = hex_gdf.geometry.intersection(area_union)

print(f"Total hexes covering area: {len(hex_gdf)}")

# -----------------------
# SPATIAL JOIN: points -> hex
# -----------------------
# Ensure hex_gdf has a clean integer id column
hex_gdf = hex_gdf.reset_index(drop=True)
hex_gdf["hex_id"] = hex_gdf.index

# Spatial join
points_with_hex = gpd.sjoin(
    points_proj,
    hex_gdf[["hex_id", "geometry"]],
    how="left",
    predicate="within"
)

# Drop points not assigned to a hex
points_with_hex = points_with_hex.dropna(subset=["hex_id"]).copy()
points_with_hex["hex_id"] = points_with_hex["hex_id"].astype(int)

print("Points assigned to hexes:", len(points_with_hex))

# -----------------------
# AGGREGATE PER HEX
# -----------------------
agg = points_with_hex.groupby("hex_id").agg(
    n_tiles = ("tile_error", "size"),
    mean_err = ("tile_error", "mean"),
    std_err = ("tile_error", "std")
).reset_index()

# join agg stats back to hex_gdf
hex_gdf = hex_gdf.merge(agg, on="hex_id", how="left")

# fill missing numeric with zeros or NaN appropriately
hex_gdf["n_tiles"] = hex_gdf["n_tiles"].fillna(0).astype(int)
hex_gdf["mean_err"] = hex_gdf["mean_err"].astype(float)
hex_gdf["std_err"] = hex_gdf["std_err"].astype(float)

# -----------------------
# HEX-LEVEL BOOTSTRAP (resample tiles within hex)
# -----------------------
def hex_bootstrap_ci(points_df, hex_id, b=B_HEX, alpha=0.05):
    """
    Given points_with_hex (projected, with 'tile_error' and 'hex_id'),
    compute bootstrap CI for mean error within the hex by resampling tiles in the hex.
    Returns (lo, hi). If hex has < MIN_SAMPLES_PER_HEX returns (nan,nan).
    """
    sub = points_df[points_df["hex_id"] == hex_id]
    n = len(sub)
    if n < MIN_SAMPLES_PER_HEX:
        return (np.nan, np.nan)
    arr = sub["tile_error"].values
    boot_means = np.empty(b)
    for i in range(b):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, 100 * (alpha / 2.0))
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2.0))
    return (lo, hi)

print("Computing hex-level bootstrap CIs (this may take some time)...")
ci_lo = []
ci_hi = []
for hid, row in hex_gdf.iterrows():
    hex_id = row["hex_id"]
    lo, hi = hex_bootstrap_ci(points_with_hex, hex_id, b=B_HEX)
    ci_lo.append(lo)
    ci_hi.append(hi)

hex_gdf["ci_lo"] = ci_lo
hex_gdf["ci_hi"] = ci_hi

# -----------------------
# SPATIAL (BLOCK) BOOTSTRAP FOR REGION TOTAL
# -----------------------
# For block bootstrap, each hex is a block. For hexes with n_tiles>0 compute total_error_hex = mean_err * n_tiles.
blocks = hex_gdf[hex_gdf["n_tiles"] > 0].copy()
if blocks.empty:
    raise ValueError("No hexes contain annotations; cannot perform block bootstrap.")

blocks["total_err_hex"] = blocks["mean_err"] * blocks["n_tiles"]
block_ids = blocks["hex_id"].values
nblocks = len(block_ids)

print(f"Performing spatial block bootstrap across {nblocks} populated hex blocks (B={B_BLOCK}) ...")
block_totals = np.empty(B_BLOCK)
# for each iteration, sample nblocks hexes with replacement, sum total_err_hex
totals_arr = blocks["total_err_hex"].values
ns_arr = blocks["n_tiles"].values
for i in range(B_BLOCK):
    sampled_idx = np.random.choice(np.arange(nblocks), size=nblocks, replace=True)
    # sum total_err_hex of sampled blocks
    block_totals[i] = totals_arr[sampled_idx].sum()

# Compute region-level estimates
region_total_est = totals_arr.sum()
alpha = 0.05
region_lo = np.percentile(block_totals, 100*(alpha/2.0))
region_hi = np.percentile(block_totals, 100*(1-alpha/2.0))

print("Region total error estimate (from observed blocks):", region_total_est)
print(f"Spatial-block bootstrap 95% CI for regional total error: [{region_lo:.1f}, {region_hi:.1f}]")

# Also compute region mean error and CI (mean per tile)
# mean_per_tile_obs = region_total_est / total_tiles_in_blocks
total_tiles_in_blocks = ns_arr.sum()
region_mean_obs = region_total_est / total_tiles_in_blocks
mean_boots = block_totals / total_tiles_in_blocks
region_mean_lo = np.percentile(mean_boots, 100*(alpha/2.0))
region_mean_hi = np.percentile(mean_boots, 100*(1-alpha/2.0))

print("Region mean tile error (observed):", region_mean_obs)
print(f"Spatial-block bootstrap 95% CI for region mean tile error: [{region_mean_lo:.3f}, {region_mean_hi:.3f}]")

# -----------------------
# EXPORT HEX SHAPE & CSV
# -----------------------
# Add helpful fields for export (rounding)
hex_gdf["mean_err"] = hex_gdf["mean_err"].round(3)
hex_gdf["std_err"] = hex_gdf["std_err"].round(3)
hex_gdf["ci_lo"] = hex_gdf["ci_lo"].round(3)
hex_gdf["ci_hi"] = hex_gdf["ci_hi"].round(3)

# Save shapefile + csv
print("Saving hex shapefile and CSV...")
# drop geometry columns potentially problematic (shapefile field name length); geopandas handles names automatically
hex_gdf.to_file(OUT_HEX_SHP)
hex_gdf.drop(columns="geometry").to_csv(OUT_HEX_CSV, index=False)

print("Saved:", OUT_HEX_SHP, OUT_HEX_CSV)

# -----------------------
# PLOT MAP: mean error with CI significance
# -----------------------
print("Creating map...")
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
# plot boundary
boundary_proj.boundary.plot(ax=ax, linewidth=0.6, color="black")

# plot hex mean
# hexes without data will be light gray
hex_gdf_plot = hex_gdf.copy()
hex_gdf_plot["plot_val"] = hex_gdf_plot["mean_err"].fillna(0.0)

# define colormap centered at zero
vmin = np.nanmin(hex_gdf_plot.loc[hex_gdf_plot["n_tiles"]>0, "plot_val"])
vmax = np.nanmax(hex_gdf_plot.loc[hex_gdf_plot["n_tiles"]>0, "plot_val"])
# if vmin==vmax, expand a little
if np.isclose(vmin, vmax):
    vmin -= 1.0
    vmax += 1.0

cax = hex_gdf_plot.plot(column="plot_val", cmap="RdBu_r", vmin=vmin, vmax=vmax, ax=ax, edgecolor="k", linewidth=0.1)

# highlight hexes whose CI does NOT contain zero (significant)
sig = hex_gdf[(hex_gdf["n_tiles"] >= MIN_SAMPLES_PER_HEX) & ((hex_gdf["ci_lo"] > 0) | (hex_gdf["ci_hi"] < 0))]
if not sig.empty:
    sig.boundary.plot(ax=ax, color="black", linewidth=1.0)

# colorbar
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Mean tile error (model - manual)")

ax.set_title("Hex-aggregated Mean Tile Error and Significant Hexes")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUT_MAP_PNG, dpi=200)
plt.close()
print("Saved map to", OUT_MAP_PNG)

# -----------------------
# SUMMARY OUTPUT
# -----------------------
summary = {
    "region_total_error_estimate": float(region_total_est),
    "region_total_error_ci_lo": float(region_lo),
    "region_total_error_ci_hi": float(region_hi),
    "region_mean_error_estimate": float(region_mean_obs),
    "region_mean_error_ci_lo": float(region_mean_lo),
    "region_mean_error_ci_hi": float(region_mean_hi),
    "n_hex_blocks": int(nblocks),
    "n_tiles_used": int(total_tiles_in_blocks)
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(OUT_DIR, "spatial_bootstrap_summary.csv"), index=False)
print("Saved summary to", os.path.join(OUT_DIR, "spatial_bootstrap_summary.csv"))

print("Done.")