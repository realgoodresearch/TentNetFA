#!/usr/bin/env python3

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
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

HEX_SIZE_M = 1000
Z = 1.96

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# HELPERS
# -----------------------
def choose_utm_crs_from_gdf(gdf):
    lonlat = gdf.to_crs("EPSG:4326")
    union = lonlat.geometry.union_all()
    cx, cy = union.centroid.x, union.centroid.y
    zone = int((cx + 180) / 6) + 1
    epsg = 32600 + zone if cy >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

def make_hexagon(center_x, center_y, a):
    angles = [0, 60, 120, 180, 240, 300]
    pts = []
    for ang in angles:
        rad = math.radians(ang)
        pts.append((center_x + a * math.cos(rad),
                    center_y + a * math.sin(rad)))
    return Polygon(pts)

def build_hex_grid(bounds, a):
    minx, miny, maxx, maxy = bounds
    horiz = 3/2 * a
    vert = math.sqrt(3) * a

    hexes = []
    x = minx - 2*a
    row = 0
    while x <= maxx + 2*a:
        y = miny - vert + (row % 2) * (vert/2)
        while y <= maxy + vert:
            hexes.append(make_hexagon(x, y, a))
            y += vert
        x += horiz
        row += 1
    return hexes

# -----------------------
# LOAD DATA
# -----------------------
print("Loading annotation CSV...")
df = pd.read_csv(ANNOTATION_CSV)
df["tile_error"] = df["model_tent_count"] - df["manual_tent_count"]

points = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
    crs="EPSG:4326"
)

print("Loading boundary...")
boundary = gpd.read_file(BOUNDARY_SHP)

utm_crs = choose_utm_crs_from_gdf(boundary)
boundary = boundary.to_crs(utm_crs)
points = points.to_crs(utm_crs)

area_union = boundary.geometry.union_all()

# -----------------------
# BUILD HEX GRID
# -----------------------
a = HEX_SIZE_M / 2.0
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
    predicate="within"
)

points_with_hex = points_with_hex.dropna(subset=["hex_id"])
points_with_hex["hex_id"] = points_with_hex["hex_id"].astype(int)

# -----------------------
# AGGREGATE PER HEX
# -----------------------
agg = points_with_hex.groupby("hex_id").agg(
    n_tiles=("tile_error", "size"),
    mean_err=("tile_error", "mean"),
    std_err=("tile_error", "std")
).reset_index()

hex_gdf = hex_gdf.merge(agg, on="hex_id", how="left")

hex_gdf["n_tiles"] = hex_gdf["n_tiles"].fillna(0).astype(int)
hex_gdf["mean_err"] = hex_gdf["mean_err"]
hex_gdf["std_err"] = hex_gdf["std_err"]

# -----------------------
# ANALYTIC CI CALCULATION
# -----------------------

# -----------------------
# ANALYTIC CI CALCULATION using hex area -> total tiles
# -----------------------

# area of single sample tile (meters^2)
TILE_SIZE_M = 100.0
TILE_AREA_M2 = TILE_SIZE_M * TILE_SIZE_M

# hex area in m^2 (projected CRS)
hex_gdf["hex_area_m2"] = hex_gdf.geometry.area

# estimate total number of 100x100m tiles in each hex (integer)
hex_gdf["N_total_tiles"] = (hex_gdf["hex_area_m2"] / TILE_AREA_M2).round().astype(int)

# keep sampled count in n_tiles (already computed)
# compute standard error of the sample mean per hex
# use ddof=1 for sample std already computed as std_err
hex_gdf["se_mean"] = hex_gdf["std_err"] / np.sqrt(hex_gdf["n_tiles"])
hex_gdf.loc[hex_gdf["n_tiles"] <= 1, "se_mean"] = np.nan  # not enough info

# 95% CI for the sample mean (per-tile)
Z = 1.96
hex_gdf["ci_lo_mean"] = hex_gdf["mean_err"] - Z * hex_gdf["se_mean"]
hex_gdf["ci_hi_mean"] = hex_gdf["mean_err"] + Z * hex_gdf["se_mean"]

# TOTAL (for the whole hex) using estimated N_total_tiles
# T = N_total_tiles * mean_err
# SE_total = N_total_tiles * se_mean
hex_gdf["total_err_est"] = hex_gdf["mean_err"] * hex_gdf["N_total_tiles"]
hex_gdf["se_total_est"] = hex_gdf["N_total_tiles"] * hex_gdf["se_mean"]

# 95% CI for the total
hex_gdf["ci_lo_total"] = hex_gdf["total_err_est"] - Z * hex_gdf["se_total_est"]
hex_gdf["ci_hi_total"] = hex_gdf["total_err_est"] + Z * hex_gdf["se_total_est"]

# If a hex has no samples, keep sample-based fields NaN and N_total_tiles still reported
# Round values for export
round_cols = [
    "hex_area_m2", "N_total_tiles",
    "mean_err","std_err","se_mean",
    "ci_lo_mean","ci_hi_mean",
    "total_err_est","se_total_est",
    "ci_lo_total","ci_hi_total"
]

for c in round_cols:
    if c in hex_gdf.columns and pd.api.types.is_numeric_dtype(hex_gdf[c]):
        # preserve integer for N_total_tiles
        if c == "N_total_tiles":
            hex_gdf[c] = hex_gdf[c].astype(int)
        else:
            hex_gdf[c] = hex_gdf[c].round(3)
# -----------------------
# EXPORT
# -----------------------
print("Saving shapefile...")
hex_gdf.to_file(OUT_HEX_SHP)
hex_gdf.drop(columns="geometry").to_csv(OUT_HEX_CSV, index=False)

# -----------------------
# PLOT MAP
# -----------------------
fig, ax = plt.subplots(figsize=(12,12))
boundary.boundary.plot(ax=ax, color="black", linewidth=0.5)

vmin = np.nanmin(hex_gdf.loc[hex_gdf["n_tiles"]>0,"mean_err"])
vmax = np.nanmax(hex_gdf.loc[hex_gdf["n_tiles"]>0,"mean_err"])

hex_gdf.plot(
    column="mean_err",
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax,
    ax=ax,
    edgecolor="k",
    linewidth=0.1
)

plt.title("Hex Mean Tile Error")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT_MAP_PNG, dpi=200)
plt.close()

print("Done.")