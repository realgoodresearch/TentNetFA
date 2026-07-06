"""Shared helpers for the model-evaluation scripts.

Every analysis in this package follows the same pattern: load the annotated
tiles, compute per-group mean tile error with a 95% CI, write a CSV and a
bar plot. The hex-grid analyses additionally aggregate tile errors onto a
hexagonal grid clipped to a boundary layer. Those shared steps live here so
each evaluate_* module only expresses how tiles are grouped.
"""

import math
import os

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import Polygon

Z_95 = 1.96


# ==========================================================
# ANNOTATION LOADING
# ==========================================================

def load_annotations(
    annotation_csv: str,
    manual_column: str,
    model_column: str,
    extra_columns: tuple = (),
) -> pd.DataFrame:
    """Read the annotation CSV, validate columns and add a tile_error column."""
    df = pd.read_csv(annotation_csv)

    required = {manual_column, model_column, *extra_columns}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    df["tile_error"] = df[model_column] - df[manual_column]
    return df


def load_annotation_points(
    annotation_csv: str,
    manual_column: str,
    model_column: str,
    extra_columns: tuple = (),
) -> gpd.GeoDataFrame:
    """Like load_annotations, but returns tile centroids as an EPSG:4326 GeoDataFrame."""
    df = load_annotations(
        annotation_csv,
        manual_column,
        model_column,
        extra_columns=("latitude", "longitude", *extra_columns),
    )
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )


# ==========================================================
# ERROR SUMMARIES
# ==========================================================

def mean_error_ci(errors, z: float = Z_95) -> dict | None:
    """Mean error with a normal-approximation CI. None when there are no values."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    n = len(errors)
    if n == 0:
        return None

    mean = float(np.mean(errors))
    margin = z * float(np.std(errors, ddof=1)) / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean_tile_error": mean,
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
# PLOTTING
# ==========================================================

def _ci_whiskers(results_df: pd.DataFrame) -> np.ndarray:
    means = results_df["mean_tile_error"].values
    lower_err = np.maximum(0, means - results_df["ci_lower"].values)
    upper_err = np.maximum(0, results_df["ci_upper"].values - means)
    return np.vstack((lower_err, upper_err))


def plot_error_bars(
    results_df: pd.DataFrame,
    label_column: str,
    title: str,
    output_plot: str,
    value_label: str = "Mean Tile-Level Prediction Error",
    figsize: tuple = (8, 5),
    rotate_labels: bool = False,
    horizontal: bool = False,
) -> None:
    """Bar plot of mean tile error per group with 95% CI whiskers."""
    if results_df.empty:
        print(f"WARNING: no groups to plot for {output_plot}")
        return

    plt.figure(figsize=figsize)

    positions = np.arange(len(results_df))
    means = results_df["mean_tile_error"].values
    whiskers = _ci_whiskers(results_df)

    labels = [
        f"{name} (n={n})"
        for name, n in zip(results_df[label_column], results_df["num_tiles"])
    ]

    if horizontal:
        plt.barh(positions, means, xerr=whiskers, capsize=5)
        plt.yticks(positions, labels)
        plt.xlabel(value_label)
    else:
        plt.bar(positions, means, yerr=whiskers, capsize=5)
        if rotate_labels:
            plt.xticks(positions, labels, rotation=45, ha="right")
        else:
            plt.xticks(positions, labels)
        plt.ylabel(value_label)
        plt.axhline(0, linestyle="--")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()


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


def plot_hex_error_map(
    hex_gdf: gpd.GeoDataFrame,
    boundary_proj: gpd.GeoDataFrame,
    output_png: str,
    title: str,
    significant: gpd.GeoDataFrame | None = None,
) -> None:
    """Choropleth of mean_err per hex over the boundary outline."""
    fig, ax = plt.subplots(figsize=(12, 12))
    boundary_proj.boundary.plot(ax=ax, color="black", linewidth=0.5)

    plot_gdf = hex_gdf.copy()
    plot_gdf["plot_val"] = plot_gdf["mean_err"].fillna(0.0)

    valid = plot_gdf["n_tiles"] > 0
    if valid.any():
        vmin = float(np.nanmin(plot_gdf.loc[valid, "plot_val"]))
        vmax = float(np.nanmax(plot_gdf.loc[valid, "plot_val"]))
    else:
        vmin, vmax = -1.0, 1.0
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    plot_gdf.plot(
        column="plot_val",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        edgecolor="k",
        linewidth=0.1,
    )

    if significant is not None and not significant.empty:
        significant.boundary.plot(ax=ax, color="black", linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean tile error")

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
