"""Hex-aggregated tile error with analytic (normal-approximation) CIs."""

import os

import numpy as np
import pandas as pd

from displacement_tracker.evaluation.scripts.plots import plot_hex_error_map
from displacement_tracker.evaluation.scripts.common import (
    Z_95,
    ensure_output_dir,
    hex_error_aggregation,
    load_annotation_points,
)

TILE_SIZE_M = 100.0
TILE_AREA_M2 = TILE_SIZE_M * TILE_SIZE_M


def evaluate_total_error(
    annotation_csv: str,
    boundary_shp: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
    hex_size_m: float = 1000.0,
    z: float = Z_95,
):
    """
    Build a hexagonal aggregation of tile-level error with analytic CIs for
    the per-hex mean and the extrapolated per-hex total.

    Outputs:
        - hex_analytic_cis.shp
        - hex_analytic_cis.csv
        - hex_analytic_mean_error_map.png

    (Named distinctly from the spatial_bootstrap_hex outputs so the two hex
    analyses can share an output directory.)
    """
    ensure_output_dir(output_dir)

    out_hex_shp = os.path.join(output_dir, "hex_analytic_cis.shp")
    out_hex_csv = os.path.join(output_dir, "hex_analytic_cis.csv")
    out_map_png = os.path.join(output_dir, "hex_analytic_mean_error_map.png")

    points = load_annotation_points(annotation_csv, manual_column, model_column)
    hex_gdf, _, boundary_proj = hex_error_aggregation(points, boundary_shp, hex_size_m)

    # Analytic CI for the mean error per hex, and for the total error
    # extrapolated to every tile the hex could contain.
    hex_gdf["hex_area_m2"] = hex_gdf.geometry.area
    hex_gdf["N_total_tiles"] = (
        (hex_gdf["hex_area_m2"] / TILE_AREA_M2).round().astype(int)
    )

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
        if pd.api.types.is_numeric_dtype(hex_gdf[c]):
            hex_gdf[c] = hex_gdf[c].round(3)

    hex_gdf.to_file(out_hex_shp)
    hex_gdf.drop(columns="geometry").to_csv(out_hex_csv, index=False)

    plot_hex_error_map(hex_gdf, boundary_proj, out_map_png, "Hex Mean Tile Error")

    return hex_gdf
