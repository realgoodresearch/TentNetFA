"""Hex-aggregated tile error with bootstrap CIs (per hex and region total)."""

import os

import numpy as np
import pandas as pd

from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    hex_error_aggregation,
    load_annotation_points,
    plot_hex_error_map,
)


def spatial_bootstrap_hex(
    annotation_csv: str,
    boundary_shp: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
    hex_size_m: float = 1000.0,
    min_samples_per_hex: int = 5,
    b_hex: int = 1000,
    b_block: int = 2000,
    random_seed: int = 42,
):
    """
    Create a hex grid, compute per-hex tile-error summaries, compute bootstrap
    CIs (per-hex and spatial-block for the region total), and export a
    shapefile, CSVs and a map.

    Outputs:
        - hex_with_cis.shp
        - hex_with_cis.csv
        - hex_mean_error_map.png
        - spatial_bootstrap_summary.csv
    """
    ensure_output_dir(output_dir)

    out_hex_shp = os.path.join(output_dir, "hex_with_cis.shp")
    out_hex_csv = os.path.join(output_dir, "hex_with_cis.csv")
    out_map_png = os.path.join(output_dir, "hex_mean_error_map.png")
    out_summary_csv = os.path.join(output_dir, "spatial_bootstrap_summary.csv")

    rng = np.random.default_rng(random_seed)
    alpha = 0.05

    points = load_annotation_points(annotation_csv, manual_column, model_column)
    hex_gdf, points_with_hex, boundary_proj = hex_error_aggregation(
        points, boundary_shp, hex_size_m
    )

    # Per-hex bootstrap CI of the mean error (hexes with too few tiles get NaN).
    errors_by_hex = points_with_hex.groupby("hex_id")["tile_error"]

    def hex_bootstrap_ci(hex_id):
        if hex_id not in errors_by_hex.groups:
            return np.nan, np.nan
        arr = errors_by_hex.get_group(hex_id).values
        n = len(arr)
        if n < min_samples_per_hex:
            return np.nan, np.nan
        boot_means = rng.choice(arr, size=(b_hex, n), replace=True).mean(axis=1)
        return (
            np.percentile(boot_means, 100 * (alpha / 2)),
            np.percentile(boot_means, 100 * (1 - alpha / 2)),
        )

    cis = [hex_bootstrap_ci(hex_id) for hex_id in hex_gdf["hex_id"]]
    hex_gdf["ci_lo"] = [lo for lo, _ in cis]
    hex_gdf["ci_hi"] = [hi for _, hi in cis]

    # Spatial (block) bootstrap for the regional total: resample whole hexes.
    blocks = hex_gdf[hex_gdf["n_tiles"] > 0].copy()
    if blocks.empty:
        raise ValueError("No hexes contain annotations; cannot perform block bootstrap.")

    blocks["total_err_hex"] = blocks["mean_err"] * blocks["n_tiles"]
    totals_arr = blocks["total_err_hex"].values
    nblocks = len(blocks)

    sampled = rng.integers(0, nblocks, size=(b_block, nblocks))
    block_totals = totals_arr[sampled].sum(axis=1)

    region_total_est = float(totals_arr.sum())
    region_lo = float(np.percentile(block_totals, 100 * (alpha / 2)))
    region_hi = float(np.percentile(block_totals, 100 * (1 - alpha / 2)))

    total_tiles_in_blocks = int(blocks["n_tiles"].sum())
    region_mean_obs = region_total_est / total_tiles_in_blocks
    mean_boots = block_totals / total_tiles_in_blocks
    region_mean_lo = float(np.percentile(mean_boots, 100 * (alpha / 2)))
    region_mean_hi = float(np.percentile(mean_boots, 100 * (1 - alpha / 2)))

    for c in ("mean_err", "std_err", "ci_lo", "ci_hi"):
        hex_gdf[c] = hex_gdf[c].round(3)

    hex_gdf.to_file(out_hex_shp)
    hex_gdf.drop(columns="geometry").to_csv(out_hex_csv, index=False)

    significant = hex_gdf[
        (hex_gdf["n_tiles"] >= min_samples_per_hex)
        & ((hex_gdf["ci_lo"] > 0) | (hex_gdf["ci_hi"] < 0))
    ]
    plot_hex_error_map(
        hex_gdf,
        boundary_proj,
        out_map_png,
        "Hex-aggregated Mean Tile Error and Significant Hexes",
        significant=significant,
    )

    summary_df = pd.DataFrame([{
        "region_total_error_estimate": region_total_est,
        "region_total_error_ci_lo": region_lo,
        "region_total_error_ci_hi": region_hi,
        "region_mean_error_estimate": region_mean_obs,
        "region_mean_error_ci_lo": region_mean_lo,
        "region_mean_error_ci_hi": region_mean_hi,
        "n_hex_blocks": nblocks,
        "n_tiles_used": total_tiles_in_blocks,
    }])
    summary_df.to_csv(out_summary_csv, index=False)

    return hex_gdf
