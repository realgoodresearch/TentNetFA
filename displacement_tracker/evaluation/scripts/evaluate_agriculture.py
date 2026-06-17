#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


# ==========================================================
# CORE FUNCTION
# ==========================================================

def evaluate_agriculture_vs_non_agriculture(
    annotation_csv: str,
    agriculture_geojson: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Evaluate model prediction error inside vs outside agriculture areas.

    Outputs:
        - agriculture_vs_non_agriculture.csv
        - agriculture_vs_non_agriculture_plot.png
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "agriculture_vs_non_agriculture.csv")
    output_plot = os.path.join(output_dir, "agriculture_vs_non_agriculture_plot.png")

    # ==========================
    # LOAD DATA
    # ==========================

    df = pd.read_csv(annotation_csv)

    required_cols = {
        "latitude",
        "longitude",
        manual_column,
        model_column,
    }

    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(
            f"Annotation CSV missing required columns: {sorted(missing)}"
        )

    df["tile_error"] = df[model_column] - df[manual_column]

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    tiles_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    agriculture_gdf = gpd.read_file(agriculture_geojson)

    # Ensure CRS alignment
    if agriculture_gdf.crs != tiles_gdf.crs:
        agriculture_gdf = agriculture_gdf.to_crs(tiles_gdf.crs)

    # Dissolve to a single geometry
    agriculture_union = agriculture_gdf.geometry.union_all()

    # ==========================
    # CLASSIFY TILES
    # ==========================

    tiles_gdf["in_agriculture"] = tiles_gdf.geometry.within(agriculture_union)

    tiles_gdf["region_type"] = np.where(
        tiles_gdf["in_agriculture"],
        "Agriculture",
        "Non-Agriculture",
    )

    # ==========================
    # METRICS
    # ==========================

    results = []

    for region_type, group in tiles_gdf.groupby("region_type"):
        errors = group["tile_error"].dropna().values
        n = len(errors)

        if n == 0:
            continue

        mean_error = float(np.mean(errors))

        if n > 1:
            std_error = float(np.std(errors, ddof=1))
            ci_margin = 1.96 * (std_error / np.sqrt(n))
        else:
            ci_margin = 0.0

        lower = mean_error - ci_margin
        upper = mean_error + ci_margin

        results.append({
            "region_type": region_type,
            "mean_tile_error": mean_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "num_tiles": n,
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values("mean_tile_error", ascending=False)

    results_df.to_csv(output_csv, index=False)

    # ==========================
    # BAR PLOT
    # ==========================

    if not results_df.empty:
        plt.figure(figsize=(8, 5))

        means = results_df["mean_tile_error"].values
        ci_lower = results_df["ci_lower"].values
        ci_upper = results_df["ci_upper"].values

        lower_err = np.maximum(0, means - ci_lower)
        upper_err = np.maximum(0, ci_upper - means)
        yerr = np.vstack((lower_err, upper_err))

        x = np.arange(len(results_df))

        plt.bar(x, means, yerr=yerr, capsize=5)

        labels = [
            f"{name} (n={n})"
            for name, n in zip(results_df["region_type"], results_df["num_tiles"])
        ]

        plt.xticks(x, labels)
        plt.ylabel("Mean Tile-Level Prediction Error")
        plt.title("Prediction Error: Agriculture vs Non-Agriculture (95% CI)")
        plt.axhline(0, linestyle="--")
        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
    else:
        print("WARNING: No agriculture comparison groups found for plot.")

    return results_df


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    evaluate_agriculture_vs_non_agriculture(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        agriculture_geojson="gaza_boundaries/layers/agriculture.json",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
    )
