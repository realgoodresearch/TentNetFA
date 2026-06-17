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

def evaluate_h3_density_bins(
    annotation_csv: str = "manual_annotation_results_with_new_model.csv",
    h3_geojson: str = "gaza_boundaries/layers/h3_density.json",
    output_dir: str = "results",
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Evaluate model prediction error by building density bins
    from h3_density polygons.

    Bins:
        1-10, 11-20, ..., 91-100, >100

    Outputs:
        - h3_density_bins.csv
        - h3_density_bins_plot.png

    Returns:
        results_df
    """
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "h3_density_bins.csv")
    output_plot = os.path.join(output_dir, "h3_density_bins_plot.png")

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
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    df["tile_error"] = df[model_column] - df[manual_column]

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    tiles_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    h3_gdf = gpd.read_file(h3_geojson)

    if "n_buildings" not in h3_gdf.columns:
        raise ValueError("h3_density layer must contain 'n_buildings' column.")

    if h3_gdf.crs != tiles_gdf.crs:
        h3_gdf = h3_gdf.to_crs(tiles_gdf.crs)

    # ==========================
    # SPATIAL JOIN
    # ==========================

    joined = gpd.sjoin(
        tiles_gdf,
        h3_gdf[["n_buildings", "geometry"]],
        how="left",
        predicate="within",
    )

    joined = joined.dropna(subset=["n_buildings"]).copy()

    # If overlapping polygons exist, take the max n_buildings per tile
    joined = (
        joined.groupby(joined.index)
        .agg({
            "tile_error": "first",
            "n_buildings": "max",
        })
        .reset_index(drop=True)
    )

    # ==========================
    # BINNING
    # ==========================

    def density_bin(n):
        if pd.isna(n) or n <= 0:
            return None
        if n > 100:
            return ">100"
        lower = ((int(n) - 1) // 10) * 10 + 1
        upper = lower + 9
        return f"{lower}-{upper}"

    joined["density_bin"] = joined["n_buildings"].apply(density_bin)
    joined = joined.dropna(subset=["density_bin"]).copy()

    bin_order = [f"{i}-{i + 9}" for i in range(1, 100, 10)] + [">100"]
    joined["density_bin"] = pd.Categorical(
        joined["density_bin"],
        categories=bin_order,
        ordered=True,
    )

    # ==========================
    # METRICS
    # ==========================

    results = []

    for bin_name, group in joined.groupby("density_bin", observed=True):
        if pd.isna(bin_name):
            continue

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
            "density_bin": str(bin_name),
            "mean_tile_error": mean_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "num_tiles": n,
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values("density_bin")
    results_df.to_csv(output_csv, index=False)

    # ==========================
    # VERTICAL BAR PLOT
    # ==========================

    if not results_df.empty:
        plt.figure(figsize=(10, 6))

        means = results_df["mean_tile_error"].values
        ci_lower = results_df["ci_lower"].values
        ci_upper = results_df["ci_upper"].values

        lower_err = np.maximum(0, means - ci_lower)
        upper_err = np.maximum(0, ci_upper - means)
        yerr = np.vstack((lower_err, upper_err))

        x = np.arange(len(results_df))

        plt.bar(x, means, yerr=yerr, capsize=5)

        labels = [
            f"{bin_name} (n={n})"
            for bin_name, n in zip(results_df["density_bin"], results_df["num_tiles"])
        ]

        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Mean Tile-Level Prediction Error")
        plt.title("Prediction Error by Building Density Bin (95% CI)")
        plt.axhline(0, linestyle="--")

        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
    else:
        print("WARNING: No density bins found for plot.")

    return results_df


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    evaluate_h3_density_bins(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        h3_geojson="gaza_boundaries/layers/h3_density.json",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
    )
