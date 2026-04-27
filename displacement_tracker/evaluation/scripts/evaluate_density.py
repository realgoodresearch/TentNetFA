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
    annotation_csv: str,
    h3_geojson: str,
    output_dir: str
):
    """
    Evaluate model prediction error by building density bins
    from h3_density polygons.

    Bins:
        1-10, 11-20, ..., 91-100, >100

    Outputs:
        - h3_density_bins.csv
        - h3_density_bins_plot.png
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
        "manual_tent_count",
        "model_tent_count"
    }

    if not required_cols.issubset(df.columns):
        raise ValueError("Annotation CSV missing required columns.")

    df["tile_error"] = df["model_tent_count"] - df["manual_tent_count"]

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
        predicate="within"
    )

    # Drop tiles not inside any density polygon
    joined = joined.dropna(subset=["n_buildings"])

    # If overlapping polygons exist, take the max n_buildings per tile
    joined = (
        joined.groupby(joined.index)
        .agg({
            "tile_error": "first",
            "n_buildings": "max"
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
    joined = joined.dropna(subset=["density_bin"])

    # Define bin order explicitly
    bin_order = [f"{i}-{i + 9}" for i in range(1, 100, 10)] + [">100"]

    joined["density_bin"] = pd.Categorical(
        joined["density_bin"],
        categories=bin_order,
        ordered=True
    )

    # ==========================
    # METRICS
    # ==========================

    results = []

    for bin_name, group in joined.groupby("density_bin"):

        if pd.isna(bin_name):
            continue

        errors = group["tile_error"].values
        n = len(errors)

        if n == 0:
            continue

        mean_error = np.mean(errors)

        if n > 1:
            std_error = np.std(errors, ddof=1)
            ci_margin = 1.96 * (std_error / np.sqrt(n))
        else:
            ci_margin = 0

        lower = mean_error - ci_margin
        upper = mean_error + ci_margin

        results.append({
            "density_bin": bin_name,
            "mean_tile_error": mean_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "num_tiles": n
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("density_bin")

    results_df.to_csv(output_csv, index=False)

    # ==========================
    # VERTICAL BAR PLOT
    # ==========================

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

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Mean Tile-Level Prediction Error")
    plt.title("Prediction Error by Building Density Bin (95% CI)")

    plt.axhline(0, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    return results_df


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":

    evaluate_h3_density_bins(
        annotation_csv="displacement_tracker/evaluation/manual_annotation_results.csv",
        h3_geojson="gaza_boundaries/layers/h3_density.json",
        output_dir="displacement_tracker/evaluation/results"
    )