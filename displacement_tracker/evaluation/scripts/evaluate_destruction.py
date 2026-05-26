import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


# ==========================================================
# CORE FUNCTION
# ==========================================================

def evaluate_destruction_vs_non_destruction(
    annotation_csv: str,
    destruction_geojson: str,
    output_dir: str
):
    """
    Evaluate model prediction error inside vs outside destruction polygons.

    A tile is considered 'Destruction' if:
        - Its centroid lies within a destruction polygon
        - destruction.date_start <= tile_date

    Outputs:
        - destruction_vs_non_destruction.csv
        - destruction_vs_non_destruction_plot.png
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "destruction_vs_non_destruction.csv")
    output_plot = os.path.join(output_dir, "destruction_vs_non_destruction_plot.png")

    # ==========================
    # LOAD DATA
    # ==========================

    df = pd.read_csv(annotation_csv)

    required_cols = {
        "latitude",
        "longitude",
        "manual_tent_count",
        "model_tent_count",
        "date"
    }

    if not required_cols.issubset(df.columns):
        raise ValueError("Annotation CSV missing required columns.")

    df["tile_error"] = df["model_tent_count"] - df["manual_tent_count"]

    # Ensure tile date is datetime
    df["date"] = pd.to_datetime(df["date"])

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    tiles_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    destruction_gdf = gpd.read_file(destruction_geojson)

    if "date_start" not in destruction_gdf.columns:
        raise ValueError("Destruction layer must contain 'date_start' column.")

    destruction_gdf["date_start"] = pd.to_datetime(destruction_gdf["date_start"])

    # CRS alignment
    if destruction_gdf.crs != tiles_gdf.crs:
        destruction_gdf = destruction_gdf.to_crs(tiles_gdf.crs)

    # ==========================
    # SPATIAL JOIN (many-to-one)
    # ==========================

    joined = gpd.sjoin(
        tiles_gdf,
        destruction_gdf[["date_start", "geometry"]],
        how="left",
        predicate="within"
    )

    # ==========================
    # TEMPORAL FILTER
    # ==========================

    # Tile is active destruction if:
    # - It spatially matched a polygon
    # - Polygon date_start <= tile date

    joined["active_destruction"] = (
        (~joined["date_start"].isna()) &
        (joined["date_start"] <= joined["date"])
    )

    # If multiple overlapping polygons exist, collapse by tile
    collapsed = (
        joined
        .groupby(joined.index)
        .agg({
            "tile_error": "first",
            "active_destruction": "max"   # any True means destruction
        })
    )

    collapsed["region_type"] = np.where(
        collapsed["active_destruction"],
        "Destruction",
        "Non-Destruction"
    )

    # ==========================
    # METRICS
    # ==========================

    results = []

    for region_type, group in collapsed.groupby("region_type"):

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
            "region_type": region_type,
            "mean_tile_error": mean_error,
            "ci_lower": lower,
            "ci_upper": upper,
            "num_tiles": n
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_tile_error", ascending=False)

    results_df.to_csv(output_csv, index=False)

    # ==========================
    # BAR PLOT
    # ==========================

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
    plt.title("Prediction Error: Destruction vs Non-Destruction (95% CI)")

    # horizontal zero reference line
    plt.axhline(0, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    return results_df


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":

    evaluate_destruction_vs_non_destruction(
        annotation_csv="displacement_tracker/evaluation/manual_annotation_results.csv",
        destruction_geojson="gaza_boundaries/layers/destruction.json",
        output_dir="displacement_tracker/evaluation/results"
    )