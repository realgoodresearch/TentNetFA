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

def evaluate_municipal_bounds(
    annotation_csv: str,
    boundary_shp: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
    name_column: str = "NAME",
):
    """
    Evaluate model prediction error per municipal polygon.

    Outputs:
        - municipal_bounds.csv
        - municipal_bounds_plot.png
        - municipal_bounds_map.png
        - municipal_scatter_subplots.png
    """

    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "municipal_bounds.csv")
    output_plot = os.path.join(output_dir, "municipal_bounds_plot.png")
    output_map = os.path.join(output_dir, "municipal_bounds_map.png")
    output_scatter = os.path.join(output_dir, "municipal_scatter_subplots.png")

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

    municipal_gdf = gpd.read_file(boundary_shp)

    if name_column not in municipal_gdf.columns:
        raise ValueError(f"{name_column} not found in boundary shapefile.")

    if municipal_gdf.crs != tiles_gdf.crs:
        municipal_gdf = municipal_gdf.to_crs(tiles_gdf.crs)

    # ==========================
    # SPATIAL JOIN
    # ==========================

    joined = gpd.sjoin(
        tiles_gdf,
        municipal_gdf[[name_column, "geometry"]],
        how="left",
        predicate="within",
    )

    # ==========================
    # REGION METRICS
    # ==========================

    results = []

    for region_name, group in joined.groupby(name_column, dropna=True):
        errors = group["tile_error"].dropna().values
        n = len(errors)

        if n == 0:
            continue

        mean_error = float(np.mean(errors))

        if n > 1:
            std_error = float(np.std(errors, ddof=1))
            ci_margin = 1.96 * (std_error / np.sqrt(n))
            se = ci_margin / 1.96
            s = se * np.sqrt(n)
        else:
            std_error = 0.0
            ci_margin = 0.0
            se = 0.0
            s = 0.0

        lower = mean_error - ci_margin
        upper = mean_error + ci_margin

        total_error = n * mean_error
        total_se = n * se
        total_lower = total_error - 1.96 * total_se
        total_upper = total_error + 1.96 * total_se

        results.append({
            name_column: region_name,
            "mean_tile_error": mean_error,
            "std_tile_error": std_error,
            "se_tile_error": se,
            "s_tile_error": s,
            "ci_lower": lower,
            "ci_upper": upper,
            "num_tiles": n,
            "total_regional_error": total_error,
            "total_regional_se": total_se,
            "total_regional_ci_lower": total_lower,
            "total_regional_ci_upper": total_upper,
        })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values("mean_tile_error", ascending=False)

    results_df.to_csv(output_csv, index=False)

    # ==========================
    # BAR PLOT
    # ==========================

    if not results_df.empty:
        plt.figure(figsize=(12, 7))

        y = np.arange(len(results_df))
        means = results_df["mean_tile_error"].values
        ci_lower = results_df["ci_lower"].values
        ci_upper = results_df["ci_upper"].values
        yerr = np.vstack((means - ci_lower, ci_upper - means))

        plt.barh(y, means, xerr=yerr, capsize=5)

        labels = [
            f"{name} (n={n})"
            for name, n in zip(results_df[name_column], results_df["num_tiles"])
        ]

        plt.yticks(y, labels)
        plt.xlabel("Mean Tile-Level Prediction Error")
        plt.title("Mean Prediction Error per Municipal Region (95% CI)")
        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
    else:
        print("WARNING: No municipal regions found for bar plot.")

    # ==========================
    # SPATIAL MAP
    # ==========================

    map_gdf = municipal_gdf.merge(results_df, on=name_column, how="left")

    if "mean_tile_error" in map_gdf.columns and map_gdf["mean_tile_error"].notna().any():
        vmax = np.nanmax(np.abs(map_gdf["mean_tile_error"].values))
        vmin = -vmax

        plt.figure(figsize=(10, 10))
        map_gdf.plot(
            column="mean_tile_error",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            legend=True,
            edgecolor="black",
        )

        plt.title("Average Model Over/Undercount per Municipality")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_map)
        plt.close()
    else:
        print("WARNING: No valid regional error values available for map plot.")

    # ==========================
    # MUNICIPAL SCATTER SUBPLOTS
    # ==========================

    unique_regions = sorted(joined[name_column].dropna().unique())
    n_regions = len(unique_regions)

    if n_regions > 0:
        n_cols = 3
        n_rows = int(np.ceil(n_regions / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for i, region in enumerate(unique_regions):
            ax = axes[i]
            subset = joined[joined[name_column] == region]

            x = subset[manual_column].values
            y = subset[model_column].values

            if len(x) == 0:
                ax.axis("off")
                continue

            ax.scatter(x, y, alpha=0.6)

            max_val = max(np.max(x), np.max(y))
            ax.plot([0, max_val], [0, max_val])

            if len(x) > 1:
                r = np.corrcoef(x, y)[0, 1]
            else:
                r = np.nan

            ax.set_title(f"{region}\n(n={len(x)}, r={r:.2f})")
            ax.set_xlabel("Manual")
            ax.set_ylabel("Model")

        for j in range(n_regions, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(output_scatter)
        plt.close()
        print(f"Saved municipal scatter subplots to {output_scatter}")
    else:
        print("WARNING: No regions available for scatter subplots.")

    return results_df


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    evaluate_municipal_bounds(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        boundary_shp="gaza_boundaries/GazaStrip_MunicipalBoundaries.shp",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
        name_column="NAME",
    )
