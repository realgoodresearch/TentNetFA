"""Evaluate model prediction error per municipal polygon."""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from displacement_tracker.evaluation.scripts.common import (
    Z_95,
    ensure_output_dir,
    group_error_summary,
    load_annotation_points,
    plot_error_bars,
)


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
    ensure_output_dir(output_dir)

    tiles_gdf = load_annotation_points(annotation_csv, manual_column, model_column)

    municipal_gdf = gpd.read_file(boundary_shp)
    if name_column not in municipal_gdf.columns:
        raise ValueError(f"{name_column} not found in boundary shapefile.")
    if municipal_gdf.crs != tiles_gdf.crs:
        municipal_gdf = municipal_gdf.to_crs(tiles_gdf.crs)

    joined = gpd.sjoin(
        tiles_gdf,
        municipal_gdf[[name_column, "geometry"]],
        how="left",
        predicate="within",
    )

    results_df = group_error_summary(joined, name_column)

    if not results_df.empty:
        # Derive per-region spread and extrapolated totals from the CI summary.
        n = results_df["num_tiles"]
        se = (results_df["mean_tile_error"] - results_df["ci_lower"]) / Z_95
        results_df["std_tile_error"] = (se * np.sqrt(n)).where(n > 1, 0.0)
        results_df["se_tile_error"] = se
        results_df["s_tile_error"] = results_df["std_tile_error"]
        results_df["total_regional_error"] = n * results_df["mean_tile_error"]
        results_df["total_regional_se"] = n * se
        results_df["total_regional_ci_lower"] = (
            results_df["total_regional_error"] - Z_95 * results_df["total_regional_se"]
        )
        results_df["total_regional_ci_upper"] = (
            results_df["total_regional_error"] + Z_95 * results_df["total_regional_se"]
        )
        results_df = results_df.sort_values("mean_tile_error", ascending=False)

    results_df.to_csv(os.path.join(output_dir, "municipal_bounds.csv"), index=False)

    plot_error_bars(
        results_df,
        label_column=name_column,
        title="Mean Prediction Error per Municipal Region (95% CI)",
        output_plot=os.path.join(output_dir, "municipal_bounds_plot.png"),
        figsize=(12, 7),
        horizontal=True,
    )

    _plot_municipal_map(
        municipal_gdf, results_df, name_column,
        os.path.join(output_dir, "municipal_bounds_map.png"),
    )
    _plot_municipal_scatter(
        joined, name_column, manual_column, model_column,
        os.path.join(output_dir, "municipal_scatter_subplots.png"),
    )

    return results_df


def _plot_municipal_map(municipal_gdf, results_df, name_column, output_map):
    """Choropleth of mean tile error per municipality."""
    map_gdf = municipal_gdf.merge(results_df, on=name_column, how="left")

    if "mean_tile_error" not in map_gdf.columns or not map_gdf["mean_tile_error"].notna().any():
        print("WARNING: No valid regional error values available for map plot.")
        return

    vmax = np.nanmax(np.abs(map_gdf["mean_tile_error"].values))

    plt.figure(figsize=(10, 10))
    map_gdf.plot(
        column="mean_tile_error",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        legend=True,
        edgecolor="black",
    )
    plt.title("Average Model Over/Undercount per Municipality")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_map)
    plt.close()


def _plot_municipal_scatter(joined, name_column, manual_column, model_column, output_scatter):
    """Per-municipality manual-vs-model scatter subplots with a 1:1 line."""
    unique_regions = sorted(joined[name_column].dropna().unique())
    if not unique_regions:
        print("WARNING: No regions available for scatter subplots.")
        return

    n_cols = 3
    n_rows = int(np.ceil(len(unique_regions) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, region in zip(axes, unique_regions):
        subset = joined[joined[name_column] == region]
        x = subset[manual_column].values
        y = subset[model_column].values

        if len(x) == 0:
            ax.axis("off")
            continue

        ax.scatter(x, y, alpha=0.6)
        max_val = max(np.max(x), np.max(y))
        ax.plot([0, max_val], [0, max_val])

        r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
        ax.set_title(f"{region}\n(n={len(x)}, r={r:.2f})")
        ax.set_xlabel("Manual")
        ax.set_ylabel("Model")

    for ax in axes[len(unique_regions):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_scatter)
    plt.close()
