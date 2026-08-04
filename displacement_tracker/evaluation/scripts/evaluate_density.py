"""Evaluate model prediction error by building-density bins from H3 polygons."""

import os

import geopandas as gpd
import pandas as pd

from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    group_error_summary,
    load_annotation_points,
    load_layer,
)
from displacement_tracker.evaluation.scripts.plots import plot_error_bars


def _density_bin(n) -> str | None:
    if pd.isna(n) or n <= 0:
        return None
    if n > 100:
        return ">100"
    lower = ((int(n) - 1) // 10) * 10 + 1
    return f"{lower}-{lower + 9}"


def evaluate_h3_density_bins(
    annotation_csv: str,
    h3_geojson: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Evaluate model prediction error by building density bins
    (1-10, 11-20, ..., 91-100, >100) from h3_density polygons.

    Outputs:
        - h3_density_bins.csv
        - h3_density_bins_plot.png
    """
    ensure_output_dir(output_dir)

    tiles_gdf = load_annotation_points(annotation_csv, manual_column, model_column)

    h3_gdf = load_layer(h3_geojson, tiles_gdf.crs, ("n_buildings",))

    joined = gpd.sjoin(
        tiles_gdf,
        h3_gdf[["n_buildings", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.dropna(subset=["n_buildings"])

    # If overlapping polygons exist, take the max n_buildings per tile.
    joined = (
        joined.groupby(joined.index)
        .agg({"tile_error": "first", "n_buildings": "max"})
        .reset_index(drop=True)
    )

    joined["density_bin"] = joined["n_buildings"].apply(_density_bin)
    joined = joined.dropna(subset=["density_bin"])

    bin_order = [f"{i}-{i + 9}" for i in range(1, 100, 10)] + [">100"]
    joined["density_bin"] = pd.Categorical(
        joined["density_bin"], categories=bin_order, ordered=True
    )

    results_df = group_error_summary(joined, "density_bin")
    if not results_df.empty:
        results_df = results_df.sort_values("density_bin")
        results_df["density_bin"] = results_df["density_bin"].astype(str)
    results_df.to_csv(os.path.join(output_dir, "h3_density_bins.csv"), index=False)

    plot_error_bars(
        results_df,
        label_column="density_bin",
        title="Prediction Error by Building Density Bin (95% CI)",
        output_plot=os.path.join(output_dir, "h3_density_bins_plot.png"),
        figsize=(10, 6),
        rotate_labels=True,
    )

    return results_df
