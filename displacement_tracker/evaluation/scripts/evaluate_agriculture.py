"""Compare model prediction error inside vs outside agriculture areas."""

import os

import geopandas as gpd
import numpy as np

from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    group_error_summary,
    load_annotation_points,
    plot_error_bars,
)


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
    ensure_output_dir(output_dir)

    tiles_gdf = load_annotation_points(annotation_csv, manual_column, model_column)

    agriculture_gdf = gpd.read_file(agriculture_geojson)
    if agriculture_gdf.crs != tiles_gdf.crs:
        agriculture_gdf = agriculture_gdf.to_crs(tiles_gdf.crs)
    agriculture_union = agriculture_gdf.geometry.union_all()

    tiles_gdf["region_type"] = np.where(
        tiles_gdf.geometry.within(agriculture_union),
        "Agriculture",
        "Non-Agriculture",
    )

    results_df = group_error_summary(tiles_gdf, "region_type")
    if not results_df.empty:
        results_df = results_df.sort_values("mean_tile_error", ascending=False)
    results_df.to_csv(
        os.path.join(output_dir, "agriculture_vs_non_agriculture.csv"), index=False
    )

    plot_error_bars(
        results_df,
        label_column="region_type",
        title="Prediction Error: Agriculture vs Non-Agriculture (95% CI)",
        output_plot=os.path.join(output_dir, "agriculture_vs_non_agriculture_plot.png"),
    )

    return results_df
