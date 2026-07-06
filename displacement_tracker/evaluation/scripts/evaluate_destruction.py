"""Compare model prediction error inside vs outside destruction polygons."""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    group_error_summary,
    load_annotation_points,
    plot_error_bars,
)


def evaluate_destruction_vs_non_destruction(
    annotation_csv: str,
    destruction_geojson: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Evaluate model prediction error inside vs outside destruction polygons.

    A tile is considered 'Destruction' if its centroid lies within a
    destruction polygon whose date_start is on or before the tile date.

    Outputs:
        - destruction_vs_non_destruction.csv
        - destruction_vs_non_destruction_plot.png
    """
    ensure_output_dir(output_dir)

    tiles_gdf = load_annotation_points(
        annotation_csv, manual_column, model_column, extra_columns=("date",)
    )
    tiles_gdf["date"] = pd.to_datetime(tiles_gdf["date"])

    destruction_gdf = gpd.read_file(destruction_geojson)
    if "date_start" not in destruction_gdf.columns:
        raise ValueError("Destruction layer must contain 'date_start' column.")
    destruction_gdf["date_start"] = pd.to_datetime(destruction_gdf["date_start"])
    if destruction_gdf.crs != tiles_gdf.crs:
        destruction_gdf = destruction_gdf.to_crs(tiles_gdf.crs)

    joined = gpd.sjoin(
        tiles_gdf,
        destruction_gdf[["date_start", "geometry"]],
        how="left",
        predicate="within",
    )

    # A tile can match several polygons; it counts as 'Destruction' if any
    # matching polygon was already destroyed by the tile date.
    joined["active_destruction"] = (
        joined["date_start"].notna() & (joined["date_start"] <= joined["date"])
    )
    collapsed = joined.groupby(joined.index).agg(
        {"tile_error": "first", "active_destruction": "max"}
    )

    collapsed["region_type"] = np.where(
        collapsed["active_destruction"], "Destruction", "Non-Destruction"
    )

    results_df = group_error_summary(collapsed, "region_type")
    if not results_df.empty:
        results_df = results_df.sort_values("mean_tile_error", ascending=False)
    results_df.to_csv(
        os.path.join(output_dir, "destruction_vs_non_destruction.csv"), index=False
    )

    plot_error_bars(
        results_df,
        label_column="region_type",
        title="Prediction Error: Destruction vs Non-Destruction (95% CI)",
        output_plot=os.path.join(output_dir, "destruction_vs_non_destruction_plot.png"),
    )

    return results_df
