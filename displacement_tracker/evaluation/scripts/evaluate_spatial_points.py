"""Hexbin map of local mean tile error."""

import os

from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    load_annotations,
)
from displacement_tracker.evaluation.scripts.plots import plot_error_hexbin


def evaluate_spatial_points(
    annotation_csv: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
    gridsize: int = 60,
):
    """
    Create a hexbin map of local mean tile error.

    Outputs:
        - spatial_tile_error_hexbin.png
    """
    ensure_output_dir(output_dir)
    output_path = os.path.join(output_dir, "spatial_tile_error_hexbin.png")

    df = load_annotations(
        annotation_csv,
        manual_column,
        model_column,
        extra_columns=("latitude", "longitude"),
    )
    if df.empty:
        raise ValueError("No rows found in annotation CSV.")

    n_hexes = plot_error_hexbin(
        df["longitude"].to_numpy(),
        df["latitude"].to_numpy(),
        df["tile_error"].to_numpy(),
        gridsize=gridsize,
        output_path=output_path,
    )

    return {
        "output_path": output_path,
        "n_points": int(len(df)),
        "n_hexes": n_hexes,
        "mean_tile_error": float(df["tile_error"].mean()),
    }
