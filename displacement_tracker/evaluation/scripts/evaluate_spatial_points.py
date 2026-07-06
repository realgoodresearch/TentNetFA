"""Hexbin map of local mean tile error."""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    load_annotations,
)


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
        annotation_csv, manual_column, model_column,
        extra_columns=("latitude", "longitude"),
    )
    if df.empty:
        raise ValueError("No rows found in annotation CSV.")

    plt.figure(figsize=(10, 10))

    hb = plt.hexbin(
        df["longitude"].to_numpy(),
        df["latitude"].to_numpy(),
        C=df["tile_error"].to_numpy(),
        reduce_C_function=np.mean,
        gridsize=gridsize,
        cmap="RdBu_r",
    )

    hex_means = hb.get_array()
    if len(hex_means) == 0:
        raise ValueError("No hex bins were created. Check data.")

    tick_min = int(np.floor(np.nanmin(hex_means) / 10.0) * 10)
    tick_max = int(np.ceil(np.nanmax(hex_means) / 10.0) * 10)
    if tick_min == tick_max:
        tick_min -= 10
        tick_max += 10

    hb.set_norm(TwoSlopeNorm(vmin=tick_min, vcenter=0, vmax=tick_max))
    hb.set_clim(tick_min, tick_max)

    cbar = plt.colorbar(hb, label="Mean Tile-Level Error")
    ticks = np.arange(tick_min, tick_max + 10, 10)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(int(t)) for t in ticks])

    plt.title("Local Mean Prediction Error (Hexbin Aggregation)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return {
        "output_path": output_path,
        "n_points": int(len(df)),
        "n_hexes": int(len(hex_means)),
        "mean_tile_error": float(df["tile_error"].mean()),
    }
