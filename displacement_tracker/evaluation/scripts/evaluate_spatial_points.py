#!/usr/bin/env python3

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ==========================================================
# CORE FUNCTION
# ==========================================================

def evaluate_spatial_points(
    annotation_csv: str = "manual_annotation_results_with_new_model.csv",
    output_dir: str = "results",
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
    gridsize: int = 60,
):
    """
    Create a hexbin map of local mean tile error.

    Outputs:
        - spatial_tile_error_hexbin.png

    Returns:
        dict with basic summary stats
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "spatial_tile_error_hexbin.png")

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

    lon = df["longitude"].to_numpy()
    lat = df["latitude"].to_numpy()
    error = df["tile_error"].to_numpy()

    if len(error) == 0:
        raise ValueError("No rows found in annotation CSV.")

    # ==========================
    # HEXBIN PLOT
    # ==========================

    plt.figure(figsize=(10, 10))

    hb = plt.hexbin(
        lon,
        lat,
        C=error,
        reduce_C_function=np.mean,
        gridsize=gridsize,
        cmap="RdBu_r",
    )

    hex_means = hb.get_array()
    if len(hex_means) == 0:
        raise ValueError("No hex bins were created. Check data.")

    vmin = np.nanmin(hex_means)
    vmax = np.nanmax(hex_means)

    tick_min = int(np.floor(vmin / 10.0) * 10)
    tick_max = int(np.ceil(vmax / 10.0) * 10)

    if tick_min == tick_max:
        tick_min -= 10
        tick_max += 10

    norm = TwoSlopeNorm(vmin=tick_min, vcenter=0, vmax=tick_max)
    hb.set_norm(norm)
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

    print(f"Saved hexbin plot to {output_path}")

    return {
        "output_path": output_path,
        "n_points": int(len(df)),
        "n_hexes": int(len(hex_means)),
        "mean_tile_error": float(np.mean(error)),
    }


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    evaluate_spatial_points(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
        gridsize=60,
    )
