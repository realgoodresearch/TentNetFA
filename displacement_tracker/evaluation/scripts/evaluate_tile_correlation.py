#!/usr/bin/env python3

import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ==========================================================
# HELPERS
# ==========================================================

def _clean_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove NaN / inf pairs from x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Compute Pearson correlation safely.
    Returns (r, p). If there are fewer than 2 valid points, returns (nan, nan).
    """
    x, y = _clean_xy(x, y)

    if len(x) < 2:
        return np.nan, np.nan

    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def _plot_scatter_with_1to1(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
):
    x, y = _clean_xy(x, y)

    plt.figure(figsize=(8, 8))

    if len(x) > 0:
        plt.scatter(x, y, alpha=0.6)

        max_val = max(np.max(x), np.max(y))
        plt.plot([0, max_val], [0, max_val])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ==========================================================
# CORE FUNCTION
# ==========================================================

def evaluate_tile_correlation(
    annotation_csv: str = "manual_annotation_results_with_new_model.csv",
    output_dir: str = "results",
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Compute and plot tile-level Pearson correlations for:
        - linear scale (all tiles)
        - log scale (all tiles)
        - linear scale (manual > 0)
        - log scale (manual > 0)

    Outputs:
        - tile_prediction_correlation_linear.png
        - tile_prediction_correlation_log.png
        - tile_prediction_correlation_linear_nonzero.png
        - tile_prediction_correlation_log_nonzero.png

    Returns:
        dict with correlation statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    output_linear = os.path.join(output_dir, "tile_prediction_correlation_linear.png")
    output_log = os.path.join(output_dir, "tile_prediction_correlation_log.png")
    output_linear_nonzero = os.path.join(output_dir, "tile_prediction_correlation_linear_nonzero.png")
    output_log_nonzero = os.path.join(output_dir, "tile_prediction_correlation_log_nonzero.png")

    # ==========================
    # LOAD DATA
    # ==========================

    df = pd.read_csv(annotation_csv)

    required_cols = {manual_column, model_column}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Annotation CSV missing required columns: {sorted(missing)}")

    x = df[manual_column].to_numpy(dtype=float)
    y = df[model_column].to_numpy(dtype=float)

    # ==========================
    # LINEAR CORRELATION (ALL)
    # ==========================

    r_linear, p_linear = _safe_pearsonr(x, y)

    _plot_scatter_with_1to1(
        x=x,
        y=y,
        xlabel="Manual Tent Count",
        ylabel="Model Tent Count",
        title=f"Tile-Level Prediction Correlation (Linear)\nPearson r = {r_linear:.3f}",
        output_path=output_linear,
    )

    # ==========================
    # LOG CORRELATION (ALL)
    # ==========================

    x_log = np.log1p(x)
    y_log = np.log1p(y)

    r_log, p_log = _safe_pearsonr(x_log, y_log)

    _plot_scatter_with_1to1(
        x=x_log,
        y=y_log,
        xlabel="log(1 + Manual Tent Count)",
        ylabel="log(1 + Model Tent Count)",
        title=f"Tile-Level Prediction Correlation (Log Scale)\nPearson r = {r_log:.3f}",
        output_path=output_log,
    )

    # ==========================
    # FILTER: MANUAL COUNT > 0
    # ==========================

    df_nonzero = df[df[manual_column].fillna(np.nan) > 0].copy()
    x_nz = df_nonzero[manual_column].to_numpy(dtype=float)
    y_nz = df_nonzero[model_column].to_numpy(dtype=float)

    # ==========================
    # LINEAR CORRELATION (NONZERO)
    # ==========================

    r_linear_nz, p_linear_nz = _safe_pearsonr(x_nz, y_nz)

    _plot_scatter_with_1to1(
        x=x_nz,
        y=y_nz,
        xlabel="Manual Tent Count",
        ylabel="Model Tent Count",
        title=(
            f"Tile-Level Prediction Correlation (Linear, Manual > 0)\n"
            f"Pearson r = {r_linear_nz:.3f}"
        ),
        output_path=output_linear_nonzero,
    )

    # ==========================
    # LOG CORRELATION (NONZERO)
    # ==========================

    x_log_nz = np.log1p(x_nz)
    y_log_nz = np.log1p(y_nz)

    r_log_nz, p_log_nz = _safe_pearsonr(x_log_nz, y_log_nz)

    _plot_scatter_with_1to1(
        x=x_log_nz,
        y=y_log_nz,
        xlabel="log(1 + Manual Tent Count)",
        ylabel="log(1 + Model Tent Count)",
        title=(
            f"Tile-Level Prediction Correlation (Log, Manual > 0)\n"
            f"Pearson r = {r_log_nz:.3f}"
        ),
        output_path=output_log_nonzero,
    )

    # ==========================
    # PRINT RESULTS
    # ==========================

    print("Saved plots:")
    print(" - Linear (all):", output_linear)
    print(" - Log (all):", output_log)
    print(" - Linear (manual > 0):", output_linear_nonzero)
    print(" - Log (manual > 0):", output_log_nonzero)

    print("\nCorrelations:")
    print(f"Linear (all)        r={r_linear:.4f}, p={p_linear:.4g}")
    print(f"Log (all)           r={r_log:.4f}, p={p_log:.4g}")
    print(f"Linear (manual>0)   r={r_linear_nz:.4f}, p={p_linear_nz:.4g}")
    print(f"Log (manual>0)      r={r_log_nz:.4f}, p={p_log_nz:.4g}")

    return {
        "linear_all": {"r": r_linear, "p": p_linear, "output": output_linear},
        "log_all": {"r": r_log, "p": p_log, "output": output_log},
        "linear_nonzero": {"r": r_linear_nz, "p": p_linear_nz, "output": output_linear_nonzero},
        "log_nonzero": {"r": r_log_nz, "p": p_log_nz, "output": output_log_nonzero},
    }


# ==========================================================
# CLI ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    evaluate_tile_correlation(
        annotation_csv="manual_annotation_results_with_new_model.csv",
        output_dir="results",
        manual_column="manual_tent_count",
        model_column="model_tent_count",
    )
