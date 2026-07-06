"""Tile-level manual-vs-model Pearson correlations on linear and log scales."""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from displacement_tracker.evaluation.scripts.common import (
    ensure_output_dir,
    load_annotations,
)


def _clean_xy(x, y):
    """Remove NaN / inf pairs from x and y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _safe_pearsonr(x, y):
    """Pearson correlation, or (nan, nan) with fewer than 2 valid points."""
    x, y = _clean_xy(x, y)
    if len(x) < 2:
        return np.nan, np.nan
    try:
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def _plot_scatter_with_1to1(x, y, xlabel, ylabel, title, output_path):
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


def evaluate_tile_correlation(
    annotation_csv: str,
    output_dir: str,
    manual_column: str = "manual_tent_count",
    model_column: str = "model_tent_count",
):
    """
    Compute and plot tile-level Pearson correlations on linear and log
    scales, over all tiles and over tiles with a nonzero manual count.

    Outputs:
        - tile_prediction_correlation_linear.png
        - tile_prediction_correlation_log.png
        - tile_prediction_correlation_linear_nonzero.png
        - tile_prediction_correlation_log_nonzero.png
    """
    ensure_output_dir(output_dir)

    df = load_annotations(annotation_csv, manual_column, model_column)

    x_all = df[manual_column].to_numpy(dtype=float)
    y_all = df[model_column].to_numpy(dtype=float)

    nonzero = df[df[manual_column] > 0]
    x_nz = nonzero[manual_column].to_numpy(dtype=float)
    y_nz = nonzero[model_column].to_numpy(dtype=float)

    variants = {
        "linear_all": (x_all, y_all, False, "Linear", "tile_prediction_correlation_linear.png"),
        "log_all": (x_all, y_all, True, "Log Scale", "tile_prediction_correlation_log.png"),
        "linear_nonzero": (x_nz, y_nz, False, "Linear, Manual > 0", "tile_prediction_correlation_linear_nonzero.png"),
        "log_nonzero": (x_nz, y_nz, True, "Log, Manual > 0", "tile_prediction_correlation_log_nonzero.png"),
    }

    results = {}
    for key, (x, y, log_scale, scale_label, filename) in variants.items():
        if log_scale:
            x, y = np.log1p(x), np.log1p(y)
            xlabel = "log(1 + Manual Tent Count)"
            ylabel = "log(1 + Model Tent Count)"
        else:
            xlabel = "Manual Tent Count"
            ylabel = "Model Tent Count"

        r, p = _safe_pearsonr(x, y)
        output_path = os.path.join(output_dir, filename)

        _plot_scatter_with_1to1(
            x, y, xlabel, ylabel,
            title=(
                f"Tile-Level Prediction Correlation ({scale_label})\n"
                f"Pearson r = {r:.3f}"
            ),
            output_path=output_path,
        )
        results[key] = {"r": r, "p": p, "output": output_path}

    return results
