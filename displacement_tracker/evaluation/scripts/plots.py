"""Shared plotting helpers for the evaluation scripts.

Kept separate from ``common.py`` so the loaders and statistics there can be
reused without importing matplotlib.
"""

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from displacement_tracker.evaluation.scripts.common import LOGGER


def _ci_whiskers(results_df: pd.DataFrame) -> np.ndarray:
    means = results_df["mean_tile_error"].values
    lower_err = np.maximum(0, means - results_df["ci_lower"].values)
    upper_err = np.maximum(0, results_df["ci_upper"].values - means)
    return np.vstack((lower_err, upper_err))


def plot_error_bars(
    results_df: pd.DataFrame,
    label_column: str,
    title: str,
    output_plot: str,
    value_label: str = "Mean Tile-Level Prediction Error",
    figsize: tuple = (8, 5),
    rotate_labels: bool = False,
    horizontal: bool = False,
) -> None:
    """Bar plot of mean tile error per group with 95% CI whiskers."""
    if results_df.empty:
        LOGGER.warning("No groups to plot for %s", output_plot)
        return

    plt.figure(figsize=figsize)

    positions = np.arange(len(results_df))
    means = results_df["mean_tile_error"].values
    whiskers = _ci_whiskers(results_df)

    labels = [
        f"{name} (n={n})"
        for name, n in zip(results_df[label_column], results_df["num_tiles"])
    ]

    if horizontal:
        plt.barh(positions, means, xerr=whiskers, capsize=5)
        plt.yticks(positions, labels)
        plt.xlabel(value_label)
    else:
        plt.bar(positions, means, yerr=whiskers, capsize=5)
        if rotate_labels:
            plt.xticks(positions, labels, rotation=45, ha="right")
        else:
            plt.xticks(positions, labels)
        plt.ylabel(value_label)
        plt.axhline(0, linestyle="--")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()


def plot_hex_error_map(
    hex_gdf: gpd.GeoDataFrame,
    boundary_proj: gpd.GeoDataFrame,
    output_png: str,
    title: str,
    significant: gpd.GeoDataFrame | None = None,
) -> None:
    """Choropleth of mean_err per hex over the boundary outline."""
    fig, ax = plt.subplots(figsize=(12, 12))
    boundary_proj.boundary.plot(ax=ax, color="black", linewidth=0.5)

    plot_gdf = hex_gdf.copy()
    plot_gdf["plot_val"] = plot_gdf["mean_err"].fillna(0.0)

    valid = plot_gdf["n_tiles"] > 0
    if valid.any():
        vmin = float(np.nanmin(plot_gdf.loc[valid, "plot_val"]))
        vmax = float(np.nanmax(plot_gdf.loc[valid, "plot_val"]))
    else:
        vmin, vmax = -1.0, 1.0
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0

    plot_gdf.plot(
        column="plot_val",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        edgecolor="k",
        linewidth=0.1,
    )

    if significant is not None and not significant.empty:
        significant.boundary.plot(ax=ax, color="black", linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean tile error")

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
