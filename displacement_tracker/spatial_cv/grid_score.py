"""100 m count score over holdout tile cores.

Bins the point-score gpkgs, keeps every cell that meets a holdout core
(including cells where both counts are 0), and writes the summary plus
the three figures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import box

from displacement_tracker.spatial_cv.folds import train_test_masks
from displacement_tracker.spatial_cv.points import MATCH_CRS

CELL_METRES = 100.0


def write_fold_grid_score(
    score_dir: Path,
    out_dir: Path,
    folds_path: Path,
    fold_id: int,
    core_metres: float,
    scheme: str,
    cell_metres: float = CELL_METRES,
) -> dict[str, Any]:
    """Score one fold from its point gpkgs. Does not load a checkpoint."""
    score_dir = Path(score_dir)
    pred_path = score_dir / "holdout_pred_points.gpkg"
    label_path = score_dir / "holdout_label_points.gpkg"
    missing = [path for path in (pred_path, label_path) if not path.exists()]
    if missing:
        listed = ", ".join(str(path) for path in missing)
        raise click.ClickException(
            f"Fold {fold_id} has no point score to bin ({listed}). Run score-points first."
        )
    folds_path = Path(folds_path)
    meta_path = folds_path.with_name("folds_meta.json")
    if not folds_path.exists() or not meta_path.exists():
        raise click.ClickException(f"Missing fold table or metadata beside {folds_path}")

    folds = pd.read_parquet(folds_path)
    meta = json.loads(meta_path.read_text())
    _, test_mask = train_test_masks(folds, int(fold_id), meta)
    holdout = folds.loc[test_mask]
    if holdout.empty:
        raise click.ClickException(f"Holdout fold {fold_id} has no tiles")

    cells = score_cells(
        _xy_from_gpkg(pred_path),
        _xy_from_gpkg(label_path),
        core_squares(holdout, core_metres),
        cell_metres=cell_metres,
    )
    summary = summarise_cells(
        cells,
        holdout_fold=int(fold_id),
        scheme=scheme,
        cell_metres=cell_metres,
        core_metres=float(core_metres),
        n_cores=int(len(holdout)),
    )
    write_grid_outputs(cells, summary, Path(out_dir))
    return summary


def core_squares(holdout: pd.DataFrame, core_metres: float) -> gpd.GeoDataFrame:
    """70 m square on the window centre, in the scoring CRS.

    The stored lon/lat box is the full window (core plus margin). The core
    is the inner square, the same region the point score keeps.
    """
    lon, lat = _window_centres(holdout)
    centres = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(MATCH_CRS)
    half = float(core_metres) / 2.0
    squares = [
        box(x - half, y - half, x + half, y + half)
        for x, y in zip(centres.x.to_numpy(), centres.y.to_numpy())
    ]
    return gpd.GeoDataFrame(geometry=squares, crs=MATCH_CRS)


def score_cells(
    pred_xy: np.ndarray,
    label_xy: np.ndarray,
    cores: gpd.GeoDataFrame,
    cell_metres: float = CELL_METRES,
) -> pd.DataFrame:
    """One row per cell that intersects a core, including both-zero cells."""
    universe = cells_intersecting_cores(cores, cell_metres)
    if not universe:
        raise click.ClickException("Holdout cores do not meet any grid cell")
    pred_counts = _bin_counts(pred_xy, universe, cell_metres)
    label_counts = _bin_counts(label_xy, universe, cell_metres)
    rows = []
    for ix, iy in sorted(universe):
        label_count = label_counts.get((ix, iy), 0)
        pred_count = pred_counts.get((ix, iy), 0)
        rows.append(
            {
                "cell_x": ix,
                "cell_y": iy,
                "x_min": ix * cell_metres,
                "y_min": iy * cell_metres,
                "label_count": label_count,
                "pred_count": pred_count,
                "residual": pred_count - label_count,
            }
        )
    return pd.DataFrame(rows)


def cells_intersecting_cores(cores: gpd.GeoDataFrame, cell_metres: float) -> set[tuple[int, int]]:
    """Cell indices whose closed 100 m square meets a core."""
    found: set[tuple[int, int]] = set()
    for geom in cores.geometry:
        minx, miny, maxx, maxy = geom.bounds
        ix0, ix1 = _index_span(minx, maxx, cell_metres)
        iy0, iy1 = _index_span(miny, maxy, cell_metres)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cell = box(
                    ix * cell_metres,
                    iy * cell_metres,
                    (ix + 1) * cell_metres,
                    (iy + 1) * cell_metres,
                )
                if cell.intersects(geom):
                    found.add((ix, iy))
    return found


def summarise_cells(
    cells: pd.DataFrame,
    *,
    holdout_fold: int,
    scheme: str,
    cell_metres: float,
    core_metres: float,
    n_cores: int,
) -> dict[str, Any]:
    label = cells["label_count"].to_numpy(dtype=np.float64)
    pred = cells["pred_count"].to_numpy(dtype=np.float64)
    residual = cells["residual"].to_numpy(dtype=np.float64)
    either = (label > 0) | (pred > 0)
    n_pred = int(pred.sum())
    n_label = int(label.sum())
    return {
        "holdout_fold": int(holdout_fold),
        "scheme": scheme,
        "cell_metres": float(cell_metres),
        "core_metres": float(core_metres),
        "crs": MATCH_CRS,
        "n_cores": int(n_cores),
        "n_cells": int(len(cells)),
        "n_pred": n_pred,
        "n_label": n_label,
        "bias": n_pred - n_label,
        "ratio": (n_pred / n_label) if n_label else None,
        "mae": _mae(residual),
        "median_ae": _median_ae(residual),
        "n_cells_either_positive": int(either.sum()),
        "mae_either_positive": _mae(residual[either]),
        "median_ae_either_positive": _median_ae(residual[either]),
        "pearson_r": _pearson(label, pred),
        "concordance": _lin_concordance(label, pred),
        "presence": {
            "both_zero": int(((label == 0) & (pred == 0)).sum()),
            "both_positive": int(((label > 0) & (pred > 0)).sum()),
            "pred_only": int(((label == 0) & (pred > 0)).sum()),
            "label_only": int(((label > 0) & (pred == 0)).sum()),
        },
    }


def write_grid_outputs(cells: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out_dir / "cells.csv", index=False)
    (out_dir / "holdout_grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _plot_count_scatter(cells, summary, out_dir / "count_scatter.png")
    _plot_residual_hist(cells, summary, out_dir / "residual_hist.png")
    _plot_residual_map(cells, summary, out_dir / "residual_map.png")


def _window_centres(holdout: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "lon_c" in holdout.columns and "lat_c" in holdout.columns:
        return holdout["lon_c"].astype(float), holdout["lat_c"].astype(float)
    lon = (holdout["lon_min"].astype(float) + holdout["lon_max"].astype(float)) / 2.0
    lat = (holdout["lat_min"].astype(float) + holdout["lat_max"].astype(float)) / 2.0
    return lon, lat


def _xy_from_gpkg(path: Path) -> np.ndarray:
    frame = gpd.read_file(path)
    if frame.empty:
        return np.zeros((0, 2), dtype=np.float64)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    projected = frame.to_crs(MATCH_CRS)
    return np.column_stack(
        [projected.geometry.x.to_numpy(), projected.geometry.y.to_numpy()]
    )


def _bin_counts(
    xy: np.ndarray,
    universe: set[tuple[int, int]],
    cell_metres: float,
) -> dict[tuple[int, int], int]:
    if len(xy) == 0:
        return {}
    ix = np.floor(xy[:, 0] / cell_metres).astype(np.int64)
    iy = np.floor(xy[:, 1] / cell_metres).astype(np.int64)
    counts: dict[tuple[int, int], int] = {}
    for key in zip(ix.tolist(), iy.tolist()):
        if key in universe:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _index_span(vmin: float, vmax: float, cell_metres: float) -> tuple[int, int]:
    """Inclusive cell indices whose closed interval meets [vmin, vmax]."""
    i_min = math.ceil(vmin / cell_metres - 1.0 - 1e-9)
    i_max = math.floor(vmax / cell_metres + 1e-9)
    return int(i_min), int(i_max)


def _mae(residual: np.ndarray) -> float | None:
    if residual.size == 0:
        return None
    return float(np.mean(np.abs(residual)))


def _median_ae(residual: np.ndarray) -> float | None:
    if residual.size == 0:
        return None
    return float(np.median(np.abs(residual)))


def _pearson(label: np.ndarray, pred: np.ndarray) -> float | None:
    if label.size < 2 or np.std(label) == 0 or np.std(pred) == 0:
        return None
    return float(np.corrcoef(label, pred)[0, 1])


def _lin_concordance(label: np.ndarray, pred: np.ndarray) -> float | None:
    """Lin's concordance on the cell counts. Variances divide by n."""
    if label.size < 2:
        return None
    mean_label = float(label.mean())
    mean_pred = float(pred.mean())
    var_label = float(np.mean((label - mean_label) ** 2))
    var_pred = float(np.mean((pred - mean_pred) ** 2))
    cov = float(np.mean((label - mean_label) * (pred - mean_pred)))
    denom = var_label + var_pred + (mean_label - mean_pred) ** 2
    if denom == 0:
        return None
    return float(2.0 * cov / denom)


def _fold_title(summary: dict[str, Any]) -> str:
    fold_id = summary.get("holdout_fold")
    if fold_id is None:
        return "Holdout"
    return f"Fold {int(fold_id)} holdout"


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _scatter_limit(label: np.ndarray, pred: np.ndarray) -> tuple[float, int]:
    """Axis max at the 99th percentile when a few cells would otherwise hide the rest."""
    if label.size == 0:
        return 1.0, 0
    peak = np.maximum(label, pred)
    data_max = float(peak.max())
    p99 = float(np.percentile(peak, 99))
    if data_max > max(p99 * 1.5, p99 + 20):
        limit = max(10.0, math.ceil(p99 / 10.0) * 10.0)
    else:
        limit = max(data_max, 1.0)
    return limit, int((peak > limit).sum())


def _panel_label(ax, letter: str) -> None:
    ax.text(
        -0.14,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _plot_count_scatter(cells: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    """Panel A is every cell. Panel B is both counts above zero, zoomed to 0–20."""
    label = cells["label_count"].to_numpy(dtype=np.float64)
    pred = cells["pred_count"].to_numpy(dtype=np.float64)
    limit, n_beyond = _scatter_limit(label, pred)
    both = (cells["label_count"] > 0) & (cells["pred_count"] > 0)
    kept = cells.loc[both]
    zoom = 20.0
    n_outside = int(((kept["label_count"] > zoom) | (kept["pred_count"] > zoom)).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    hex_all = ax.hexbin(
        label,
        pred,
        gridsize=35,
        mincnt=1,
        bins="log",
        cmap="Blues",
        linewidths=0.2,
        edgecolors="white",
    )
    ax.plot([0, limit], [0, limit], color="#2c3e50", lw=1.0, label="1:1")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Label count")
    ax.set_ylabel("Predicted count")
    title_a = f"All cells, per {summary['cell_metres']:.0f} m"
    title_a += f"\n{n_beyond} beyond {limit:.0f}" if n_beyond else "\ncolour scale is log"
    if n_beyond:
        title_a += ", colour scale is log"
    ax.set_title(title_a)
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    _panel_label(ax, "A")
    fig.colorbar(hex_all, ax=ax, label="Cells (log)", fraction=0.046, pad=0.04)

    ax = axes[1]
    if len(kept):
        hex_pos = ax.hexbin(
            kept["label_count"],
            kept["pred_count"],
            gridsize=20,
            mincnt=1,
            extent=(0, zoom, 0, zoom),
            cmap="Blues",
            linewidths=0.2,
            edgecolors="white",
        )
        fig.colorbar(hex_pos, ax=ax, label="Cells", fraction=0.046, pad=0.04)
    ax.plot([0, zoom], [0, zoom], color="#2c3e50", lw=1.0, label="1:1")
    ax.set_xlim(0, zoom)
    ax.set_ylim(0, zoom)
    ax.set_aspect("equal")
    ax.set_xlabel("Label count")
    ax.set_ylabel("Predicted count")
    ax.set_title(
        f"Both counts above 0, zoomed to {zoom:.0f}"
        f"\n{len(kept):,} cells, {n_outside:,} outside this frame"
    )
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    _panel_label(ax, "B")
    _save(fig, path)


def _plot_residual_hist(cells: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    residual = cells["residual"].to_numpy(dtype=np.float64)
    lo = int(np.floor(residual.min()))
    hi = int(np.ceil(residual.max()))
    bins = np.arange(lo, hi + 2) - 0.5
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor("white")
    counts, edges, _ = ax.hist(
        residual, bins=bins, color="#1a5276", edgecolor="white", linewidth=0.4
    )
    ax.axvline(0, color="#c0392b", lw=1.0)
    x_lo, x_hi = -5, 30
    y_hi = 250
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, y_hi)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        centre = (left + right) / 2.0
        if count <= 0 or centre < x_lo or centre > x_hi:
            continue
        label = f"{int(count)}"
        if count >= y_hi - 18:
            ax.text(
                centre, y_hi - 4, label,
                rotation=90, ha="center", va="top", color="white", fontsize=7,
            )
        else:
            ax.text(
                centre, count + 2, label,
                rotation=90, ha="center", va="bottom", color="#1a1a1a", fontsize=7,
            )
    ax.set_xlabel("Predicted count − label count")
    ax.set_ylabel("Cells")
    ax.set_title(f"{_fold_title(summary)} — residual")
    _style_axes(ax)
    _save(fig, path)


def _plot_residual_map(cells: pd.DataFrame, summary: dict[str, Any], path: Path) -> None:
    ix = cells["cell_x"].to_numpy(dtype=np.int64)
    iy = cells["cell_y"].to_numpy(dtype=np.int64)
    ix0, ix1 = int(ix.min()), int(ix.max())
    iy0, iy1 = int(iy.min()), int(iy.max())
    grid = np.full((iy1 - iy0 + 1, ix1 - ix0 + 1), np.nan)
    grid[iy - iy0, ix - ix0] = cells["residual"].to_numpy(dtype=np.float64)
    cell_m = float(summary["cell_metres"])
    x_edges = np.arange(ix0, ix1 + 2) * cell_m
    y_edges = np.arange(iy0, iy1 + 2) * cell_m
    span = float(np.nanmax(np.abs(grid)))
    if not np.isfinite(span) or span == 0:
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    else:
        norm = TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    fig.patch.set_facecolor("white")
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{_fold_title(summary)} — residual, {cell_m:.0f} m cells")
    _style_axes(ax)
    fig.colorbar(mesh, ax=ax, label="Predicted − label")
    _save(fig, path)
