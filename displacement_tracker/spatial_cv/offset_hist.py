"""Nearest-neighbour offset figure for one holdout point score."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from displacement_tracker.spatial_cv.points import nearest_offset_metrics, upsert_point_summary

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fold_label(summary: dict) -> str:
    fold_id = summary.get("holdout_fold")
    if fold_id is None:
        return "Holdout"
    return f"Fold {int(fold_id)} holdout vs snapshot"


def write_offset_figure(
    fold_dir: Path,
    pred_lon: np.ndarray,
    pred_lat: np.ndarray,
    lab_lon: np.ndarray,
    lab_lat: np.ndarray,
    summary: dict,
    max_dist: float,
    summary_csv: Path | None = None,
) -> Path:
    """Histogram, score bars, and confusion matrix next to the point summary."""
    distances, nn_stats = nearest_offset_metrics(
        pred_lon, pred_lat, lab_lon, lab_lat, max_dist
    )
    row = dict(summary)
    row.update(nn_stats)
    row["max_dist_m"] = float(max_dist)

    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / "holdout_point_summary.json"
    summary_path.write_text(json.dumps(row, indent=2) + "\n")
    if summary_csv is not None:
        upsert_point_summary(Path(summary_csv), row)

    dist_csv = fold_dir / "offset_distances.csv"
    pd.DataFrame({"distance_m": distances}).to_csv(dist_csv, index=False)
    plot_summary = fold_dir / "offset_summary.csv"
    pd.DataFrame(
        [
            {
                "name": _fold_label(row),
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "mean_pos_error_m": row["mean_pos_error_m"],
                "nn_median_m": row["nn_median_m"],
                "nn_pct_beyond_gate": row["nn_pct_beyond_gate"],
                "nn_mean_beyond_m": row["nn_mean_beyond_m"],
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
            }
        ]
    ).to_csv(plot_summary, index=False)

    out_png = fold_dir / "offset_hist.png"
    _run_r(dist_csv, plot_summary, out_png, max_dist)
    return out_png


def write_offset_figure_from_points(
    fold_dir: Path,
    pred_points: list[tuple],
    label_lonlat: list[tuple[float, float]],
    summary: dict,
    max_dist: float,
    summary_csv: Path | None = None,
) -> Path:
    pred_lat = np.array([p[0] for p in pred_points], dtype=np.float64)
    pred_lon = np.array([p[1] for p in pred_points], dtype=np.float64)
    lab_lon = np.array([p[0] for p in label_lonlat], dtype=np.float64)
    lab_lat = np.array([p[1] for p in label_lonlat], dtype=np.float64)
    return write_offset_figure(
        fold_dir,
        pred_lon,
        pred_lat,
        lab_lon,
        lab_lat,
        summary,
        max_dist,
        summary_csv=summary_csv,
    )


def write_offset_figure_from_gpkg(
    fold_dir: Path,
    max_dist: float | None = None,
    summary_csv: Path | None = None,
) -> Path:
    """Redraw from holdout_pred_points.gpkg and holdout_label_points.gpkg."""
    fold_dir = Path(fold_dir)
    summary_path = fold_dir / "holdout_point_summary.json"
    summary = json.loads(summary_path.read_text())
    if max_dist is None:
        max_dist = float(summary.get("max_dist_m", 5.0))
    pred = gpd.read_file(fold_dir / "holdout_pred_points.gpkg")
    labels = gpd.read_file(fold_dir / "holdout_label_points.gpkg")
    return write_offset_figure(
        fold_dir,
        pred.geometry.x.to_numpy(),
        pred.geometry.y.to_numpy(),
        labels.geometry.x.to_numpy(),
        labels.geometry.y.to_numpy(),
        summary,
        max_dist,
        summary_csv=summary_csv,
    )


def _run_r(dist_csv: Path, summary_csv: Path, out_png: Path, gate: float) -> None:
    r_script = Path(__file__).with_name("plot_offset_hist.R")
    rlib = REPO_ROOT / ".Rlibrary"
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(rlib)
    result = subprocess.run(
        [
            "Rscript",
            str(r_script),
            str(dist_csv),
            str(summary_csv),
            str(out_png),
            str(gate),
            str(rlib),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"R histogram failed (exit {result.returncode})\n{result.stdout}\n{result.stderr}"
        )
