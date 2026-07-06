"""Shared utilities for validating predicted point sets against reference data.

The validation flow resolves predictions and a reference source (see
``util/reference_data.py``) onto a master grid, restricted to the convex
hull of the predictions, then derives per-cell error metrics inside that
hull.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features, mask
from rasterio.transform import rowcol
from scipy.stats import spearmanr

from displacement_tracker.util.reference_data import (
    PointsSource,
    ReferenceSource,
)
from displacement_tracker.util.thresholding import (
    passes_threshold,
    rescale_adjusted_peak,
)


# Direction of optimization for each metric: "min" = lower is better.
METRIC_DIRECTIONS: Dict[str, str] = {
    "rms": "min",
    "mae": "min",
    "rmsle": "min",
    "abs_total_diff": "min",
    "abs_total_pdiff": "min",
    "spearman": "max",
}


def list_point_files(directory: str) -> List[str]:
    """Sorted point-set files (predictions) in a directory."""
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".gpkg", ".geojson", ".json"))
    )


def prepare_grouped_cell_inputs(
    pred_gdf: gpd.GeoDataFrame,
    reference: Union[ReferenceSource, gpd.GeoDataFrame],
    src_grid: rasterio.io.DatasetReader,
    nodata_val: float = -9999.0,
) -> Dict[str, object]:
    """Build one-time geometry/grid products and per-point cell assignments.

    ``reference`` is any :class:`ReferenceSource`; a plain point
    GeoDataFrame is accepted as a convenience and wrapped on the fly.
    """
    if isinstance(reference, gpd.GeoDataFrame):
        reference = PointsSource(reference)

    prediction_extent_geom = pred_gdf.union_all().convex_hull

    out_image, out_transform = mask.mask(
        src_grid,
        [prediction_extent_geom],
        crop=True,
        nodata=np.nan,
    )
    grid_shape = (out_image.shape[1], out_image.shape[2])

    mask_array = features.geometry_mask(
        [prediction_extent_geom],
        out_shape=grid_shape,
        transform=out_transform,
        invert=True,
    )

    val_raster = reference.counts_on_grid(
        grid_shape,
        out_transform,
        src_grid.crs,
        clip_geom=prediction_extent_geom,
    )

    xs = pred_gdf.geometry.x.to_numpy()
    ys = pred_gdf.geometry.y.to_numpy()
    rows, cols = rowcol(out_transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)

    in_bounds = (
        (rows >= 0)
        & (rows < grid_shape[0])
        & (cols >= 0)
        & (cols < grid_shape[1])
    )

    pred_prepped = pred_gdf.loc[in_bounds, ["peak_value", "adjusted_peak"]].copy()
    pred_prepped["row"] = rows[in_bounds]
    pred_prepped["col"] = cols[in_bounds]

    return {
        "pred_prepped": pred_prepped,
        "val_raster": val_raster,
        "mask_array": mask_array,
        "out_transform": out_transform,
        "grid_shape": grid_shape,
        "nodata_val": nodata_val,
    }


def process_grouped_cells(
    pred_rows: np.ndarray,
    pred_cols: np.ndarray,
    val_raster: np.ndarray,
    mask_array: np.ndarray,
    grid_shape: Tuple[int, int],
    nodata_val: float = -9999.0,
) -> Dict[str, np.ndarray]:
    """Build a prediction raster from pre-grouped cells and derive diff/error rasters.

    `val_raster` is mutated in place to hold `nodata_val` outside the mask, so the
    caller must pass a copy if the pristine raster is needed afterwards.
    """
    pred_raster = np.zeros(grid_shape, dtype=np.float32)
    if pred_rows.size > 0:
        np.add.at(pred_raster, (pred_rows, pred_cols), 1.0)

    diff = pred_raster - val_raster
    error_raster = np.log1p(pred_raster) - np.log1p(val_raster)

    for arr in [pred_raster, val_raster, diff, error_raster]:
        arr[~mask_array] = nodata_val

    return {
        "pred_raster": pred_raster,
        "val_raster": val_raster,
        "diff": diff,
        "error_raster": error_raster,
        "mask_array": mask_array,
    }


def keep_mask_from_params(pred_prepped, factor: float, cutoff: float) -> np.ndarray:
    """Return a boolean keep-mask for predictions given a rescaling factor and cutoff.

    The rescaled peak is `peak_value + factor * (adjusted_peak - peak_value)`;
    a point is kept iff its rescaled peak is >= `cutoff`.
    """
    rescaled = rescale_adjusted_peak(
        pred_prepped["peak_value"], pred_prepped["adjusted_peak"], factor
    )
    return passes_threshold(rescaled, cutoff).to_numpy()


def compute_metrics(
    pred_raster: np.ndarray,
    val_raster: np.ndarray,
    error_raster: np.ndarray,
    mask_array: np.ndarray,
) -> Dict[str, float]:
    """Compute per-tile error metrics restricted to the analysis mask."""
    pred_in = pred_raster[mask_array]
    val_in = val_raster[mask_array]
    err_in = error_raster[mask_array]
    diff_in = pred_in - val_in

    total_pred = float(pred_in.sum())
    total_val = float(val_in.sum())
    total_diff = total_pred - total_val
    total_pdiff = total_diff / total_val if total_val > 0 else 0.0

    if diff_in.size > 0:
        rms = float(np.sqrt(np.mean(np.square(diff_in))))
        mae = float(np.mean(np.abs(diff_in)))
        rmsle = float(np.sqrt(np.mean(np.square(err_in))))
    else:
        rms = mae = rmsle = float("inf")

    if (
        pred_in.size > 1
        and np.any(pred_in != pred_in[0])
        and np.any(val_in != val_in[0])
    ):
        spearman = float(spearmanr(pred_in, val_in).correlation)
    else:
        spearman = float("nan")

    return {
        "n_cells": int(diff_in.size),
        "total_pred": total_pred,
        "total_val": total_val,
        "total_diff": total_diff,
        "abs_total_diff": abs(total_diff),
        "total_pdiff": total_pdiff,
        "abs_total_pdiff": abs(total_pdiff),
        "rms": rms,
        "mae": mae,
        "rmsle": rmsle,
        "spearman": spearman,
    }


def write_output_rasters(
    out_dir: str,
    base_name: str,
    pred_raster: np.ndarray,
    val_raster: np.ndarray,
    diff_masked: np.ndarray,
    src_grid: rasterio.io.DatasetReader,
    grid_shape: Tuple[int, int],
    out_transform: rasterio.Affine,
    nodata_val: float = -9999.0,
) -> None:
    """Save predicted-count, validation-count, and diff rasters for one tile."""
    meta = src_grid.meta.copy()
    meta.update(
        {
            "height": grid_shape[0],
            "width": grid_shape[1],
            "transform": out_transform,
            "dtype": "float32",
            "nodata": nodata_val,
            "count": 1,
        }
    )

    outputs = {
        "diff": diff_masked,
        "pred_count": pred_raster,
        "val_count": val_raster,
    }
    for suffix, data in outputs.items():
        out_path = Path(out_dir) / f"{suffix}_100m" / f"{base_name}_{suffix}_100m.tif"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dest:
            dest.write(data, 1)


def initial_best_value(metric: str) -> float:
    """Sentinel value worse than any real measurement for the given metric."""
    return float("inf") if METRIC_DIRECTIONS[metric] == "min" else -float("inf")


def is_better(metric: str, candidate: float, current_best: float) -> bool:
    """Return True if `candidate` improves on `current_best` for this metric."""
    if not np.isfinite(candidate):
        return False
    if METRIC_DIRECTIONS[metric] == "min":
        return candidate < current_best
    return candidate > current_best
