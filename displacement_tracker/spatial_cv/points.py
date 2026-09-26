"""Holdout tent points vs the snapshot coordinates those tiles were labelled with."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from displacement_tracker.d_train_cnn import custom_collate
from displacement_tracker.e_predict_json import (
    NMS_SIGMA_FRACTION,
    PIXEL_METRES,
    extract_tile_nms,
)
from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.point_match.match import match_points, nearest_distances
from displacement_tracker.simple_cnn import SimpleCNN
from displacement_tracker.spatial_cv.folds import train_test_masks
from displacement_tracker.util.deduplication import merge_close_points_global
from displacement_tracker.util.thresholding import filter_points_by_adjusted_peak

MATCH_CRS = "EPSG:32636"
LABEL_SIGMA = 3.0


def detection_settings(predict_params: dict) -> dict[str, Any]:
    """NMS and peak filter shared with the predict/merge flow.

    ``crop_pixels`` and ``nms_sigma`` follow ``e_predict_json``: they are
    derived from ``processing.margin_metres``, not stored in the YAML.
    """
    pred = predict_params.get("prediction") or {}
    selection = dict(pred.get("selection") or {})
    merge = predict_params.get("merge") or {}
    processing = predict_params.get("processing") or {}
    if "margin_metres" not in processing:
        raise click.ClickException("Missing required config key: processing.margin_metres")
    margin_pixels = int(round(float(processing["margin_metres"]) / PIXEL_METRES))
    agreement = merge.get("agreement", 1)
    if isinstance(agreement, bool):
        agreement = 2 if agreement else 1
    min_distance = merge.get("min_distance_m", selection.get("min_distance_m", 3.0))
    return {
        "threshold": float(selection.get("threshold", 0.0001)),
        "factor": float(selection.get("factor", 1.0)),
        "nms_kernel_size": int(selection.get("nms_kernel_size", 7)),
        "crop_pixels": margin_pixels,
        "nms_sigma": float(NMS_SIGMA_FRACTION * margin_pixels),
        "min_distance_m": float(min_distance),
        "min_adj_peak": float(merge.get("min_adj_peak", 0.0)),
        "adjustment_factor": float(merge.get("adjustment_factor", 1.0)),
        "agreement": int(agreement),
        "batch_size": int(pred.get("batch_size", 8)),
    }


def resolve_device(name: str | None) -> torch.device:
    if torch.cuda.is_available() and (name is None or name == "cuda"):
        return torch.device("cuda")
    if name is None or name == "cpu":
        return torch.device("cpu")
    raise click.ClickException(f"Could not find device {name}")


def load_fold_model(checkpoint: str | Path, device: torch.device) -> SimpleCNN:
    model = SimpleCNN.from_pth(
        str(checkpoint),
        map_location=device,
        model_args={"n_channels": 9, "n_classes": 1},
    )
    model.to(device)
    model.eval()
    return model


def holdout_row_indices(
    n_rows: int,
    folds_path: str | Path,
    fold_id: int,
) -> tuple[list[int], dict[str, Any]]:
    folds_path = Path(folds_path)
    folds = pd.read_parquet(folds_path)
    meta_path = folds_path.with_name("folds_meta.json")
    if not meta_path.exists():
        raise click.ClickException(f"Missing fold metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if len(folds) != n_rows:
        raise click.ClickException(
            f"Fold table has {len(folds)} rows but the manifest has {n_rows}; "
            "re-run assign-folds on the same training manifest."
        )
    _, test_mask = train_test_masks(folds, int(fold_id), meta)
    indices = [i for i, keep in enumerate(test_mask.tolist()) if keep]
    if not indices:
        raise click.ClickException(f"Holdout fold {fold_id} has no tiles")
    return indices, meta


def open_labelled_dataset(manifest: str | Path) -> PairedImageDataset:
    """Training view of the manifest: dataset-wide standardisation, σ=3 labels."""
    return PairedImageDataset(
        manifest,
        sigma=LABEL_SIGMA,
        per_tile_standardisation=False,
    )


def row_features(dataset: PairedImageDataset, row: dict[str, Any]) -> list[dict[str, Any]]:
    labels_path = row.get("labels_path")
    feature_ids = row.get("label_feature_ids")
    if not labels_path or not feature_ids:
        return []
    features = dataset._labels_cache.get(labels_path, [])
    return [features[i] for i in feature_ids if 0 <= i < len(features)]


def _lon_lat(feat: dict[str, Any]) -> tuple[float, float] | None:
    geom = feat.get("geometry") or {}
    coords = geom.get("coordinates")
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    lon, lat = coords[0], coords[1]
    if isinstance(lon, (list, tuple)) or isinstance(lat, (list, tuple)):
        return None
    try:
        return float(lon), float(lat)
    except (TypeError, ValueError):
        return None


def _pixel(lon: float, lat: float, row: dict[str, Any]) -> tuple[int, int] | None:
    """Same pixel mapping as ``create_label_from_feats``."""
    h = int(row["r1"]) - int(row["r0"])
    w = int(row["c1"]) - int(row["c0"])
    lon_min = float(row["lon_min"])
    lon_max = float(row["lon_max"])
    lat_min = float(row["lat_min"])
    lat_max = float(row["lat_max"])
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    if h <= 0 or w <= 0 or lon_span == 0 or lat_span == 0:
        return None
    local_col = int(round((lon - lon_min) / lon_span * (w - 1)))
    local_row = int(round((lat_max - lat) / lat_span * (h - 1)))
    if not (0 <= local_row < h and 0 <= local_col < w):
        return None
    return local_row, local_col


def _in_core(local_row: int, local_col: int, row: dict[str, Any], crop_pixels: int) -> bool:
    h = int(row["r1"]) - int(row["r0"])
    w = int(row["c1"]) - int(row["c0"])
    crop = int(crop_pixels)
    if crop <= 0:
        return True
    return crop <= local_row < h - crop and crop <= local_col < w - crop


def snapshot_core_hits(
    row: dict[str, Any],
    features: list[dict[str, Any]],
    crop_pixels: int,
) -> list[tuple[int, int, float, float]]:
    """Snapshot tents inside the tile core: (pixel_row, pixel_col, lon, lat)."""
    hits = []
    for feat in features:
        lon_lat = _lon_lat(feat)
        if lon_lat is None:
            continue
        lon, lat = lon_lat
        pixel = _pixel(lon, lat, row)
        if pixel is None:
            continue
        local_row, local_col = pixel
        if _in_core(local_row, local_col, row, crop_pixels):
            hits.append((local_row, local_col, lon, lat))
    return hits


def core_snapshot_counts(
    dataset: PairedImageDataset,
    row_indices: list[int],
    crop_pixels: int,
) -> list[tuple[int, int]]:
    """(manifest row index, core snapshot count) for every holdout tile."""
    counts = []
    for idx in row_indices:
        row = dataset._rows[idx]
        n = len(snapshot_core_hits(row, row_features(dataset, row), crop_pixels))
        counts.append((idx, n))
    return counts


def collect_snapshot_points(
    dataset: PairedImageDataset,
    row_indices: list[int],
    crop_pixels: int,
) -> list[tuple[float, float]]:
    """Unique (lon, lat) snapshot tents that fall in the core of a holdout tile.

    A tent in the margin of one tile and the core of the next is kept once.
    """
    seen: dict[tuple[float, float], tuple[float, float]] = {}
    for idx in row_indices:
        row = dataset._rows[idx]
        for _r, _c, lon, lat in snapshot_core_hits(
            row, row_features(dataset, row), crop_pixels
        ):
            seen.setdefault((round(lon, 7), round(lat, 7)), (lon, lat))
    return list(seen.values())


def detect_tile_points(probs_np: np.ndarray, bounds: dict, settings: dict) -> list[tuple]:
    """NMS peaks on one density map, then the adjusted-peak filter."""
    coords = extract_tile_nms(
        probs_np,
        bounds,
        settings["threshold"],
        factor=settings["factor"],
        kernel_size=settings["nms_kernel_size"],
        sigma=settings["nms_sigma"],
        crop_pixels=settings["crop_pixels"],
    )
    return filter_points_by_adjusted_peak(
        coords,
        settings["min_adj_peak"],
        settings["adjustment_factor"],
    )


def predict_tile_detections(
    dataset: PairedImageDataset,
    row_indices: list[int],
    model: SimpleCNN,
    device: torch.device,
    settings: dict,
) -> list[tuple[int, list]]:
    """(manifest row, detections on that tile) before tiles are merged."""
    dataset.indices = list(row_indices)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(settings["batch_size"])),
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate,
        pin_memory=device.type == "cuda",
    )
    tiles: list[tuple[int, list]] = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for entry in tqdm(loader, desc="holdout predict", unit="batch"):
            feature = entry["feature"]
            prewar = entry["prewar"]
            diff = feature - prewar
            feats = torch.cat((feature, prewar, diff), dim=1).to(device)
            probs = model(feats).detach().cpu().numpy()
            for b in range(probs.shape[0]):
                idx = row_indices[offset + b]
                bounds = json.loads(entry["meta"][b])
                detected = detect_tile_points(probs[b, 0], bounds, settings)
                tiles.append((idx, detected))
            offset += int(probs.shape[0])
    return tiles


def merged_tile_detections(tiles: list[tuple[int, list]], settings: dict) -> list[tuple]:
    """Global merge of per-tile detections. This is what the fold score uses."""
    flat = [point for _idx, detected in tiles for point in detected]
    return merge_close_points_global(
        flat,
        min_distance_m=settings["min_distance_m"],
        agreement=settings["agreement"],
    )


def predict_holdout_points(
    dataset: PairedImageDataset,
    row_indices: list[int],
    model: SimpleCNN,
    device: torch.device,
    settings: dict,
) -> list[tuple]:
    """Detected (lat, lon, peak, adjusted_peak) on the holdout tiles, merged."""
    tiles = predict_tile_detections(dataset, row_indices, model, device, settings)
    return merged_tile_detections(tiles, settings)


def _xy(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if len(lon) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    frame = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    ).to_crs(MATCH_CRS)
    return np.column_stack([frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()])


def match_lonlat(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    lab_lon: np.ndarray,
    lab_lat: np.ndarray,
    max_dist: float,
) -> dict[str, Any]:
    """Greedy 1-to-1 match in the scoring CRS. ``matched_pairs`` is (pred, label, metres)."""
    pred_lat = np.asarray(pred_lat, dtype=np.float64).reshape(-1)
    pred_lon = np.asarray(pred_lon, dtype=np.float64).reshape(-1)
    lab_lon = np.asarray(lab_lon, dtype=np.float64).reshape(-1)
    lab_lat = np.asarray(lab_lat, dtype=np.float64).reshape(-1)
    return match_points(
        _xy(pred_lon, pred_lat),
        _xy(lab_lon, lab_lat),
        max_dist=max_dist,
    )


def score_points(
    pred_points: list[tuple],
    label_lonlat: list[tuple[float, float]],
    max_dist: float,
) -> dict[str, Any]:
    pred_lat = np.array([p[0] for p in pred_points], dtype=np.float64)
    pred_lon = np.array([p[1] for p in pred_points], dtype=np.float64)
    lab_lon = np.array([p[0] for p in label_lonlat], dtype=np.float64)
    lab_lat = np.array([p[1] for p in label_lonlat], dtype=np.float64)
    metrics = match_lonlat(pred_lat, pred_lon, lab_lon, lab_lat, max_dist)
    return {
        "n_pred": int(len(pred_points)),
        "n_label": int(len(label_lonlat)),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mean_pos_error_m": metrics["mean_pos_error_m"],
        "rmse_pos_error_m": metrics["rmse_pos_error_m"],
        "max_dist_m": float(max_dist),
    }


def nearest_offset_metrics(
    pred_lon: np.ndarray,
    pred_lat: np.ndarray,
    lab_lon: np.ndarray,
    lab_lat: np.ndarray,
    max_dist: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Nearest snapshot tent for every prediction, including those outside the gate."""
    nn = nearest_distances(
        _xy(np.asarray(pred_lon, dtype=np.float64), np.asarray(pred_lat, dtype=np.float64)),
        _xy(np.asarray(lab_lon, dtype=np.float64), np.asarray(lab_lat, dtype=np.float64)),
    )
    finite = nn[np.isfinite(nn)]
    beyond = finite[finite > max_dist]
    n = int(finite.size)
    stats = {
        "nn_median_m": float(np.median(finite)) if n else None,
        "nn_mean_m": float(np.mean(finite)) if n else None,
        "nn_pct_within_gate": float((finite <= max_dist).mean()) if n else None,
        "nn_pct_beyond_gate": float((finite > max_dist).mean()) if n else None,
        "nn_n_beyond": int(beyond.size),
        "nn_mean_beyond_m": float(beyond.mean()) if beyond.size else None,
        "nn_median_beyond_m": float(np.median(beyond)) if beyond.size else None,
    }
    return finite, stats


def write_pred_gpkg(path: Path, points: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points:
        rows = [
            {
                "geometry": Point(lon, lat),
                "name": "tents",
                "peak_value": peak,
                "adjusted_peak": adj_peak,
            }
            for lat, lon, peak, adj_peak in points
        ]
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    else:
        frame = gpd.GeoDataFrame(
            {
                "name": [],
                "peak_value": [],
                "adjusted_peak": [],
                "geometry": gpd.GeoSeries([], crs="EPSG:4326"),
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
    frame.to_file(path, driver="GPKG")


def write_label_gpkg(path: Path, lonlat: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if lonlat:
        frame = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([p[0] for p in lonlat], [p[1] for p in lonlat]),
            crs="EPSG:4326",
        )
    else:
        frame = gpd.GeoDataFrame(
            {"geometry": gpd.GeoSeries([], crs="EPSG:4326")},
            geometry="geometry",
            crs="EPSG:4326",
        )
    frame.to_file(path, driver="GPKG")


def upsert_point_summary(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        frame = frame.loc[frame["holdout_fold"] != row["holdout_fold"]]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    else:
        frame = pd.DataFrame([row])
    frame = frame.sort_values("holdout_fold")
    frame.to_csv(csv_path, index=False)
