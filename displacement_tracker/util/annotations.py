from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import date, datetime
from typing import Any

import numpy as np
import rasterio


def parse_date_safe(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def extract_date_from_filename(path: str) -> str | None:
    name = os.path.splitext(os.path.basename(path))[0]
    for part in name.split("_"):
        if part.isdigit() and len(part) == 8:
            return part
    return None


def filter_tents_by_target_date(
    features: list[dict[str, Any]], target: date
) -> list[dict[str, Any]]:
    """
    Keep features that match the TIFF date by either:
      - start == target, OR
      - start <= target <= end (when end exists)
    """
    out: list[dict[str, Any]] = []
    for f in features:
        props = f.get("properties", {}) or {}
        start = parse_date_safe(props.get("date_start"))
        end = parse_date_safe(props.get("date_end"))

        if start is None:
            continue
        if start == target:
            out.append(f)
            continue
        if end and start <= target <= end:
            out.append(f)
    return out


def group_coords(
    features: list[dict[str, Any]], step: float
) -> dict[tuple[float, float], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) != 2:
            continue
        lon, lat = coords
        base_lon = math.floor(lon / step) * step
        base_lat = math.floor(lat / step) * step
        # 3x5 neighbourhood to ensure all subtiles are labelled in the dataset
        # (previously, upper subtiles had labels = 0 which caused underfitting)
        for i in (-1, 0, 1):
            for j in (-2, -1, 0, 1, 2):
                grouped[
                    (round(base_lon + i * step, 5), round(base_lat + j * step, 5))
                ].append(feat)
    return grouped


def is_high_quality_tile(
    feats: list[dict[str, Any]],
    date_target_str: str,
    src: rasterio.io.DatasetReader,
    lon: float,
    lat: float,
    step: float,
    start_threshold: float,
    max_missing_end: float,
    min_valid_fraction: float,
    transformer,
) -> bool:
    """
    Checks date distributions and raster valid-pixel fraction for the tile.
    """
    from displacement_tracker.util.tile_builder import world_window

    if not feats or not date_target_str:
        return False
    try:
        date_target = datetime.strptime(date_target_str, "%Y%m%d").date()
    except Exception:
        return False

    start_matches = sum(
        parse_date_safe(f.get("properties", {}).get("date_start")) == date_target
        for f in feats
    )
    missing_end = sum(
        parse_date_safe(f.get("properties", {}).get("date_end")) is None for f in feats
    )
    n = len(feats)
    if n == 0:
        return False
    if (start_matches / n) < start_threshold or (missing_end / n) > max_missing_end:
        return False

    window = world_window(src, lon, lat, step, transformer)
    if not window:
        return False
    try:
        data = src.read(
            [1, 2, 3], window=((window[0], window[1]), (window[2], window[3]))
        )
        if data.size == 0:
            return False
        nodata = src.nodata
        if nodata is not None:
            valid_mask = (~np.isnan(data)) & (data != nodata)
        else:
            valid_mask = (~np.isnan(data)) & (data != 0)
        valid_fraction = np.count_nonzero(valid_mask) / data.size
        return valid_fraction >= min_valid_fraction
    except Exception:
        return False
