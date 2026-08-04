from __future__ import annotations

import math
import os
from collections import defaultdict
from datetime import date, datetime
from typing import Any


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
    features: list[dict[str, Any]],
    core_m: float,
    margin_m: float,
    transformer_to_src,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Bin features by source-CRS metric cell (i, j), where the tile centred at
    `(i*core_m, j*core_m)` spans `core_m + 2*margin_m` metres on a side.

    Each feature is added to every cell whose tile geometrically contains it
    — which for `margin_m < core_m` is at most a 3x3 neighbourhood, but the
    inclusion test here is exact and works for any margin.
    """
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    half = (core_m + 2.0 * margin_m) / 2.0
    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) != 2:
            continue
        lon, lat = coords
        fx, fy = transformer_to_src.transform(lon, lat)
        i_min = math.ceil((fx - half) / core_m)
        i_max = math.floor((fx + half) / core_m)
        j_min = math.ceil((fy - half) / core_m)
        j_max = math.floor((fy + half) / core_m)
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                grouped[(i, j)].append(feat)
    return grouped
