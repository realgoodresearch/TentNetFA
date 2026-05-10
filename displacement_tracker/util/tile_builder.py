from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform

from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.raster_processing import read_rgb

LOGGER = setup_logging("tile_builder")


@dataclass(frozen=True)
class TileWindow:
    r0: int
    r1: int
    c0: int
    c1: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    valid_fraction: float


def _compute_valid_fraction(data: np.ndarray, nodata) -> float:
    if nodata is not None:
        valid_mask = (~np.isnan(data)) & (data != nodata)
    else:
        valid_mask = (~np.isnan(data)) & (data != 0)
    return float(np.count_nonzero(valid_mask) / data.size)


def tile_pixel_size(src, span_m: float) -> int:
    """Square tile side in pixels given a metric span and raster resolution."""
    px = abs(src.transform.a)
    return int(round(span_m / px))


def compute_tile_window(
    src: rasterio.io.DatasetReader,
    x: float,
    y: float,
    core_m: float,
    margin_m: float,
    min_valid_fraction: float = 0.0,
) -> TileWindow | None:
    """Snap to a square pixel window, validate via RGB, return geo bbox + validity.

    `(x, y)` is the desired tile centre in the source raster's CRS (metres for
    a UTM raster). Callers iterate in source CRS so no degree-step is needed.
    """
    window = world_window(src, x, y, core_m, margin_m)
    if not window:
        return None

    rgb = read_rgb(src, window)
    if rgb is None:
        return None

    valid_fraction = _compute_valid_fraction(rgb, src.nodata)
    if valid_fraction < float(min_valid_fraction):
        return None

    r0, r1, c0, c1 = window
    x_ul, y_ul = xy(src.transform, r0, c0, offset="ul")
    x_lr, y_lr = xy(src.transform, r1 - 1, c1 - 1, offset="lr")
    lons, lats = transform(src.crs, "EPSG:4326", [x_ul, x_lr], [y_ul, y_lr])

    return TileWindow(
        r0=int(r0),
        r1=int(r1),
        c0=int(c0),
        c1=int(c1),
        lon_min=float(min(lons)),
        lon_max=float(max(lons)),
        lat_min=float(min(lats)),
        lat_max=float(max(lats)),
        valid_fraction=valid_fraction,
    )


def world_window(src, x: float, y: float, core_m: float, margin_m: float):
    """Compute a square pixel window centred on `(x, y)` in source-CRS metres.

    The window spans `core_m + 2*margin_m` metres on a side; pixel count is
    derived from `src.transform`, so the result is square-in-pixels by
    construction (assuming square pixels in the source CRS).
    """
    try:
        span_m = float(core_m) + 2.0 * float(margin_m)
        tile_px = tile_pixel_size(src, span_m)
        if tile_px <= 0 or tile_px > src.height or tile_px > src.width:
            return None

        col_f, row_f = (~src.transform) * (x, y)
        r_center = int(round(row_f))
        c_center = int(round(col_f))

        half = tile_px // 2
        r0 = max(0, min(r_center - half, src.height - tile_px))
        c0 = max(0, min(c_center - half, src.width - tile_px))

        return (r0, r0 + tile_px, c0, c0 + tile_px)
    except Exception:
        return None


def create_label_from_feats(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    feats: list[dict[str, Any]],
    label_shape,
):
    label = np.zeros(label_shape, dtype=np.uint8)
    h, w = label.shape
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    if lon_span == 0 or lat_span == 0:
        return label

    for feat in feats:
        feat_lon, feat_lat = feat["geometry"]["coordinates"]
        local_col = int(round((feat_lon - lon_min) / lon_span * (w - 1)))
        local_row = int(round((lat_max - feat_lat) / lat_span * (h - 1)))

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = local_row + dr
                cc = local_col + dc
                if 0 <= rr < h and 0 <= cc < w:
                    label[rr, cc] = 255
    return label


def _read_prewar_tile(
    prewar_src: rasterio.io.DatasetReader,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    span_m: float,
    min_valid_fraction: float,
) -> np.ndarray | None:
    """Read a prewar RGB tile centred on the saved WGS84 bbox.

    Sizes the window from the prewar raster's own pixel resolution and the
    metric tile side (`span_m = core_m + 2*margin_m`), so output is square in
    pixels regardless of how the source tile's pixels and prewar pixels relate.
    """
    try:
        tile_px = tile_pixel_size(prewar_src, span_m)
        if tile_px <= 0 or tile_px > prewar_src.height or tile_px > prewar_src.width:
            return None

        lon_c = 0.5 * (lon_min + lon_max)
        lat_c = 0.5 * (lat_min + lat_max)
        pxs, pys = transform(
            "EPSG:4326", prewar_src.crs, [lon_c], [lat_c]
        )
        col_f, row_f = (~prewar_src.transform) * (pxs[0], pys[0])
        r_center = int(round(row_f))
        c_center = int(round(col_f))

        half = tile_px // 2
        rpre0 = max(0, min(r_center - half, prewar_src.height - tile_px))
        cpre0 = max(0, min(c_center - half, prewar_src.width - tile_px))
        rpre1 = rpre0 + tile_px
        cpre1 = cpre0 + tile_px

        pre_data = prewar_src.read(
            [1, 2, 3], window=((rpre0, rpre1), (cpre0, cpre1))
        )
        if pre_data.size == 0:
            return None

        prewar_tile_rgb = pre_data.astype(np.float32)

        try:
            nodata_pre = prewar_src.nodata
            if nodata_pre is not None:
                valid_mask = (~np.isnan(prewar_tile_rgb)) & (
                    prewar_tile_rgb != nodata_pre
                )
            else:
                valid_mask = (~np.isnan(prewar_tile_rgb)) & (prewar_tile_rgb != 0)
            valid_fraction_pre = (
                np.count_nonzero(valid_mask) / prewar_tile_rgb.size
            )
        except Exception:
            valid_fraction_pre = 0.0

        if valid_fraction_pre < float(min_valid_fraction):
            return None

        return prewar_tile_rgb
    except Exception:
        return None
