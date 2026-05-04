from __future__ import annotations

from typing import Any

import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform

from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.raster_processing import read_rgb

LOGGER = setup_logging("tile_builder")

# Canonical tile size shared by all flows. These were previously module-level
# globals lazily set inside helper functions; pinning them as constants keeps
# the same behaviour without the global-state shuffle.
TILE_WIDTH: int = 191
TILE_HEIGHT: int = 224


def world_window(src, lon, lat, step, transformer):
    """Compute a fixed-size pixel window centred on (lon, lat)."""
    try:
        lons = [lon - step, lon - step, lon + 2 * step, lon + 2 * step]
        lats = [lat - step, lat + 2 * step, lat - step, lat + 2 * step]

        xs, ys = transformer.transform(lons, lats)

        rows: list[int] = []
        cols: list[int] = []
        for x, y in zip(xs, ys):
            try:
                r, c = src.index(x, y)
                rows.append(int(r))
                cols.append(int(c))
            except Exception:
                continue

        if not rows or not cols:
            return None

        r0 = max(0, min(rows))
        r1 = min(src.height, max(rows))
        c0 = max(0, min(cols))
        c1 = min(src.width, max(cols))

        if r0 >= r1 or c0 >= c1:
            return None

        tile_w = min(TILE_WIDTH, src.width)
        tile_h = min(TILE_HEIGHT, src.height)
        w = c1 - c0
        h = r1 - r0

        if w != tile_w:
            desired_c1 = c0 + tile_w
            if desired_c1 <= src.width:
                c1 = desired_c1
            else:
                c0 = max(0, src.width - tile_w)
                c1 = c0 + tile_w

        if h != tile_h:
            desired_r1 = r0 + tile_h
            if desired_r1 <= src.height:
                r1 = desired_r1
            else:
                r0 = max(0, src.height - tile_h)
                r1 = r0 + tile_h

        if r0 >= r1 or c0 >= c1:
            return None

        return (r0, r1, c0, c1)
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
    min_valid_fraction: float,
) -> np.ndarray | None:
    """Read & size-normalise a prewar RGB tile aligned to the same WGS84 bbox."""
    try:
        pxs, pys = transform(
            "EPSG:4326", prewar_src.crs, [lon_min, lon_max], [lat_min, lat_max]
        )
        rpre_a, cpre_a = prewar_src.index(pxs[0], pys[0])
        rpre_b, cpre_b = prewar_src.index(pxs[1], pys[1])
        rpre0, rpre1 = sorted([int(rpre_a), int(rpre_b)])
        cpre0, cpre1 = sorted([int(cpre_a), int(cpre_b)])

        rpre0 = max(0, rpre0)
        rpre1 = min(prewar_src.height, rpre1)
        cpre0 = max(0, cpre0)
        cpre1 = min(prewar_src.width, cpre1)
        if rpre0 >= rpre1 or cpre0 >= cpre1:
            return None

        tile_w = min(TILE_WIDTH, prewar_src.width)
        tile_h = min(TILE_HEIGHT, prewar_src.height)

        if (cpre1 - cpre0) != tile_w:
            desired_c1 = cpre0 + tile_w
            if desired_c1 <= prewar_src.width:
                cpre1 = desired_c1
            else:
                cpre0 = max(0, prewar_src.width - tile_w)
                cpre1 = cpre0 + tile_w

        if (rpre1 - rpre0) != tile_h:
            desired_r1 = rpre0 + tile_h
            if desired_r1 <= prewar_src.height:
                rpre1 = desired_r1
            else:
                rpre0 = max(0, prewar_src.height - tile_h)
                rpre1 = rpre0 + tile_h

        if rpre0 >= rpre1 or cpre0 >= cpre1:
            return None

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


def process_group(
    src: rasterio.io.DatasetReader,
    feats: list[dict[str, Any]],
    lon: float,
    lat: float,
    step: float,
    origin_image: str,
    origin_date: str,
    transformer,
    prewar_src: rasterio.io.DatasetReader | None = None,
    min_valid_fraction: float = 0.0,
) -> tuple[
    np.ndarray | None, np.ndarray | None, dict[str, Any] | None, np.ndarray | None
]:
    try:
        window = world_window(src, lon, lat, step, transformer)
        if not window:
            return None, None, None, None

        rgb = read_rgb(src, window)
        if rgb is None:
            return None, None, None, None

        try:
            nodata = src.nodata
            if nodata is not None:
                valid_mask = (~np.isnan(rgb)) & (rgb != nodata)
            else:
                valid_mask = (~np.isnan(rgb)) & (rgb != 0)
            valid_fraction = np.count_nonzero(valid_mask) / rgb.size
        except Exception:
            valid_fraction = 0.0

        if valid_fraction < float(min_valid_fraction):
            return None, None, None, None

        r0, r1, c0, c1 = window

        x_ul, y_ul = xy(src.transform, r0, c0, offset="ul")
        x_lr, y_lr = xy(src.transform, r1 - 1, c1 - 1, offset="lr")

        lons, lats = transform(src.crs, "EPSG:4326", [x_ul, x_lr], [y_ul, y_lr])
        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)

        label = create_label_from_feats(
            lon_min, lat_min, lon_max, lat_max, feats, (rgb.shape[1], rgb.shape[2])
        )
        if (rgb.shape[1], rgb.shape[2]) != label.shape:
            LOGGER.warning(
                "shape mismatch for %s: rgb %s vs label %s; skipping tile",
                origin_image,
                (rgb.shape[1], rgb.shape[2]),
                label.shape,
            )
            return None, None, None, None

        meta = {
            "origin_image": origin_image,
            "origin_date": origin_date,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "lat_min": lat_min,
            "lat_max": lat_max,
        }

        prewar_tile_rgb = None
        if prewar_src:
            try:
                prewar_tile_rgb = _read_prewar_tile(
                    prewar_src, lon_min, lon_max, lat_min, lat_max, min_valid_fraction
                )
            except Exception:
                LOGGER.exception("Failed to build prewar tile for %s", origin_image)

        h = rgb.shape[1]
        w = rgb.shape[2]
        if prewar_tile_rgb is None:
            # Previously this allowed for training tiles without prewar data;
            # we now require a prewar tile to keep dataset shape consistent.
            return None, None, None, None

        ph, pw = prewar_tile_rgb.shape[1], prewar_tile_rgb.shape[2]
        if (ph, pw) != (h, w):
            adjusted = np.zeros((3, h, w), dtype=np.float16)
            copy_h = min(ph, h)
            copy_w = min(pw, w)
            adjusted[:, :copy_h, :copy_w] = prewar_tile_rgb[:, :copy_h, :copy_w]
            prewar_rgb = adjusted
        else:
            prewar_rgb = prewar_tile_rgb.astype(np.float16, copy=False)

        return rgb, label, meta, prewar_rgb
    except Exception:
        LOGGER.exception("Failed to process group")
        return None, None, None, None
