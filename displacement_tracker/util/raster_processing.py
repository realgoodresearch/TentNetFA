from __future__ import annotations

import os

import fiona
import numpy as np
import rasterio
from rasterio import mask
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_geom
from rasterio.windows import Window

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("raster_processing")

# GDAL metadata tag used to mark a GeoTIFF as already cropped to the project
# boundaries shapefile. Set on the new dataset after a successful crop and
# checked at the start of `crop_src_to_boundaries` to make the operation
# idempotent across re-runs.
_CROPPED_TAG_KEY = "DT_CROPPED_TO_BOUNDARIES"
_CROPPED_TAG_VALUE = "1"

# Tunables for the on-disk standardisation pipeline.
# Larger CHUNK_SIZE reduces Python/GDAL overhead per call; 1024×1024 floats
# across N bands is ~12 MB per chunk for 3 bands — comfortable.
_STANDARDISE_CHUNK_SIZE = 1024
# Sample every Nth chunk for the stats pass. Per-channel mean/std are very
# stable from a quarter-sample, and skipping 75% of the read I/O is the
# single biggest win for large rasters.
_STATS_SAMPLE_STRIDE = 20


def open_raster(path: str):
    name = os.path.basename(path)
    LOGGER.info(f"[open:{name}] opening GeoTIFF: {path}")
    try:
        ds = rasterio.open(path)
    except RasterioIOError:
        LOGGER.exception(f"[open:{name}] FAILED to open GeoTIFF: {path}")
        return None
    LOGGER.info(
        f"[open:{name}] opened: dims={ds.width}x{ds.height}, bands={ds.count}, "
        f"dtype={ds.dtypes[0] if ds.dtypes else 'unknown'}, crs={ds.crs}, "
        f"nodata={ds.nodata}"
    )
    return ds


def read_rgb(src: rasterio.io.DatasetReader, window):
    """
    Read RGB bands from src for the given window and return a float32 array
    with shape (3, H, W) or None if the window contains no valid data.
    """
    data = src.read([1, 2, 3], window=((window[0], window[1]), (window[2], window[3])))
    if data.size == 0 or np.all(np.isnan(data)) or np.all(data == 0):
        return None
    return data.astype(np.float32)


def _iter_chunk_windows(width: int, height: int, chunk: int):
    for row_off in range(0, height, chunk):
        h = min(chunk, height - row_off)
        for col_off in range(0, width, chunk):
            w = min(chunk, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def _accumulate_chunk_stats(
    data: np.ndarray, nodata, sums, sumsq, counts
) -> None:
    """Vectorised per-band sum/sumsq/count update for one chunk."""
    valid = np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    n_bands = data.shape[0]
    flat = data.reshape(n_bands, -1).astype(np.float64, copy=False)
    mask_flat = valid.reshape(n_bands, -1)
    masked = np.where(mask_flat, flat, 0.0)
    sums += masked.sum(axis=1)
    sumsq += np.square(masked).sum(axis=1)
    counts += mask_flat.sum(axis=1)


def _stream_per_channel_stats(
    src: rasterio.io.DatasetReader,
    chunk_size: int = _STANDARDISE_CHUNK_SIZE,
    sample_stride: int = _STATS_SAMPLE_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std from sampled chunks; one streaming pass over a subset."""
    n_bands = src.count
    nodata = src.nodata
    sums = np.zeros(n_bands, dtype=np.float64)
    sumsq = np.zeros(n_bands, dtype=np.float64)
    counts = np.zeros(n_bands, dtype=np.int64)

    sampled_any = False
    for i, window in enumerate(_iter_chunk_windows(src.width, src.height, chunk_size)):
        if sample_stride > 1 and i % sample_stride != 0:
            continue
        sampled_any = True
        data = src.read(window=window)
        _accumulate_chunk_stats(data, nodata, sums, sumsq, counts)

    # If sampling skipped everything (very small raster), force a full pass.
    if not sampled_any:
        for window in _iter_chunk_windows(src.width, src.height, chunk_size):
            data = src.read(window=window)
            _accumulate_chunk_stats(data, nodata, sums, sumsq, counts)

    safe_counts = np.maximum(counts, 1)
    means = sums / safe_counts
    var = np.maximum(0.0, sumsq / safe_counts - means ** 2)
    stds = np.sqrt(var)
    means[counts == 0] = 0.0
    stds[counts == 0] = 0.0
    return means, stds


def compute_standardisation_stats(
    src: rasterio.io.DatasetReader,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean / std for `src`, sampled in chunks.

    Standardisation itself happens at tile read time inside the dataset using
    these values stashed in the Parquet manifest's metadata.
    """
    src_name = os.path.basename(src.name)
    LOGGER.info(
        f"[stats:{src_name}] computing per-channel mean/std "
        f"(dims={src.width}x{src.height}, bands={src.count}, "
        f"chunk={_STANDARDISE_CHUNK_SIZE}, sample_stride={_STATS_SAMPLE_STRIDE})"
    )
    means, stds = _stream_per_channel_stats(
        src, _STANDARDISE_CHUNK_SIZE, _STATS_SAMPLE_STRIDE
    )
    LOGGER.info(
        f"[stats:{src_name}] ready "
        f"(means={np.round(means, 3).tolist()}, stds={np.round(stds, 3).tolist()})"
    )

    # Remove last element, alpha channel
    return means.astype(np.float32, copy=False)[:-1], stds.astype(np.float32, copy=False)[:-1]


def standardise_window(
    data: np.ndarray, means: np.ndarray, stds: np.ndarray, nodata
) -> np.ndarray:
    """Apply standardisation in place. Mirrors the old streaming write loop:
    invalid pixels (nan / nodata) become 0 in the standardised output.
    """
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=True)
    else:
        data = data.copy()

    means_b = means.reshape(-1, 1, 1).astype(np.float32)
    denom_b = (stds + 1e-6).reshape(-1, 1, 1).astype(np.float32)

    if nodata is not None:
        invalid = (~np.isfinite(data)) | (data == nodata)
    else:
        invalid = ~np.isfinite(data)

    np.subtract(data, means_b, out=data)
    np.divide(data, denom_b, out=data)
    if invalid.any():
        data[invalid] = 0.0
    return data


def crop_src_to_boundaries(
    src: rasterio.io.DatasetReader, boundaries_path: str
) -> rasterio.io.DatasetReader | None:
    src_name = os.path.basename(src.name)
    LOGGER.info(
        f"[crop:{src_name}] starting crop pipeline (boundaries={boundaries_path})"
    )

    if src.tags().get(_CROPPED_TAG_KEY) == _CROPPED_TAG_VALUE:
        LOGGER.info(
            f"[crop:{src_name}] already cropped (metadata tag present); skipping."
        )
        return src

    # --- Stage 1: read boundaries shapefile ---
    LOGGER.info(f"[crop:{src_name}] stage 1/6: reading boundaries shapefile")
    try:
        with fiona.open(boundaries_path, "r") as shp:
            shp_geoms = [feat["geometry"] for feat in shp]
            shp_crs = shp.crs
        LOGGER.info(
            f"[crop:{src_name}] stage 1/6: loaded {len(shp_geoms)} geometries "
            f"(crs={shp_crs})"
        )
    except Exception:
        LOGGER.exception(
            f"[crop:{src_name}] stage 1/6 FAILED: could not read boundaries shapefile; "
            "proceeding without cropping"
        )
        return src

    # --- Stage 2: reproject geometries to source CRS ---
    LOGGER.info(
        f"[crop:{src_name}] stage 2/6: transforming {len(shp_geoms)} geometries "
        f"from {shp_crs} -> {src.crs}"
    )
    transformed = []
    fallback_count = 0
    failed_count = 0
    for g in shp_geoms:
        try:
            transformed.append(transform_geom(shp_crs, src.crs, g))
        except Exception:
            try:
                transformed.append(transform_geom("EPSG:4326", src.crs, g))
                fallback_count += 1
            except Exception:
                failed_count += 1
                LOGGER.exception(
                    f"[crop:{src_name}] stage 2/6: failed to transform a geometry; skipping it"
                )
    LOGGER.info(
        f"[crop:{src_name}] stage 2/6: transformed={len(transformed)} "
        f"(fallback_to_wgs84={fallback_count}, failed={failed_count})"
    )
    if not transformed:
        LOGGER.warning(
            f"[crop:{src_name}] stage 2/6: no valid transformed geometries; skipping cropping."
        )
        return src

    # --- Stage 3: rasterio mask + crop ---
    LOGGER.info(
        f"[crop:{src_name}] stage 3/6: applying mask.mask "
        f"(input dims={src.width}x{src.height}, bands={src.count})"
    )
    try:
        out_image, out_transform = mask.mask(src, transformed, crop=True)
    except ValueError:
        LOGGER.info(
            f"[crop:{src_name}] stage 3/6: no overlap between raster and boundaries; "
            "skipping file."
        )
        src.close()
        return None
    except Exception:
        LOGGER.exception(
            f"[crop:{src_name}] stage 3/6 FAILED: unexpected error while cropping; "
            "proceeding with original raster"
        )
        return src
    LOGGER.info(
        f"[crop:{src_name}] stage 3/6: cropped array shape={out_image.shape}, "
        f"dtype={out_image.dtype}"
    )

    # --- Stage 4/4: overwrite the source GeoTIFF with cropped data ---
    out_meta = src.meta.copy()
    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        }
    )

    target_path = src.name
    tmp_path = f"{target_path}.crop.tmp"
    out_bytes = int(out_image.nbytes)
    LOGGER.info(
        f"[crop:{src_name}] stage 4/4: overwriting source raster -> {target_path} "
        f"(~{out_bytes / 1e6:.1f} MB on disk)"
    )

    # Release the original handle before replacing the file underneath it;
    # write to a sibling temp first and atomically rename so a crash mid-write
    # can't leave the source raster in a half-cropped state.
    src.close()
    try:
        with rasterio.open(tmp_path, "w", **out_meta) as ds:
            ds.write(out_image)
            ds.update_tags(**{_CROPPED_TAG_KEY: _CROPPED_TAG_VALUE})
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                LOGGER.exception(
                    f"[crop:{src_name}] failed to clean up temp file {tmp_path}"
                )
        raise
    del out_image

    new_src = rasterio.open(target_path)
    LOGGER.info(
        f"[crop:{src_name}] crop pipeline complete; new bounds={new_src.bounds}, "
        f"dims={new_src.width}x{new_src.height}"
    )
    return new_src
