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

_STANDARDIZED_SUFFIX = "_standardized"

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


def standardise_array(data: np.ndarray, nodata=None) -> np.ndarray:
    """Standardise a (C, H, W) array per-channel over valid pixels (zero mean, unit variance)."""
    data = data.astype(np.float32, copy=True)
    for i in range(data.shape[0]):
        channel = data[i]
        valid_mask = np.isfinite(channel)
        if nodata is not None:
            valid_mask &= channel != nodata
        valid_vals = channel[valid_mask]
        if valid_vals.size == 0:
            continue
        m = float(valid_vals.mean())
        s = float(valid_vals.std())
        data[i] = (channel - m) / (s + 1e-6)
    return data


def _standardized_sibling_path(src_path: str) -> str:
    """Return `<dir>/<stem>_standardized.tif` next to the source raster."""
    parent = os.path.dirname(os.path.abspath(src_path)) or "."
    stem, ext = os.path.splitext(os.path.basename(src_path))
    if not ext:
        ext = ".tif"
    return os.path.join(parent, f"{stem}{_STANDARDIZED_SUFFIX}{ext}")


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


def standardise_src(src: rasterio.io.DatasetReader) -> rasterio.io.DatasetReader:
    """
    Standardise per-channel (zero mean, unit variance) and write the result to a
    sibling GeoTIFF next to the source (same directory, `_standardized` suffix).

    Direct read→standardise→write loop: one chunk in flight at a time, with
    in-place arithmetic and a tiled+BIGTIFF output so neither Python nor GDAL's
    block cache holds large intermediates. Memory footprint is bounded by a
    single chunk (~tens of MB) regardless of source raster size.

    The stats pass samples chunks (default ~5%) for speed.
    """
    src_name = os.path.basename(src.name)
    LOGGER.info(
        f"[standardise:{src_name}] starting pipeline "
        f"(dims={src.width}x{src.height}, bands={src.count}, "
        f"chunk={_STANDARDISE_CHUNK_SIZE}, sample_stride={_STATS_SAMPLE_STRIDE})"
    )
    try:
        chunk_size = _STANDARDISE_CHUNK_SIZE

        # --- Stage 1: streaming stats pass (sampled chunks) ---
        LOGGER.info(
            f"[standardise:{src_name}] stage 1/3: computing per-channel mean/std "
            f"via sampled chunks"
        )
        means, stds = _stream_per_channel_stats(src, chunk_size, _STATS_SAMPLE_STRIDE)
        LOGGER.info(
            f"[standardise:{src_name}] stage 1/3: stats ready "
            f"(means={np.round(means, 3).tolist()}, stds={np.round(stds, 3).tolist()})"
        )
        nodata = src.nodata

        means_b = means.reshape(-1, 1, 1).astype(np.float32)
        denom_b = (stds + 1e-6).reshape(-1, 1, 1).astype(np.float32)

        out_meta = src.meta.copy()
        # Tiled + BIGTIFF: keeps GDAL's block cache footprint bounded for
        # huge outputs, and lets per-window writes hit a small set of tiles
        # rather than touching full-width strips.
        out_meta.update(
            {
                "dtype": "float32",
                "nodata": None,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "deflate",
                "BIGTIFF": "IF_SAFER",
            }
        )

        path = _standardized_sibling_path(src.name)
        windows = list(_iter_chunk_windows(src.width, src.height, chunk_size))

        # --- Stage 2: synchronous read → standardise → write loop ---
        LOGGER.info(
            f"[standardise:{src_name}] stage 2/3: writing standardized output "
            f"-> {path} ({len(windows)} chunks of ~{chunk_size}x{chunk_size}, "
            f"tiled 512x512, BIGTIFF=IF_SAFER, compress=deflate)"
        )
        chunks_done = 0
        log_every = max(1, len(windows) // 10)
        with rasterio.open(path, "w", **out_meta) as out_ds:
            for window in windows:
                # Read this chunk and convert to float32. Source array is freed
                # immediately after astype returns.
                raw = src.read(window=window)
                data = raw.astype(np.float32, copy=False)
                if data is raw:
                    # Source already float32 → make our own copy so in-place
                    # arithmetic doesn't mutate any GDAL-internal buffer.
                    data = data.copy()
                del raw

                # Mark invalid pixels before we overwrite them in place.
                if nodata is not None:
                    invalid = (~np.isfinite(data)) | (data == nodata)
                else:
                    invalid = ~np.isfinite(data)

                # In-place standardise: data := (data - mean) / std
                np.subtract(data, means_b, out=data)
                np.divide(data, denom_b, out=data)

                # Invalid pixels become 0 (the standardised output has
                # nodata=None, and downstream filters treat 0 as invalid —
                # matches the existing convention without needing to keep a
                # second full-size copy of the original values).
                if invalid.any():
                    data[invalid] = 0.0
                del invalid

                out_ds.write(data, window=window)
                del data

                chunks_done += 1
                if chunks_done % log_every == 0 or chunks_done == len(windows):
                    LOGGER.info(
                        f"[standardise:{src_name}] stage 2/3: "
                        f"{chunks_done}/{len(windows)} chunks written"
                    )
        LOGGER.info(
            f"[standardise:{src_name}] stage 2/3: write complete "
            f"({chunks_done}/{len(windows)} chunks)"
        )

        # --- Stage 3: reopen the standardized sibling file as the new source ---
        LOGGER.info(
            f"[standardise:{src_name}] stage 3/3: reopening standardized dataset "
            f"for downstream use"
        )
        new_src = rasterio.open(path)
        LOGGER.info(
            f"[standardise:{src_name}] pipeline complete -> {path} "
            f"(dims={new_src.width}x{new_src.height}, bands={new_src.count}, dtype=float32)"
        )
        return new_src
    except Exception:
        LOGGER.exception(
            f"[standardise:{src_name}] FAILED; returning original raster"
        )
        return src


def crop_src_to_boundaries(
    src: rasterio.io.DatasetReader, boundaries_path: str
) -> rasterio.io.DatasetReader | None:
    src_name = os.path.basename(src.name)
    LOGGER.info(
        f"[crop:{src_name}] starting crop pipeline (boundaries={boundaries_path})"
    )

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

    # --- Stage 4: per-channel standardisation of the cropped array ---
    LOGGER.info(f"[crop:{src_name}] stage 4/6: standardising cropped array")
    out_image = standardise_array(out_image, nodata=src.nodata)
    LOGGER.info(
        f"[crop:{src_name}] stage 4/6: standardisation done (dtype={out_image.dtype})"
    )

    out_meta = src.meta.copy()
    out_meta.update(
        {
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "dtype": "float32",
            "nodata": None,
        }
    )

    # --- Stage 5: write the cropped+standardised output to a sibling GeoTIFF ---
    path = _standardized_sibling_path(src.name)
    out_bytes = int(out_image.nbytes)
    LOGGER.info(
        f"[crop:{src_name}] stage 5/6: writing standardized GeoTIFF -> {path} "
        f"(~{out_bytes / 1e6:.1f} MB on disk)"
    )
    with rasterio.open(path, "w", **out_meta) as ds:
        ds.write(out_image)
    del out_image
    LOGGER.info(f"[crop:{src_name}] stage 5/6: standardized GeoTIFF write complete")

    src.close()

    # --- Stage 6: reopen the standardized sibling GeoTIFF as the new source ---
    LOGGER.info(
        f"[crop:{src_name}] stage 6/6: reopening standardized dataset for downstream use"
    )
    new_src = rasterio.open(path)
    LOGGER.info(
        f"[crop:{src_name}] crop pipeline complete; new bounds={new_src.bounds}, "
        f"dims={new_src.width}x{new_src.height}"
    )
    return new_src
