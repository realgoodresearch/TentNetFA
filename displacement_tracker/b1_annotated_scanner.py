"""Flow: scan a TIFF using GeoJSON annotations (with quality gating)."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any

import click
import numpy as np
from pyproj import Transformer
from rasterio.warp import transform
from tqdm import tqdm

from displacement_tracker.util.annotations import (
    extract_date_from_filename,
    filter_tents_by_target_date,
    group_coords,
    is_high_quality_tile,
)
from displacement_tracker.util.env_loader import load_yaml_with_env
from displacement_tracker.util.hdf5_writer import HDF5Writer
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.raster_processing import (
    crop_src_to_boundaries,
    open_raster,
    standardise_src,
)
from displacement_tracker.util.scan_orchestrator import (
    collect_tif_files,
    require_keys,
    run_scans,
)
from displacement_tracker.util.tile_builder import process_group

LOGGER = setup_logging("annotated_scanner")


def _resolve_min_valid(quality_thresholds: dict[str, Any] | None) -> float:
    if isinstance(quality_thresholds, dict):
        return float(quality_thresholds.get("min_valid_fraction", 0.0))
    return 0.0


def _scan_complete_raster(
    src,
    grouped: dict[tuple[float, float], list[dict[str, Any]]],
    base_name: str,
    date_target: str | None,
    step: float,
    transformer,
    prewar_src,
    min_valid: float,
    hdf5_writer: HDF5Writer,
) -> int:
    bounds = src.bounds
    lon_bounds, lat_bounds = transform(
        src.crs,
        "EPSG:4326",
        [bounds.left, bounds.right],
        [bounds.bottom, bounds.top],
    )
    LOGGER.info(
        f"[{base_name}] Raster bounds (wgs84): {lon_bounds[0]}..{lon_bounds[1]} x {lat_bounds[0]}..{lat_bounds[1]}"
    )

    processed_count = 0
    lon_iter = np.arange(lon_bounds[0], lon_bounds[1], step)
    lat_iter = np.arange(lat_bounds[0], lat_bounds[1], step)
    for lon in tqdm(lon_iter, desc=f"{base_name} lon"):
        for lat in lat_iter:
            base_lon = round(math.floor(lon / step) * step, 5)
            base_lat = round(math.floor(lat / step) * step, 5)
            feats_for_tile = grouped.get((base_lon, base_lat), [])

            feature, label, meta, prewar_tile = process_group(
                src,
                feats_for_tile,
                lon,
                lat,
                step,
                base_name,
                date_target or "",
                transformer,
                prewar_src,
                min_valid_fraction=min_valid,
            )

            if (
                feature is not None
                and label is not None
                and meta is not None
                and prewar_tile is not None
            ):
                hdf5_writer.add_entry(feature, label, meta, prewar_tile)
                processed_count += 1
    return processed_count


def _scan_grouped_tiles(
    src,
    grouped: dict[tuple[float, float], list[dict[str, Any]]],
    base_name: str,
    date_target: str | None,
    step: float,
    transformer,
    prewar_src,
    quality_thresholds: dict[str, Any],
    min_valid: float,
    hdf5_writer: HDF5Writer,
) -> int:
    processed_count = 0
    for (lon, lat), feats in tqdm(
        grouped.items(), desc=f"{base_name} processing tiles"
    ):
        if not is_high_quality_tile(
            feats,
            date_target,
            src,
            lon,
            lat,
            step,
            **quality_thresholds,
            transformer=transformer,
        ):
            continue
        feature, label, meta, prewar_tile = process_group(
            src,
            feats,
            lon,
            lat,
            step,
            base_name,
            date_target or "",
            transformer,
            prewar_src,
            min_valid_fraction=min_valid,
        )
        if (
            feature is not None
            and label is not None
            and meta is not None
            and prewar_tile is not None
        ):
            hdf5_writer.add_entry(feature, label, meta, prewar_tile)
            processed_count += 1
    return processed_count


def scan_grouped_coordinates(
    geotiff_path: str,
    geojson_path: str,
    hdf5_writer: HDF5Writer,
    quality_thresholds: dict[str, Any],
    step: float,
    date_target: str | None,
    prewar_path: str | None = None,
    boundaries_path: str | None = None,
    complete_list: list[str] | None = None,
    prediction_only: bool = False,
) -> None:
    src = open_raster(geotiff_path)
    if src is None:
        return

    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    if boundaries_path:
        cropped = crop_src_to_boundaries(src, boundaries_path)
        if cropped is None:
            return
        src = cropped
    else:
        src = standardise_src(src)

    prewar_src = open_raster(prewar_path) if prewar_path else None
    if prewar_src is not None:
        prewar_src = standardise_src(prewar_src)

    try:
        with open(geojson_path, "r") as f:
            features = json.load(f).get("features", [])
    except Exception:
        LOGGER.exception(f"Failed to read geojson: {geojson_path}")
        src.close()
        if prewar_src:
            prewar_src.close()
        return

    base_name = os.path.basename(geotiff_path)
    is_complete = base_name in (complete_list or []) or prediction_only

    min_valid = _resolve_min_valid(quality_thresholds)

    if date_target:
        try:
            date_obj = datetime.strptime(date_target, "%Y%m%d").date()
            features = filter_tents_by_target_date(features, date_obj)
            LOGGER.info(
                f"[{base_name}] Filtered to {len(features)} features for date {date_target}."
            )
        except Exception:
            LOGGER.exception(f"Date parsing error for {geotiff_path}")
            src.close()
            if prewar_src:
                prewar_src.close()
            return

    grouped = group_coords(features, step)

    try:
        if is_complete:
            LOGGER.info(
                f"[{base_name}] marked as COMPLETE - quality gating disabled; scanning entire raster grid."
            )
            processed_count = _scan_complete_raster(
                src,
                grouped,
                base_name,
                date_target,
                step,
                transformer,
                prewar_src,
                min_valid,
                hdf5_writer,
            )
            LOGGER.info(f"[{base_name}] wrote {processed_count} tiles (complete=True).")
            if processed_count == 0:
                LOGGER.warning(f"No valid tiles written for COMPLETE TIFF {base_name}")
            return

        LOGGER.info(
            f"[{base_name}] marked as INCOMPLETE - applying quality gating from YAML."
        )
        LOGGER.info(
            f"[{base_name}] Found {len(grouped)} coordinate groups with tents."
        )
        LOGGER.info(
            f"GeoTIFF bounds: min_lon={src.bounds.left}, min_lat={src.bounds.bottom}, max_lon={src.bounds.right}, max_lat={src.bounds.top}"
        )

        processed_count = _scan_grouped_tiles(
            src,
            grouped,
            base_name,
            date_target,
            step,
            transformer,
            prewar_src,
            quality_thresholds,
            min_valid,
            hdf5_writer,
        )
        LOGGER.info(f"[{base_name}] wrote {processed_count} tiles (complete=False).")
        if processed_count == 0:
            LOGGER.warning(f"No valid high-quality tiles found in {base_name}")
    finally:
        src.close()
        if prewar_src:
            prewar_src.close()


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def cli(config):
    """Run the annotated (GeoJSON + image) scan flow from a YAML config."""
    params = load_yaml_with_env(config)
    try:
        require_keys(params, ("geotiff_dir", "geojson", "processing"))
    except KeyError as e:
        raise click.ClickException(str(e))

    proc = params["processing"]
    step = proc["step"]
    quality_thresholds = proc["quality_thresholds"]
    complete_list = proc.get("complete", []) or []
    individual = bool(proc.get("individual", False))
    prediction_only = bool(proc.get("prediction_only", False))
    prewar_path = params.get("prewar_gaza")
    boundaries_path = params.get("boundaries")
    geojson = params["geojson"]

    if individual and not params.get("hdf5_folder"):
        raise click.ClickException(
            "Missing required config key: hdf5_folder when processing.individual is true"
        )
    if not individual and "hdf5" not in params:
        raise click.ClickException("Missing required config key: hdf5")

    tif_files = collect_tif_files(params["geotiff_dir"], config)
    if not tif_files:
        raise click.ClickException(f"No .tif files found in {params['geotiff_dir']}")

    def scan_one(tif_path: str, writer: HDF5Writer) -> None:
        date_target = extract_date_from_filename(tif_path)
        if not date_target:
            LOGGER.warning(f"Skipping {tif_path} (no date found in filename).")
            return
        scan_grouped_coordinates(
            tif_path,
            geojson,
            writer,
            quality_thresholds,
            step,
            date_target,
            prewar_path,
            boundaries_path,
            complete_list,
            prediction_only=prediction_only,
        )

    run_scans(
        tif_files,
        scan_one,
        hdf5=params.get("hdf5"),
        hdf5_folder=params.get("hdf5_folder"),
        individual=individual,
    )


if __name__ == "__main__":
    cli()

# EXAMPLE CLI USAGE:
# poetry run annotated-scanner config.yaml
