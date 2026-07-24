"""Flow: scan a TIFF using GeoJSON annotations (with quality gating).

Builds a per-TIFF Parquet manifest of accepted tiles plus a sibling labels
JSON containing the date-filtered tent features. The runtime dataset reads
tiles from the standardised raster on demand and rasterises labels via
`create_label_from_feats(...)` against the JSON-stored feature list.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any

import click
from pyproj import Transformer
from tqdm import tqdm

from displacement_tracker.util.annotations import (
    extract_date_from_filename,
    filter_tents_by_target_date,
    group_coords,
)
from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.manifest_writer import (
    ManifestWriter,
    compute_tile_id,
    labels_sibling_path,
    write_labels_json,
)
from displacement_tracker.util.raster_processing import (
    compute_standardisation_stats,
    crop_src_to_boundaries,
    open_raster,
)
from displacement_tracker.util.scan_orchestrator import (
    collect_tif_files,
    require_keys,
    run_scans,
)
from displacement_tracker.util.tile_builder import (
    _read_prewar_tile,
    compute_tile_window,
)

LOGGER = setup_logging("annotated_scanner")


def _resolve_min_valid(quality_thresholds: dict[str, Any] | None) -> float:
    if isinstance(quality_thresholds, dict):
        return float(quality_thresholds.get("min_valid_fraction", 0.0))
    return 0.0


def _build_row(
    tile,
    feats: list[dict[str, Any]],
    base_name: str,
    date_target: str | None,
    raster_path: str,
    prewar_path: str,
    labels_path: str,
    is_complete: bool,
) -> dict[str, Any]:
    feature_ids = [int(f["_idx"]) for f in feats if isinstance(f, dict) and "_idx" in f]
    return {
        "tile_id": compute_tile_id(raster_path, tile.r0, tile.c0),
        "raster_path": raster_path,
        "prewar_path": prewar_path,
        "labels_path": labels_path,
        "r0": tile.r0,
        "r1": tile.r1,
        "c0": tile.c0,
        "c1": tile.c1,
        "lon_min": tile.lon_min,
        "lon_max": tile.lon_max,
        "lat_min": tile.lat_min,
        "lat_max": tile.lat_max,
        "origin_image": base_name,
        "origin_date": date_target or "",
        "valid_fraction": tile.valid_fraction,
        "is_complete": is_complete,
        "label_feature_ids": feature_ids,
    }


def _scan_complete_raster(
    src,
    grouped: dict[tuple[int, int], list[dict[str, Any]]],
    base_name: str,
    date_target: str | None,
    core_m: float,
    margin_m: float,
    prewar_src,
    min_valid: float,
    manifest_writer: ManifestWriter,
    raster_path: str,
    prewar_path: str,
    labels_path: str,
) -> int:
    bounds = src.bounds
    LOGGER.info(
        f"[{base_name}] Raster bounds (src CRS): {bounds.left}..{bounds.right} x {bounds.bottom}..{bounds.top}"
    )

    span_m = core_m + 2.0 * margin_m
    i_start = math.floor(bounds.left / core_m)
    i_end = math.ceil(bounds.right / core_m)
    j_start = math.floor(bounds.bottom / core_m)
    j_end = math.ceil(bounds.top / core_m)

    processed_count = 0
    for i in tqdm(range(i_start, i_end), desc=f"{base_name} i"):
        x_centre = i * core_m
        for j in range(j_start, j_end):
            y_centre = j * core_m
            feats_for_tile = grouped.get((i, j), [])

            tile = compute_tile_window(
                src, x_centre, y_centre, core_m, margin_m, min_valid
            )
            if tile is None:
                continue
            if prewar_src is not None:
                prewar_check = _read_prewar_tile(
                    prewar_src,
                    tile.lon_min,
                    tile.lon_max,
                    tile.lat_min,
                    tile.lat_max,
                    span_m,
                    min_valid,
                )
                if prewar_check is None:
                    continue

            manifest_writer.add_row(
                _build_row(
                    tile,
                    feats_for_tile,
                    base_name,
                    date_target,
                    raster_path,
                    prewar_path,
                    labels_path,
                    is_complete=True,
                )
            )
            processed_count += 1
    return processed_count


def _scan_grouped_tiles(
    src,
    grouped: dict[tuple[int, int], list[dict[str, Any]]],
    base_name: str,
    date_target: str | None,
    core_m: float,
    margin_m: float,
    prewar_src,
    min_valid: float,
    manifest_writer: ManifestWriter,
    raster_path: str,
    prewar_path: str,
    labels_path: str,
) -> int:
    span_m = core_m + 2.0 * margin_m
    processed_count = 0
    for (i, j), feats in tqdm(grouped.items(), desc=f"{base_name} processing tiles"):
        if not feats:
            continue
        x_centre = i * core_m
        y_centre = j * core_m
        tile = compute_tile_window(src, x_centre, y_centre, core_m, margin_m, min_valid)
        if tile is None:
            continue
        if prewar_src is not None:
            prewar_check = _read_prewar_tile(
                prewar_src,
                tile.lon_min,
                tile.lon_max,
                tile.lat_min,
                tile.lat_max,
                span_m,
                min_valid,
            )
            if prewar_check is None:
                continue

        manifest_writer.add_row(
            _build_row(
                tile,
                feats,
                base_name,
                date_target,
                raster_path,
                prewar_path,
                labels_path,
                is_complete=False,
            )
        )
        processed_count += 1
    return processed_count


def scan_grouped_coordinates(
    geotiff_path: str,
    geojson_path: str,
    manifest_writer: ManifestWriter,
    quality_thresholds: dict[str, Any],
    core_m: float,
    margin_m: float,
    date_target: str | None,
    prewar_path: str | None = None,
    boundaries_path: str | None = None,
    complete_list: list[str] | None = None,
) -> None:
    src = open_raster(geotiff_path)
    if src is None:
        return

    if boundaries_path:
        cropped = crop_src_to_boundaries(src, boundaries_path)
        if cropped is None:
            return
        src = cropped

    wgs84_to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    src_means, src_stds = compute_standardisation_stats(src)
    manifest_writer.set_raster_stats(src.name, src_means, src_stds, nodata=src.nodata)

    prewar_src = open_raster(prewar_path) if prewar_path else None
    if prewar_src is not None:
        prewar_means, prewar_stds = compute_standardisation_stats(prewar_src)
        manifest_writer.set_raster_stats(
            prewar_src.name, prewar_means, prewar_stds, nodata=prewar_src.nodata
        )

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
    is_complete_tif = base_name in (complete_list or [])

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

    for i, feat in enumerate(features):
        feat["_idx"] = i

    grouped = group_coords(features, core_m, margin_m, wgs84_to_src)

    raster_path = src.name
    prewar_path_for_row = (prewar_src.name if prewar_src is not None else "") or ""
    labels_path = str(labels_sibling_path(manifest_writer.path).resolve())
    write_labels_json(manifest_writer.path, base_name, features)

    try:
        if is_complete_tif:
            LOGGER.info(
                f"[{base_name}] marked as COMPLETE - quality gating disabled; scanning entire raster grid."
            )
            processed_count = _scan_complete_raster(
                src,
                grouped,
                base_name,
                date_target,
                core_m,
                margin_m,
                prewar_src,
                min_valid,
                manifest_writer,
                raster_path,
                prewar_path_for_row,
                labels_path,
            )
            LOGGER.info(f"[{base_name}] wrote {processed_count} tiles (complete=True).")
            if processed_count == 0:
                LOGGER.warning(f"No valid tiles written for COMPLETE TIFF {base_name}")
            return

        LOGGER.info(
            f"[{base_name}] marked as INCOMPLETE - applying quality gating from YAML."
        )
        LOGGER.info(f"[{base_name}] Found {len(grouped)} coordinate groups with tents.")
        LOGGER.info(
            f"GeoTIFF bounds: min_lon={src.bounds.left}, min_lat={src.bounds.bottom}, max_lon={src.bounds.right}, max_lat={src.bounds.top}"
        )

        processed_count = _scan_grouped_tiles(
            src,
            grouped,
            base_name,
            date_target,
            core_m,
            margin_m,
            prewar_src,
            min_valid,
            manifest_writer,
            raster_path,
            prewar_path_for_row,
            labels_path,
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
@flow_option(default="train")
def cli(config, flow):
    """Run the annotated (GeoJSON + image) scan flow from a YAML config."""
    params = load_flow_config(config, flow)
    try:
        require_keys(params, ("geotiff_dir", "geojson", "processing"))
    except KeyError as e:
        raise click.ClickException(str(e))

    proc = params["processing"]
    core_m = float(proc["core_metres"])
    margin_m = float(proc["margin_metres"])
    quality_thresholds = proc["quality_thresholds"]
    complete_list = proc.get("complete", []) or []
    prewar_path = params.get("prewar_gaza")
    boundaries_path = params.get("boundaries")
    geojson = params["geojson"]

    manifest_folder = params.get("manifest_folder")
    if not manifest_folder:
        raise click.ClickException("Missing required config key: manifest_folder")

    tif_files = collect_tif_files(params["geotiff_dir"], params)
    if not tif_files:
        raise click.ClickException(f"No .tif files found in {params['geotiff_dir']}")

    def scan_one(tif_path: str, writer: ManifestWriter) -> None:
        date_target = extract_date_from_filename(tif_path)
        if not date_target:
            LOGGER.warning(f"Skipping {tif_path} (no date found in filename).")
            return
        scan_grouped_coordinates(
            tif_path,
            geojson,
            writer,
            quality_thresholds,
            core_m,
            margin_m,
            date_target,
            prewar_path,
            boundaries_path,
            complete_list,
        )

    run_scans(tif_files, scan_one, manifest_folder=manifest_folder)


if __name__ == "__main__":
    cli()

# EXAMPLE CLI USAGE:
# poetry run annotated-scanner config.yaml
