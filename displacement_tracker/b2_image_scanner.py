"""Flow: scan a TIFF without annotations (image-only, full-grid).

Each worker computes tile windows + validity and emits manifest rows. The main
process aggregates rows and writes a single Parquet manifest at end-of-TIFF.
The runtime dataset reads tiles directly from the standardised raster.
"""

from __future__ import annotations

import inspect
import math
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from itertools import islice
from typing import Any, Iterable, Iterator

import click
import rasterio
from tqdm import tqdm

from displacement_tracker.util.annotations import extract_date_from_filename
from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.manifest_writer import (
    ManifestWriter,
    compute_tile_id,
)
from displacement_tracker.util.raster_processing import (
    compute_standardisation_stats,
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

LOGGER = setup_logging("image_scanner")

_WORKER_STATE: dict[str, Any] = {}


def _init_worker(src_path: str, prewar_path: str | None) -> None:
    """Open rasterio handles once per worker process and cache them."""
    src = rasterio.open(src_path)
    _WORKER_STATE["src"] = src
    _WORKER_STATE["prewar"] = rasterio.open(prewar_path) if prewar_path else None


def _process_batch(
    coords: list[tuple[float, float]],
    core_m: float,
    margin_m: float,
    base_name: str,
    date_target: str,
    raster_path: str,
    prewar_path: str,
    min_valid_fraction: float,
) -> list[dict[str, Any]]:
    src = _WORKER_STATE["src"]
    prewar = _WORKER_STATE["prewar"]
    span_m = core_m + 2.0 * margin_m
    out: list[dict[str, Any]] = []
    for x_centre, y_centre in coords:
        tile = compute_tile_window(
            src, x_centre, y_centre, core_m, margin_m, min_valid_fraction
        )
        if tile is None:
            continue
        if prewar is not None:
            prewar_check = _read_prewar_tile(
                prewar,
                tile.lon_min,
                tile.lon_max,
                tile.lat_min,
                tile.lat_max,
                span_m,
                min_valid_fraction,
            )
            if prewar_check is None:
                continue
        out.append(
            {
                "tile_id": compute_tile_id(raster_path, tile.r0, tile.c0),
                "raster_path": raster_path,
                "prewar_path": prewar_path,
                "labels_path": "",
                "r0": tile.r0,
                "r1": tile.r1,
                "c0": tile.c0,
                "c1": tile.c1,
                "lon_min": tile.lon_min,
                "lon_max": tile.lon_max,
                "lat_min": tile.lat_min,
                "lat_max": tile.lat_max,
                "origin_image": base_name,
                "origin_date": date_target,
                "valid_fraction": tile.valid_fraction,
                "is_complete": True,
                "label_feature_ids": [],
            }
        )
    return out


def _chunked(seq: Iterable, n: int) -> Iterator[list]:
    it = iter(seq)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def _supports_max_tasks_per_child() -> bool:
    """ProcessPoolExecutor.max_tasks_per_child was added in Python 3.11."""
    try:
        return (
            "max_tasks_per_child"
            in inspect.signature(ProcessPoolExecutor.__init__).parameters
        )
    except (TypeError, ValueError):
        return sys.version_info >= (3, 11)


def _make_executor(
    n_workers: int,
    src_path: str,
    prewar_worker_path: str | None,
    max_tasks_per_child: int | None,
) -> ProcessPoolExecutor:
    kwargs: dict[str, Any] = {
        "max_workers": n_workers,
        "initializer": _init_worker,
        "initargs": (src_path, prewar_worker_path),
    }
    if max_tasks_per_child and _supports_max_tasks_per_child():
        kwargs["max_tasks_per_child"] = max_tasks_per_child
    return ProcessPoolExecutor(**kwargs)


def scan_all_coordinates(
    geotiff_path: str,
    manifest_writer: ManifestWriter,
    date_target: str | None,
    core_m: float,
    margin_m: float,
    prewar_path: str | None = None,
    min_valid_fraction: float = 0.0,
    max_workers: int | None = None,
    batch_size: int = 64,
    max_tasks_per_child: int | None = 32,
    max_pool_restarts: int = 3,
) -> None:
    base_name = os.path.basename(geotiff_path)
    LOGGER.info(
        f"Scanning entire raster grid for {base_name} with core_m={core_m}, "
        f"margin_m={margin_m}, min_valid_fraction={min_valid_fraction}"
    )

    src = open_raster(geotiff_path)
    if src is None:
        return

    src_means, src_stds = compute_standardisation_stats(src)
    manifest_writer.set_raster_stats(src.name, src_means, src_stds, nodata=src.nodata)

    bounds = src.bounds
    LOGGER.info(
        f"GeoTIFF bounds (src CRS): min_x={bounds.left}, min_y={bounds.bottom}, "
        f"max_x={bounds.right}, max_y={bounds.top}"
    )

    prewar_src = open_raster(prewar_path) if prewar_path else None
    if prewar_src is not None:
        prewar_means, prewar_stds = compute_standardisation_stats(prewar_src)
        manifest_writer.set_raster_stats(
            prewar_src.name, prewar_means, prewar_stds, nodata=prewar_src.nodata
        )

    try:
        i_start = math.floor(bounds.left / core_m)
        i_end = math.ceil(bounds.right / core_m)
        j_start = math.floor(bounds.bottom / core_m)
        j_end = math.ceil(bounds.top / core_m)
        coords = [
            (i * core_m, j * core_m)
            for i in range(i_start, i_end)
            for j in range(j_start, j_end)
        ]
        if not coords:
            return

        batches = list(_chunked(coords, batch_size))
        n_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
        in_flight_cap = max(2 * n_workers, 4)

        raster_path = src.name
        prewar_worker_path = prewar_src.name if prewar_src is not None else None
        prewar_path_for_row = prewar_worker_path or ""

        LOGGER.info(
            f"Dispatching {len(batches)} batches of up to {batch_size} tiles to "
            f"{n_workers} workers (max_tasks_per_child={max_tasks_per_child})."
        )

        date_arg = date_target or ""

        remaining: list[list[tuple[float, float]]] = list(batches)
        in_progress: dict = {}
        completed = 0
        failed_batches = 0
        restarts = 0

        executor = _make_executor(
            n_workers, raster_path, prewar_worker_path, max_tasks_per_child
        )

        def _restart_pool(reason: str) -> bool:
            nonlocal executor, restarts, in_progress
            if restarts >= max_pool_restarts:
                LOGGER.error(
                    f"Worker pool died ({reason}) after {restarts} restarts; "
                    "giving up on remaining batches."
                )
                return False
            restarts += 1
            lost = list(in_progress.values())
            in_progress = {}
            remaining[:0] = lost
            LOGGER.warning(
                f"Worker pool died ({reason}); restart {restarts}/{max_pool_restarts}, "
                f"requeued {len(lost)} in-flight batches "
                f"(remaining={len(remaining)})."
            )
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            executor = _make_executor(
                n_workers, raster_path, prewar_worker_path, max_tasks_per_child
            )
            return True

        try:
            with tqdm(total=len(batches), desc=f"{base_name} batches") as pbar:
                while remaining or in_progress:
                    while remaining and len(in_progress) < in_flight_cap:
                        batch = remaining.pop(0)
                        try:
                            fut = executor.submit(
                                _process_batch,
                                batch,
                                core_m,
                                margin_m,
                                base_name,
                                date_arg,
                                raster_path,
                                prewar_path_for_row,
                                min_valid_fraction,
                            )
                        except BrokenProcessPool:
                            remaining.insert(0, batch)
                            if not _restart_pool("submit"):
                                return
                            continue
                        in_progress[fut] = batch

                    if not in_progress:
                        continue

                    try:
                        done, _pending = wait(
                            list(in_progress), return_when=FIRST_COMPLETED
                        )
                    except BrokenProcessPool:
                        if not _restart_pool("wait"):
                            return
                        continue

                    pool_broken = False
                    for fut in done:
                        batch = in_progress.pop(fut)
                        try:
                            rows = fut.result()
                        except BrokenProcessPool:
                            remaining.insert(0, batch)
                            pool_broken = True
                            break
                        except Exception:
                            LOGGER.exception(
                                "Worker batch failed; dropping its tiles and continuing"
                            )
                            failed_batches += 1
                            rows = []

                        manifest_writer.extend(rows)
                        completed += 1
                        pbar.update(1)

                    if pool_broken:
                        if not _restart_pool("future"):
                            return
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        LOGGER.info(
            f"[{base_name}] batches complete: {completed}/{len(batches)} "
            f"(failed={failed_batches}, pool restarts={restarts}, "
            f"manifest rows={len(manifest_writer)})"
        )
    finally:
        src.close()
        if prewar_src:
            prewar_src.close()


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@flow_option(default="predict")
def cli(config, flow):
    """Run the image-only (no annotations) scan flow from a YAML config."""
    params = load_flow_config(config, flow)
    try:
        require_keys(params, ("geotiff_dir", "processing"))
    except KeyError as e:
        raise click.ClickException(str(e))

    proc = params["processing"]
    core_m = float(proc["core_metres"])
    margin_m = float(proc["margin_metres"])
    quality_thresholds = proc.get("quality_thresholds") or {}
    min_valid_fraction = (
        quality_thresholds.get("min_valid_fraction", 0.0)
        if isinstance(quality_thresholds, dict)
        else 0.0
    )
    prewar_path = params.get("prewar_gaza")
    max_workers = proc.get("max_workers")
    batch_size = int(proc.get("batch_size", 64))
    max_tasks_per_child = proc.get("max_tasks_per_child", 32)
    if max_tasks_per_child is not None:
        max_tasks_per_child = int(max_tasks_per_child) or None
    max_pool_restarts = int(proc.get("max_pool_restarts", 3))

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
        scan_all_coordinates(
            tif_path,
            writer,
            date_target,
            core_m,
            margin_m,
            prewar_path,
            min_valid_fraction=min_valid_fraction,
            max_workers=max_workers,
            batch_size=batch_size,
            max_tasks_per_child=max_tasks_per_child,
            max_pool_restarts=max_pool_restarts,
        )

    run_scans(tif_files, scan_one, manifest_folder=manifest_folder)


if __name__ == "__main__":
    cli()

# EXAMPLE CLI USAGE:
# poetry run image-scanner config.yaml
