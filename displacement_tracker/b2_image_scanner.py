"""Flow: scan a TIFF without annotations (image-only, full-grid).

Tile production is parallelised across worker processes; each worker re-opens
the standardised raster from disk so rasterio handles aren't shared. The HDF5
writer stays on the main process (h5py serialises writes anyway) and consumes
batches as they complete.
"""
from __future__ import annotations

import inspect
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from itertools import islice
from typing import Any, Iterable, Iterator

import click
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import transform
from tqdm import tqdm

from displacement_tracker.util.annotations import extract_date_from_filename
from displacement_tracker.util.env_loader import load_yaml_with_env
from displacement_tracker.util.hdf5_writer import HDF5Writer
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.raster_processing import (
    open_raster,
    standardise_src,
)
from displacement_tracker.util.scan_orchestrator import (
    collect_tif_files,
    require_keys,
    run_scans,
)
from displacement_tracker.util.tile_builder import process_group

LOGGER = setup_logging("image_scanner")

# Per-worker process state (set by the pool initializer).
_WORKER_STATE: dict[str, Any] = {}


def _init_worker(src_path: str, prewar_path: str | None) -> None:
    """Open rasterio handles once per worker process and cache them."""
    src = rasterio.open(src_path)
    _WORKER_STATE["src"] = src
    _WORKER_STATE["prewar"] = rasterio.open(prewar_path) if prewar_path else None
    _WORKER_STATE["transformer"] = Transformer.from_crs(
        "EPSG:4326", src.crs, always_xy=True
    )


def _process_batch(
    coords: list[tuple[float, float]],
    step: float,
    base_name: str,
    date_target: str,
    min_valid_fraction: float,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]]:
    src = _WORKER_STATE["src"]
    prewar = _WORKER_STATE["prewar"]
    transformer = _WORKER_STATE["transformer"]
    out: list[tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray | None]] = []
    for lon, lat in coords:
        feature, label, meta, prewar_img = process_group(
            src,
            [],
            lon,
            lat,
            step,
            base_name,
            date_target,
            transformer,
            prewar,
            min_valid_fraction=min_valid_fraction,
        )
        if feature is not None and label is not None and meta is not None:
            out.append((feature, label, meta, prewar_img))
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
        return "max_tasks_per_child" in inspect.signature(
            ProcessPoolExecutor.__init__
        ).parameters
    except (TypeError, ValueError):
        return sys.version_info >= (3, 11)


def _make_executor(
    n_workers: int,
    src_path: str,
    prewar_worker_path: str | None,
    max_tasks_per_child: int | None,
) -> ProcessPoolExecutor:
    """Build a process pool, recycling workers periodically when supported."""
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
    hdf5_writer: HDF5Writer,
    date_target: str | None,
    step: float = 0.001,
    prewar_path: str | None = None,
    min_valid_fraction: float = 0.0,
    normalize_cfg: dict[str, Any] | None = None,
    max_workers: int | None = None,
    batch_size: int = 64,
    max_tasks_per_child: int | None = 16,
    max_pool_restarts: int = 10,
) -> None:
    base_name = os.path.basename(geotiff_path)
    LOGGER.info(
        f"Scanning entire raster grid for {base_name} with step {step} and min_valid_fraction {min_valid_fraction}"
    )

    src = open_raster(geotiff_path)
    if src is None:
        return

    src = standardise_src(src)

    bounds = src.bounds
    lon_bounds, lat_bounds = transform(
        src.crs, "EPSG:4326", [bounds.left, bounds.right], [bounds.bottom, bounds.top]
    )
    LOGGER.info(
        f"GeoTIFF bounds: min_lon={lon_bounds[0]}, min_lat={lat_bounds[0]}, max_lon={lon_bounds[1]}, max_lat={lat_bounds[1]}"
    )

    prewar_src = open_raster(prewar_path) if prewar_path else None
    if prewar_src is not None and "std" not in prewar_path:
        prewar_src = standardise_src(prewar_src)

    try:
        lon_iter = np.arange(lon_bounds[0], lon_bounds[1], step)
        lat_iter = np.arange(lat_bounds[0], lat_bounds[1], step)
        coords = [
            (float(lon), float(lat)) for lon in lon_iter for lat in lat_iter
        ]
        if not coords:
            return

        batches = list(_chunked(coords, batch_size))
        n_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
        # Cap in-flight futures so result memory doesn't grow unbounded.
        in_flight_cap = max(2 * n_workers, 4)

        src_path = src.name
        prewar_worker_path = prewar_src.name if prewar_src is not None else None

        LOGGER.info(
            f"Dispatching {len(batches)} batches of up to {batch_size} tiles to "
            f"{n_workers} workers (max_tasks_per_child={max_tasks_per_child})."
        )

        date_arg = date_target or ""

        # Resilient submit/drain loop: if a worker dies (e.g. OOM-killed by the
        # OS) the whole ProcessPoolExecutor enters a Broken state and every
        # pending future raises BrokenProcessPool. We track each in-flight
        # batch alongside its future so we can rebuild the pool and resubmit
        # any batches that were mid-flight or never even submitted.
        remaining: list[list[tuple[float, float]]] = list(batches)
        in_progress: dict = {}  # Future -> batch
        completed = 0
        failed_batches = 0
        restarts = 0

        executor = _make_executor(
            n_workers, src_path, prewar_worker_path, max_tasks_per_child
        )

        def _restart_pool(reason: str) -> bool:
            """Tear down the broken pool, requeue lost work, spin up a new one."""
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
            # Lost batches go back to the front of the queue.
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
                n_workers, src_path, prewar_worker_path, max_tasks_per_child
            )
            return True

        try:
            with tqdm(total=len(batches), desc=f"{base_name} batches") as pbar:
                while remaining or in_progress:
                    # Top up the in-flight queue.
                    while remaining and len(in_progress) < in_flight_cap:
                        batch = remaining.pop(0)
                        try:
                            fut = executor.submit(
                                _process_batch,
                                batch,
                                step,
                                base_name,
                                date_arg,
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
                            entries = fut.result()
                        except BrokenProcessPool:
                            # Requeue this batch and tear down the pool.
                            remaining.insert(0, batch)
                            pool_broken = True
                            break
                        except Exception:
                            LOGGER.exception(
                                "Worker batch failed; dropping its tiles and continuing"
                            )
                            failed_batches += 1
                            entries = []

                        for feature, label, meta, prewar_img in entries:
                            hdf5_writer.add_entry(feature, label, meta, prewar_img)
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
            f"(failed={failed_batches}, pool restarts={restarts})"
        )
    finally:
        src.close()
        if prewar_src:
            prewar_src.close()


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
def cli(config):
    """Run the image-only (no annotations) scan flow from a YAML config."""
    params = load_yaml_with_env(config)
    try:
        require_keys(params, ("geotiff_dir", "processing"))
    except KeyError as e:
        raise click.ClickException(str(e))

    proc = params["processing"]
    step = proc["step"]
    quality_thresholds = proc.get("quality_thresholds") or {}
    min_valid_fraction = (
        quality_thresholds.get("min_valid_fraction", 0.0)
        if isinstance(quality_thresholds, dict)
        else 0.0
    )
    individual = bool(proc.get("individual", False))
    prewar_path = params.get("prewar_gaza")
    normalize_cfg = proc.get("normalize")
    max_workers = proc.get("max_workers")
    batch_size = int(proc.get("batch_size", 64))
    max_tasks_per_child = proc.get("max_tasks_per_child", 32)
    if max_tasks_per_child is not None:
        max_tasks_per_child = int(max_tasks_per_child) or None
    max_pool_restarts = int(proc.get("max_pool_restarts", 3))

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
        scan_all_coordinates(
            tif_path,
            writer,
            date_target,
            step,
            prewar_path,
            min_valid_fraction=min_valid_fraction,
            normalize_cfg=normalize_cfg,
            max_workers=max_workers,
            batch_size=batch_size,
            max_tasks_per_child=max_tasks_per_child,
            max_pool_restarts=max_pool_restarts,
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
# poetry run image-scanner config.yaml
