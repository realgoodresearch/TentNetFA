"""Shared per-TIFF orchestration: file collection, output layout, writer lifecycle."""
from __future__ import annotations

import glob
import os
from typing import Any, Callable

import rasterio

from displacement_tracker.util.env_loader import load_yaml_with_env
from displacement_tracker.util.hdf5_writer import HDF5Writer
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("scan_orchestrator")

ScanCallback = Callable[[str, HDF5Writer], None]


def collect_tif_files(geotiff_dir: str, config: str | None = None) -> list[str]:
    if config:
        cfg = load_yaml_with_env(config)
        search_files = cfg.get("loading", {}).get("files", [])
        if search_files:
            all_tifs = glob.glob(os.path.join(geotiff_dir, "*.tif"))
            return [
                p
                for p in all_tifs
                if any(s in os.path.basename(p) for s in search_files)
            ]
    return glob.glob(os.path.join(geotiff_dir, "*.tif"))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def hdf5_suffix(hdf5: str | None) -> str:
    if not hdf5:
        return ".hdf5"
    suffix = os.path.splitext(hdf5)[1]
    return suffix or ".hdf5"


def run_scans(
    tif_files: list[str],
    scan_one: ScanCallback,
    *,
    hdf5: str | None,
    hdf5_folder: str | None,
    individual: bool,
) -> None:
    """Drive the per-TIFF scan loop, handling individual vs bundled output."""
    if not tif_files:
        LOGGER.error("No .tif files supplied")
        return

    with rasterio.Env(GDAL_CACHEMAX=256):
        if individual:
            if not hdf5_folder:
                raise ValueError(
                    "hdf5_folder must be provided when processing individual TIFF outputs"
                )
            os.makedirs(hdf5_folder, exist_ok=True)
            suffix = hdf5_suffix(hdf5)

            for tif_path in tif_files:
                base = os.path.splitext(os.path.basename(tif_path))[0]
                out_h5 = os.path.join(hdf5_folder, f"{base}{suffix}")
                ensure_parent_dir(out_h5)

                LOGGER.info(f"Processing {tif_path} → {out_h5}")
                writer = HDF5Writer(out_h5)
                try:
                    scan_one(tif_path, writer)
                finally:
                    writer.write()
                LOGGER.info(f"Saved dataset to {out_h5}")
            return

        if not hdf5:
            raise ValueError(
                "hdf5 must be provided when processing.individual is false"
            )

        ensure_parent_dir(hdf5)
        LOGGER.info(f"Processing {len(tif_files)} TIFF files into {hdf5}")
        writer = HDF5Writer(hdf5)
        try:
            for tif_path in tif_files:
                scan_one(tif_path, writer)
        finally:
            writer.write()
        LOGGER.info(f"Saved dataset to {hdf5}")


def require_keys(params: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in params:
            raise KeyError(f"Missing required config key: {key}")
