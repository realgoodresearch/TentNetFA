"""Shared per-TIFF orchestration: file collection, output layout, writer lifecycle."""

from __future__ import annotations

import glob
import os
from typing import Any, Callable

import rasterio

from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.manifest_writer import ManifestWriter

LOGGER = setup_logging("scan_orchestrator")

ScanCallback = Callable[[str, ManifestWriter], None]


def collect_tif_files(geotiff_dir: str, params: dict | None = None) -> list[str]:
    """List .tif files in ``geotiff_dir``, filtered by ``loading.files`` if set."""
    search_files = ((params or {}).get("loading") or {}).get("files") or []
    if search_files:
        all_tifs = glob.glob(os.path.join(geotiff_dir, "*.tif"))
        return [
            p for p in all_tifs if any(s in os.path.basename(p) for s in search_files)
        ]
    return glob.glob(os.path.join(geotiff_dir, "*.tif"))


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_scans(
    tif_files: list[str],
    scan_one: ScanCallback,
    *,
    manifest_folder: str,
) -> None:
    """Drive the per-TIFF scan loop, writing one Parquet manifest per TIFF."""
    if not tif_files:
        LOGGER.error("No .tif files supplied")
        return

    os.makedirs(manifest_folder, exist_ok=True)

    with rasterio.Env(GDAL_CACHEMAX=256):
        for tif_path in tif_files:
            base = os.path.splitext(os.path.basename(tif_path))[0]
            out_manifest = os.path.join(manifest_folder, f"{base}.parquet")
            ensure_parent_dir(out_manifest)

            LOGGER.info(f"Processing {tif_path} → {out_manifest}")
            writer = ManifestWriter(out_manifest)
            try:
                scan_one(tif_path, writer)
            finally:
                row_count = len(writer)
                writer.close()
            LOGGER.info(f"Saved {row_count} rows to {out_manifest}")


def require_keys(params: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in params:
            raise KeyError(f"Missing required config key: {key}")
