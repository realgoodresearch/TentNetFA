"""Manifest loading helpers used by `PairedImageDataset` and the resampler.

Manifest rows are returned as plain dicts so per-row access in `__getitem__`
doesn't depend on pyarrow at hot-path time. Memory cost: ~150 bytes per row
× O(10^5) tiles per TIFF ≈ tens of MB — comfortable to hold in worker RAM.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from displacement_tracker.util.manifest_writer import (
    MANIFEST_COLUMNS,
    MANIFEST_STATS_KEY,
    labels_sibling_path,
)


def _expand_paths(paths: str | os.PathLike[str] | list) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    expanded: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            expanded.extend(sorted(p.glob("*.parquet")))
        elif p.suffix == ".parquet":
            expanded.append(p)
        else:
            raise ValueError(f"Not a parquet file or directory: {p}")
    if not expanded:
        raise ValueError(f"No parquet manifests found at {paths}")
    return expanded


def load_manifest_table(
    paths: str | os.PathLike[str] | list,
) -> tuple[pa.Table, list[Path]]:
    """Read one or more manifests and concatenate them into a single Table."""
    files = _expand_paths(paths)
    tables = [pq.read_table(p) for p in files]
    return pa.concat_tables(tables, promote_options="default"), files


def load_manifest_rows(
    paths: str | os.PathLike[str] | list,
) -> tuple[list[dict[str, Any]], list[Path], dict[str, dict[str, Any]]]:
    """Load manifest(s) as row dicts, source files, and merged raster stats."""
    files = _expand_paths(paths)
    raster_stats: dict[str, dict[str, Any]] = {}
    tables: list[pa.Table] = []
    for path in files:
        table = pq.read_table(path)
        tables.append(table)
        meta = table.schema.metadata or {}
        raw = meta.get(MANIFEST_STATS_KEY.encode("utf-8")) or meta.get(
            MANIFEST_STATS_KEY
        )
        if raw is not None:
            stats = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            for raster_path, entry in stats.items():
                raster_stats[raster_path] = {
                    "means": np.asarray(entry["means"], dtype=np.float32),
                    "stds": np.asarray(entry["stds"], dtype=np.float32),
                    "nodata": entry.get("nodata"),
                }

    table = pa.concat_tables(tables, promote_options="default")
    missing = [c for c in MANIFEST_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(
            f"Manifest is missing required columns {missing} (files={files})"
        )
    return table.to_pylist(), files, raster_stats


def load_labels_for_manifest(
    manifest_path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Load the sibling labels JSON for a per-TIFF manifest. Empty list if absent."""
    labels_path = labels_sibling_path(manifest_path)
    if not labels_path.exists():
        return []
    return _read_labels_file(labels_path)


def load_labels_by_path(
    labels_path: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Load a labels JSON given its absolute path (the column carried per row)."""
    p = Path(labels_path)
    if not p.exists() or p.is_dir():
        return []
    return _read_labels_file(p)


def _read_labels_file(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "features" in data:
        return list(data["features"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected labels JSON shape at {path}")
