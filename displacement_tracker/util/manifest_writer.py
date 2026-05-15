"""Per-TIFF Parquet manifest writer.

Replaces the gzip-chunked HDF5 tile dump with a small columnar manifest. Each
row describes a tile by reference (raster path + pixel window + bbox) rather
than carrying the raw pixels, so the dataset can stream tiles directly from
the source GeoTIFFs at training/inference time. Per-channel standardisation
stats live in the Parquet file metadata so the dataset can normalise windows
on the fly.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("tile_id", pa.uint64()),
        pa.field("raster_path", pa.string()),
        pa.field("prewar_path", pa.string()),
        pa.field("labels_path", pa.string()),
        pa.field("r0", pa.int32()),
        pa.field("r1", pa.int32()),
        pa.field("c0", pa.int32()),
        pa.field("c1", pa.int32()),
        pa.field("lon_min", pa.float64()),
        pa.field("lon_max", pa.float64()),
        pa.field("lat_min", pa.float64()),
        pa.field("lat_max", pa.float64()),
        pa.field("origin_image", pa.string()),
        pa.field("origin_date", pa.string()),
        pa.field("valid_fraction", pa.float32()),
        pa.field("is_complete", pa.bool_()),
        pa.field("label_feature_ids", pa.list_(pa.int32())),
    ]
)

MANIFEST_COLUMNS: tuple[str, ...] = tuple(MANIFEST_SCHEMA.names)

MANIFEST_STATS_KEY = "raster_stats"


def compute_tile_id(raster_path: str, r0: int, c0: int) -> int:
    """Deterministic uint64 hash so split caches stay stable across re-runs."""
    digest = hashlib.blake2b(
        f"{raster_path}|{r0}|{c0}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def labels_sibling_path(manifest_path: str | os.PathLike[str]) -> Path:
    """Convention: labels JSON lives at `<manifest_dir>/labels/<stem>.json`."""
    p = Path(manifest_path)
    return p.parent / "labels" / f"{p.stem}.json"


class ManifestWriter:
    """Buffer manifest rows in memory, atomically write Parquet on close.

    For per-TIFF outputs the row count is bounded (raster_extent / step)^2 —
    well below memory limits — so a single end-of-TIFF write is simpler than
    incremental flushing and keeps the on-disk file a single Parquet object.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._rows: list[dict[str, Any]] = []
        self._raster_stats: dict[str, dict[str, Any]] = {}
        self._closed = False

    def add_row(self, row: dict[str, Any]) -> None:
        missing = [c for c in MANIFEST_COLUMNS if c not in row]
        if missing:
            raise KeyError(f"Manifest row missing columns: {missing}")
        self._rows.append(row)

    def extend(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.add_row(row)

    def set_raster_stats(
        self,
        raster_path: str,
        means: np.ndarray | list[float],
        stds: np.ndarray | list[float],
        nodata: Any = None,
    ) -> None:
        """Stash per-channel mean/std for a raster the dataset will read.
        Stored in the Parquet file metadata, keyed by `raster_path`.
        """
        self._raster_stats[str(raster_path)] = {
            "means": [float(x) for x in np.asarray(means).ravel()],
            "stds": [float(x) for x in np.asarray(stds).ravel()],
            "nodata": None if nodata is None else float(nodata),
        }

    def __len__(self) -> int:
        return len(self._rows)

    def close(self) -> pa.Table:
        if self._closed:
            raise RuntimeError("ManifestWriter already closed")
        self._closed = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._rows, schema=MANIFEST_SCHEMA)
        if self._raster_stats:
            metadata = {
                MANIFEST_STATS_KEY: json.dumps(
                    self._raster_stats, ensure_ascii=False
                ).encode("utf-8")
            }
            table = table.replace_schema_metadata(metadata)
        tmp = self.path.with_name(self.path.name + ".tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, self.path)
        return table

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close()


def write_labels_json(
    manifest_path: str | os.PathLike[str],
    origin_image: str,
    features: list[dict[str, Any]],
) -> Path:
    """Write the per-TIFF features list referenced by `label_feature_ids`."""
    out_path = labels_sibling_path(manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"origin_image": origin_image, "features": features}
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, out_path)
    return out_path
