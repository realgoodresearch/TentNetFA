"""Rebalance per-TIFF manifests by class (positive / negative tiles).

Reads each per-TIFF Parquet manifest under `manifest_folder`, partitions rows
by whether `label_feature_ids` is empty (negative) or non-empty (positive),
samples negatives at `null_keep_fraction`, and concatenates kept rows into a
single output Parquet. No raster I/O — runs in seconds.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.manifest_writer import MANIFEST_STATS_KEY

LOGGER = setup_logging("resample-manifest")


def resample_and_merge(config_path: str, flow: str | None = "train") -> None:
    config = load_flow_config(config_path, flow)

    loading_files = config.get("loading", {}).get("files", []) or []
    rebal_config = config.get("rebalancing") or {}

    manifest_dir = config.get("manifest_folder")
    if not manifest_dir:
        raise click.ClickException("Missing required config key: manifest_folder")
    out_path = Path(rebal_config.get("out") or "")
    if not str(out_path):
        raise click.ClickException("Missing required config key: rebalancing.out")
    if out_path.suffix != ".parquet":
        out_path = out_path.with_suffix(".parquet")

    rng_seed = int(rebal_config.get("rng_seed", 42))
    null_keep_fraction = float(rebal_config.get("null_keep_fraction", 0.25))

    rng = np.random.default_rng(rng_seed)
    manifest_dir_path = Path(manifest_dir)
    if not manifest_dir_path.is_dir():
        print(
            f"Error: manifest directory not found: {manifest_dir_path}", file=sys.stderr
        )
        sys.exit(1)

    if loading_files:
        candidate_paths = [
            manifest_dir_path / f"{Path(name).stem}.parquet" for name in loading_files
        ]
    else:
        candidate_paths = sorted(manifest_dir_path.glob("*.parquet"))

    selected_tables: list[pa.Table] = []
    merged_raster_stats: dict[str, dict[str, Any]] = {}
    total_kept = 0
    total_seen = 0

    for parquet_path in tqdm(candidate_paths, desc="Resampling manifests"):
        if not parquet_path.exists():
            print(f"Warning: {parquet_path} not found, skipping")
            continue

        table = pq.read_table(parquet_path)
        if table.num_rows == 0:
            continue

        ids_col = table.column("label_feature_ids").to_pylist()
        is_null = np.array([len(ids) == 0 for ids in ids_col], dtype=bool)
        null_idx = np.where(is_null)[0]
        non_null_idx = np.where(~is_null)[0]

        keep_null_n = int(null_keep_fraction * len(null_idx))
        if keep_null_n > 0:
            keep_null_idx = rng.choice(null_idx, size=keep_null_n, replace=False)
        else:
            keep_null_idx = np.array([], dtype=int)

        keep_idx = np.sort(np.concatenate([non_null_idx, keep_null_idx]))
        if keep_idx.size == 0:
            continue

        selected_tables.append(table.take(pa.array(keep_idx)))
        total_kept += int(keep_idx.size)
        total_seen += int(table.num_rows)

        meta = table.schema.metadata or {}
        raw = meta.get(MANIFEST_STATS_KEY.encode("utf-8")) or meta.get(
            MANIFEST_STATS_KEY
        )
        if raw is not None:
            stats = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            for raster_path, entry in stats.items():
                existing = merged_raster_stats.get(raster_path)
                if existing is not None and existing != entry:
                    LOGGER.warning(
                        f"Conflicting raster_stats for {raster_path} across manifests; "
                        f"keeping entry from {parquet_path}"
                    )
                merged_raster_stats[raster_path] = entry

    if not selected_tables:
        print("No rows selected. Aborting.", file=sys.stderr)
        sys.exit(1)

    merged = pa.concat_tables(selected_tables)
    if merged_raster_stats:
        metadata = {
            MANIFEST_STATS_KEY: json.dumps(
                merged_raster_stats, ensure_ascii=False
            ).encode("utf-8")
        }
        merged = merged.replace_schema_metadata(metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    pq.write_table(merged, tmp, compression="zstd")
    os.replace(tmp, out_path)

    LOGGER.info(
        f"Saved merged + balanced manifest to {out_path} "
        f"(kept {total_kept}/{total_seen} rows across {len(selected_tables)} files)"
    )


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@flow_option(default="train")
def cli(config_path: str, flow: str) -> None:
    resample_and_merge(config_path, flow)


if __name__ == "__main__":
    cli()
