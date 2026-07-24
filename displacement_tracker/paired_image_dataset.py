from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.manifest_reader import (
    load_labels_by_path,
    load_manifest_rows,
)
from displacement_tracker.util.raster_processing import standardise_window
from displacement_tracker.util.tile_builder import (
    _read_prewar_tile,
    create_label_from_feats,
)

LOGGER = setup_logging("paired-image-ds")


class PairedImageDataset(Dataset):
    """Stream tiles directly from the standardised GeoTIFFs at access time.

    Backed by a Parquet manifest (or directory of them). Each row references a
    pixel window in a `_standardized.tif`, the shared prewar raster, and an
    optional labels JSON. No rasters are opened in `__init__` — handles are
    created lazily inside each DataLoader worker so the dataset is fork-safe.
    """

    def __init__(
        self,
        manifest_path: str | os.PathLike[str] | list,
        indices: list[int] | None = None,
        sigma: float = 3.0,
        gdal_cache_mb: int = 512,
        per_tile_standardisation: bool = False,
    ):
        self.manifest_path = manifest_path
        self.sigma = sigma
        self.gdal_cache_mb = int(gdal_cache_mb)
        self.indices: list[int] | None = indices
        self.per_tile_standardisation = per_tile_standardisation

        rows, files, raster_stats = load_manifest_rows(manifest_path)
        self._rows: list[dict[str, Any]] = rows
        self._manifest_files: list[Path] = files
        self._raster_stats: dict[str, dict[str, Any]] = raster_stats

        unique_label_paths = {
            row["labels_path"] for row in self._rows if row["labels_path"]
        }
        self._labels_cache: dict[str, list[dict[str, Any]]] = {
            lp: load_labels_by_path(lp) for lp in unique_label_paths
        }

        self._handles: dict[str, rasterio.io.DatasetReader] = {}

        LOGGER.info(
            f"Initialized PairedImageDataset over {len(self._rows)} rows "
            f"from {len(self._manifest_files)} manifest file(s) (sigma={self.sigma})."
        )

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return
        dataset = info.dataset
        dataset._handles = {}
        os.environ.setdefault("GDAL_CACHEMAX", str(dataset.gdal_cache_mb))

    def _get_handle(self, path: str) -> rasterio.io.DatasetReader:
        h = self._handles.get(path)
        if h is None:
            h = rasterio.open(path)
            self._handles[path] = h
        return h

    @staticmethod
    def feat_transform(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr)

    def label_transform(self, arr: np.ndarray) -> torch.Tensor:
        arr = arr.astype(np.float32)
        blurred = gaussian_filter(arr, sigma=self.sigma)
        TENT_MASS = 255.0 * 9.0
        blurred = blurred / TENT_MASS
        return torch.from_numpy(blurred).unsqueeze(0)

    def __len__(self) -> int:
        if self.indices is not None:
            return len(self.indices)
        return len(self._rows)

    def _standardise(
        self, raster_path: str, data: np.ndarray, per_tile: bool = False
    ) -> np.ndarray:
        if per_tile:
            return standardise_window(
                data, np.mean(data, axis=(1, 2)), np.std(data, axis=(1, 2)), None
            )
        stats = self._raster_stats.get(raster_path)
        if stats is None:
            LOGGER.warning(
                "No stats found for raster %s; returning unstandardised data",
                raster_path,
            )
            return data.astype(np.float32, copy=False)
        return standardise_window(
            data, stats["means"], stats["stds"], stats.get("nodata")
        )

    def _materialise_row(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]

        raster_path = row["raster_path"]
        raster_h = self._get_handle(raster_path)
        window = ((row["r0"], row["r1"]), (row["c0"], row["c1"]))
        rgb_raw = raster_h.read([1, 2, 3], window=window)
        rgb = self._standardise(
            raster_path, rgb_raw, per_tile=self.per_tile_standardisation
        )

        h = row["r1"] - row["r0"]
        w = row["c1"] - row["c0"]
        span_m = float(w) * abs(raster_h.transform.a)

        prewar_path = row["prewar_path"]
        if prewar_path:
            prewar_h = self._get_handle(prewar_path)
            prewar = _read_prewar_tile(
                prewar_h,
                row["lon_min"],
                row["lon_max"],
                row["lat_min"],
                row["lat_max"],
                span_m=span_m,
                min_valid_fraction=0.0,
            )
        else:
            prewar = None
        if prewar is None:
            LOGGER.warning(
                "No prewar data for row %d (prewar_path=%s); using zeros",
                idx,
                prewar_path,
            )
            prewar = np.zeros((3, h, w), dtype=np.float32)
        else:
            prewar = self._standardise(
                prewar_path, prewar, per_tile=self.per_tile_standardisation
            )
        feats_for_tile: list[dict[str, Any]] = []
        labels_path = row["labels_path"]
        feature_ids = row["label_feature_ids"] or []
        if labels_path and feature_ids:
            features = self._labels_cache.get(labels_path, [])
            feats_for_tile = [
                features[i] for i in feature_ids if 0 <= i < len(features)
            ]
        label = create_label_from_feats(
            row["lon_min"],
            row["lat_min"],
            row["lon_max"],
            row["lat_max"],
            feats_for_tile,
            (h, w),
        )

        meta = {
            "origin_image": row["origin_image"],
            "origin_date": row["origin_date"],
            "lon_min": row["lon_min"],
            "lon_max": row["lon_max"],
            "lat_min": row["lat_min"],
            "lat_max": row["lat_max"],
        }
        return {
            "feature": self.feat_transform(rgb),
            "label": self.label_transform(label),
            "prewar": self.feat_transform(prewar),
            "meta": json.dumps(meta, ensure_ascii=False),
        }

    def __getitem__(self, idx: int, remap_idx: bool = True) -> dict[str, Any]:
        if self.indices is not None and remap_idx:
            idx = self.indices[idx]
        return self._materialise_row(idx)

    def label_is_negative(self, i: int) -> bool:
        idx = self.indices[i] if self.indices is not None else i
        feature_ids = self._rows[idx].get("label_feature_ids") or []
        return len(feature_ids) == 0

    def create_subsets(
        self,
        splits: list[float],
        shuffle: bool = True,
        save_loc: str | None = None,
        regenerate_splits: bool = False,
        seed: int | None = None,
    ) -> tuple[list["PairedImageDataset"], list[list[int]]]:
        if save_loc is None:
            cache_valid = False
        else:
            split_file = Path(save_loc) / "splits.csv"
            cache_valid = split_file.exists() and not regenerate_splits

        idcs_list: list[list[int]] = []
        if cache_valid:
            LOGGER.info("Found cached splits, using those.")
            with split_file.open("r") as f:
                data = f.readlines()

            fracs = [float(field.strip()) for field in data[0].strip().split(",")]
            if np.allclose(fracs, splits):
                for row in data[1:]:
                    idcs_list.append(
                        [int(field.strip()) for field in row.strip().split(",")]
                    )
            else:
                LOGGER.warn("Cached splits don't match args, ignoring cache")
                cache_valid = False

        idcs = list(range(len(self)))
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(idcs)

        start_idx = 0
        datasets: list[PairedImageDataset] = []
        for i, split in enumerate(splits):
            end_idx = start_idx + int(len(self) * split)
            subset_indices = idcs[start_idx:end_idx]
            if not cache_valid:
                idcs_list.append(subset_indices)
            else:
                subset_indices = idcs_list[i]
            datasets.append(
                PairedImageDataset(
                    self.manifest_path,
                    indices=subset_indices,
                    sigma=self.sigma,
                    gdal_cache_mb=self.gdal_cache_mb,
                    per_tile_standardisation=self.per_tile_standardisation,
                )
            )
            start_idx = end_idx

        return datasets, idcs_list

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}
