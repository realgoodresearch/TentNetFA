from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from displacement_tracker.util.tile_builder import TILE_HEIGHT, TILE_WIDTH


class HDF5Writer:
    """
    Writes tiles by appending into extensible chunked datasets:
      - feature : (N, C, H, W) dtype matches feature
      - label   : (N, H, W) dtype=uint8
      - prewar  : (N, 3, Hp, Wp) float16
      - meta    : (N,) variable-length JSON strings

    Datasets are created lazily on the first add_entry (so we know tile shape).
    Periodically flushes to disk to release internal buffers.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = h5py.File(self.path, "w")
        self._created = False
        self.tile_idx = 0
        self._d_feature = None
        self._d_label = None
        self._d_prewar = None
        self._d_meta = None

    def _create_datasets(
        self, feature: np.ndarray, label: np.ndarray, prewar: np.ndarray | None
    ):
        channels, h, w = feature.shape
        chunks = (1, channels, h, w)
        maxshape = (None, channels, h, w)

        self._d_feature = self._file.create_dataset(
            "feature",
            shape=(0, channels, h, w),
            maxshape=maxshape,
            chunks=chunks,
            dtype=feature.dtype,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        self._d_label = self._file.create_dataset(
            "label",
            shape=(0, h, w),
            maxshape=(None, h, w),
            chunks=(1, h, w),
            dtype=label.dtype,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        self._d_prewar = self._file.create_dataset(
            "prewar",
            shape=(0, 3, TILE_HEIGHT, TILE_WIDTH),
            maxshape=(None, 3, TILE_HEIGHT, TILE_WIDTH),
            chunks=(1, 3, TILE_HEIGHT, TILE_WIDTH),
            dtype=np.float16,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        dt = h5py.special_dtype(vlen=str)
        self._d_meta = self._file.create_dataset(
            "meta", shape=(0,), maxshape=(None,), dtype=dt, chunks=(1,)
        )
        self._created = True

    def add_entry(
        self,
        feature: np.ndarray,
        label: np.ndarray,
        meta: dict[str, Any],
        prewar: np.ndarray | None = None,
    ):
        if not self._created:
            self._create_datasets(feature, label, prewar)

        idx = self._d_feature.shape[0]

        self._d_feature.resize(idx + 1, axis=0)
        self._d_feature[idx, :, :, :] = feature.astype(self._d_feature.dtype, copy=False)

        self._d_label.resize(idx + 1, axis=0)
        self._d_label[idx, :, :] = label

        if self._d_prewar is not None:
            self._d_prewar.resize(idx + 1, axis=0)
            if prewar is None:
                self._d_prewar[idx] = np.zeros(
                    self._d_prewar.shape[1:], dtype=self._d_prewar.dtype
                )
            else:
                self._d_prewar[idx] = prewar.astype(self._d_prewar.dtype, copy=False)

        meta_json = json.dumps(meta, ensure_ascii=False)
        self._d_meta.resize(idx + 1, axis=0)
        self._d_meta[idx] = meta_json

        self.tile_idx += 1
        try:
            if self.tile_idx % 1000 == 0:
                self._file.flush()
        except Exception:
            pass

    def write(self):
        self._file.close()
