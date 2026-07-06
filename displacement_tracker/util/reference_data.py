"""Generic interface for ground-truth reference data.

Validation compares predicted point sets against reference counts resolved
onto the master grid. Historically the reference was a directory of
date-stamped UNOSAT exports, paired to each prediction file by nearest
timestamp; that implicit pairing is gone. A reference is now declared
explicitly — in config (``tuning.reference``) or on a CLI — and any source
that can resolve counts onto a window of the master grid plugs in through
``ReferenceSource``::

    reference:
      type: vector            # vector | unosat | raster
      path: ${DATA_DIR}/data/reference/annotations.geojson

Built-in source types (see ``SOURCE_TYPES``):

``vector``
    Point annotations in any OGR-readable file (GeoJSON, GPKG, SHP, ...);
    non-point geometries are reduced to centroids. Optional ``layer`` and
    ``where`` (OGR SQL) narrow the selection.
``unosat``
    A UNOSAT shelter export. Like ``vector``, but ``path`` may also be a
    directory of exports, in which case an explicit ``date`` picks exactly
    one file — the reference is never inferred from prediction timestamps.
``raster``
    Counts already resolved on the master grid (a single-band raster
    aligned with it; anything not aligned is warped cell-to-cell, which
    does not preserve sums — resolve counts onto the master grid upstream).
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("reference_data")

VECTOR_SUFFIXES = (".geojson", ".json", ".gpkg", ".shp", ".fgb", ".gdb")
RASTER_SUFFIXES = (".tif", ".tiff", ".vrt")


class ReferenceSource(ABC):
    """Ground truth that can be resolved onto a window of the master grid."""

    @abstractmethod
    def counts_on_grid(
        self,
        grid_shape: Tuple[int, int],
        transform: rasterio.Affine,
        crs,
        clip_geom=None,
    ) -> np.ndarray:
        """Reference counts per cell (float32) for a master-grid window.

        ``clip_geom`` (in ``crs``) restricts the reference to a geometry —
        typically the convex hull of the predictions — so cells partially
        covered by the analysis area do not pick up outside objects.
        """


class PointsSource(ReferenceSource):
    """An in-memory point GeoDataFrame: one row per reference object."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        self._gdf = gdf

    def counts_on_grid(self, grid_shape, transform, crs, clip_geom=None):
        gdf = self._gdf
        if gdf.crs is not None and crs is not None:
            gdf = gdf.to_crs(crs)
        if clip_geom is not None:
            gdf = gdf.clip(clip_geom)
        return rasterize_point_counts(gdf, grid_shape, transform)


class VectorReferenceSource(PointsSource):
    """Point annotations from an OGR-readable vector file."""

    def __init__(
        self,
        path: str,
        layer: Optional[str] = None,
        where: Optional[str] = None,
    ):
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Reference data not found: {source_path}")
        read_kwargs = {}
        if layer is not None:
            read_kwargs["layer"] = layer
        if where is not None:
            read_kwargs["where"] = where
        gdf = gpd.read_file(source_path, **read_kwargs)
        non_points = gdf.geom_type != "Point"
        if non_points.any():
            LOGGER.info(
                "Reference %s: reducing %d non-point geometries to centroids.",
                source_path.name,
                int(non_points.sum()),
            )
            gdf = gdf.assign(geometry=gdf.geometry.centroid)
        LOGGER.info(
            "Loaded %d reference points from %s", len(gdf), source_path
        )
        super().__init__(gdf)


class UnosatReferenceSource(VectorReferenceSource):
    """A UNOSAT shelter export, selected explicitly.

    ``path`` is either a single export or a directory of exports; a
    directory additionally requires ``date`` (``YYYY-MM-DD`` or
    ``YYYYMMDD``) naming which export to use.
    """

    def __init__(
        self,
        path: str,
        date: Optional[str] = None,
        layer: Optional[str] = None,
        where: Optional[str] = None,
    ):
        super().__init__(_select_export(path, date), layer=layer, where=where)


def _select_export(path: str, date: Optional[str]) -> str:
    """Resolve a UNOSAT export path, requiring an explicit date for directories."""
    source_path = Path(path)
    if not source_path.is_dir():
        return str(source_path)
    if not date:
        raise ValueError(
            f"Reference path {source_path} is a directory of exports; "
            "set reference.date (YYYY-MM-DD) to pick one explicitly."
        )
    wanted = str(date).replace("-", "")
    matches = sorted(
        p
        for p in source_path.iterdir()
        if p.suffix.lower() in VECTOR_SUFFIXES
        and wanted in p.name.replace("-", "")
    )
    if len(matches) != 1:
        available = sorted(
            p.name
            for p in source_path.iterdir()
            if p.suffix.lower() in VECTOR_SUFFIXES
        )
        raise ValueError(
            f"Expected exactly one export dated {date} in {source_path}, "
            f"found {len(matches)}. Available: {available}"
        )
    return str(matches[0])


class RasterReferenceSource(ReferenceSource):
    """A counts raster already resolved on (aligned with) the master grid."""

    def __init__(self, path: str, band: int = 1):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Reference raster not found: {self.path}")
        self.band = int(band)

    def counts_on_grid(self, grid_shape, transform, crs, clip_geom=None):
        # A WarpedVRT pinned to the requested window resolves CRS,
        # registration and out-of-bounds fill in one step; for a raster
        # already on the master grid this is a plain aligned read.
        with rasterio.open(self.path) as src:
            with WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                width=grid_shape[1],
                height=grid_shape[0],
                resampling=Resampling.nearest,
                nodata=0.0,
            ) as vrt:
                data = vrt.read(self.band).astype(np.float32)
        data[~np.isfinite(data)] = 0.0
        np.clip(data, 0.0, None, out=data)
        if clip_geom is not None:
            keep = features.geometry_mask(
                [clip_geom],
                out_shape=grid_shape,
                transform=transform,
                invert=True,
            )
            data[~keep] = 0.0
        return data


def rasterize_point_counts(
    gdf: gpd.GeoDataFrame,
    out_shape: Tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    """Rasterize point geometries into counts per cell."""
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.float32)
    shapes = ((geom, 1) for geom in gdf.geometry)
    return features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        merge_alg=rasterio.enums.MergeAlg.add,
        fill=0,
        dtype="float32",
    )


SOURCE_TYPES = {
    "vector": VectorReferenceSource,
    "geojson": VectorReferenceSource,  # alias: plain annotation files
    "unosat": UnosatReferenceSource,
    "raster": RasterReferenceSource,
}


def _infer_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in RASTER_SUFFIXES:
        return "raster"
    if suffix in VECTOR_SUFFIXES:
        return "vector"
    raise ValueError(
        f"Cannot infer reference type from {path!r}; set reference.type "
        f"to one of: {', '.join(sorted(SOURCE_TYPES))}."
    )


def build_reference_source(cfg) -> ReferenceSource:
    """Build a :class:`ReferenceSource` from config.

    ``cfg`` is either a bare path (type inferred from the suffix) or a
    mapping with ``path``, optional ``type`` and any type-specific keys
    (``date``, ``layer``, ``where``, ``band``). ``None`` values are treated
    as unset so optional keys can be left as ``null`` in YAML.
    """
    if isinstance(cfg, (str, Path)):
        cfg = {"path": str(cfg)}
    if not isinstance(cfg, dict):
        raise ValueError(
            "Reference config must be a path or a mapping with a 'path' key."
        )
    cfg = {k: v for k, v in cfg.items() if v is not None}
    path = cfg.pop("path", None)
    if not path:
        raise ValueError("Reference config is missing required key: path")
    source_type = cfg.pop("type", None) or _infer_type(path)
    try:
        factory = SOURCE_TYPES[source_type]
    except KeyError:
        raise ValueError(
            f"Unknown reference type {source_type!r}; expected one of: "
            f"{', '.join(sorted(SOURCE_TYPES))}."
        ) from None
    # Only pass what the type understands, so options left over from
    # another type (e.g. a unosat `date` after switching to vector) don't
    # abort the run.
    accepted_params = inspect.signature(factory.__init__).parameters
    accepted = {k: v for k, v in cfg.items() if k in accepted_params}
    dropped = sorted(set(cfg) - set(accepted))
    if dropped:
        LOGGER.warning(
            "Reference type %r ignores option(s): %s", source_type, dropped
        )
    return factory(path, **accepted)
