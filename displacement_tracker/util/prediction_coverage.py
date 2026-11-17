"""Prediction coverage artifacts and merged footprint writing."""

import math
from pathlib import Path
from uuid import uuid4

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp, MaskFlags, Resampling
from rasterio.features import shapes
from rasterio.warp import reproject
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.source_images import source_image_key

LOGGER = setup_logging("prediction_coverage")
MAX_FOOTPRINT_DIMENSION = 10_000


def _minimum_resample(src, index, out_shape, transform):
    """Downsample one band without ignoring zero-valued alpha pixels."""
    destination = np.empty(out_shape, dtype=src.dtypes[index - 1])
    reproject(
        source=rasterio.band(src, index),
        destination=destination,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=src.crs,
        resampling=Resampling.min,
    )
    return destination


def coverage_record(meta: dict) -> dict:
    """Return one feature describing an inference tile actually evaluated."""
    return {
        "source_image": str(meta.get("origin_image") or ""),
        "source_date": str(meta.get("origin_date") or ""),
        "raster_path": str(meta.get("raster_path") or ""),
        "geometry": box(
            float(meta["lon_min"]),
            float(meta["lat_min"]),
            float(meta["lon_max"]),
            float(meta["lat_max"]),
        ),
    }


def _valid_raster_footprint(raster_path: Path):
    """Return the valid-pixel footprint in EPSG:4326.

    Alpha is authoritative when present. Otherwise use a GDAL dataset/nodata
    mask, then fall back to the scanner's nonzero-RGB convention. Large rasters
    are reduced conservatively: a coarse cell is valid only when all source
    pixels represented by it are valid, avoiding coverage beyond image edges.
    """
    with rasterio.open(raster_path) as src:
        factor = max(
            1,
            math.ceil(max(src.width, src.height) / MAX_FOOTPRINT_DIMENSION),
        )
        out_height = math.ceil(src.height / factor)
        out_width = math.ceil(src.width / factor)
        out_shape = (out_height, out_width)
        footprint_transform = src.transform * Affine.scale(
            src.width / out_width,
            src.height / out_height,
        )

        alpha_indexes = [
            index
            for index, color in enumerate(src.colorinterp, start=1)
            if color == ColorInterp.alpha
        ]
        if alpha_indexes:
            # A coarse cell is valid only if all contributing alpha pixels
            # are nonzero. This works for integer and floating-point alpha.
            alpha = (
                src.read(alpha_indexes[0])
                if factor == 1
                else _minimum_resample(
                    src, alpha_indexes[0], out_shape, footprint_transform
                )
            )
            valid = np.isfinite(alpha) & (alpha > 0)
        elif any(MaskFlags.all_valid not in flags for flags in src.mask_flag_enums[:3]):
            dataset_mask = src.dataset_mask(
                out_shape=out_shape,
                resampling=Resampling.average,
            )
            valid = dataset_mask > 0 if factor == 1 else dataset_mask >= 255
        else:
            indexes = list(range(1, min(src.count, 3) + 1))
            rgb = (
                src.read(indexes)
                if factor == 1
                else np.stack(
                    [
                        _minimum_resample(src, index, out_shape, footprint_transform)
                        for index in indexes
                    ]
                )
            )
            valid = np.any(np.isfinite(rgb) & (rgb != 0), axis=0)

        if not valid.any():
            return None

        polygons = [
            shape(geometry)
            for geometry, value in shapes(
                valid.astype("uint8"),
                mask=valid,
                transform=footprint_transform,
            )
            if value == 1
        ]
        if not polygons:
            return None
        footprint = gpd.GeoSeries([unary_union(polygons)], crs=src.crs).to_crs(
            "EPSG:4326"
        )
        return footprint.iloc[0]


def save_prediction_coverage(records: list[dict], output_path: Path) -> None:
    """Write inferred coverage clipped to each source's valid pixels."""
    if not records:
        output_path.unlink(missing_ok=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage = gpd.GeoDataFrame(records, crs="EPSG:4326")
    rows = []
    for keys, group in coverage.groupby(
        ["source_image", "source_date", "raster_path"], dropna=False
    ):
        source_image, source_date, raster_path = keys
        geometry = unary_union(group.geometry)
        if raster_path:
            try:
                valid_footprint = _valid_raster_footprint(Path(raster_path))
                if valid_footprint is None:
                    LOGGER.warning(
                        "No valid pixels found in %s; omitting its coverage",
                        raster_path,
                    )
                    continue
                geometry = geometry.intersection(valid_footprint)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not clip prediction coverage to valid pixels in "
                    f"{raster_path}"
                ) from exc
        if geometry.is_empty:
            continue
        rows.append(
            {
                "source_image": source_image,
                "source_date": source_date,
                "raster_path": raster_path,
                "geometry": geometry,
            }
        )
    if not rows:
        output_path.unlink(missing_ok=True)
        return
    gpd.GeoDataFrame(rows, crs="EPSG:4326").to_file(
        output_path, layer="inference_tiles", driver="GPKG"
    )


def _load_coverage(coverage_paths: list[Path]) -> gpd.GeoDataFrame:
    frames = [gpd.read_file(path, layer="inference_tiles") for path in coverage_paths]
    if not frames:
        return gpd.GeoDataFrame(
            columns=["source_image", "source_date", "geometry"], crs="EPSG:4326"
        )
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")


def _dissolve_by_source(coverage: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return coverage.dissolve(by=["source_image", "source_date"], as_index=False)


def filter_points_by_priority_coverage(
    points: list[tuple], coverage_paths: list[Path], priorities: list[str] | None
) -> list[tuple]:
    """Remove points covered by an actually inferred, higher-priority source.

    Points are ``(lat, lon, peak, adjusted_peak, source_image)`` tuples. An
    unlisted source has lower priority than every listed source, while sources
    at the same priority do not suppress one another.
    """
    coverage = _load_coverage(coverage_paths)
    if coverage.empty:
        return points
    by_source = coverage.dissolve(by="source_image", as_index=False)
    priorities = list(priorities or [])
    ranks = {source_image_key(name): index for index, name in enumerate(priorities)}
    fallback_rank = len(priorities)
    source_geometries = {
        source_image_key(row.source_image): row.geometry
        for row in by_source.itertuples(index=False)
    }
    higher_coverage: dict[int, object] = {}
    covered = None
    for rank in range(fallback_rank + 1):
        higher_coverage[rank] = covered
        geometries = [
            geometry
            for source, geometry in source_geometries.items()
            if ranks.get(source, fallback_rank) == rank
        ]
        if geometries:
            current = unary_union(geometries)
            covered = current if covered is None else covered.union(current)

    retained = []
    for point in points:
        source = source_image_key(point[4]) if len(point) > 4 else ""
        higher = higher_coverage.get(ranks.get(source, fallback_rank))
        if higher is not None and higher.covers(
            Point(float(point[1]), float(point[0]))
        ):
            continue
        retained.append(point)
    return retained


def _preferred_coverage(
    by_source: gpd.GeoDataFrame, priorities: list[str]
) -> gpd.GeoDataFrame:
    """Make non-overlapping coverage by subtracting earlier-priority sources."""
    rank = {source_image_key(name): index for index, name in enumerate(priorities)}
    ordered = sorted(
        by_source.itertuples(index=False),
        key=lambda row: (
            rank.get(source_image_key(row.source_image), len(priorities)),
            row.source_image,
        ),
    )
    covered = None
    rows = []
    for row in ordered:
        geometry = row.geometry if covered is None else row.geometry.difference(covered)
        if not geometry.is_empty:
            rows.append(
                {
                    "source_image": row.source_image,
                    "source_date": row.source_date,
                    "priority_rank": rank.get(
                        source_image_key(row.source_image), len(priorities)
                    ),
                    "geometry": geometry,
                }
            )
        covered = row.geometry if covered is None else covered.union(row.geometry)
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def write_merged_footprint(
    coverage_paths: list[Path], output_path: Path, *, priorities: list[str] | None
) -> bool:
    """Write dissolved-source and preferred coverage layers."""
    coverage = _load_coverage(coverage_paths)
    if coverage.empty:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_source = _dissolve_by_source(coverage)
    preferred = _preferred_coverage(by_source, list(priorities or []))
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp.gpkg"
    )
    try:
        by_source.to_file(temporary_path, layer="coverage_by_source", driver="GPKG")
        preferred.to_file(
            temporary_path,
            layer="preferred_coverage",
            driver="GPKG",
            mode="a",
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True
