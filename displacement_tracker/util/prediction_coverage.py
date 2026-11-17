"""Prediction coverage artifacts and merged footprint writing."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

from displacement_tracker.util.source_images import source_image_key


def coverage_record(meta: dict) -> dict:
    """Return one feature describing an inference tile actually evaluated."""
    return {
        "source_image": str(meta.get("origin_image") or ""),
        "source_date": str(meta.get("origin_date") or ""),
        "geometry": box(
            float(meta["lon_min"]),
            float(meta["lat_min"]),
            float(meta["lon_max"]),
            float(meta["lat_max"]),
        ),
    }


def save_prediction_coverage(records: list[dict], output_path: Path) -> None:
    """Write per-tile inference coverage beside one prediction GeoJSON."""
    if not records:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(records, crs="EPSG:4326").to_file(
        output_path, layer="inference_tiles", driver="GPKG"
    )


def _load_coverage(coverage_paths: list[Path]) -> gpd.GeoDataFrame:
    frames = [gpd.read_file(path, layer="inference_tiles") for path in coverage_paths]
    if not frames:
        return gpd.GeoDataFrame(columns=["source_image", "source_date", "geometry"], crs="EPSG:4326")
    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs="EPSG:4326"
    )


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
        source_image_key(row.source_image): row.geometry for row in by_source.itertuples(index=False)
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
        if higher is not None and higher.covers(Point(float(point[1]), float(point[0]))):
            continue
        retained.append(point)
    return retained


def _preferred_coverage(by_source: gpd.GeoDataFrame, priorities: list[str]) -> gpd.GeoDataFrame:
    """Make non-overlapping coverage by subtracting earlier-priority sources."""
    rank = {source_image_key(name): index for index, name in enumerate(priorities)}
    ordered = sorted(
        by_source.itertuples(index=False),
        key=lambda row: (rank.get(source_image_key(row.source_image), len(priorities)), row.source_image),
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
                    "priority_rank": rank.get(source_image_key(row.source_image), len(priorities)),
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
    by_source.to_file(output_path, layer="coverage_by_source", driver="GPKG")
    _preferred_coverage(by_source, list(priorities or [])).to_file(
        output_path, layer="preferred_coverage", driver="GPKG", mode="a"
    )
    return True
