"""Prediction coverage artifacts and merged footprint writing."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


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


def _preferred_coverage(by_source: gpd.GeoDataFrame, priorities: list[str]) -> gpd.GeoDataFrame:
    """Make non-overlapping coverage by subtracting earlier-priority sources."""
    rank = {name: index for index, name in reversed(list(enumerate(priorities)))}
    ordered = sorted(
        by_source.itertuples(index=False),
        key=lambda row: (rank.get(row.source_image, len(priorities)), row.source_image),
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
                    "priority_rank": rank.get(row.source_image, len(priorities)),
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
