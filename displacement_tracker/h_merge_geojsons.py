"""
merge_geojsons.py

Merge multiple point-cloud GeoJSON files from a folder into a single deduplicated
GeoPackage (.gpkg), using the same merge_close_points_global function used during
prediction.

Usage:
    poetry run merge-geojsons config.yaml

All settings come from the ``merge`` section of the config; the input folder
defaults to ``prediction.output_folder`` so the stage chains onto the
prediction output without extra configuration.
"""

import json
from pathlib import Path

import click
import geopandas as gpd
from shapely.geometry import Point
import yaml

from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.deduplication import merge_close_points_global
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.thresholding import filter_points_by_adjusted_peak

LOGGER = setup_logging("merge_geojsons")


def load_thresholds(thresholds_config: str | None) -> dict[str, float]:
    """
    Load per-file adj_peak thresholds from a YAML file.

    Expected format::

        default: 0.05          # optional global fallback (overrides --min-adj-peak)
        per_file:
          some_file.json: 0.10
          other_file.geojson: 0.20
    """
    if not thresholds_config:
        return {}
    path = Path(thresholds_config)
    if not path.exists():
        raise click.ClickException(f"Thresholds config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_threshold(
    filename: str,
    thresholds_data: dict,
    global_threshold: float,
) -> float:
    """Return the effective adj_peak threshold for a given filename."""
    per_file = thresholds_data.get("per_file") or {}
    if filename in per_file:
        return float(per_file[filename])
    if "default" in thresholds_data:
        return float(thresholds_data["default"])
    return global_threshold


def load_points_from_geojson(path: Path) -> list[tuple]:
    """
    Load all Point features from a GeoJSON file.
    Returns a list of (lat, lon, peak_value, adjusted_peak) tuples.
    adjusted_peak defaults to 0.0 when the field is absent.
    """
    with path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    points = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = feat.get("properties") or {}
        peak = float(props.get("peak_value", 0.0))
        adj_raw = props.get("adjusted_peak", 0.0)
        adj_peak = float(adj_raw) if adj_raw is not None else 0.0
        points.append([lat, lon, peak, adj_peak])

    return points


def load_zone_geometry(zones_path: str | None, label: str):
    """Load and unify polygon geometries from a shapefile or GeoPackage file."""
    if not zones_path:
        return None

    path = Path(zones_path)
    if not path.exists():
        raise click.ClickException(f"{label} zones file not found: {path}")

    try:
        zones_gdf = gpd.read_file(path)
    except Exception as exc:
        raise click.ClickException(
            f"Failed to read {label} zones file {path}: {exc}"
        ) from exc

    if zones_gdf.empty:
        LOGGER.warning("%s zones file is empty: %s", label, path)
        return None

    if zones_gdf.crs is None:
        LOGGER.warning(
            "%s zones CRS missing, assuming EPSG:4326: %s", label, path
        )
        zones_gdf = zones_gdf.set_crs("EPSG:4326")
    else:
        zones_gdf = zones_gdf.to_crs("EPSG:4326")

    geom = zones_gdf.geometry.union_all()
    LOGGER.info("Loaded %s zones from %s", label, path)
    return geom


def filter_points_by_exclusion(points: list[tuple], exclusion_geom) -> list[tuple]:
    """Drop points that lie inside exclusion geometry."""
    if exclusion_geom is None:
        return points

    kept = []
    for lat, lon, peak, adj_peak in points:
        pt = Point(lon, lat)
        if exclusion_geom.contains(pt):
            continue
        kept.append((lat, lon, peak, adj_peak))
    return kept


def filter_points_by_inclusion(points: list[tuple], inclusion_geom) -> list[tuple]:
    """Drop points that lie outside inclusion geometry."""
    if inclusion_geom is None:
        return points

    kept = []
    for lat, lon, peak, adj_peak in points:
        pt = Point(lon, lat)
        if not inclusion_geom.contains(pt):
            continue
        kept.append((lat, lon, peak, adj_peak))
    return kept


def save_merged_gpkg(points: list[tuple], out_path: Path) -> None:
    """Save merged (lat, lon, peak, adj_peak) tuples to a GeoPackage file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "geometry": Point(lon, lat),
            "name": "tents",
            "peak_value": peak,
            "adjusted_peak": adj_peak,
        }
        for lat, lon, peak, adj_peak in points
    ]

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf.to_file(str(out_path), driver="GPKG")

    LOGGER.info("Merged GeoPackage saved to %s (%d points)", out_path, len(gdf))


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@flow_option(default="predict")
def cli(config: str, flow: str) -> None:
    """Merge prediction GeoJSONs into one deduplicated GeoPackage.

    Reads the ``merge`` section of the YAML config; the input folder
    defaults to ``prediction.output_folder``.
    """
    params = load_flow_config(config, flow)
    merge_cfg = params.get("merge") or {}

    input_folder = merge_cfg.get("input_folder") or (
        params.get("prediction") or {}
    ).get("output_folder")
    if not input_folder:
        raise click.ClickException(
            "Missing required config key: merge.input_folder "
            "(or prediction.output_folder as fallback)"
        )
    output_gpkg = merge_cfg.get("output")
    if not output_gpkg:
        raise click.ClickException("Missing required config key: merge.output")

    merge_geojsons(
        input_folder,
        output_gpkg,
        min_distance_m=float(merge_cfg.get("min_distance_m", 3.0)),
        agreement=int(merge_cfg.get("agreement", 1)),
        min_adj_peak=float(merge_cfg.get("min_adj_peak", 0.0)),
        adjustment_factor=float(merge_cfg.get("adjustment_factor", 1.0)),
        thresholds_config=merge_cfg.get("thresholds_config"),
        exclusion_zones_gpkg=merge_cfg.get("exclusion_zones_gpkg"),
        inclusion_zone=merge_cfg.get("inclusion_zone"),
    )


def merge_geojsons(
    input_folder: str,
    output_gpkg: str,
    *,
    min_distance_m: float = 3.0,
    agreement: int = 1,
    min_adj_peak: float = 0.0,
    adjustment_factor: float = 1.0,
    thresholds_config: str | None = None,
    exclusion_zones_gpkg: str | None = None,
    inclusion_zone: str | None = None,
) -> None:
    input_dir = Path(input_folder)
    if not input_dir.is_dir():
        raise click.ClickException(f"Input folder not found: {input_dir}")
    geojson_files = sorted(input_dir.glob("*.geojson")) + sorted(input_dir.glob("*.json"))

    if not geojson_files:
        raise click.ClickException(f"No GeoJSON files found in {input_dir}")

    LOGGER.info("Found %d GeoJSON files in %s", len(geojson_files), input_dir)

    thresholds_data = load_thresholds(thresholds_config)
    exclusion_geom = load_zone_geometry(exclusion_zones_gpkg, "exclusion")
    inclusion_geom = load_zone_geometry(inclusion_zone, "inclusion")

    flat: list[tuple] = []
    for path in geojson_files:
        pts = load_points_from_geojson(path)
        threshold = resolve_threshold(path.name, thresholds_data, min_adj_peak)
        loaded = len(pts)
        pts = filter_points_by_adjusted_peak(pts, threshold, adjustment_factor)
        LOGGER.info(
            "  %s: %d points loaded, %d kept (adj_peak >= %.4f)",
            path.name, loaded, len(pts), threshold,
        )

        if exclusion_geom is not None:
            before_exclusion = len(pts)
            pts = filter_points_by_exclusion(pts, exclusion_geom)
            LOGGER.info(
                "  %s: %d kept after exclusion filtering",
                path.name,
                len(pts),
            )
            if before_exclusion != len(pts):
                LOGGER.info(
                    "  %s: %d points removed by exclusion zones",
                    path.name,
                    before_exclusion - len(pts),
                )

        if inclusion_geom is not None:
            before_inclusion = len(pts)
            pts = filter_points_by_inclusion(pts, inclusion_geom)
            LOGGER.info(
                "  %s: %d kept after inclusion filtering",
                path.name,
                len(pts),
            )
            if before_inclusion != len(pts):
                LOGGER.info(
                    "  %s: %d points removed outside inclusion zone",
                    path.name,
                    before_inclusion - len(pts),
                )

        flat.extend(pts)

    LOGGER.info("Total points before merge: %d", len(flat))

    merged = merge_close_points_global(
        flat, min_distance_m=min_distance_m, agreement=agreement
    )

    LOGGER.info("Total points after merge: %d", len(merged))
    save_merged_gpkg(merged, Path(output_gpkg))


if __name__ == "__main__":
    cli()
