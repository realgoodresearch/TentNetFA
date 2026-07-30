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
from displacement_tracker.util.thresholding import (
    PredictedPoint,
    adjustment_signal_from_peaks,
    filter_points_by_rescaled_peak,
)

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


def load_points_from_geojson(path: Path) -> list[PredictedPoint]:
    """
    Load all Point features from a GeoJSON file, applying the missing-field
    rules documented on PredictedPoint. Logs once per file when any point's
    raw adjustment signal had to be derived rather than read.
    """
    with path.open("r", encoding="utf-8") as f:
        gj = json.load(f)

    points = []
    derived_signals = 0
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        props = feat.get("properties") or {}
        peak_raw = props.get("peak_value")
        peak = float(peak_raw) if peak_raw is not None else 0.0
        adj_raw = props.get("adjusted_peak")
        # A point with no adjusted peak recorded was never adjusted.
        adj_peak = float(adj_raw) if adj_raw is not None else peak
        signal_raw = props.get("adjustment_signal")
        if signal_raw is None:
            adj_signal = adjustment_signal_from_peaks(peak, adj_peak)
            derived_signals += 1
        else:
            adj_signal = float(signal_raw)
        points.append(PredictedPoint(lat, lon, peak, adj_peak, adj_signal))

    if derived_signals:
        LOGGER.warning(
            "  %s: %d/%d points carry no adjustment_signal; derived it as "
            "(adjusted_peak - peak_value), which assumes the predictions were "
            "made with selection.factor=1.0",
            path.name, derived_signals, len(points),
        )

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


def filter_points_by_geometry(
    points: list[PredictedPoint], geom, *, keep_inside: bool
) -> list[PredictedPoint]:
    """Keep points inside (or outside) ``geom``; a null geometry keeps everything."""
    if geom is None:
        return points

    return [
        point
        for point in points
        if bool(geom.contains(Point(point.lon, point.lat))) == keep_inside
    ]


def save_merged_gpkg(points: list[PredictedPoint], out_path: Path) -> None:
    """Save merged points to a GeoPackage file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "geometry": Point(point.lon, point.lat),
            "name": "tents",
            "peak_value": point.peak_value,
            "adjusted_peak": point.adjusted_peak,
            "adjustment_signal": point.adjustment_signal,
        }
        for point in points
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

    # only pass keys the config actually sets — merge_geojsons() signature
    # defaults are the single source of truth for the rest
    kwargs = {
        key: merge_cfg[key]
        for key in (
            "min_distance_m",
            "agreement",
            "min_adj_peak",
            "adjustment_factor",
            "thresholds_config",
            "exclusion_zones_gpkg",
            "inclusion_zone",
        )
        if merge_cfg.get(key) is not None
    }
    merge_geojsons(input_folder, output_gpkg, **kwargs)


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

    flat: list[PredictedPoint] = []
    for path in geojson_files:
        pts = load_points_from_geojson(path)
        threshold = resolve_threshold(path.name, thresholds_data, min_adj_peak)
        loaded = len(pts)
        pts = filter_points_by_rescaled_peak(pts, threshold, adjustment_factor)
        LOGGER.info(
            "  %s: %d points loaded, %d kept (peak + %.4g * adjustment_signal >= %.4f)",
            path.name, loaded, len(pts), adjustment_factor, threshold,
        )

        for geom, keep_inside, label in (
            (exclusion_geom, False, "inside exclusion zones"),
            (inclusion_geom, True, "outside the inclusion zone"),
        ):
            if geom is None:
                continue
            before = len(pts)
            pts = filter_points_by_geometry(pts, geom, keep_inside=keep_inside)
            LOGGER.info(
                "  %s: %d kept, %d removed %s",
                path.name, len(pts), before - len(pts), label,
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
