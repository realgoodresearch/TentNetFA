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
import re
import shutil
from datetime import datetime
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

DATE_PATTERN = re.compile(r"_(\d{8})(?=[_.])")

GEOJSON_SUFFIXES = {".json", ".geojson"}


def parse_compact_date(date_str: str) -> bool:
    """True when date_str is a real YYYYMMDD calendar date."""
    try:
        datetime.strptime(date_str, "%Y%m%d")
        return True
    except ValueError:
        return False


def list_geojson_files(directory: Path) -> list[Path]:
    """Sorted GeoJSON/JSON files in a directory, case-insensitive on suffix."""
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in GEOJSON_SUFFIXES
    )


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


def filter_points_by_zone(
    points: list[tuple], zone_geom, keep_inside: bool
) -> list[tuple]:
    """Keep only points inside (keep_inside) or outside the zone geometry."""
    if zone_geom is None:
        return points

    return [
        (lat, lon, peak, adj_peak)
        for lat, lon, peak, adj_peak in points
        if zone_geom.contains(Point(lon, lat)) == keep_inside
    ]


def save_merged_gpkg(points: list[tuple], out_path: Path) -> None:
    """Save merged (lat, lon, peak, adj_peak) tuples to a GeoPackage file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    if not points:
        LOGGER.warning("No points to write; skipping %s", out_path)
        return

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


def process_geojson_folder(
    input_dir: Path,
    output_gpkg: Path,
    min_distance_m: float,
    agreement: int,
    min_adj_peak: float,
    adjustment_factor: float,
    thresholds_data: dict,
    exclusion_geom,
    inclusion_geom,
) -> bool:
    """Merge all GeoJSON/JSON files in one folder into one GPKG.

    Returns False when the folder contains no GeoJSON files. Files that
    cannot be parsed are logged and skipped; if every file fails, raises.
    """
    geojson_files = list_geojson_files(input_dir)

    if not geojson_files:
        LOGGER.warning("No GeoJSON files found in %s", input_dir)
        return False

    LOGGER.info("Found %d GeoJSON files in %s", len(geojson_files), input_dir)

    flat: list[tuple] = []
    unreadable = 0
    for path in geojson_files:
        try:
            pts = load_points_from_geojson(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.error("Skipping unreadable GeoJSON %s: %s", path, exc)
            unreadable += 1
            continue
        threshold = resolve_threshold(path.name, thresholds_data, min_adj_peak)
        loaded = len(pts)
        pts = filter_points_by_adjusted_peak(pts, threshold, adjustment_factor)
        LOGGER.info(
            "  %s: %d points loaded, %d kept (adj_peak >= %.4f)",
            path.name, loaded, len(pts), threshold,
        )

        for label, zone_geom, keep_inside in (
            ("exclusion", exclusion_geom, False),
            ("inclusion", inclusion_geom, True),
        ):
            if zone_geom is None:
                continue
            before = len(pts)
            pts = filter_points_by_zone(pts, zone_geom, keep_inside)
            LOGGER.info(
                "  %s: %d kept after %s filtering (%d removed)",
                path.name,
                len(pts),
                label,
                before - len(pts),
            )

        flat.extend(pts)

    if unreadable == len(geojson_files):
        raise ValueError(
            f"All {unreadable} GeoJSON files in {input_dir} failed to load."
        )

    LOGGER.info("Total points before merge: %d", len(flat))

    merged = merge_close_points_global(
        flat, min_distance_m=min_distance_m, agreement=agreement
    )

    LOGGER.info("Total points after merge: %d", len(merged))
    save_merged_gpkg(merged, output_gpkg)
    return True


def sort_preds_by_date(base_dir: Path) -> None:
    """
    Move root-level JSON/GeoJSON files into date-named subfolders.

    A file like:
        prediction_20240115_abc.json

    becomes:
        base_dir/20240115/prediction_20240115_abc.json
    """
    for path in list_geojson_files(base_dir):
        match = DATE_PATTERN.search(path.name)
        if not match:
            LOGGER.warning("Skipping file without date pattern: %s", path.name)
            continue

        date_str = match.group(1)
        if not parse_compact_date(date_str):
            LOGGER.warning(
                "Skipping file with invalid date %s: %s", date_str, path.name
            )
            continue

        date_folder = base_dir / date_str
        date_folder.mkdir(parents=True, exist_ok=True)

        destination = date_folder / path.name
        if destination.exists():
            LOGGER.warning(
                "Not moving %s: %s already exists; the root-level copy is "
                "left in place and will NOT be merged.",
                path.name,
                destination,
            )
            continue

        shutil.move(str(path), str(destination))
        LOGGER.info("Moved: %s -> %s", path.name, date_folder)


def iter_date_folders(base_dir: Path) -> list[Path]:
    """Return only valid-YYYYMMDD subfolders under base_dir."""
    return sorted(
        p
        for p in base_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{8}", p.name) and parse_compact_date(p.name)
    )


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
    if not output_gpkg and not merge_cfg.get("process_by_date"):
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
            "process_by_date",
        )
        if merge_cfg.get(key) is not None
    }
    merge_geojsons(input_folder, output_gpkg, **kwargs)


def merge_geojsons(
    input_folder: str,
    output_gpkg: str | None,
    *,
    min_distance_m: float = 3.0,
    agreement: int = 1,
    min_adj_peak: float = 0.0,
    adjustment_factor: float = 1.0,
    thresholds_config: str | None = None,
    exclusion_zones_gpkg: str | None = None,
    inclusion_zone: str | None = None,
    process_by_date: bool = False,
) -> None:
    """Merge a folder of prediction GeoJSONs into deduplicated GeoPackage(s).

    With ``process_by_date`` set, root-level predictions are first sorted
    into YYYYMMDD folders and each date is merged separately into
    ``<input_folder>/YYYYMMDD.gpkg`` (``output_gpkg`` is unused); otherwise
    the whole folder is merged into ``output_gpkg``.
    """
    input_dir = Path(input_folder)
    if not input_dir.is_dir():
        raise click.ClickException(f"Input folder not found: {input_dir}")

    thresholds_data = load_thresholds(thresholds_config)
    exclusion_geom = load_zone_geometry(exclusion_zones_gpkg, "exclusion")
    inclusion_geom = load_zone_geometry(inclusion_zone, "inclusion")

    merge_kwargs = dict(
        min_distance_m=min_distance_m,
        agreement=agreement,
        min_adj_peak=min_adj_peak,
        adjustment_factor=adjustment_factor,
        thresholds_data=thresholds_data,
        exclusion_geom=exclusion_geom,
        inclusion_geom=inclusion_geom,
    )

    if process_by_date:
        sort_preds_by_date(input_dir)

        date_folders = iter_date_folders(input_dir)
        if not date_folders:
            raise click.ClickException(
                f"No date folders found in {input_dir} after sorting."
            )

        failed: list[str] = []
        for date_dir in date_folders:
            out_path = input_dir / f"{date_dir.name}.gpkg"
            LOGGER.info("Processing date folder %s -> %s", date_dir, out_path)
            try:
                process_geojson_folder(date_dir, out_path, **merge_kwargs)
            except Exception:
                LOGGER.exception("Failed to process date folder %s", date_dir)
                failed.append(date_dir.name)

        if failed:
            raise click.ClickException(
                f"Failed to process {len(failed)} of {len(date_folders)} "
                f"date folder(s): {', '.join(failed)}"
            )
        return

    if not output_gpkg:
        raise click.ClickException(
            "merge.output is required unless merge.process_by_date is set."
        )

    if not process_geojson_folder(input_dir, Path(output_gpkg), **merge_kwargs):
        raise click.ClickException(f"No GeoJSON files found in {input_dir}")


if __name__ == "__main__":
    cli()
