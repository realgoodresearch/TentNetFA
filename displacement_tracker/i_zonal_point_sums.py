"""Aggregate merged tent points into zonal summaries.

This script reads the final merged points GeoPackage produced by
``h_merge_geojsons.py`` and computes per-zone point counts and value distribution stats.
Input/output defaults are resolved from the ``DATA_DIR`` environment variable.
"""

import os
from pathlib import Path

import click
import geopandas as gpd
import pandas as pd

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("zonal_point_sums")


def resolve_data_dir() -> Path:
    """Return DATA_DIR as a Path, raising a CLI error if it is not set."""
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise click.ClickException("Environment variable DATA_DIR is required.")
    return Path(data_dir)


def resolve_path(base_dir: Path, path_value: str) -> Path:
    """Resolve absolute path or DATA_DIR-relative path."""
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path)


def load_gdf(
    path: Path, label: str, fallback_crs: str | None = None
) -> gpd.GeoDataFrame:
    """Load a geodataframe and optionally set a fallback CRS when missing."""
    if not path.exists():
        raise click.ClickException(f"{label} not found: {path}")
    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        raise click.ClickException(f"Failed to read {label} at {path}: {exc}") from exc

    if gdf.empty:
        raise click.ClickException(f"{label} is empty: {path}")

    if gdf.crs is None and fallback_crs is not None:
        LOGGER.warning("%s has no CRS; assuming %s (%s)", label, fallback_crs, path)
        gdf = gdf.set_crs(fallback_crs)

    return gdf


def summarize_points_by_zone(
    points_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    zone_name: str,
    zone_id_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Return per-zone summary table and value column names."""
    if zone_id_column not in zones_gdf.columns:
        raise click.ClickException(
            f"{zone_name} is missing required zone id column: {zone_id_column}"
        )

    zones_cols = zones_gdf[[zone_id_column, "geometry"]].copy()
    joined = gpd.sjoin(
        points_gdf,
        zones_cols,
        how="left",
        predicate="within",
    )

    value_cols = [
        col for col in ["peak_value", "adjusted_peak"] if col in joined.columns
    ]

    grouped = joined.groupby(zone_id_column, dropna=False).agg({"geometry": "count"}).reset_index()
    grouped = grouped.rename(columns={"geometry": "tent_count"})

    for value_col in value_cols:
        series = joined.groupby(zone_id_column, dropna=False)[value_col]
        median_series = series.median()
        iqr_series = series.quantile(0.75) - series.quantile(0.25)
        grouped[f"{value_col}_median"] = grouped[zone_id_column].map(median_series)
        grouped[f"{value_col}_iqr"] = grouped[zone_id_column].map(iqr_series)

    grouped = grouped.sort_values("tent_count", ascending=False)
    return grouped, value_cols


def write_zone_summary(
    points_gdf: gpd.GeoDataFrame,
    zones_path: Path,
    zone_name: str,
    zone_id_column: str,
    output_dir: Path,
) -> None:
    """Compute and write one zonal summary CSV and GeoPackage."""
    zones_gdf = load_gdf(
        zones_path,
        label=f"{zone_name} zones",
        fallback_crs=points_gdf.crs.to_string() if points_gdf.crs else "EPSG:4326",
    )

    if points_gdf.crs != zones_gdf.crs:
        zones_gdf = zones_gdf.to_crs(points_gdf.crs)

    summary_df, value_cols = summarize_points_by_zone(
        points_gdf=points_gdf,
        zones_gdf=zones_gdf,
        zone_name=zone_name,
        zone_id_column=zone_id_column,
    )

    zones_with_summary = zones_gdf.merge(summary_df, on=zone_id_column, how="left")
    zones_with_summary["tent_count"] = zones_with_summary["tent_count"].fillna(0).astype(int)
    if "peak_value" in value_cols:
        zones_with_summary["peak_value_median"] = zones_with_summary["peak_value_median"].fillna(0.0)
        zones_with_summary["peak_value_iqr"] = zones_with_summary["peak_value_iqr"].fillna(0.0)
    if "adjusted_peak" in value_cols:
        zones_with_summary["adjusted_peak_median"] = zones_with_summary["adjusted_peak_median"].fillna(0.0)
        zones_with_summary["adjusted_peak_iqr"] = zones_with_summary["adjusted_peak_iqr"].fillna(0.0)

    output_csv = output_dir / f"zonal_sum_{zone_name}.csv"
    output_gpkg = output_dir / f"zonal_sum_{zone_name}.gpkg"
    summary_df.to_csv(output_csv, index=False)
    zones_with_summary.to_file(output_gpkg, driver="GPKG")
    LOGGER.info("Saved %s zonal summary CSV: %s", zone_name, output_csv)
    LOGGER.info("Saved %s zonal summary GPKG: %s", zone_name, output_gpkg)


@click.command()
@click.option(
    "--points-gpkg",
    default="results/TentNetFA/final/merged_points.gpkg",
    show_default=True,
    help="DATA_DIR-relative or absolute path to merged point GeoPackage.",
)
@click.option(
    "--neighbourhood-zones",
    default="boundaries/neighbourhoods.gpkg",
    show_default=True,
    help="DATA_DIR-relative or absolute path to neighbourhood polygons (placeholder).",
)
@click.option(
    "--municipality-zones",
    default="boundaries/municipalities.gpkg",
    show_default=True,
    help="DATA_DIR-relative or absolute path to municipality polygons (placeholder).",
)
@click.option(
    "--governorate-zones",
    default="boundaries/governorates.gpkg",
    show_default=True,
    help="DATA_DIR-relative or absolute path to governorate polygons (placeholder).",
)
@click.option(
    "--neighbourhood-id-column",
    default="name",
    show_default=True,
    help="Zone identifier column for neighbourhood polygons.",
)
@click.option(
    "--municipality-id-column",
    default="NAME",
    show_default=True,
    help="Zone identifier column for municipality polygons.",
)
@click.option(
    "--governorate-id-column",
    default="name",
    show_default=True,
    help="Zone identifier column for governorate polygons.",
)
@click.option(
    "--output-dir",
    default="results/TentNetFA/final/zonal_summaries",
    show_default=True,
    help="DATA_DIR-relative or absolute output directory for CSV and GPKG summaries.",
)
def cli(
    points_gpkg: str,
    neighbourhood_zones: str,
    municipality_zones: str,
    governorate_zones: str,
    neighbourhood_id_column: str,
    municipality_id_column: str,
    governorate_id_column: str,
    output_dir: str,
) -> None:
    """Generate zonal point summaries for neighbourhood, municipality, and governorate."""
    data_dir = resolve_data_dir()

    points_path = resolve_path(data_dir, points_gpkg)
    neighbourhood_path = resolve_path(data_dir, neighbourhood_zones)
    municipality_path = resolve_path(data_dir, municipality_zones)
    governorate_path = resolve_path(data_dir, governorate_zones)
    output_path = resolve_path(data_dir, output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    points_gdf = load_gdf(
        points_path, "merged point GeoPackage", fallback_crs="EPSG:4326"
    )

    write_zone_summary(
        points_gdf=points_gdf,
        zones_path=neighbourhood_path,
        zone_name="neighbourhood",
        zone_id_column=neighbourhood_id_column,
        output_dir=output_path,
    )
    write_zone_summary(
        points_gdf=points_gdf,
        zones_path=municipality_path,
        zone_name="municipality",
        zone_id_column=municipality_id_column,
        output_dir=output_path,
    )
    write_zone_summary(
        points_gdf=points_gdf,
        zones_path=governorate_path,
        zone_name="governorate",
        zone_id_column=governorate_id_column,
        output_dir=output_path,
    )

    LOGGER.info("All zonal summaries completed in: %s", output_path)


if __name__ == "__main__":
    cli()
