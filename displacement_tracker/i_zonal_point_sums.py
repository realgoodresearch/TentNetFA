"""Aggregate merged tent points into zonal summaries.

This script reads the final merged points GeoPackage produced by
``h_merge_geojsons.py`` and computes per-zone point counts and value distribution stats.
Input/output defaults are resolved from the ``DATA_DIR`` environment variable.
"""

import os
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("zonal_point_sums")


def resolve_data_dir() -> Path:
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        raise click.ClickException("Environment variable DATA_DIR is required.")
    return Path(data_dir)


def resolve_path(base_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base_dir / path)


def load_gdf(path: Path, label: str, fallback_crs: str | None = None) -> gpd.GeoDataFrame:
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


def attach_distribution_stats(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Attach median and IQR stats computed from raw grouped values."""
    for value_col in value_cols:
        grouped = raw_df.groupby(group_col, dropna=False)[value_col]
        medians = grouped.median()
        iqrs = grouped.quantile(0.75) - grouped.quantile(0.25)
        summary_df[f"{value_col}_median"] = summary_df[group_col].map(medians)
        summary_df[f"{value_col}_iqr"] = summary_df[group_col].map(iqrs)
    return summary_df


def summarize_points_by_zone(
    points_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    zone_name: str,
    zone_id_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    if zone_id_column not in zones_gdf.columns:
        raise click.ClickException(
            f"{zone_name} is missing required zone id column: {zone_id_column}"
        )

    zones_cols = zones_gdf[[zone_id_column, "geometry"]].copy()
    joined = gpd.sjoin(points_gdf, zones_cols, how="left", predicate="within")

    value_cols = [col for col in ["peak_value", "adjusted_peak"] if col in joined.columns]
    grouped = (
        joined.groupby(zone_id_column, dropna=False)
        .agg({"geometry": "count"})
        .reset_index()
        .rename(columns={"geometry": "tent_count"})
    )

    grouped = attach_distribution_stats(grouped, joined, zone_id_column, value_cols)
    grouped = grouped.sort_values("tent_count", ascending=False)
    return grouped, value_cols


def write_zone_summary(
    points_gdf: gpd.GeoDataFrame,
    zones_path: Path,
    zone_name: str,
    zone_id_column: str,
    output_dir: Path,
) -> None:
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


def write_master_grid_tent_count_tiff(points_gdf: gpd.GeoDataFrame, master_grid_path: Path, output_dir: Path) -> None:
    if not master_grid_path.exists():
        raise click.ClickException(f"Master grid raster not found: {master_grid_path}")

    with rasterio.open(master_grid_path) as src:
        if src.crs is None:
            raise click.ClickException(f"Master grid CRS is missing: {master_grid_path}")

        points_in_grid = points_gdf.to_crs(src.crs)
        xs = points_in_grid.geometry.x.to_numpy()
        ys = points_in_grid.geometry.y.to_numpy()
        rows, cols = rowcol(src.transform, xs, ys)

        rows = np.asarray(rows)
        cols = np.asarray(cols)
        in_bounds = (
            (rows >= 0)
            & (rows < src.height)
            & (cols >= 0)
            & (cols < src.width)
        )

        counts = np.zeros((src.height, src.width), dtype=np.int32)
        if np.any(in_bounds):
            np.add.at(counts, (rows[in_bounds], cols[in_bounds]), 1)

        profile = src.profile.copy()
        profile.update(dtype="int32", count=1, nodata=0)

        output_tif = output_dir / "zonal_sum_mastergrid_tent_count.tif"
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(counts, 1)

    LOGGER.info("Saved master-grid tent-count GeoTIFF: %s", output_tif)


@click.command()
@click.option("--points-gpkg", default="results/TentNetFA/final/merged_points.gpkg", show_default=True)
@click.option("--neighbourhood-zones", default="boundaries/neighbourhoods.gpkg", show_default=True)
@click.option("--municipality-zones", default="boundaries/municipalities.gpkg", show_default=True)
@click.option("--governorate-zones", default="boundaries/governorates.gpkg", show_default=True)
@click.option(
    "--master-grid",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional DATA_DIR-relative or absolute path to a master-grid raster for grid-cell tent-count GeoTIFF output.",
)
@click.option("--neighbourhood-id-column", default="name", show_default=True)
@click.option("--municipality-id-column", default="NAME", show_default=True)
@click.option("--governorate-id-column", default="name", show_default=True)
@click.option("--output-dir", default="results/TentNetFA/final/zonal_summaries", show_default=True)
def cli(
    points_gpkg: str,
    neighbourhood_zones: str,
    municipality_zones: str,
    governorate_zones: str,
    master_grid: str | None,
    neighbourhood_id_column: str,
    municipality_id_column: str,
    governorate_id_column: str,
    output_dir: str,
) -> None:
    data_dir = resolve_data_dir()

    points_path = resolve_path(data_dir, points_gpkg)
    neighbourhood_path = resolve_path(data_dir, neighbourhood_zones)
    municipality_path = resolve_path(data_dir, municipality_zones)
    governorate_path = resolve_path(data_dir, governorate_zones)
    master_grid_path = resolve_path(data_dir, master_grid) if master_grid else None
    output_path = resolve_path(data_dir, output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    points_gdf = load_gdf(points_path, "merged point GeoPackage", fallback_crs="EPSG:4326")

    write_zone_summary(points_gdf, neighbourhood_path, "neighbourhood", neighbourhood_id_column, output_path)
    write_zone_summary(points_gdf, municipality_path, "municipality", municipality_id_column, output_path)
    write_zone_summary(points_gdf, governorate_path, "governorate", governorate_id_column, output_path)

    if master_grid_path is not None:
        write_master_grid_tent_count_tiff(points_gdf, master_grid_path, output_path)

    LOGGER.info("All zonal summaries completed in: %s", output_path)


if __name__ == "__main__":
    cli()
