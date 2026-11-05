"""Remove isolated detections from a merged prediction GeoPackage with DBSCAN.

This is intentionally downstream of :mod:`h_merge_geojsons`: merging removes
duplicate detections caused by overlapping tiles, while this stage removes
spatially isolated detections from the resulting single reference-date layer.
"""

from __future__ import annotations

from pathlib import Path

import click
import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN

from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("dbscan_postprocess")
EARTH_RADIUS_M = 6_371_000.0


def dbscan_retain_non_noise(
    points: gpd.GeoDataFrame, *, eps_m: float = 40.0, min_samples: int = 3
) -> gpd.GeoDataFrame:
    """Return the DBSCAN non-noise points, retaining all source attributes.

    Distances use the haversine metric, so this works correctly regardless of
    the source CRS. Output retains the input CRS and row order.
    """
    if eps_m <= 0:
        raise click.ClickException("postprocess.dbscan.eps_m must be greater than zero")
    if min_samples < 1:
        raise click.ClickException(
            "postprocess.dbscan.min_samples must be at least one"
        )
    if points.empty:
        return points.copy()
    if points.crs is None:
        LOGGER.warning("Input CRS is missing; assuming EPSG:4326")
        points = points.set_crs("EPSG:4326")
    if not points.geometry.geom_type.eq("Point").all():
        raise click.ClickException("DBSCAN post-processing requires Point geometries")

    geographic = points.to_crs("EPSG:4326")
    coords = np.column_stack(
        [
            geographic.geometry.y.to_numpy(dtype=float),
            geographic.geometry.x.to_numpy(dtype=float),
        ]
    )
    if not np.isfinite(coords).all():
        raise click.ClickException("DBSCAN post-processing requires finite point coordinates")

    labels = DBSCAN(
        eps=float(eps_m) / EARTH_RADIUS_M,
        min_samples=int(min_samples),
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(np.radians(coords))
    return points.iloc[np.flatnonzero(labels != -1)].copy()


def postprocess_gpkg(
    input_path: str | Path,
    output_path: str | Path,
    *,
    eps_m: float = 40.0,
    min_samples: int = 3,
) -> tuple[int, int]:
    """Filter ``input_path`` and write a separate GeoPackage to ``output_path``."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise click.ClickException(f"Merged GeoPackage not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise click.ClickException("DBSCAN output must differ from its input")

    points = gpd.read_file(input_path)
    retained = dbscan_retain_non_noise(
        points, eps_m=eps_m, min_samples=min_samples
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    retained.to_file(output_path, driver="GPKG")
    return len(points), len(retained)


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@flow_option(default="predict")
def cli(config: str, flow: str) -> None:
    """Apply DBSCAN cleaning to the merged prediction GeoPackage."""
    params = load_flow_config(config, flow)
    postprocess_cfg = params.get("postprocess") or {}
    dbscan_cfg = postprocess_cfg.get("dbscan") or {}
    input_path = postprocess_cfg.get("input") or (params.get("merge") or {}).get(
        "output"
    )
    output_path = postprocess_cfg.get("output")
    if not input_path:
        raise click.ClickException(
            "Missing required config key: postprocess.input (or merge.output as fallback)"
        )
    if not output_path:
        raise click.ClickException("Missing required config key: postprocess.output")

    n_input, n_retained = postprocess_gpkg(
        input_path,
        output_path,
        eps_m=float(dbscan_cfg.get("eps_m", 40.0)),
        min_samples=int(dbscan_cfg.get("min_samples", 3)),
    )
    LOGGER.info(
        "DBSCAN post-processing complete: input=%d, retained=%d, removed=%d; output=%s",
        n_input,
        n_retained,
        n_input - n_retained,
        output_path,
    )


if __name__ == "__main__":
    cli()
