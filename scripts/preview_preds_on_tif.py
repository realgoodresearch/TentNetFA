#!/usr/bin/env python3
"""Overlay prediction points on a satellite GeoTIFF and save a PNG preview."""

from pathlib import Path

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling


@click.command()
@click.argument("points_path", type=click.Path(exists=True))
@click.argument("tif_path", type=click.Path(exists=True))
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output PNG path (default: next to points as preview_on_imagery.png)",
)
@click.option("--max-dim", default=2000, show_default=True, help="Max preview image dimension (px).")
@click.option("--markersize", default=4.0, show_default=True)
def main(points_path, tif_path, out_path, max_dim, markersize):
    points_path = Path(points_path)
    tif_path = Path(tif_path)
    out = Path(out_path) if out_path else points_path.with_name("preview_on_imagery.png")
    if points_path.suffix.lower() == ".gpkg" and out_path is None:
        out = points_path.parent / "preview_on_imagery.png"

    gdf = gpd.read_file(points_path)
    if gdf.empty:
        raise click.ClickException(f"No features in {points_path}")

    with rasterio.open(tif_path) as src:
        scale = max(1, max(src.height, src.width) // max_dim)
        data = src.read(
            out_shape=(src.count, src.height // scale, src.width // scale),
            resampling=Resampling.bilinear,
        )
        transform = src.transform * src.transform.scale(
            src.width / data.shape[-1],
            src.height / data.shape[-2],
        )

        if data.shape[0] >= 3:
            rgb = np.moveaxis(data[:3], 0, -1).astype(np.float32)
        else:
            rgb = np.repeat(np.moveaxis(data[:1], 0, -1), 3, axis=-1).astype(np.float32)

        for i in range(3):
            band = rgb[..., i]
            valid = band > 0
            if np.any(valid):
                lo, hi = np.percentile(band[valid], [2, 98])
            else:
                lo, hi = 0.0, 1.0
            rgb[..., i] = np.clip((band - lo) / (hi - lo + 1e-6), 0, 1)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        extent = [
            transform.c,
            transform.c + transform.a * data.shape[-1],
            transform.f + transform.e * data.shape[-2],
            transform.f,
        ]

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb, extent=extent, origin="upper")
        gdf.plot(
            ax=ax,
            markersize=markersize,
            color="cyan",
            edgecolor="black",
            linewidth=0.2,
            alpha=0.85,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"{points_path.name} on {tif_path.name} ({len(gdf)} points)")
        ax.set_aspect("equal")
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)

    click.echo(f"Saved {out} ({len(gdf)} points)")


if __name__ == "__main__":
    main()
