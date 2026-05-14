"""Direct validation of predicted GeoJSON/GPKG point sets against ground truth.

For each prediction file, the temporally nearest validation file is rasterized
onto a master grid restricted to the convex hull of the predictions, and per-tile
RMS, MAE, RMSLE, Spearman correlation, total counts and total difference are
reported. A single (factor, cutoff) is applied to the predictions; for sweeping
that parameter pair see g_scan_validation.py.
"""

import os

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from displacement_tracker.util.validation_core import (
    compute_metrics,
    discover_pred_val_pairs,
    keep_mask_from_params,
    prepare_grouped_cell_inputs,
    process_grouped_cells,
    write_output_rasters,
)


def validate_one_tile(
    pred_gdf: gpd.GeoDataFrame,
    val_gdf: gpd.GeoDataFrame,
    src_grid: rasterio.io.DatasetReader,
    factor: float,
    cutoff: float,
):
    """Run the full validation pipeline for one prediction/reference pair."""
    grouped = prepare_grouped_cell_inputs(pred_gdf, val_gdf, src_grid)
    pred_prepped = grouped["pred_prepped"]
    keep = keep_mask_from_params(pred_prepped, factor=factor, cutoff=cutoff)

    processed = process_grouped_cells(
        pred_rows=pred_prepped["row"].to_numpy(dtype=np.int32)[keep],
        pred_cols=pred_prepped["col"].to_numpy(dtype=np.int32)[keep],
        val_raster=grouped["val_raster"].copy(),
        mask_array=grouped["mask_array"],
        grid_shape=grouped["grid_shape"],
        nodata_val=grouped["nodata_val"],
    )
    metrics = compute_metrics(
        processed["pred_raster"],
        processed["val_raster"],
        processed["error_raster"],
        processed["mask_array"],
    )
    return grouped, processed, metrics


@click.command()
@click.option("--pred-dir", type=click.Path(exists=True), required=True)
@click.option("--val-dir", type=click.Path(exists=True), required=True)
@click.option("--master-grid", type=click.Path(exists=True), required=True)
@click.option("--out-dir", type=click.Path(), default="validation_results")
@click.option(
    "--exclusion-zones",
    type=click.Path(exists=True),
    default=None,
    help="Optional gpkg file of exclusion zones; predictions are clipped to its union.",
)
@click.option(
    "--factor",
    type=float,
    default=1.0,
    show_default=True,
    help="Peak rescaling factor: 0 -> peak_value, 1 -> adjusted_peak.",
)
@click.option(
    "--cutoff",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum rescaled peak required to keep a predicted point.",
)
def cli(pred_dir, val_dir, master_grid, out_dir, exclusion_zones, factor, cutoff):
    """Validate predictions against ground truth at a fixed (factor, cutoff)."""
    os.makedirs(out_dir, exist_ok=True)

    exclusion_geom = None
    if exclusion_zones:
        exclusion_geom = gpd.read_file(exclusion_zones).geometry.union_all()

    pairs = discover_pred_val_pairs(pred_dir, val_dir)
    results = []

    with rasterio.open(master_grid) as src_grid:
        raster_crs = src_grid.crs

        for pred_path, val_path, pred_date, val_date in pairs:
            pred_file = os.path.basename(pred_path)
            base_name = os.path.splitext(pred_file)[0]

            pred_gdf = gpd.read_file(pred_path).to_crs(raster_crs)
            val_gdf = gpd.read_file(val_path).to_crs(raster_crs)
            if exclusion_geom is not None:
                pred_gdf = pred_gdf.clip(exclusion_geom)
            if pred_gdf.empty:
                continue

            try:
                grouped, processed, metrics = validate_one_tile(
                    pred_gdf, val_gdf, src_grid, factor=factor, cutoff=cutoff
                )
            except Exception as exc:
                click.echo(f"Skipping {pred_file}: {exc}")
                continue

            results.append(
                {
                    "file": pred_file,
                    "pred_date": pred_date.strftime("%Y-%m-%d"),
                    "val_date": val_date.strftime("%Y-%m-%d"),
                    "factor": factor,
                    "cutoff": cutoff,
                    **metrics,
                }
            )
            click.echo(
                f"{base_name}: RMS={metrics['rms']:.3f}  MAE={metrics['mae']:.3f}  "
                f"total_diff={metrics['total_diff']:.1f}  "
                f"spearman={metrics['spearman']:.3f}"
            )

            diff_masked = processed["diff"] * processed["mask_array"].astype(np.int32)
            write_output_rasters(
                out_dir=out_dir,
                base_name=base_name,
                pred_raster=processed["pred_raster"],
                val_raster=processed["val_raster"],
                diff_masked=diff_masked,
                src_grid=src_grid,
                grid_shape=grouped["grid_shape"],
                out_transform=grouped["out_transform"],
                nodata_val=grouped["nodata_val"],
            )

    if not results:
        click.echo("No results to summarize.")
        return

    df = pd.DataFrame(results)
    summary_path = os.path.join(out_dir, "validation_summary.csv")
    df.to_csv(summary_path, index=False)
    click.echo(f"\nSummary report saved to: {summary_path}")
    click.echo(f"Global RMS:      {df['rms'].mean():.4f}")
    click.echo(f"Global MAE:      {df['mae'].mean():.4f}")
    click.echo(f"Global RMSLE:    {df['rmsle'].mean():.4f}")
    click.echo(f"Global Spearman: {df['spearman'].mean(skipna=True):.4f}")


if __name__ == "__main__":
    cli()
