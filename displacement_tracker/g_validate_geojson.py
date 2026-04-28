import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features, mask
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from tqdm import tqdm

@dataclass(frozen=True)
class RasterMetadata:
    """Stores the necessary metadata for the output difference raster."""

    transform: rasterio.Affine
    width: int
    height: int
    crs: rasterio.crs.CRS


def extract_date_from_path(path: str) -> Optional[datetime]:
    """Matches YYYY-MM-DD or YYYYMMDD in filenames."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})|(\d{8})", os.path.basename(path))
    if not match:
        return None
    date_str = match.group(0).replace("-", "")
    return datetime.strptime(date_str, "%Y%m%d")


def get_point_counts(
    gdf: gpd.GeoDataFrame, out_shape: Tuple[int, int], transform: rasterio.Affine
) -> np.ndarray:
    """Rasterizes point geometries into counts per cell."""
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.float32)

    # Create (geometry, 1) pairs. rasterize expects geometries in the target CRS.
    shapes = ((geom, 1) for geom in gdf.geometry)

    return features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        merge_alg=rasterio.enums.MergeAlg.add,
        fill=0,
        dtype="float32",
    )


def process_prediction_validation(
    pred_gdf: gpd.GeoDataFrame,
    val_gdf: gpd.GeoDataFrame,
    src_grid: rasterio.io.DatasetReader,
    nodata_val: float = -9999.0,
) -> Dict[str, object]:
    """Rasterize, compare, and mask prediction/validation points for one tile."""
    # analysis mask (convex hull around predicted tents)
    prediction_extent_geom = pred_gdf.union_all().convex_hull

    # apply mask to master grid
    out_image, out_transform = mask.mask(
        src_grid,
        [prediction_extent_geom],
        crop=True,
        nodata=np.nan,
    )

    # out_image shape is (bands, rows, cols)
    grid_shape = (out_image.shape[1], out_image.shape[2])

    # rasterize predictions
    pred_raster = get_point_counts(pred_gdf, grid_shape, out_transform)

    # clip validation points to the analysis mask and rasterize
    val_in_hull = val_gdf.clip(prediction_extent_geom)
    val_raster = get_point_counts(val_in_hull, grid_shape, out_transform)

    # prediction-minus-validation and log-space error rasters
    diff = pred_raster - val_raster
    error_raster = np.log1p(pred_raster) - np.log1p(val_raster)

    # outside the mask is False
    mask_array = features.geometry_mask(
        [prediction_extent_geom],
        out_shape=grid_shape,
        transform=out_transform,
        invert=True,
    )

    # mask all arrays outside the analysis area
    for r in [pred_raster, val_raster, diff, error_raster]:
        r = r[mask_array]

    return {
        "pred_raster": pred_raster,
        "val_raster": val_raster,
        "diff": diff,
        "error_raster": error_raster,
        "mask_array": mask_array,
        "out_transform": out_transform,
        "grid_shape": grid_shape,
        "nodata_val": nodata_val,
    }


def prepare_grouped_cell_inputs(
    pred_gdf: gpd.GeoDataFrame,
    val_gdf: gpd.GeoDataFrame,
    src_grid: rasterio.io.DatasetReader,
    nodata_val: float = -9999.0,
) -> Dict[str, object]:
    """Prepare one-time geometry/grid products and per-point cell assignments."""
    prediction_extent_geom = pred_gdf.union_all().convex_hull

    out_image, out_transform = mask.mask(
        src_grid,
        [prediction_extent_geom],
        crop=True,
        nodata=np.nan,
    )
    grid_shape = (out_image.shape[1], out_image.shape[2])

    mask_array = features.geometry_mask(
        [prediction_extent_geom],
        out_shape=grid_shape,
        transform=out_transform,
        invert=True,
    )

    val_in_hull = val_gdf.clip(prediction_extent_geom)
    val_raster = get_point_counts(val_in_hull, grid_shape, out_transform)

    xs = pred_gdf.geometry.x.to_numpy()
    ys = pred_gdf.geometry.y.to_numpy()
    rows, cols = rowcol(out_transform, xs, ys)
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)

    in_bounds = (
        (rows >= 0)
        & (rows < grid_shape[0])
        & (cols >= 0)
        & (cols < grid_shape[1])
    )

    pred_prepped = pred_gdf.loc[in_bounds, ["peak_value", "adjusted_peak"]].copy()
    pred_prepped["row"] = rows[in_bounds]
    pred_prepped["col"] = cols[in_bounds]

    return {
        "pred_prepped": pred_prepped,
        "val_raster": val_raster,
        "mask_array": mask_array,
        "out_transform": out_transform,
        "grid_shape": grid_shape,
        "nodata_val": nodata_val,
    }


def process_grouped_cells(
    pred_rows: np.ndarray,
    pred_cols: np.ndarray,
    val_raster: np.ndarray,
    mask_array: np.ndarray,
    grid_shape: Tuple[int, int],
    nodata_val: float,
) -> Dict[str, np.ndarray]:
    """Build prediction raster from pre-grouped cells and derive diff/error rasters."""
    pred_raster = np.zeros(grid_shape, dtype=np.float32)
    if pred_rows.size > 0:
        np.add.at(pred_raster, (pred_rows, pred_cols), 1.0)

    diff = pred_raster - val_raster
    error_raster = np.log1p(pred_raster) - np.log1p(val_raster)

    for arr in [pred_raster, val_raster, diff, error_raster]:
        arr[~mask_array] = nodata_val

    return {
        "pred_raster": pred_raster,
        "val_raster": val_raster,
        "diff": diff,
        "error_raster": error_raster,
        "mask_array": mask_array,
    }


@click.command()
@click.option("--pred-dir", type=click.Path(exists=True), required=True)
@click.option("--val-dir", type=click.Path(exists=True), required=True)
@click.option("--master-grid", type=click.Path(exists=True), required=True)
@click.option("--out-dir", type=click.Path(), default="diff_results")
@click.option("--exclusion-zones", type=click.Path(exists=True), default=None, help="Optional path to gpkg file containing exclusion zones.")
def cli(pred_dir: str, val_dir: str, master_grid: str, out_dir: str, exclusion_zones: str):
    """
    Validates point predictions against ground truth by differencing
    rasterized counts within a master grid framework.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if exclusion_zones:
        zones_gdf = gpd.read_file(exclusion_zones)
        exclusion_geom = zones_gdf.geometry.union_all()

    # List to store stats for each tile
    results_summary = []

    # validation files and dates
    val_paths = [
        os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".gpkg") or f.endswith(".geojson") or f.endswith(".json")
    ]
    val_map = {
        extract_date_from_path(p): p for p in val_paths if extract_date_from_path(p)
    }

    if not val_map:
        raise click.ClickException("No date-stamped validation files found.")

    with rasterio.open(master_grid) as src_grid:
        # src_grid = rasterio.open(master_grid)

        raster_crs = src_grid.crs

        pred_files = [
            f
            for f in os.listdir(pred_dir)
            if (f.endswith(".geojson") or (f.endswith(".gpkg")) or f.endswith(".json"))
        ]

        print(pred_files)

        for pred_file in pred_files:
            # pred_file = pred_files[0]

            # prediction filepath and date
            pred_path = os.path.join(pred_dir, pred_file)
            pred_date = extract_date_from_path(pred_path)

            if not pred_date:
                continue

            # temporal matching to validation data
            val_dates = list(val_map.keys())
            closest_date = min(val_dates, key=lambda d: abs(d - pred_date))

            # load and project prediction and validation data
            pred_gdf = gpd.read_file(pred_path).to_crs(raster_crs)
            val_gdf = gpd.read_file(val_map[closest_date]).to_crs(raster_crs)

            if pred_gdf.empty:
                continue

            best_td = float("inf")
            td_cutoff = None
            td_factor = None
            best_rms = float("inf")
            rms_cutoff = None
            rms_factor = None
            best_keep = None

            pred_gdf = pred_gdf.clip(exclusion_geom) if exclusion_zones else pred_gdf

            if pred_gdf.empty:
                continue

            try:
                grouped = prepare_grouped_cell_inputs(
                    pred_gdf=pred_gdf,
                    val_gdf=val_gdf,
                    src_grid=src_grid,
                )
            except Exception:
                click.echo(f"Skipping {pred_file}: No overlap.")
                continue

            pred_prepped = grouped["pred_prepped"]
            val_raster_base = grouped["val_raster"]
            mask_array = grouped["mask_array"]
            out_transform = grouped["out_transform"]
            grid_shape = grouped["grid_shape"]
            nodata_val = grouped["nodata_val"]

            parameter_map_rms = []
            parameter_map_td = []

            factor_scale = 0.01
            factor_steps = 1000
            factor_start_steps = 0
            factor_min = factor_scale * factor_start_steps
            factor_max = factor_scale * (factor_steps + factor_start_steps)


            cutoff_step = 0.0001
            cutoff_min = 5
            cutoff_max = 50

            for c in tqdm(range(cutoff_min, cutoff_max + 1)):
                row_rms = []
                row_td = []
                for f in range(factor_start_steps, factor_steps + 1 + factor_start_steps):
                    delta = pred_prepped["adjusted_peak"] - pred_prepped["peak_value"]
                    factor = f * factor_scale
                    rescaled_peak = pred_prepped["peak_value"] + factor * delta
                    cut_off = c *cutoff_step
                    keep = (rescaled_peak >= cut_off).to_numpy()

                    try:
                        processed = process_grouped_cells(
                            pred_rows=pred_prepped["row"].to_numpy(dtype=np.int32)[keep],
                            pred_cols=pred_prepped["col"].to_numpy(dtype=np.int32)[keep],
                            val_raster=val_raster_base.copy(),
                            mask_array=mask_array,
                            grid_shape=grid_shape,
                            nodata_val=nodata_val,
                        )
                        diff_in_mask = processed["diff"][mask_array]
                        # rms = np.sqrt(np.mean(np.square(diff_in_mask))) if diff_in_mask.size > 0 else np.inf
                        rms = np.mean(np.abs(diff_in_mask)) if diff_in_mask.size > 0 else np.inf
                        if rms < best_rms:
                            best_rms = rms
                            rms_cutoff = cut_off
                            rms_factor = factor
                            best_keep = keep
                        td = np.sum(diff_in_mask) if diff_in_mask.size > 0 else np.inf
                        if abs(td) < abs(best_td):
                            best_td = td
                            td_cutoff = cut_off
                            td_factor = factor
                        
                        row_rms.append(rms)
                        row_td.append(td)
                    except Exception:
                        continue
                parameter_map_rms.append(row_rms)
                parameter_map_td.append(row_td)
            click.echo(f"{pred_file.split('20260220')[0].strip('_')} | Best RMS: {best_rms:.3f} at cutoff {rms_cutoff:.4f} & factor {rms_factor:.2f} | Best TD: {best_td:.2f} at cutoff {td_cutoff:.4f} & factor {td_factor:.2f}")
            fig = plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            im = plt.imshow(parameter_map_rms, aspect='auto', origin='lower', extent=[factor_min, factor_max, cutoff_min * cutoff_step, cutoff_max * cutoff_step])
            plt.scatter(rms_factor, rms_cutoff, color='white', label='Best RMS', edgecolor='black', marker="*", s=100)
            plt.colorbar(im, label='RMS')
            plt.xlabel('Factor')
            plt.ylabel('Cutoff')
            plt.title(f'RMS across parameters for {pred_file.split("20260220")[0].strip("_")}')
            plt.subplot(1, 2, 2)
            im = plt.imshow(parameter_map_td, aspect='auto', origin='lower', extent=[factor_min, factor_max, cutoff_min * cutoff_step, cutoff_max * cutoff_step], cmap="seismic", vmin=-max(map(abs, np.array(parameter_map_td).flatten())), vmax=max(map(abs, np.array(parameter_map_td).flatten())))
            plt.scatter(td_factor, td_cutoff, color='white', label='Best TD', edgecolor='black', marker="*", s=100)
            plt.colorbar(im, label='Total Difference')
            plt.xlabel('Factor')
            plt.ylabel('Cutoff')
            plt.title(f'Total Difference across parameters for {pred_file.split("20260220")[0].strip("_")}')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{pred_file.split('20260220')[0].strip('_')}_parameter_map.png")


            processed = process_grouped_cells(
                pred_rows=pred_prepped["row"].to_numpy(dtype=np.int32)[best_keep],
                pred_cols=pred_prepped["col"].to_numpy(dtype=np.int32)[best_keep],
                val_raster=val_raster_base.copy(),
                mask_array=mask_array,
                grid_shape=grid_shape,
                nodata_val=nodata_val,
            )

            p_valid = processed["pred_raster"]
            v_valid = processed["val_raster"]
            d_valid = processed["diff"]
            diff = processed["diff"] * mask_array.astype(np.int32)
            e_valid = processed["error_raster"]
            out_transform = grouped["out_transform"]
            grid_shape = grouped["grid_shape"]
            nodata_val = grouped["nodata_val"]


            # pre-calculate positive cell indices
            valpos_idx = (v_valid > 0) & mask_array
            predpos_idx = (p_valid > 0) & mask_array

            # diff for positive cells
            diff_valpos = d_valid[valpos_idx]
            diff_predpos = d_valid[predpos_idx]

            # proportional difference
            pdiff = np.zeros_like(d_valid)
            if diff_valpos.size > 0:
                pdiff[valpos_idx] = diff_valpos / v_valid[valpos_idx]

            # extract specific pdiff segment for stats
            pdiff_valpos = pdiff[valpos_idx]
            stats = {
                "file": pred_file,
                "pred_date": pred_date.strftime("%Y-%m-%d"),
                "val_date": closest_date.strftime("%Y-%m-%d"),
                "total_pred": np.sum(p_valid[predpos_idx]),
                "total_val": np.sum(v_valid[valpos_idx]),
                "total_diff": np.sum(v_valid[valpos_idx]) - np.sum(p_valid[predpos_idx]),
                "total_pdiff": (np.sum(v_valid[valpos_idx]) - np.sum(p_valid[predpos_idx])) / np.sum(v_valid[valpos_idx])
                if np.sum(v_valid[valpos_idx]) > 0
                else 0,
                "rmsle": np.sqrt(np.mean(np.square(e_valid))),
                # "exp(rmsle)": np.exp(np.sqrt(np.mean(np.square(e_valid)))),
                # "bias": np.mean(d_valid),
                # "impr": np.std(d_valid),
                # "inac": np.abs(np.mean(d_valid)),
                # # Validation-positive stats (Omission/Sensitivity focus)
                # "bias_valpos": np.mean(diff_valpos) if diff_valpos.size > 0 else 0,
                # "impr_valpos": np.std(diff_valpos) if diff_valpos.size > 0 else 0,
                # "inac_valpos": np.abs(np.mean(diff_valpos))
                # if diff_valpos.size > 0
                # else 0,
                # # Percent Validation-positive stats
                # "pbias_valpos": np.mean(pdiff_valpos) if pdiff_valpos.size > 0 else 0,
                # "pimpr_valpos": np.std(pdiff_valpos) if pdiff_valpos.size > 0 else 0,
                # "pinac_valpos": np.abs(np.mean(pdiff_valpos))
                # if pdiff_valpos.size > 0
                # else 0,
                # # Prediction-positive stats (Commission/Precision focus)
                # "bias_predpos": np.mean(diff_predpos) if diff_predpos.size > 0 else 0,
                # "impr_predpos": np.std(diff_predpos) if diff_predpos.size > 0 else 0,
                # "inac_predpos": np.abs(np.mean(diff_predpos))
                # if diff_predpos.size > 0
                # else 0,
            }
            results_summary.append(stats)

            # --- save rasters ---#

            # save three rasters (predicted count, validation count, and difference)
            meta = src_grid.meta.copy()
            meta.update(
                {
                    "height": grid_shape[0],
                    "width": grid_shape[1],
                    "transform": out_transform,
                    "dtype": "float32",
                    "nodata": nodata_val,
                    "count": 1,
                }
            )

            base_name = os.path.splitext(pred_file)[0]
            outputs = {
                "diff": diff,
                # "error": error_raster,
                "pred_count": pred_raster,
                "val_count": val_raster,
            }

            for suffix, data in outputs.items():
                out_path = (
                    Path(out_dir) / f"{suffix}_100m" / f"{base_name}_{suffix}_100m.tif"
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)

                with rasterio.open(out_path, "w", **meta) as dest:
                    dest.write(data, 1)

            click.echo(f"Exported set for: {base_name}")

    # --- save summary ---
    if results_summary:
        import pandas as pd

        # convert summarty to dataframe
        df_results = pd.DataFrame(results_summary)

        # save to disk
        summary_path = os.path.join(out_dir, "validation_summary.csv")
        df_results.to_csv(summary_path, index=False)

        click.echo(f"\nSummary report saved to: {summary_path}")

        # print overal bias
        overall_error = df_results["rmsle"].mean()
        click.echo(f"Global Mean RMSLE: {overall_error:.4f}")
    else:
        click.echo("No results to summarize.")


if __name__ == "__main__":
    cli()
