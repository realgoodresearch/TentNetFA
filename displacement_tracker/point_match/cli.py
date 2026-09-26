"""Run 5 m greedy matching of v2.2 tent preds against UNOSAT points."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from displacement_tracker.point_match.match import match_points, nearest_distances

MATCH_CRS = "EPSG:32636"
REPO_ROOT = Path(__file__).resolve().parents[2]

PAIRS = (
    {
        "name": "20251201",
        "pred": (
            "results/TentNetFA/model_v2.2/20251201/preds_final/"
            "merged_tents_20251201_min_adj_peak_0.003_factor_10_excl_yellow_line.gpkg"
        ),
        "gt": "data/unosat/tent_mapping/UNOSAT_Gaza_IDP_Tent_points_20251212.gpkg",
    },
    {
        "name": "20260220",
        "pred": (
            "results/TentNetFA/model_v2.2/20260220/preds_final/"
            "merged_tents_20260220_min_adj_peak_0.003_factor_10_excl_yellow_line.gpkg"
        ),
        "gt": "data/unosat/tent_mapping/UNOSAT_Gaza_IDP_Tent_points_20260111.gpkg",
    },
    {
        "name": "20260325",
        "pred": (
            "results/TentNetFA/model_v2.2/20260325/preds_final/"
            "merged_tents_20260325_min_adj_peak_0.003_factor_10_excl_yellow_line.gpkg"
        ),
        "gt": "data/unosat/tent_mapping/UNOSAT_Gaza_IDP_Tent_points_20260327.gpkg",
    },
)


def _data_dir() -> Path:
    load_dotenv(REPO_ROOT / ".env")
    return Path(os.getenv("DATA_DIR", "/data/projects/gazatents"))


def _xy(gdf: gpd.GeoDataFrame) -> np.ndarray:
    projected = gdf.to_crs(MATCH_CRS)
    geom = projected.geometry
    projected = projected[~geom.is_empty]
    xy = np.column_stack(
        [projected.geometry.x.to_numpy(), projected.geometry.y.to_numpy()]
    )
    return xy[np.isfinite(xy).all(axis=1)]


def _drop_inside_zone(gdf: gpd.GeoDataFrame, zone_path: Path) -> gpd.GeoDataFrame:
    """Drop points inside the yellow-line polygon (same rule as merge)."""
    zone = gpd.read_file(zone_path)
    if zone.crs is None:
        zone = zone.set_crs("EPSG:4326")
    gdf_ll = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    geom = zone.to_crs("EPSG:4326").geometry.union_all()
    return gdf.loc[~gdf_ll.geometry.within(geom)].copy()


def _print_metrics(
    name: str,
    metrics: dict,
    n_pred: int,
    n_gt: int,
    n_gt_raw: int,
    nn: np.ndarray,
    gate: float,
) -> None:
    beyond = nn[np.isfinite(nn) & (nn > gate)]
    click.echo(f"--- {name} ---")
    click.echo(f"Pred points: {n_pred:,}")
    click.echo(f"UNOSAT points (after yellow-line clip): {n_gt:,}  (raw {n_gt_raw:,})")
    click.echo(f"Matched (TP ≤ {gate:.0f} m): {metrics['tp']:,}")
    click.echo(f"False alarms (FP): {metrics['fp']:,}")
    click.echo(f"Missed (FN): {metrics['fn']:,}")
    click.echo(f"Precision: {metrics['precision']:.2%}")
    click.echo(f"Recall:    {metrics['recall']:.2%}")
    click.echo(f"F1:        {metrics['f1']:.2%}")
    click.echo(f"Mean offset (1-to-1, ≤{gate:.0f} m): {metrics['mean_pos_error_m']:.2f} m")
    click.echo(f"RMSE offset (1-to-1, ≤{gate:.0f} m): {metrics['rmse_pos_error_m']:.2f} m")
    click.echo(
        f"NN to UNOSAT: median={np.median(nn):.2f} m  "
        f"mean={np.mean(nn):.2f} m  "
        f"pct≤{gate:.0f}m={(nn <= gate).mean():.1%}  "
        f"pct>{gate:.0f}m={(nn > gate).mean():.1%}"
    )
    if beyond.size:
        click.echo(
            f"NN beyond {gate:.0f} m: n={beyond.size:,}  "
            f"mean={beyond.mean():.2f} m  median={np.median(beyond):.2f} m"
        )
    click.echo("")


def _run_overlay_hist(dist_csv: Path, out_png: Path, gate: float) -> None:
    r_script = Path(__file__).with_name("plot_offset_hist.R")
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(REPO_ROOT / ".Rlibrary")
    rlib = REPO_ROOT / ".Rlibrary"
    result = subprocess.run(
        [
            "Rscript",
            str(r_script),
            str(dist_csv),
            str(out_png),
            str(gate),
            str(rlib),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(result.stdout)
        click.echo(result.stderr)
        raise click.ClickException(f"R histogram failed (exit {result.returncode})")
    click.echo(f"Wrote {out_png}")


@click.command()
@click.option("--data-dir", "data_dir_opt", default=None)
@click.option(
    "--out-dir",
    "out_dir_opt",
    default=None,
    help="Default: $DATA_DIR/results/TentNetFA/evaluation/v2.2_vs_unosat",
)
@click.option("--max-dist", default=5.0, show_default=True, help="Match gate in metres.")
@click.option(
    "--exclusion-zone",
    default=None,
    help="Drop UNOSAT points inside this polygon (default: yellow-line-zone.json).",
)
def cli(data_dir_opt, out_dir_opt, max_dist, exclusion_zone):
    """Greedy 5 m 1-to-1 match of v2.2 preds vs UNOSAT tent points."""
    base = Path(data_dir_opt) if data_dir_opt else _data_dir()
    out_dir = (
        Path(out_dir_opt)
        if out_dir_opt
        else base / "results" / "TentNetFA" / "evaluation" / "v2.2_vs_unosat"
    )
    zone_path = Path(exclusion_zone) if exclusion_zone else (
        base / "results" / "TentNetFA" / "tent_postprocessing" / "layers" / "yellow-line-zone.json"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    dist_frames = []
    for pair in PAIRS:
        pred_path = base / pair["pred"]
        gt_path = base / pair["gt"]
        if not pred_path.exists():
            raise click.ClickException(f"Missing pred: {pred_path}")
        if not gt_path.exists():
            raise click.ClickException(f"Missing UNOSAT: {gt_path}")

        pred_gdf = gpd.read_file(pred_path)
        gt_gdf = gpd.read_file(gt_path)
        n_gt_raw = len(gt_gdf)
        if zone_path.exists():
            gt_gdf = _drop_inside_zone(gt_gdf, zone_path)
        else:
            click.echo(f"Warning: exclusion zone not found ({zone_path}); UNOSAT unclipped")

        pred_xy = _xy(pred_gdf)
        gt_xy = _xy(gt_gdf)
        metrics = match_points(pred_xy, gt_xy, max_dist=max_dist)
        nn = nearest_distances(pred_xy, gt_xy)
        _print_metrics(pair["name"], metrics, len(pred_xy), len(gt_xy), n_gt_raw, nn, max_dist)
        dist_frames.append(pd.DataFrame({"date": pair["name"], "distance_m": nn}))
        beyond = nn[np.isfinite(nn) & (nn > max_dist)]
        rows.append(
            {
                "name": pair["name"],
                "n_pred": len(pred_xy),
                "n_unosat_raw": n_gt_raw,
                "n_unosat_clipped": len(gt_xy),
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mean_pos_error_m": metrics["mean_pos_error_m"],
                "rmse_pos_error_m": metrics["rmse_pos_error_m"],
                "max_dist_m": max_dist,
                "nn_median_m": float(np.median(nn)) if nn.size else np.nan,
                "nn_mean_m": float(np.mean(nn)) if nn.size else np.nan,
                "nn_pct_within_gate": float((nn <= max_dist).mean()) if nn.size else np.nan,
                "nn_pct_beyond_gate": float((nn > max_dist).mean()) if nn.size else np.nan,
                "nn_n_beyond": int(beyond.size),
                "nn_mean_beyond_m": float(beyond.mean()) if beyond.size else np.nan,
                "nn_median_beyond_m": float(np.median(beyond)) if beyond.size else np.nan,
            }
        )

    summary = out_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    click.echo(f"Wrote {summary}")

    dist_csv = out_dir / "offset_distances.csv"
    pd.concat(dist_frames, ignore_index=True).to_csv(dist_csv, index=False)
    click.echo(f"Wrote {dist_csv}")
    _run_overlay_hist(dist_csv, out_dir / "offset_hist_overlay.png", max_dist)


if __name__ == "__main__":
    cli()
