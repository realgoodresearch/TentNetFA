"""Ridge-aware search for the best (factor, cutoff) per metric.

For a fixed factor, the best cutoff is approximately linear in factor; off that
ridge the metric degrades on either side. We exploit that structure instead of
running a dense grid:

  1. Probe the ridge at ``n_probes`` factors spread across the range, using
     bounded 1-D scalar minimization over cutoff at each (Brent's method).
  2. Fit ``cutoff = a*factor + b`` through the resulting ridge points.
  3. Walk along the fitted ridge with another bounded 1-D scalar minimization
     over factor.
  4. Refine locally in 2-D with Nelder-Mead seeded at the best point so far.

Every evaluation computes all metrics in one shot, so optimizing one metric also
updates the bests of the others (no redundant raster work).

For straight validation at a single fixed (factor, cutoff), see
g2_validate_geojson.py.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.optimize import minimize, minimize_scalar
from tqdm import tqdm

from displacement_tracker.util.validation_core import (
    METRIC_DIRECTIONS,
    compute_metrics,
    discover_pred_val_pairs,
    initial_best_value,
    is_better,
    keep_mask_from_params,
    prepare_grouped_cell_inputs,
    process_grouped_cells,
    write_output_rasters,
)

SCAN_METRICS_DEFAULT = ("rms", "mae", "abs_total_diff")
LARGE_PENALTY = 1e12
# Per-call upper estimate for scipy's bounded Brent. Real counts depend on xtol
# but typically sit in the 15-25 range; used only to size the progress bar.
BRENT_EVAL_BUDGET = 25


def _budget_per_metric(n_probes: int, refine_maxiter: int) -> int:
    """Estimated upper bound on evaluations spent optimizing one metric.

    n_probes 1-D Brent searches (Phase 1) + one along-ridge Brent (Phase 3)
    + Nelder-Mead refinement (Phase 4, capped at refine_maxiter).
    """
    return n_probes * BRENT_EVAL_BUDGET + BRENT_EVAL_BUDGET + max(refine_maxiter, 0)


def _make_evaluator(grouped, scan_metrics, bests, trace, progress=None):
    """Return an evaluate(factor, cutoff) -> metrics function.

    Side effects: every call appends to `trace`, updates `bests[m]` for any
    metric that improves, and ticks `progress()` if provided. Returns None on
    failure so callers can penalize.
    """
    pred_prepped = grouped["pred_prepped"]
    val_raster_base = grouped["val_raster"]
    mask_array = grouped["mask_array"]
    grid_shape = grouped["grid_shape"]
    nodata_val = grouped["nodata_val"]
    rows_arr = pred_prepped["row"].to_numpy(dtype=np.int32)
    cols_arr = pred_prepped["col"].to_numpy(dtype=np.int32)

    def evaluate(factor: float, cutoff: float) -> Optional[Dict[str, float]]:
        keep = keep_mask_from_params(pred_prepped, factor=float(factor), cutoff=float(cutoff))
        try:
            processed = process_grouped_cells(
                pred_rows=rows_arr[keep],
                pred_cols=cols_arr[keep],
                val_raster=val_raster_base.copy(),
                mask_array=mask_array,
                grid_shape=grid_shape,
            )
        except Exception:
            return None
        metrics = compute_metrics(
            processed["pred_raster"],
            processed["val_raster"],
            processed["error_raster"],
            processed["mask_array"],
        )
        trace.append(
            {"factor": float(factor), "cutoff": float(cutoff),
             **{m: metrics[m] for m in scan_metrics}}
        )
        for m in scan_metrics:
            v = metrics[m]
            if is_better(m, v, bests[m]["value"]):
                bests[m] = {
                    "value": float(v),
                    "factor": float(factor),
                    "cutoff": float(cutoff),
                    "keep": keep,
                }
        if progress is not None:
            progress()
        return metrics

    return evaluate


def _objective(evaluate: Callable, metric: str, factor: float, cutoff: float) -> float:
    """Sign-adjusted scalar objective (always minimized) with penalty on failure."""
    metrics = evaluate(factor, cutoff)
    if metrics is None:
        return LARGE_PENALTY
    v = metrics.get(metric, np.nan)
    if not np.isfinite(v):
        return LARGE_PENALTY
    return float(v) if METRIC_DIRECTIONS[metric] == "min" else float(-v)


def _optimize_metric(
    evaluate: Callable,
    metric: str,
    bests: Dict[str, dict],
    factor_bounds: Tuple[float, float],
    cutoff_bounds: Tuple[float, float],
    n_probes: int,
    xtol_factor: float,
    xtol_cutoff: float,
    refine_maxiter: int,
) -> Tuple[float, float]:
    """Ridge-aware search for one metric. Returns the fitted ridge (a, b).

    The "best" is tracked via side effects in `bests` — callers can read
    `bests[metric]` after this returns.
    """
    fb_lo, fb_hi = factor_bounds
    cb_lo, cb_hi = cutoff_bounds

    # --- Phase 1: probe the ridge at n_probes factors ---
    probe_factors = np.linspace(fb_lo, fb_hi, n_probes)
    ridge_pts: List[Tuple[float, float]] = []
    for f in probe_factors:
        res = minimize_scalar(
            lambda c, _f=float(f): _objective(evaluate, metric, _f, c),
            bounds=(cb_lo, cb_hi),
            method="bounded",
            options={"xatol": xtol_cutoff},
        )
        ridge_pts.append((float(res.x), float(f)))

    # --- Phase 2: fit cutoff = a*factor + b ---
    cs = np.array([p[0] for p in ridge_pts])
    fs = np.array([p[1] for p in ridge_pts])
    if fs.size >= 2 and np.std(fs) > 0:
        a, b = np.polyfit(fs, cs, 1)
    else:
        a, b = 0.0, float(cs.mean())

    # --- Phase 3: 1-D search along the fitted ridge ---
    def along_ridge(f: float) -> float:
        c = float(np.clip(a * f + b, cb_lo, cb_hi))
        return _objective(evaluate, metric, f, c)

    minimize_scalar(
        along_ridge,
        bounds=(fb_lo, fb_hi),
        method="bounded",
        options={"xatol": xtol_factor},
    )

    # --- Phase 4: 2-D Nelder-Mead refinement from the best point so far ---
    cur = bests[metric]
    if cur["factor"] is not None and refine_maxiter > 0:
        # Out-of-bounds is clipped inside the objective so NM can't escape the
        # admissible rectangle.
        def obj2d(x):
            f = float(np.clip(x[0], fb_lo, fb_hi))
            c = float(np.clip(x[1], cb_lo, cb_hi))
            return _objective(evaluate, metric, f, c)

        minimize(
            obj2d,
            x0=np.array([cur["factor"], cur["cutoff"]]),
            method="Nelder-Mead",
            options={
                "xatol": max(xtol_factor, xtol_cutoff),
                "fatol": xtol_cutoff,
                "maxiter": refine_maxiter,
            },
        )

    return float(a), float(b)


def scan_tile(
    grouped,
    factor_bounds: Tuple[float, float],
    cutoff_bounds: Tuple[float, float],
    scan_metrics,
    n_probes: int,
    xtol_factor: float,
    xtol_cutoff: float,
    refine_maxiter: int,
    progress=None,
):
    """Optimize every scan metric on a prepared tile.

    Returns (bests, trace, ridges) where:
      bests[m]   = {value, factor, cutoff, keep}
      trace      = list of {factor, cutoff, <metric>: value, ...}
      ridges[m]  = (a, b) of the fitted ridge cutoff = a*factor + b
    """
    bests = {
        m: {"value": initial_best_value(m), "factor": None, "cutoff": None, "keep": None}
        for m in scan_metrics
    }
    trace: List[Dict[str, float]] = []

    evaluate = _make_evaluator(grouped, scan_metrics, bests, trace, progress=progress)
    ridges: Dict[str, Tuple[float, float]] = {}
    for m in scan_metrics:
        a, b = _optimize_metric(
            evaluate=evaluate,
            metric=m,
            bests=bests,
            factor_bounds=factor_bounds,
            cutoff_bounds=cutoff_bounds,
            n_probes=n_probes,
            xtol_factor=xtol_factor,
            xtol_cutoff=xtol_cutoff,
            refine_maxiter=refine_maxiter,
        )
        ridges[m] = (a, b)
    return bests, trace, ridges


def plot_search_trace(
    trace,
    bests,
    ridges,
    factor_bounds,
    cutoff_bounds,
    out_path,
    title_prefix,
):
    """One subplot per metric: scatter of every visited point, fitted ridge, best."""
    if not trace:
        return

    factors = np.array([t["factor"] for t in trace])
    cutoffs = np.array([t["cutoff"] for t in trace])
    fb_lo, fb_hi = factor_bounds
    cb_lo, cb_hi = cutoff_bounds

    metrics = list(bests.keys())
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        values = np.array([t[metric] for t in trace], dtype=float)
        finite = np.isfinite(values)
        cmap = "viridis" if METRIC_DIRECTIONS[metric] == "max" else "viridis_r"
        kwargs = dict(cmap=cmap, s=14)
        if metric == "spearman":
            kwargs.update(vmin=-1, vmax=1)
        sc = ax.scatter(factors[finite], cutoffs[finite], c=values[finite], **kwargs)
        fig.colorbar(sc, ax=ax, label=metric)

        a, b = ridges.get(metric, (None, None))
        if a is not None:
            fs = np.linspace(fb_lo, fb_hi, 100)
            cs = np.clip(a * fs + b, cb_lo, cb_hi)
            ax.plot(fs, cs, color="black", lw=1.0, ls="--", label="fitted ridge")

        bv = bests[metric]
        if bv["factor"] is not None:
            ax.scatter(
                bv["factor"], bv["cutoff"],
                color="white", edgecolor="black", marker="*", s=140,
                label=f"best {metric}",
            )

        ax.set_xlim(fb_lo, fb_hi)
        ax.set_ylim(cb_lo, cb_hi)
        ax.set_xlabel("factor")
        ax.set_ylabel("cutoff")
        ax.set_title(f"{title_prefix} — {metric}")
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _parse_metrics(value: str):
    metrics = [m.strip() for m in value.split(",") if m.strip()]
    if not metrics:
        return list(SCAN_METRICS_DEFAULT)
    unknown = [m for m in metrics if m not in METRIC_DIRECTIONS]
    if unknown:
        raise click.BadParameter(
            f"Unknown metric(s): {unknown}. Choose from {sorted(METRIC_DIRECTIONS)}."
        )
    return metrics


@click.command()
@click.option("--pred-dir", type=click.Path(exists=True), required=True)
@click.option("--val-dir", type=click.Path(exists=True), required=True)
@click.option("--master-grid", type=click.Path(exists=True), required=True)
@click.option("--out-dir", type=click.Path(), default="scan_results")
@click.option(
    "--exclusion-zones",
    type=click.Path(exists=True),
    default=None,
    help="Optional gpkg file of exclusion zones; predictions are clipped to its union.",
)
@click.option("--factor-min", type=float, default=0.0, show_default=True)
@click.option("--factor-max", type=float, default=10.0, show_default=True)
@click.option("--cutoff-min", type=float, default=0.0001, show_default=True)
@click.option("--cutoff-max", type=float, default=0.01, show_default=True)
@click.option(
    "--ridge-probes", type=int, default=5, show_default=True,
    help="Number of cutoffs at which the ridge is probed in Phase 1.",
)
@click.option(
    "--xtol-factor", type=float, default=1e-3, show_default=True,
    help="Absolute tolerance for the 1-D factor search (Brent's xatol).",
)
@click.option(
    "--xtol-cutoff", type=float, default=1e-6, show_default=True,
    help="Absolute tolerance for the 1-D cutoff search (Brent's xatol).",
)
@click.option(
    "--refine-maxiter", type=int, default=60, show_default=True,
    help="Max Nelder-Mead iterations for the 2-D refinement (0 disables).",
)
@click.option(
    "--metrics",
    default=",".join(SCAN_METRICS_DEFAULT),
    show_default=True,
    help=(
        "Comma-separated metrics to optimize. Choices: "
        + ", ".join(sorted(METRIC_DIRECTIONS))
    ),
)
@click.option(
    "--export-best",
    type=str,
    default="rms",
    show_default=True,
    help="Metric whose best parameters drive raster export. Must be in --metrics.",
)
def cli(
    pred_dir, val_dir, master_grid, out_dir, exclusion_zones,
    factor_min, factor_max, cutoff_min, cutoff_max,
    ridge_probes, xtol_factor, xtol_cutoff, refine_maxiter,
    metrics, export_best,
):
    """Ridge-aware search for the best (factor, cutoff) per metric."""
    os.makedirs(out_dir, exist_ok=True)
    scan_metrics = _parse_metrics(metrics)
    if export_best not in scan_metrics:
        raise click.BadParameter(
            f"--export-best={export_best!r} must be among --metrics ({scan_metrics})."
        )
    if factor_min >= factor_max or cutoff_min >= cutoff_max:
        raise click.BadParameter("factor/cutoff min must be strictly less than max.")

    factor_bounds = (factor_min, factor_max)
    cutoff_bounds = (cutoff_min, cutoff_max)

    exclusion_geom = None
    if exclusion_zones:
        exclusion_geom = gpd.read_file(exclusion_zones).geometry.union_all()

    pairs = discover_pred_val_pairs(pred_dir, val_dir)
    results = []

    total_evals = (
        _budget_per_metric(ridge_probes, refine_maxiter)
        * len(scan_metrics)
        * len(pairs)
    )
    pbar = tqdm(total=total_evals, desc="evals", unit="eval")

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
                grouped = prepare_grouped_cell_inputs(pred_gdf, val_gdf, src_grid)
            except Exception:
                click.echo(f"Skipping {pred_file}: no overlap with master grid.")
                continue

            bests, trace, ridges = scan_tile(
                grouped=grouped,
                factor_bounds=factor_bounds,
                cutoff_bounds=cutoff_bounds,
                scan_metrics=scan_metrics,
                n_probes=ridge_probes,
                xtol_factor=xtol_factor,
                xtol_cutoff=xtol_cutoff,
                refine_maxiter=refine_maxiter,
                progress=lambda: pbar.update(1),
            )

            best_summary = " | ".join(
                f"{m}: {bests[m]['value']:.4f} @ f={bests[m]['factor']:.3f}, c={bests[m]['cutoff']:.4f}"
                for m in scan_metrics
                if bests[m]["factor"] is not None
            )
            click.echo(f"{base_name} ({len(trace)} evals) -> {best_summary}")

            plot_search_trace(
                trace=trace, bests=bests, ridges=ridges,
                factor_bounds=factor_bounds, cutoff_bounds=cutoff_bounds,
                out_path=Path(out_dir) / f"{base_name}_search_trace.png",
                title_prefix=base_name,
            )

            chosen = bests[export_best]
            final_metrics: Dict[str, float] = {}
            if chosen["keep"] is not None:
                pred_prepped = grouped["pred_prepped"]
                processed = process_grouped_cells(
                    pred_rows=pred_prepped["row"].to_numpy(dtype=np.int32)[chosen["keep"]],
                    pred_cols=pred_prepped["col"].to_numpy(dtype=np.int32)[chosen["keep"]],
                    val_raster=grouped["val_raster"].copy(),
                    mask_array=grouped["mask_array"],
                    grid_shape=grouped["grid_shape"],
                    nodata_val=grouped["nodata_val"],
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
                )
                final_metrics = compute_metrics(
                    processed["pred_raster"],
                    processed["val_raster"],
                    processed["error_raster"],
                    processed["mask_array"],
                )

            row = {
                "file": pred_file,
                "pred_date": pred_date.strftime("%Y-%m-%d"),
                "val_date": val_date.strftime("%Y-%m-%d"),
                "export_best_metric": export_best,
                "n_evals": len(trace),
            }
            for m in scan_metrics:
                row[f"best_{m}"] = bests[m]["value"]
                row[f"best_{m}_factor"] = bests[m]["factor"]
                row[f"best_{m}_cutoff"] = bests[m]["cutoff"]
                a, b = ridges.get(m, (None, None))
                row[f"ridge_{m}_a"] = a
                row[f"ridge_{m}_b"] = b
            row.update({f"final_{k}": v for k, v in final_metrics.items()})
            results.append(row)

    if not results:
        click.echo("No results to summarize.")
        return

    df = pd.DataFrame(results)
    summary_path = os.path.join(out_dir, "scan_summary.csv")
    df.to_csv(summary_path, index=False)
    click.echo(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    cli()
