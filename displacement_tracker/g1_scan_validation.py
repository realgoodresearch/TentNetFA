"""Scan stage of the tuning pipeline: ridge-aware search for the best
(adjustment_factor, min_adj_peak) merge hyperparameters per metric.

The unthresholded merged predictions (``merge.output`` of the tune flow) are
validated against an explicitly configured reference source (point
annotations, a UNOSAT export, or a counts raster on the master grid — see
``util/reference_data.py``) while sweeping the thresholding pair the merge
stage calls (``adjustment_factor``, ``min_adj_peak``) and the scan calls
(factor, cutoff).

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

The optimum of ``tuning.metric`` is written to ``tuning.best_params`` (YAML),
which the final stage (h2_merge_tuned) feeds back into the merge. For straight
validation at a single fixed (factor, cutoff), see g2_validate_geojson.py.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import click
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy.optimize import minimize, minimize_scalar
from tqdm import tqdm

from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.reference_data import build_reference_source
from displacement_tracker.util.validation_core import (
    METRIC_DIRECTIONS,
    compute_metrics,
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


def _resolve_metrics(tuning_cfg: dict) -> Tuple[str, List[str]]:
    """Return (eval metric, metrics to track); the eval metric is always tracked."""
    metric = tuning_cfg.get("metric", "rms")
    metrics = list(tuning_cfg.get("metrics") or SCAN_METRICS_DEFAULT)
    if metric not in metrics:
        metrics.insert(0, metric)
    unknown = [m for m in metrics if m not in METRIC_DIRECTIONS]
    if unknown:
        raise click.ClickException(
            f"Unknown metric(s): {unknown}. Choose from {sorted(METRIC_DIRECTIONS)}."
        )
    return metric, metrics


def _require(cfg: dict, section: str, key: str):
    value = cfg.get(key)
    if not value:
        raise click.ClickException(f"Missing required config key: {section}.{key}")
    return value


@dataclass(frozen=True)
class ScanSettings:
    """Validated inputs of one threshold scan, extracted from the config."""

    input_path: str
    master_grid: str
    reference: object  # config for build_reference_source
    out_dir: str
    best_params_path: str
    metric: str
    scan_metrics: List[str]
    factor_bounds: Tuple[float, float]
    cutoff_bounds: Tuple[float, float]
    ridge_probes: int
    xtol_factor: float
    xtol_cutoff: float
    refine_maxiter: int
    exclusion_zones: Optional[str]

    @property
    def base_name(self) -> str:
        return os.path.splitext(os.path.basename(self.input_path))[0]


def _scan_settings(params: dict) -> ScanSettings:
    """Extract and validate the scan settings from a resolved (flat) config."""
    tuning = params.get("tuning") or {}
    merge_cfg = params.get("merge") or {}

    input_path = tuning.get("input") or merge_cfg.get("output")
    if not input_path:
        raise click.ClickException(
            "Missing required config key: tuning.input "
            "(or merge.output as fallback)"
        )
    out_dir = tuning.get("out_dir") or "scan_results"
    metric, scan_metrics = _resolve_metrics(tuning)

    factor_bounds = (
        float(tuning.get("factor_min", 0.0)),
        float(tuning.get("factor_max", 10.0)),
    )
    cutoff_bounds = (
        float(tuning.get("cutoff_min", 0.0001)),
        float(tuning.get("cutoff_max", 0.01)),
    )
    if factor_bounds[0] >= factor_bounds[1] or cutoff_bounds[0] >= cutoff_bounds[1]:
        raise click.ClickException(
            "tuning.factor_min/cutoff_min must be strictly less than their max."
        )

    return ScanSettings(
        input_path=input_path,
        master_grid=_require(tuning, "tuning", "master_grid"),
        reference=_require(tuning, "tuning", "reference"),
        out_dir=out_dir,
        best_params_path=tuning.get("best_params")
        or os.path.join(out_dir, "best_params.yaml"),
        metric=metric,
        scan_metrics=scan_metrics,
        factor_bounds=factor_bounds,
        cutoff_bounds=cutoff_bounds,
        ridge_probes=int(tuning.get("ridge_probes", 5)),
        xtol_factor=float(tuning.get("xtol_factor", 1e-3)),
        xtol_cutoff=float(tuning.get("xtol_cutoff", 1e-6)),
        refine_maxiter=int(tuning.get("refine_maxiter", 60)),
        exclusion_zones=tuning.get("exclusion_zones"),
    )


def _load_predictions(settings: ScanSettings, raster_crs) -> gpd.GeoDataFrame:
    """Read the merged raw predictions, optionally clipped to the scan zones."""
    pred_gdf = gpd.read_file(settings.input_path).to_crs(raster_crs)
    if settings.exclusion_zones:
        exclusion_geom = gpd.read_file(settings.exclusion_zones).geometry.union_all()
        pred_gdf = pred_gdf.clip(exclusion_geom)
    if pred_gdf.empty:
        raise click.ClickException(
            f"No predictions left to scan in {settings.input_path}"
        )
    return pred_gdf


def _export_best(settings: ScanSettings, grouped, chosen, src_grid) -> Dict[str, float]:
    """Write the rasters at the chosen optimum; return its final metrics."""
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
        out_dir=settings.out_dir,
        base_name=settings.base_name,
        pred_raster=processed["pred_raster"],
        val_raster=processed["val_raster"],
        diff_masked=diff_masked,
        src_grid=src_grid,
        grid_shape=grouped["grid_shape"],
        out_transform=grouped["out_transform"],
    )
    return compute_metrics(
        processed["pred_raster"],
        processed["val_raster"],
        processed["error_raster"],
        processed["mask_array"],
    )


def _write_summary_csv(settings: ScanSettings, bests, ridges, trace, final_metrics):
    row = {
        "file": os.path.basename(settings.input_path),
        "metric": settings.metric,
        "n_evals": len(trace),
    }
    for m in settings.scan_metrics:
        row[f"best_{m}"] = bests[m]["value"]
        row[f"best_{m}_factor"] = bests[m]["factor"]
        row[f"best_{m}_cutoff"] = bests[m]["cutoff"]
        a, b = ridges.get(m, (None, None))
        row[f"ridge_{m}_a"] = a
        row[f"ridge_{m}_b"] = b
    row.update({f"final_{k}": v for k, v in final_metrics.items()})
    summary_path = os.path.join(settings.out_dir, "scan_summary.csv")
    pd.DataFrame([row]).to_csv(summary_path, index=False)
    click.echo(f"Summary saved to: {summary_path}")


def _write_best_params(settings: ScanSettings, bests, chosen) -> dict:
    # The merge stage names the pair (adjustment_factor, min_adj_peak);
    # h2_merge_tuned reads exactly these keys back.
    best_params = {
        "metric": settings.metric,
        "value": float(chosen["value"]),
        "adjustment_factor": float(chosen["factor"]),
        "min_adj_peak": float(chosen["cutoff"]),
        "input": str(settings.input_path),
        "bests": {
            m: {
                "value": bests[m]["value"],
                "adjustment_factor": bests[m]["factor"],
                "min_adj_peak": bests[m]["cutoff"],
            }
            for m in settings.scan_metrics
        },
    }
    path = settings.best_params_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(best_params, f, sort_keys=False)
    click.echo(
        f"Best parameters ({settings.metric}={chosen['value']:.4f}) saved to: {path}"
    )
    return best_params


def run_scan(params: dict) -> dict:
    """Run the threshold scan described by a resolved (flat) config.

    Reads the ``tuning`` section (input defaults to ``merge.output``, the
    unthresholded merge of the tune flow), writes the search trace plot,
    ``scan_summary.csv``, best-parameter rasters and ``best_params.yaml``
    into ``tuning.out_dir``, and returns the tuned parameters.
    """
    settings = _scan_settings(params)
    os.makedirs(settings.out_dir, exist_ok=True)
    reference = build_reference_source(settings.reference)

    total_evals = (
        _budget_per_metric(settings.ridge_probes, settings.refine_maxiter)
        * len(settings.scan_metrics)
    )
    pbar = tqdm(total=total_evals, desc="evals", unit="eval")

    with rasterio.open(settings.master_grid) as src_grid:
        pred_gdf = _load_predictions(settings, src_grid.crs)
        try:
            grouped = prepare_grouped_cell_inputs(pred_gdf, reference, src_grid)
        except Exception as exc:
            raise click.ClickException(
                f"Could not resolve {settings.input_path} onto the master "
                f"grid: {exc}"
            ) from exc

        bests, trace, ridges = scan_tile(
            grouped=grouped,
            factor_bounds=settings.factor_bounds,
            cutoff_bounds=settings.cutoff_bounds,
            scan_metrics=settings.scan_metrics,
            n_probes=settings.ridge_probes,
            xtol_factor=settings.xtol_factor,
            xtol_cutoff=settings.xtol_cutoff,
            refine_maxiter=settings.refine_maxiter,
            progress=lambda: pbar.update(1),
        )
        pbar.close()

        best_summary = " | ".join(
            f"{m}: {bests[m]['value']:.4f} @ f={bests[m]['factor']:.3f}, c={bests[m]['cutoff']:.4f}"
            for m in settings.scan_metrics
            if bests[m]["factor"] is not None
        )
        click.echo(f"{settings.base_name} ({len(trace)} evals) -> {best_summary}")

        plot_search_trace(
            trace=trace, bests=bests, ridges=ridges,
            factor_bounds=settings.factor_bounds,
            cutoff_bounds=settings.cutoff_bounds,
            out_path=Path(settings.out_dir) / f"{settings.base_name}_search_trace.png",
            title_prefix=settings.base_name,
        )

        chosen = bests[settings.metric]
        if chosen["factor"] is None:
            raise click.ClickException(
                f"The scan found no finite value for metric {settings.metric!r}; "
                "check the reference data and the factor/cutoff bounds."
            )
        final_metrics = _export_best(settings, grouped, chosen, src_grid)

    _write_summary_csv(settings, bests, ridges, trace, final_metrics)
    return _write_best_params(settings, bests, chosen)


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@flow_option(default="tune")
def cli(config: str, flow: str) -> None:
    """Scan for the best merge hyperparameters against reference data.

    Reads the ``tuning`` section of the YAML config; the input defaults to
    ``merge.output`` (the unthresholded merge of the tune flow).
    """
    params = load_flow_config(config, flow)
    run_scan(params)


if __name__ == "__main__":
    cli()
