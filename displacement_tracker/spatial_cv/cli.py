"""Spatial cross-validation: north–south strips, east–west strips, or governorates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv

from displacement_tracker.spatial_cv.folds import (
    SCHEMES,
    assign_fold_ids,
    fold_summary,
    load_manifest_frame,
    write_folds,
)
from displacement_tracker.spatial_cv.plot import plot_folds
from displacement_tracker.util.config import load_flow_config

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen after north_south fold-3 pilot (2026-09-22). See README.md.
DEFAULT_SCHEME = "north_south"
DEFAULT_EPOCHS = 83
DEFAULT_PATIENCE = 30
DEFAULT_INNER_VAL_FRAC = 0.1


def _data_dir() -> Path:
    load_dotenv(REPO_ROOT / ".env")
    return Path(os.getenv("DATA_DIR", "/data/projects/gazatents"))


def _eval_dir(data_dir: Path) -> Path:
    return data_dir / "results" / "TentNetFA" / "evaluation" / "spatial_cv"


def _scheme_eval_dir(data_dir: Path, scheme: str) -> Path:
    return _eval_dir(data_dir) / scheme


def _run_root(artifact_dir: str | Path, scheme: str = "north_south") -> Path:
    path = Path(artifact_dir)
    if not path.is_absolute():
        path = REPO_ROOT / path
    root = path / "spatial_cv"
    if scheme == "north_south":
        return root
    return root / scheme


def _admin_boundaries(data_dir: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else REPO_ROOT / path
    return data_dir / "data" / "ocha" / "admin" / "gaza_governorate_boundaries.gpkg"


def _fold_dir(run_root: Path, fold_id: int) -> Path:
    return run_root / f"fold{fold_id}"


def _score_dir(run_root: Path, fold_id: int, min_adj_peak: float | None = None) -> Path:
    name = "score_points"
    if min_adj_peak is not None:
        name = f"score_points_{min_adj_peak:g}"
    return _fold_dir(run_root, fold_id) / name


def _grid_dir(run_root: Path, fold_id: int, min_adj_peak: float | None = None) -> Path:
    name = "score_grid"
    if min_adj_peak is not None:
        name = f"score_grid_{min_adj_peak:g}"
    return _fold_dir(run_root, fold_id) / name


def _core_metres(params: dict) -> float:
    processing = params.get("processing") or {}
    if "core_metres" not in processing:
        raise click.ClickException("Missing required config key: processing.core_metres")
    return float(processing["core_metres"])


def _echo_grid_score(summary: dict, out_dir: Path) -> None:
    concordance = summary["concordance"]
    concordance_text = "n/a" if concordance is None else f"{concordance:.3f}"
    mae = summary["mae"]
    mae_text = "n/a" if mae is None else f"{mae:.2f}"
    click.echo(
        f"Grid: cells={summary['n_cells']:,}  "
        f"pred={summary['n_pred']:,}  label={summary['n_label']:,}  "
        f"bias={summary['bias']:+,}  MAE={mae_text}  concordance={concordance_text}"
    )
    click.echo(f"Wrote {out_dir / 'holdout_grid_summary.json'}")
    click.echo(f"Wrote {out_dir / 'count_scatter.png'}")
    click.echo(f"Wrote {out_dir / 'residual_hist.png'}")
    click.echo(f"Wrote {out_dir / 'residual_map.png'}")


def _fold_summary_path(run_root: Path, fold_id: int) -> Path:
    return _fold_dir(run_root, fold_id) / "cv_fold_summary.json"


def _spatial_settings(params: dict, epochs, early_stop_patience, scheme):
    """CLI overrides, then config.yaml train.spatial_cv, then frozen defaults."""
    cfg = dict(params.get("spatial_cv") or {})
    if scheme is None:
        scheme = cfg.get("scheme", DEFAULT_SCHEME)
    if epochs is None:
        epochs = cfg.get("epochs", DEFAULT_EPOCHS)
    if early_stop_patience is None:
        early_stop_patience = cfg.get("early_stop_patience", DEFAULT_PATIENCE)
    inner_val_frac = float(cfg.get("inner_val_frac", DEFAULT_INNER_VAL_FRAC))
    return str(scheme), int(epochs), int(early_stop_patience), inner_val_frac


def _scheme_option():
    return click.option(
        "--scheme",
        type=click.Choice(SCHEMES),
        default="north_south",
        show_default=True,
        help="north_south latitude strips, east_west longitude strips, or admin governorates.",
    )


@click.group()
def cli():
    """Spatial CV for SimpleCNN: latitude strips, longitude strips, or governorates."""


@cli.command("assign-folds")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@_scheme_option()
@click.option("--data-dir", "data_dir_opt", default=None)
@click.option(
    "--boundaries",
    "boundaries_opt",
    default=None,
    help="Governorate polygons for --scheme admin.",
)
def assign_folds_cmd(config, scheme, data_dir_opt, boundaries_opt):
    """Write locked fold ids from the training manifest."""
    params = load_flow_config(config, "train")
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    manifest = Path(params["manifest"])
    if not manifest.exists():
        raise click.ClickException(f"Missing training manifest: {manifest}")
    frame = load_manifest_frame(manifest)
    boundaries = _admin_boundaries(data_dir, boundaries_opt) if scheme == "admin" else None
    folded, meta = assign_fold_ids(frame, scheme=scheme, boundaries=boundaries)
    out_dir = _scheme_eval_dir(data_dir, scheme)
    out_parquet = out_dir / "tiles_with_folds.parquet"
    write_folds(folded, meta, out_parquet)
    click.echo(f"Wrote {out_parquet}")
    click.echo(f"Wrote {out_parquet.with_name('folds_meta.json')}")
    click.echo(fold_summary(folded).to_string(index=False))


@cli.command("plot-strips")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@_scheme_option()
@click.option("--data-dir", "data_dir_opt", default=None)
def plot_strips_cmd(config, scheme, data_dir_opt):
    """Research-area map of the folds for one scheme."""
    params = load_flow_config(config, "train")
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    out_dir = _scheme_eval_dir(data_dir, scheme)
    folds_path = out_dir / "tiles_with_folds.parquet"
    meta_path = out_dir / "folds_meta.json"
    if not folds_path.exists():
        raise click.ClickException(f"Run assign-folds --scheme {scheme} first; missing {folds_path}")
    folds = pd.read_parquet(folds_path)
    meta = json.loads(meta_path.read_text())
    bounds = Path(params.get("boundaries") or "gaza_boundaries/GazaStrip_MunicipalBoundaries.shp")
    if not bounds.is_absolute():
        bounds = REPO_ROOT / bounds
    geotiff_dir = Path(params["geotiff_dir"]) if params.get("geotiff_dir") else None
    map_name = "folds_map.png" if scheme == "admin" else "strips_map.png"
    plot_folds(folds, meta, out_dir / map_name, bounds, geotiff_dir)


@cli.command("train-folds")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@click.option("--folds", default="1,2,3,4,5", help="Comma-separated holdout fold ids.")
@click.option("--epochs", type=int, default=None, help="Default: train.spatial_cv.epochs (83).")
@click.option(
    "--early-stop-patience",
    type=int,
    default=None,
    help="Default: train.spatial_cv.early_stop_patience (30).",
)
@click.option(
    "--scheme",
    type=click.Choice(SCHEMES),
    default=None,
    help="Default: train.spatial_cv.scheme (north_south).",
)
@click.option("--data-dir", "data_dir_opt", default=None)
@click.option(
    "--force",
    is_flag=True,
    help="Retrain even if cv_fold_summary.json already exists for a fold.",
)
def train_folds_cmd(config, folds, epochs, early_stop_patience, scheme, data_dir_opt, force):
    """Train from scratch for each holdout strip; write cv_summary.csv.

    Resume is the default: a fold with cv_fold_summary.json is skipped.
    Incomplete dirs (e.g. a killed run) are trained from scratch.
    """
    import gc

    import torch

    from displacement_tracker.d_train_cnn import train
    from displacement_tracker.util.config import load_flow_config as _load

    params = _load(config, "train")
    scheme, epochs, early_stop_patience, inner_val_frac = _spatial_settings(
        params, epochs, early_stop_patience, scheme
    )
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    folds_path = _scheme_eval_dir(data_dir, scheme) / "tiles_with_folds.parquet"
    if not folds_path.exists():
        raise click.ClickException(f"Run assign-folds --scheme {scheme} first; missing {folds_path}")
    artifact_dir = params.get("artifact_dir", "runs")
    run_root = _run_root(artifact_dir, scheme)
    training = dict(params["training"])
    for key in ("holdout_fold", "spatial_folds_path", "early_stop_patience", "inner_val_frac", "checkpoint"):
        training.pop(key, None)
    training["epochs"] = int(epochs)
    training["memory"] = True
    fold_ids = [int(x.strip()) for x in folds.split(",") if x.strip()]
    rows = []
    for fold_id in fold_ids:
        done = _fold_summary_path(run_root, fold_id)
        if done.exists() and not force:
            click.echo(f"=== resume: skip fold {fold_id} ({done}) ===")
            rows.append(json.loads(done.read_text()))
            continue
        fold_dir = _fold_dir(run_root, fold_id)
        if fold_dir.exists() and not done.exists():
            click.echo(
                f"=== fold {fold_id} incomplete (no cv_fold_summary.json); "
                "training from scratch ==="
            )
        else:
            click.echo(f"=== spatial CV holdout fold {fold_id} ({scheme}) ===")
        summary = train(
            params["manifest"],
            artifact_dir=artifact_dir,
            holdout_fold=fold_id,
            spatial_folds_path=str(folds_path),
            early_stop_patience=int(early_stop_patience),
            inner_val_frac=inner_val_frac,
            **training,
        )
        rows.append(summary)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out_csv = _write_cv_summary(data_dir, run_root, rows, scheme)
    click.echo(f"Wrote {out_csv}")


@cli.command("run")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@click.option("--folds", default="1,2,3,4,5", help="Comma-separated holdout fold ids.")
@click.option("--epochs", type=int, default=None)
@click.option("--early-stop-patience", type=int, default=None)
@click.option("--scheme", type=click.Choice(SCHEMES), default=None)
@click.option("--data-dir", "data_dir_opt", default=None)
@click.option("--force", is_flag=True)
@click.option(
    "--boundaries",
    "boundaries_opt",
    default=None,
    help="Governorate polygons for --scheme admin.",
)
@click.pass_context
def run_cmd(
    ctx, config, folds, epochs, early_stop_patience, scheme, data_dir_opt, force, boundaries_opt
):
    """Assign folds if needed, map them, then train (resume-safe).

    Defaults come from train.spatial_cv in config.yaml.
    """
    params = load_flow_config(config, "train")
    scheme, _, _, _ = _spatial_settings(params, epochs, early_stop_patience, scheme)
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    folds_path = _scheme_eval_dir(data_dir, scheme) / "tiles_with_folds.parquet"
    if not folds_path.exists():
        ctx.invoke(
            assign_folds_cmd,
            config=config,
            scheme=scheme,
            data_dir_opt=data_dir_opt,
            boundaries_opt=boundaries_opt,
        )
    ctx.invoke(plot_strips_cmd, config=config, scheme=scheme, data_dir_opt=data_dir_opt)
    ctx.invoke(
        train_folds_cmd,
        config=config,
        folds=folds,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        scheme=scheme,
        data_dir_opt=data_dir_opt,
        force=force,
    )


@cli.command("score-points")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@click.option("--folds", default="1,2,3,4,5", help="Comma-separated holdout fold ids.")
@click.option("--max-dist", default=5.0, show_default=True, help="Match gate in metres.")
@click.option(
    "--min-adj-peak",
    type=float,
    default=None,
    help="Override predict.merge.min_adj_peak. Writes fold{K}/score_points_<value>/ and leaves the default score in place.",
)
@click.option(
    "--n-tiles",
    type=int,
    default=16,
    show_default=True,
    help="Example tiles in the match figure. 9, 16, and 25 fill a square. 0 skips it.",
)
@click.option("--scheme", type=click.Choice(SCHEMES), default=None)
@click.option("--data-dir", "data_dir_opt", default=None)
def score_points_cmd(config, folds, max_dist, min_adj_peak, n_tiles, scheme, data_dir_opt):
    """Predict a finished fold, score the point match, then draw the figures.

    Loads best_model.pth, predicts the left-out strip, matches those points to
    the snapshot tents, and writes the score, the offset figure, the example
    grid, four pure TP, TN, FP, and FN grids, and a zoomed view of pairs that
    sit on the match gate. Then bins those points into the 100 m grid score.
    """
    import gc

    import numpy as np
    import torch

    from displacement_tracker.spatial_cv.points import (
        collect_snapshot_points,
        detection_settings,
        holdout_row_indices,
        load_fold_model,
        merged_tile_detections,
        open_labelled_dataset,
        predict_tile_detections,
        resolve_device,
        score_points,
        write_label_gpkg,
        write_pred_gpkg,
    )

    train_params = load_flow_config(config, "train")
    predict_params = load_flow_config(config, "predict")
    scheme, _, _, _ = _spatial_settings(train_params, None, None, scheme)
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    if n_tiles < 0:
        raise click.ClickException("--n-tiles cannot be negative")
    if max_dist <= 0:
        raise click.ClickException("--max-dist must be positive")
    if min_adj_peak is not None and min_adj_peak < 0:
        raise click.ClickException("--min-adj-peak cannot be negative")
    folds_path = _scheme_eval_dir(data_dir, scheme) / "tiles_with_folds.parquet"
    if not folds_path.exists():
        raise click.ClickException(f"Run assign-folds --scheme {scheme} first; missing {folds_path}")
    settings = detection_settings(predict_params)
    if min_adj_peak is not None:
        settings["min_adj_peak"] = float(min_adj_peak)
        click.echo(f"min_adj_peak override: {settings['min_adj_peak']:g}")
    core_metres = _core_metres(predict_params)
    device = resolve_device((train_params.get("training") or {}).get("device"))
    run_root = _run_root(train_params.get("artifact_dir", "runs"), scheme)
    fold_ids = [int(x.strip()) for x in folds.split(",") if x.strip()]
    jobs = []
    missing = []
    for fold_id in fold_ids:
        checkpoint = _fold_dir(run_root, fold_id) / "best_model.pth"
        if not checkpoint.exists():
            click.echo(f"error: fold {fold_id} has no best_model.pth ({checkpoint})", err=True)
            missing.append(fold_id)
        else:
            jobs.append((fold_id, checkpoint))

    summary_csv = _scheme_eval_dir(data_dir, scheme) / "cv_point_summary.csv"
    if jobs:
        dataset = open_labelled_dataset(train_params["manifest"])
        try:
            for fold_id, checkpoint in jobs:
                click.echo(f"=== score points, holdout fold {fold_id} ({scheme}) ===")
                indices, _meta = holdout_row_indices(len(dataset._rows), folds_path, fold_id)
                model = load_fold_model(checkpoint, device)
                tile_detections = predict_tile_detections(
                    dataset, indices, model, device, settings
                )
                pred_points = merged_tile_detections(tile_detections, settings)
                label_points = collect_snapshot_points(
                    dataset, indices, settings["crop_pixels"]
                )
                metrics = score_points(pred_points, label_points, max_dist)
                row = {
                    "holdout_fold": fold_id,
                    "scheme": scheme,
                    "n_holdout": len(indices),
                    "checkpoint": str(checkpoint),
                    "min_adj_peak": settings["min_adj_peak"],
                    "adjustment_factor": settings["adjustment_factor"],
                    **metrics,
                }
                score_dir = _score_dir(run_root, fold_id, min_adj_peak)
                write_pred_gpkg(score_dir / "holdout_pred_points.gpkg", pred_points)
                write_label_gpkg(score_dir / "holdout_label_points.gpkg", label_points)
                from displacement_tracker.spatial_cv.offset_hist import (
                    write_offset_figure_from_points,
                )

                summary_path = score_dir / "holdout_point_summary.json"
                offset_png = write_offset_figure_from_points(
                    score_dir,
                    pred_points,
                    label_points,
                    row,
                    max_dist,
                    summary_csv=None if min_adj_peak is not None else summary_csv,
                )
                click.echo(
                    f"Fold {fold_id}: pred={metrics['n_pred']:,}  "
                    f"label={metrics['n_label']:,}  "
                    f"TP={metrics['tp']:,}  FP={metrics['fp']:,}  FN={metrics['fn']:,}  "
                    f"P={metrics['precision']:.2%}  R={metrics['recall']:.2%}  "
                    f"F1={metrics['f1']:.2%}  "
                    f"mean offset={metrics['mean_pos_error_m']:.2f} m"
                )
                click.echo(f"Wrote {summary_path}")
                click.echo(f"Wrote {offset_png}")
                from displacement_tracker.spatial_cv.gate_view import write_gate_figure

                gate_png = score_dir / f"gate_{max_dist:g}m.png"
                write_gate_figure(
                    dataset,
                    indices,
                    np.array([p[0] for p in pred_points], dtype=float),
                    np.array([p[1] for p in pred_points], dtype=float),
                    np.array([p[0] for p in label_points], dtype=float),
                    np.array([p[1] for p in label_points], dtype=float),
                    gate_png,
                    fold_id,
                    gate_m=max_dist,
                )
                click.echo(f"Wrote {gate_png}")
                if n_tiles:
                    from displacement_tracker.spatial_cv.inspect import (
                        build_match_panels,
                        save_match_figure,
                        select_moderate_tiles,
                    )

                    chosen = select_moderate_tiles(
                        dataset, indices, settings["crop_pixels"], n=n_tiles
                    )
                    click.echo(
                        "Example tiles (manifest row, core tents): "
                        + ", ".join(f"{idx}:{count}" for idx, count in chosen)
                    )
                    panels = build_match_panels(
                        dataset, chosen, model, device, settings, max_dist
                    )
                    match_png = score_dir / "inspect_matches.png"
                    save_match_figure(panels, match_png, fold_id, max_dist)
                    click.echo(f"Wrote {match_png}")
                from displacement_tracker.spatial_cv.inspect import (
                    write_class_match_figures,
                )

                write_class_match_figures(
                    dataset,
                    tile_detections,
                    model,
                    device,
                    settings,
                    max_dist,
                    score_dir,
                    fold_id,
                )
                from displacement_tracker.spatial_cv.grid_score import write_fold_grid_score

                grid_dir = _grid_dir(run_root, fold_id, min_adj_peak)
                grid_summary = write_fold_grid_score(
                    score_dir,
                    grid_dir,
                    folds_path,
                    fold_id,
                    core_metres,
                    scheme,
                )
                _echo_grid_score(grid_summary, grid_dir)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            dataset.close()
        if min_adj_peak is None:
            click.echo(f"Wrote {summary_csv}")
        else:
            click.echo(
                "Left cv_point_summary.csv unchanged; "
                f"this run is under score_points_{min_adj_peak:g}/"
            )
    if missing:
        listed = ", ".join(str(fold_id) for fold_id in missing)
        raise click.ClickException(f"No best_model.pth for fold(s): {listed}")


@cli.command("score-grid")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@click.option("--folds", default="1,2,3,4,5", help="Comma-separated holdout fold ids.")
@click.option(
    "--min-adj-peak",
    type=float,
    default=None,
    help="Read fold{K}/score_points_<value>/ and write score_grid_<value>/. Default reads score_points/ and writes score_grid/.",
)
@click.option("--scheme", type=click.Choice(SCHEMES), default=None)
@click.option("--data-dir", "data_dir_opt", default=None)
def score_grid_cmd(config, folds, min_adj_peak, scheme, data_dir_opt):
    """Bin an existing point score into 100 m cells and write the count figures.

    Reads the holdout point gpkgs. Does not load the checkpoint. Empty cells
    inside the holdout cores are kept.
    """
    from displacement_tracker.spatial_cv.grid_score import write_fold_grid_score

    train_params = load_flow_config(config, "train")
    predict_params = load_flow_config(config, "predict")
    scheme, _, _, _ = _spatial_settings(train_params, None, None, scheme)
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    if min_adj_peak is not None and min_adj_peak < 0:
        raise click.ClickException("--min-adj-peak cannot be negative")
    folds_path = _scheme_eval_dir(data_dir, scheme) / "tiles_with_folds.parquet"
    if not folds_path.exists():
        raise click.ClickException(f"Run assign-folds --scheme {scheme} first; missing {folds_path}")
    core_metres = _core_metres(predict_params)
    run_root = _run_root(train_params.get("artifact_dir", "runs"), scheme)
    fold_ids = [int(x.strip()) for x in folds.split(",") if x.strip()]
    if not fold_ids:
        raise click.ClickException("Pass at least one fold id in --folds")
    for fold_id in fold_ids:
        click.echo(f"=== score grid, holdout fold {fold_id} ({scheme}) ===")
        score_dir = _score_dir(run_root, fold_id, min_adj_peak)
        grid_dir = _grid_dir(run_root, fold_id, min_adj_peak)
        summary = write_fold_grid_score(
            score_dir,
            grid_dir,
            folds_path,
            fold_id,
            core_metres,
            scheme,
        )
        _echo_grid_score(summary, grid_dir)


@cli.command("inspect-tiles")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@click.option("--fold", "fold_id", type=int, required=True, help="Holdout fold to draw.")
@click.option("--n-tiles", type=int, default=4, show_default=True)
@click.option("--scheme", type=click.Choice(SCHEMES), default=None)
@click.option("--data-dir", "data_dir_opt", default=None)
def inspect_tiles_cmd(config, fold_id, n_tiles, scheme, data_dir_opt):
    """Draw a few moderate-density holdout tiles through the label pipeline."""
    from displacement_tracker.spatial_cv.inspect import (
        build_panels,
        save_inspect_figure,
        select_moderate_tiles,
    )
    from displacement_tracker.spatial_cv.points import (
        detection_settings,
        holdout_row_indices,
        load_fold_model,
        open_labelled_dataset,
        resolve_device,
    )

    if n_tiles < 1:
        raise click.ClickException("--n-tiles must be at least 1")
    train_params = load_flow_config(config, "train")
    predict_params = load_flow_config(config, "predict")
    scheme, _, _, _ = _spatial_settings(train_params, None, None, scheme)
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    folds_path = _scheme_eval_dir(data_dir, scheme) / "tiles_with_folds.parquet"
    if not folds_path.exists():
        raise click.ClickException(f"Run assign-folds --scheme {scheme} first; missing {folds_path}")
    run_root = _run_root(train_params.get("artifact_dir", "runs"), scheme)
    checkpoint = _fold_dir(run_root, fold_id) / "best_model.pth"
    if not checkpoint.exists():
        raise click.ClickException(f"No best_model.pth for fold {fold_id}: {checkpoint}")
    settings = detection_settings(predict_params)
    device = resolve_device((train_params.get("training") or {}).get("device"))
    dataset = open_labelled_dataset(train_params["manifest"])
    try:
        indices, _meta = holdout_row_indices(len(dataset._rows), folds_path, fold_id)
        chosen = select_moderate_tiles(
            dataset, indices, settings["crop_pixels"], n=n_tiles
        )
        click.echo(
            "Tiles (manifest row, core tents): "
            + ", ".join(f"{idx}:{count}" for idx, count in chosen)
        )
        model = load_fold_model(checkpoint, device)
        panels = build_panels(dataset, chosen, model, device, settings)
    finally:
        dataset.close()
    out_path = _fold_dir(run_root, fold_id) / "inspect_tiles.png"
    save_inspect_figure(panels, out_path, fold_id)
    click.echo(f"Wrote {out_path}")


@cli.command("inspect-matches")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@click.option("--fold", "fold_id", type=int, required=True, help="Holdout fold to draw.")
@click.option(
    "--n-tiles",
    type=int,
    default=16,
    show_default=True,
    help="How many tiles. 9, 16, and 25 fill a square grid.",
)
@click.option("--max-dist", default=5.0, show_default=True, help="Match gate in metres.")
@click.option(
    "--min-adj-peak",
    type=float,
    default=None,
    help="Override predict.merge.min_adj_peak. Writes fold{K}/score_points_<value>/.",
)
@click.option("--scheme", type=click.Choice(SCHEMES), default=None)
@click.option("--data-dir", "data_dir_opt", default=None)
def inspect_matches_cmd(config, fold_id, n_tiles, max_dist, min_adj_peak, scheme, data_dir_opt):
    """Draw holdout tiles with the point match: pairs, misses, and extras."""
    from displacement_tracker.spatial_cv.inspect import (
        build_match_panels,
        save_match_figure,
        select_moderate_tiles,
        write_class_match_figures,
    )
    from displacement_tracker.spatial_cv.points import (
        detection_settings,
        holdout_row_indices,
        load_fold_model,
        open_labelled_dataset,
        predict_tile_detections,
        resolve_device,
    )

    if n_tiles < 1:
        raise click.ClickException("--n-tiles must be at least 1")
    if max_dist <= 0:
        raise click.ClickException("--max-dist must be positive")
    if min_adj_peak is not None and min_adj_peak < 0:
        raise click.ClickException("--min-adj-peak cannot be negative")
    train_params = load_flow_config(config, "train")
    predict_params = load_flow_config(config, "predict")
    scheme, _, _, _ = _spatial_settings(train_params, None, None, scheme)
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    folds_path = _scheme_eval_dir(data_dir, scheme) / "tiles_with_folds.parquet"
    if not folds_path.exists():
        raise click.ClickException(f"Run assign-folds --scheme {scheme} first; missing {folds_path}")
    run_root = _run_root(train_params.get("artifact_dir", "runs"), scheme)
    checkpoint = _fold_dir(run_root, fold_id) / "best_model.pth"
    if not checkpoint.exists():
        raise click.ClickException(f"No best_model.pth for fold {fold_id}: {checkpoint}")
    settings = detection_settings(predict_params)
    if min_adj_peak is not None:
        settings["min_adj_peak"] = float(min_adj_peak)
        click.echo(f"min_adj_peak override: {settings['min_adj_peak']:g}")
    device = resolve_device((train_params.get("training") or {}).get("device"))
    dataset = open_labelled_dataset(train_params["manifest"])
    try:
        indices, _meta = holdout_row_indices(len(dataset._rows), folds_path, fold_id)
        model = load_fold_model(checkpoint, device)
        tile_detections = predict_tile_detections(
            dataset, indices, model, device, settings
        )
        chosen = select_moderate_tiles(
            dataset, indices, settings["crop_pixels"], n=n_tiles
        )
        click.echo(
            "Tiles (manifest row, core tents): "
            + ", ".join(f"{idx}:{count}" for idx, count in chosen)
        )
        panels = build_match_panels(
            dataset, chosen, model, device, settings, max_dist
        )
        score_dir = _score_dir(run_root, fold_id, min_adj_peak)
        out_path = score_dir / "inspect_matches.png"
        save_match_figure(panels, out_path, fold_id, max_dist)
        click.echo(f"Wrote {out_path}")
        write_class_match_figures(
            dataset,
            tile_detections,
            model,
            device,
            settings,
            max_dist,
            score_dir,
            fold_id,
        )
    finally:
        dataset.close()


@cli.command("inspect-gate")
@click.argument("config", type=click.Path(exists=True), default="config.yaml")
@click.option("--fold", "fold_id", type=int, required=True, help="Holdout fold to draw.")
@click.option("--max-dist", default=5.0, show_default=True, help="Match gate in metres.")
@click.option(
    "--band",
    default=0.5,
    show_default=True,
    help="How far either side of the gate to look, in metres.",
)
@click.option("--n-each", type=int, default=4, show_default=True, help="Crops on each side of the gate.")
@click.option("--crop-m", default=40.0, show_default=True, help="Window width in metres.")
@click.option(
    "--min-adj-peak",
    type=float,
    default=None,
    help="Read fold{K}/score_points_<value>/. Default reads score_points/.",
)
@click.option("--scheme", type=click.Choice(SCHEMES), default=None)
@click.option("--data-dir", "data_dir_opt", default=None)
def inspect_gate_cmd(
    config, fold_id, max_dist, band, n_each, crop_m, min_adj_peak, scheme, data_dir_opt
):
    """Draw pairs that sit on the match gate, from an existing point score."""
    import geopandas as gpd

    from displacement_tracker.spatial_cv.gate_view import write_gate_figure
    from displacement_tracker.spatial_cv.points import holdout_row_indices, open_labelled_dataset

    if max_dist <= 0:
        raise click.ClickException("--max-dist must be positive")
    if band <= 0:
        raise click.ClickException("--band must be positive")
    if n_each < 1:
        raise click.ClickException("--n-each must be at least 1")
    train_params = load_flow_config(config, "train")
    scheme, _, _, _ = _spatial_settings(train_params, None, None, scheme)
    data_dir = Path(data_dir_opt) if data_dir_opt else _data_dir()
    folds_path = _scheme_eval_dir(data_dir, scheme) / "tiles_with_folds.parquet"
    if not folds_path.exists():
        raise click.ClickException(f"Run assign-folds --scheme {scheme} first; missing {folds_path}")
    run_root = _run_root(train_params.get("artifact_dir", "runs"), scheme)
    score_dir = _score_dir(run_root, fold_id, min_adj_peak)
    pred_path = score_dir / "holdout_pred_points.gpkg"
    label_path = score_dir / "holdout_label_points.gpkg"
    if not pred_path.exists() or not label_path.exists():
        raise click.ClickException(f"Run score-points first; missing points in {score_dir}")
    pred = gpd.read_file(pred_path)
    label = gpd.read_file(label_path)
    dataset = open_labelled_dataset(train_params["manifest"])
    try:
        indices, _meta = holdout_row_indices(len(dataset._rows), folds_path, fold_id)
        out_path = score_dir / f"gate_{max_dist:g}m.png"
        write_gate_figure(
            dataset,
            indices,
            pred.geometry.y.to_numpy(),
            pred.geometry.x.to_numpy(),
            label.geometry.x.to_numpy(),
            label.geometry.y.to_numpy(),
            out_path,
            fold_id,
            gate_m=max_dist,
            band_m=band,
            n_each=n_each,
            crop_m=crop_m,
        )
    finally:
        dataset.close()
    click.echo(f"Wrote {out_path}")


def _write_cv_summary(
    data_dir: Path,
    run_root: Path,
    extra_rows: list[dict] | None = None,
    scheme: str = "north_south",
) -> Path:
    """Merge on-disk fold summaries (including prior pilots) into cv_summary.csv."""
    by_fold: dict[int, dict] = {}
    for path in sorted(run_root.glob("fold*/cv_fold_summary.json")):
        row = json.loads(path.read_text())
        if row.get("holdout_fold") is not None:
            by_fold[int(row["holdout_fold"])] = row
    for row in extra_rows or []:
        if row.get("holdout_fold") is not None:
            by_fold[int(row["holdout_fold"])] = row
    out_csv = _scheme_eval_dir(data_dir, scheme) / "cv_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([by_fold[k] for k in sorted(by_fold)]).to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    cli()
