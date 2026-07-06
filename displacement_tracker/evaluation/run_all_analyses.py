"""Run the full model-evaluation suite from a JSON config file.

Usage:
    poetry run run-evaluation --config displacement_tracker/evaluation/analysis_config.json

All relative paths in the config are resolved against the directory that
contains the config file, so the command can be run from anywhere.

If the config contains `prediction_dir`, `sample_tif` and `new_model_column`,
the new model's counts are first joined onto the annotation CSV (see
scripts/add_new_model_results.py) and evaluated. Otherwise the existing
`model_column` of the annotation CSV is evaluated directly.
"""

import json
from pathlib import Path

import click

from displacement_tracker.evaluation.scripts.add_new_model_results import (
    add_new_model_results,
)
from displacement_tracker.evaluation.scripts.evaluate_agriculture import (
    evaluate_agriculture_vs_non_agriculture,
)
from displacement_tracker.evaluation.scripts.evaluate_density import (
    evaluate_h3_density_bins,
)
from displacement_tracker.evaluation.scripts.evaluate_destruction import (
    evaluate_destruction_vs_non_destruction,
)
from displacement_tracker.evaluation.scripts.evaluate_month import (
    evaluate_error_by_month,
)
from displacement_tracker.evaluation.scripts.evaluate_municipal_bounds import (
    evaluate_municipal_bounds,
)
from displacement_tracker.evaluation.scripts.evaluate_spatial_points import (
    evaluate_spatial_points,
)
from displacement_tracker.evaluation.scripts.evaluate_tile_correlation import (
    evaluate_tile_correlation,
)
from displacement_tracker.evaluation.scripts.spatial_bootstrap_hex import (
    spatial_bootstrap_hex,
)
from displacement_tracker.evaluation.scripts.total_error import evaluate_total_error

PATH_KEYS = (
    "annotation_csv",
    "output_csv",
    "prediction_dir",
    "sample_tif",
    "output_dir",
    "boundary_shp",
    "agriculture_geojson",
    "h3_geojson",
    "destruction_geojson",
)

NEW_MODEL_KEYS = ("prediction_dir", "sample_tif", "new_model_column")


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Resolve relative paths against the config file's directory.
    base = config_path.parent
    for key in PATH_KEYS:
        if cfg.get(key):
            cfg[key] = str((base / cfg[key]).resolve())
    return cfg


def require(cfg: dict, key: str):
    if not cfg.get(key):
        raise click.ClickException(f"Missing required config key: {key}")
    return cfg[key]


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to JSON config file.",
)
def cli(config_path: Path) -> None:
    """Run all evaluation analyses from a config file."""
    cfg = load_config(config_path)

    annotation_csv = require(cfg, "annotation_csv")
    output_dir = require(cfg, "output_dir")
    boundary_shp = require(cfg, "boundary_shp")
    agriculture_geojson = require(cfg, "agriculture_geojson")
    h3_geojson = require(cfg, "h3_geojson")
    destruction_geojson = require(cfg, "destruction_geojson")

    manual_column = cfg.get("manual_column", "manual_tent_count")
    hex_size_m = float(cfg.get("hex_size_m", 1000.0))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if any(cfg.get(key) for key in NEW_MODEL_KEYS):
        # Join the new model's per-date predictions onto the annotations
        # first, then evaluate the newly added column.
        for key in NEW_MODEL_KEYS:
            require(cfg, key)
        annotation_csv, model_column = add_new_model_results(
            annotation_csv=annotation_csv,
            output_csv=require(cfg, "output_csv"),
            prediction_dir=cfg["prediction_dir"],
            sample_tif=cfg["sample_tif"],
            new_model_column=cfg["new_model_column"],
        )
    else:
        model_column = cfg.get("model_column", "model_tent_count")

    common = {
        "annotation_csv": annotation_csv,
        "output_dir": output_dir,
        "manual_column": manual_column,
        "model_column": model_column,
    }

    click.echo("Running total error analysis...")
    evaluate_total_error(boundary_shp=boundary_shp, hex_size_m=hex_size_m, **common)

    click.echo("Running spatial points analysis...")
    evaluate_spatial_points(**common)

    click.echo("Running spatial bootstrap analysis...")
    spatial_bootstrap_hex(boundary_shp=boundary_shp, hex_size_m=hex_size_m, **common)

    click.echo("Running tile correlation analysis...")
    evaluate_tile_correlation(**common)

    click.echo("Running agriculture analysis...")
    evaluate_agriculture_vs_non_agriculture(
        agriculture_geojson=agriculture_geojson, **common
    )

    click.echo("Running building density analysis...")
    evaluate_h3_density_bins(h3_geojson=h3_geojson, **common)

    click.echo("Running destruction analysis...")
    evaluate_destruction_vs_non_destruction(
        destruction_geojson=destruction_geojson, **common
    )

    click.echo("Running municipal bounds analysis...")
    evaluate_municipal_bounds(boundary_shp=boundary_shp, **common)

    click.echo("Running error by month analysis...")
    evaluate_error_by_month(**common)

    click.echo(f"Done. Results written to {output_dir}")


if __name__ == "__main__":
    cli()
