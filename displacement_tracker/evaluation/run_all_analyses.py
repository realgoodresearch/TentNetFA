"""Run the full model-evaluation suite from the shared pipeline config.

Usage:
    poetry run run-evaluation config.yaml

All settings come from the ``evaluation`` section of the config, except the
municipal boundaries, which are read from ``boundaries`` so the suite scores
predictions against the same boundary layer the scan stages tile against.

Paths are used as configured — relative ones resolve against the working
directory, as in every other stage — so a run directory's resolved
``config.yaml`` evaluates that run.

If the ``evaluation.new_model`` section sets anything, the new model's counts
are first joined onto the annotation CSV (see
scripts/add_new_model_results.py) and the added column is evaluated;
otherwise the existing ``model_column`` of the annotation CSV is evaluated
directly.
"""

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
from displacement_tracker.util.config import (
    deep_get,
    flow_option,
    forwarded,
    load_flow_config,
    require,
)

# Spatial context layers, keyed by the analysis that reads each one.
LAYER_NAMES = ("agriculture", "h3_density", "destruction")

# Column overrides forwarded to every analysis. Keys the config leaves unset
# are omitted so the analyses' own signature defaults apply.
COLUMN_KEYS = ("manual_column", "model_column")

# Any of these being set marks the run as a new-model run; prediction_dir is
# excluded because it falls back to prediction.output_folder, which a predict
# config always sets.
NEW_MODEL_TRIGGER_KEYS = ("column", "sample_tif", "output_csv")


def resolve_new_model(params: dict) -> dict | None:
    """The new-model join arguments, or None for a plain model_column run."""
    new_model_cfg = deep_get(params, "evaluation.new_model") or {}
    if not any(new_model_cfg.get(key) for key in NEW_MODEL_TRIGGER_KEYS):
        return None

    prediction_dir = new_model_cfg.get("prediction_dir") or deep_get(
        params, "prediction.output_folder"
    )
    if not prediction_dir:
        raise click.ClickException(
            "Missing required config key: evaluation.new_model.prediction_dir "
            "(or prediction.output_folder as fallback)"
        )

    return {
        "output_csv": require(params, "evaluation.new_model.output_csv"),
        "prediction_dir": prediction_dir,
        "sample_tif": require(params, "evaluation.new_model.sample_tif"),
        "new_model_column": require(params, "evaluation.new_model.column"),
    }


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@flow_option(default="predict")
def cli(config: str, flow: str) -> None:
    """Run all evaluation analyses from the YAML config.

    Reads the ``evaluation`` section; the municipal boundaries come from
    ``boundaries`` and the new-model prediction folder defaults to
    ``prediction.output_folder``.
    """
    params = load_flow_config(config, flow)
    eval_cfg = params.get("evaluation") or {}

    boundary_shp = require(params, "boundaries")
    annotation_csv = require(params, "evaluation.annotation_csv")
    output_dir = require(params, "evaluation.output_dir")
    layers = {
        name: require(params, f"evaluation.layers.{name}") for name in LAYER_NAMES
    }

    missing = [
        p
        for p in (annotation_csv, boundary_shp, *layers.values())
        if not Path(p).exists()
    ]
    if missing:
        raise click.ClickException(
            "Missing input file(s):\n  "
            + "\n  ".join(missing)
            + "\nThe spatial context layers are not committed to the "
            "repository; see the 'Evaluate Model Predictions' section of the "
            "README for how to obtain them."
        )

    new_model = resolve_new_model(params)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    common = {
        "annotation_csv": annotation_csv,
        "output_dir": output_dir,
        **forwarded(eval_cfg, *COLUMN_KEYS),
    }

    if new_model is not None:
        # Join the new model's per-date predictions onto the annotations
        # first, then evaluate the newly added column instead of whichever
        # model_column the config or the CSV already carries.
        common["annotation_csv"], common["model_column"] = add_new_model_results(
            annotation_csv=annotation_csv, **new_model
        )

    # hex_size_m, if the config sets it; omitted lets the two hex analyses
    # use their own signature default.
    hexes = forwarded(eval_cfg, "hex_size_m")

    click.echo("Running total error analysis...")
    evaluate_total_error(boundary_shp=boundary_shp, **hexes, **common)

    click.echo("Running spatial points analysis...")
    evaluate_spatial_points(**common)

    click.echo("Running spatial bootstrap analysis...")
    spatial_bootstrap_hex(boundary_shp=boundary_shp, **hexes, **common)

    click.echo("Running tile correlation analysis...")
    evaluate_tile_correlation(**common)

    click.echo("Running agriculture analysis...")
    evaluate_agriculture_vs_non_agriculture(
        agriculture_geojson=layers["agriculture"], **common
    )

    click.echo("Running building density analysis...")
    evaluate_h3_density_bins(h3_geojson=layers["h3_density"], **common)

    click.echo("Running destruction analysis...")
    evaluate_destruction_vs_non_destruction(
        destruction_geojson=layers["destruction"], **common
    )

    click.echo("Running municipal bounds analysis...")
    evaluate_municipal_bounds(boundary_shp=boundary_shp, **common)

    click.echo("Running error by month analysis...")
    evaluate_error_by_month(**common)

    click.echo(f"Done. Results written to {output_dir}")


if __name__ == "__main__":
    cli()
