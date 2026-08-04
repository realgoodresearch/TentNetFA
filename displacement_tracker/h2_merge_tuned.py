"""Final stage of the tuning pipeline: re-run the merge with tuned thresholds.

Reads the ``best_params.yaml`` written by g1_scan_validation and repeats the
merge of the raw prediction GeoJSONs with the tuned ``min_adj_peak`` /
``adjustment_factor``, producing the final thresholded output at
``tuning.final_output``. All other merge settings (distance, agreement,
zones) come from the ``merge`` section, so the only difference from the
first, unthresholded pass is the tuned thresholding pair.

Usage:
    poetry run merge-tuned config.yaml
"""

from pathlib import Path

import click
import yaml

from displacement_tracker.h_merge_geojsons import (
    merge_geojsons,
    merge_kwargs_from_config,
)
from displacement_tracker.util.config import flow_option, load_flow_config
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("merge_tuned")


def load_best_params(path: str) -> dict:
    """Load and validate the tuned parameters written by the scan stage."""
    params_path = Path(path)
    if not params_path.exists():
        raise click.ClickException(
            f"Tuned parameters not found: {params_path} — "
            "run the scan stage (scan-validation) first."
        )
    with params_path.open("r", encoding="utf-8") as f:
        best = yaml.safe_load(f) or {}
    missing = [k for k in ("min_adj_peak", "adjustment_factor") if k not in best]
    if missing:
        raise click.ClickException(
            f"Tuned parameters file {params_path} is missing keys: {missing}"
        )
    return best


@click.command()
@click.argument("config", type=click.Path(exists=True, dir_okay=False))
@flow_option(default="tune")
def cli(config: str, flow: str) -> None:
    """Merge prediction GeoJSONs with the thresholds found by the tuning scan."""
    params = load_flow_config(config, flow)
    merge_cfg = params.get("merge") or {}
    tuning = params.get("tuning") or {}

    input_folder = merge_cfg.get("input_folder")
    if not input_folder:
        raise click.ClickException("Missing required config key: merge.input_folder")
    output = tuning.get("final_output")
    if not output:
        raise click.ClickException("Missing required config key: tuning.final_output")
    best_params_path = tuning.get("best_params")
    if not best_params_path and tuning.get("out_dir"):
        best_params_path = str(Path(tuning["out_dir"]) / "best_params.yaml")
    if not best_params_path:
        raise click.ClickException("Missing required config key: tuning.best_params")

    best = load_best_params(best_params_path)
    LOGGER.info(
        "Tuned thresholds from %s: min_adj_peak=%.6f, adjustment_factor=%.4f "
        "(metric=%s, value=%s)",
        best_params_path,
        float(best["min_adj_peak"]),
        float(best["adjustment_factor"]),
        best.get("metric", "?"),
        best.get("value", "?"),
    )

    # The tuned pair replaces the config's thresholds; per-file
    # thresholds_config is deliberately omitted, since per-file entries
    # would shadow the tuned global threshold (resolve_threshold prefers
    # them).
    kwargs = merge_kwargs_from_config(
        merge_cfg,
        ("min_distance_m", "agreement", "exclusion_zones_gpkg", "inclusion_zone"),
    )
    merge_geojsons(
        input_folder,
        output,
        min_adj_peak=float(best["min_adj_peak"]),
        adjustment_factor=float(best["adjustment_factor"]),
        **kwargs,
    )


if __name__ == "__main__":
    cli()
