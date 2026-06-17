#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from scripts.add_new_model_results import add_new_model_results
from scripts.total_error import evaluate_total_error
from scripts.spatial_bootstrap_hex import spatial_bootstrap_hex
from scripts.evaluate_spatial_points import evaluate_spatial_points
from scripts.evaluate_tile_correlation import evaluate_tile_correlation
from scripts.evaluate_agriculture import evaluate_agriculture_vs_non_agriculture
from scripts.evaluate_density import evaluate_h3_density_bins
from scripts.evaluate_destruction import evaluate_destruction_vs_non_destruction
from scripts.evaluate_municipal_bounds import evaluate_municipal_bounds
from scripts.evaluate_month import evaluate_error_by_month


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def require(cfg: Dict[str, Any], key: str):
    if key not in cfg or cfg[key] in (None, ""):
        raise ValueError(f"Missing required config key: {key}")
    return cfg[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all analyses from a config file.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    annotation_csv = require(cfg, "annotation_csv")
    output_csv = require(cfg, "output_csv")
    prediction_dir = require(cfg, "prediction_dir")
    sample_tif = require(cfg, "sample_tif")
    new_model_column = require(cfg, "new_model_column")

    output_dir = require(cfg, "output_dir")
    boundary_shp = require(cfg, "boundary_shp")
    agriculture_geojson = require(cfg, "agriculture_geojson")
    h3_geojson = require(cfg, "h3_geojson")
    destruction_geojson = require(cfg, "destruction_geojson")

    manual_column = cfg.get("manual_column", "manual_tent_count")
    hex_size_m = float(cfg.get("hex_size_m", 1000.0))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_csv, model_col = add_new_model_results(
        annotation_csv=annotation_csv,
        output_csv=output_csv,
        prediction_dir=prediction_dir,
        sample_tif=sample_tif,
        new_model_column=new_model_column,
    )

    evaluate_total_error(
        annotation_csv=output_csv,
        boundary_shp=boundary_shp,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
        hex_size_m=hex_size_m,
    )

    evaluate_spatial_points(
        annotation_csv=output_csv,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
    )

    spatial_bootstrap_hex(
        annotation_csv=output_csv,
        boundary_shp=boundary_shp,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
        hex_size_m=hex_size_m,
    )

    evaluate_tile_correlation(
        annotation_csv=output_csv,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
    )

    evaluate_agriculture_vs_non_agriculture(
        annotation_csv=output_csv,
        agriculture_geojson=agriculture_geojson,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
    )

    evaluate_h3_density_bins(
        annotation_csv=output_csv,
        h3_geojson=h3_geojson,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
    )

    evaluate_destruction_vs_non_destruction(
        annotation_csv=output_csv,
        destruction_geojson=destruction_geojson,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
    )

    evaluate_municipal_bounds(
        annotation_csv=output_csv,
        boundary_shp=boundary_shp,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
    )

    evaluate_error_by_month(
        annotation_csv=output_csv,
        output_dir=output_dir,
        manual_column=manual_column,
        model_column=model_col,
    )


if __name__ == "__main__":
    main()
