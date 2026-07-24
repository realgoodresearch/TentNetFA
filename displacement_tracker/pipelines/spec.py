"""Declarative descriptions of the end-to-end pipelines.

This module is the single source of truth for:

- which stages a pipeline runs (and the module executed for each stage),
- which config parameters are exposed for interactive override,
- where artifacts land inside a run directory (fixed subfolder names).

Frontends (Streamlit app, plain CLI, future TUIs) render forms from these
specs instead of hardcoding config keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from displacement_tracker.util.config import FLOWS


@dataclass(frozen=True)
class Param:
    """A single overridable config value, addressed by dotted path."""

    path: str  # dotted path into the config, e.g. "training.batch_size"
    label: str
    type: str  # one of: str, path, int, float, bool
    group: str  # section header used by frontends
    help: str = ""
    optional: bool = False  # empty input maps to None instead of ""


@dataclass(frozen=True)
class Stage:
    """One executable step of a pipeline.

    Every stage is run as ``python -m <module> <resolved_config>``.
    """

    key: str
    label: str
    module: str
    default_enabled: bool = True


@dataclass(frozen=True)
class Pipeline:
    key: str
    label: str
    # Default base YAML, relative to the repo root. The pipeline's flow
    # section (named by its key) is resolved from it before use.
    base_config: str
    stages: tuple[Stage, ...]
    params: tuple[Param, ...]
    # Dotted config path -> path relative to the run directory. These are
    # forced into the run directory so every run is self-contained.
    artifact_paths: dict[str, str] = field(default_factory=dict)
    # Config sections injected if absent from the base YAML (e.g. the merge
    # settings, which historically lived in CLI flags rather than config).
    extra_defaults: dict = field(default_factory=dict)
    # Dotted config path -> value applied after base config and overrides,
    # like artifact_paths: invariants the pipeline depends on, not defaults.
    forced_values: dict = field(default_factory=dict)

    @property
    def subfolders(self) -> list[str]:
        """Fixed subfolder names created inside every run directory."""
        names = {"logs"}
        for rel in self.artifact_paths.values():
            # "dataset/balanced.parquet" -> "dataset"; "manifests" -> itself
            head = rel.split("/", 1)[0]
            names.add(head)
        return sorted(names)


PREDICT = Pipeline(
    key="predict",
    label="Prediction pipeline",
    base_config="config.yaml",
    stages=(
        Stage(
            "download",
            "Download GeoTIFFs from Drive",
            "displacement_tracker.a_tif_loader",
            default_enabled=False,
        ),
        Stage(
            "scan",
            "Scan imagery & build manifests",
            "displacement_tracker.b2_image_scanner",
        ),
        Stage("predict", "Run model inference", "displacement_tracker.e_predict_json"),
        Stage(
            "merge",
            "Merge & deduplicate predictions",
            "displacement_tracker.h_merge_geojsons",
        ),
    ),
    params=(
        Param("geotiff_dir", "GeoTIFF directory", "path", "Inputs"),
        Param(
            "loading.files",
            "GeoTIFF files / search strings",
            "list",
            "Inputs",
            optional=True,
            help="One entry per line. Download stage: entries are Drive search "
            "strings — if empty, the stage downloads nothing. Scan stage: "
            "entries are filename filters — if empty, ALL .tif files in "
            "the GeoTIFF directory are scanned.",
        ),
        Param("boundaries", "Municipal boundaries (.shp)", "path", "Inputs"),
        Param("prewar_gaza", "Pre-war reference raster", "path", "Inputs"),
        Param("prediction.model", "Model checkpoint (.pth)", "path", "Inputs"),
        Param("processing.core_metres", "Tile core size (m)", "int", "Processing"),
        Param("processing.margin_metres", "Tile margin (m)", "int", "Processing"),
        Param(
            "processing.quality_thresholds.min_valid_fraction",
            "Min valid fraction",
            "float",
            "Processing",
            help="Minimum non-black/NaN fraction for a tile to be kept.",
        ),
        Param("prediction.batch_size", "Batch size", "int", "Prediction"),
        Param("prediction.num_workers", "Data-loader workers", "int", "Prediction"),
        Param(
            "prediction.per_tile_standardisation",
            "Per-tile standardisation",
            "bool",
            "Prediction",
        ),
        Param("prediction.sample.enable", "Sample tiles only", "bool", "Prediction"),
        Param("prediction.sample.size", "Sample size", "int", "Prediction"),
        Param("prediction.sample.seed", "Sample seed", "int", "Prediction"),
        Param(
            "prediction.selection.threshold",
            "Selection threshold",
            "float",
            "Selection",
        ),
        Param("prediction.selection.factor", "Selection factor", "float", "Selection"),
        Param(
            "prediction.selection.nms_kernel_size",
            "NMS kernel size (px)",
            "int",
            "Selection",
            help="Max-pool kernel for NMS peak picking. For the 'centroid' "
            "method set selection.min_area via the YAML overrides instead.",
        ),
        Param(
            "prediction.selection.min_distance_m",
            "Min peak distance (m)",
            "float",
            "Selection",
        ),
        Param("merge.min_distance_m", "Merge distance (m)", "float", "Merge"),
        Param("merge.agreement", "Min cluster size", "int", "Merge"),
        Param("merge.min_adj_peak", "Min adjusted peak", "float", "Merge"),
        Param("merge.adjustment_factor", "Adjustment factor", "float", "Merge"),
        Param(
            "merge.thresholds_config",
            "Per-file thresholds YAML",
            "path",
            "Merge",
            optional=True,
        ),
        Param(
            "merge.exclusion_zones_gpkg",
            "Exclusion zones",
            "path",
            "Merge",
            optional=True,
        ),
        Param("merge.inclusion_zone", "Inclusion zone", "path", "Merge", optional=True),
    ),
    artifact_paths={
        "manifest_folder": "manifests",
        "prediction.input_folder": "manifests",
        "prediction.output_folder": "preds",
        "merge.output": "merged/merged.gpkg",
    },
    extra_defaults={
        "merge": {
            "min_distance_m": 3.0,
            "agreement": 1,
            "min_adj_peak": 0.0,
            "adjustment_factor": 1.0,
            "thresholds_config": None,
            "exclusion_zones_gpkg": None,
            "inclusion_zone": None,
        },
    },
)


TRAIN = Pipeline(
    key="train",
    label="Training pipeline",
    base_config="config.yaml",
    stages=(
        Stage(
            "download",
            "Download GeoTIFFs from Drive",
            "displacement_tracker.a_tif_loader",
            default_enabled=False,
        ),
        Stage(
            "scan",
            "Scan annotated imagery",
            "displacement_tracker.b1_annotated_scanner",
        ),
        Stage(
            "rebalance",
            "Rebalance manifest",
            "displacement_tracker.c_resample_manifest",
        ),
        Stage("train", "Train CNN", "displacement_tracker.d_train_cnn"),
    ),
    params=(
        Param("geotiff_dir", "GeoTIFF directory", "path", "Inputs"),
        Param(
            "loading.files",
            "GeoTIFF files / search strings",
            "list",
            "Inputs",
            optional=True,
            help="One entry per line. Download stage: entries are Drive search "
            "strings — if empty, the stage downloads nothing. Scan stage: "
            "entries are filename filters — if empty, ALL .tif files in "
            "the GeoTIFF directory are scanned.",
        ),
        Param("geojson", "Tent annotations (.geojson)", "path", "Inputs"),
        Param("boundaries", "Municipal boundaries (.shp)", "path", "Inputs"),
        Param("prewar_gaza", "Pre-war reference raster", "path", "Inputs"),
        Param("processing.core_metres", "Tile core size (m)", "int", "Processing"),
        Param("processing.margin_metres", "Tile margin (m)", "int", "Processing"),
        Param("processing.max_workers", "Scan workers", "int", "Processing"),
        Param(
            "processing.quality_thresholds.min_valid_fraction",
            "Min valid fraction",
            "float",
            "Processing",
        ),
        Param("rebalancing.rng_seed", "Rebalancing seed", "int", "Rebalancing"),
        Param(
            "rebalancing.null_keep_fraction",
            "Null keep fraction",
            "float",
            "Rebalancing",
        ),
        Param(
            "training.checkpoint",
            "Resume checkpoint (.pth)",
            "path",
            "Training",
            optional=True,
            help="Leave empty to train from scratch.",
        ),
        Param("training.device", "Device", "str", "Training"),
        Param("training.epochs", "Epochs", "int", "Training"),
        Param("training.batch_size", "Batch size", "int", "Training"),
        Param("training.learning_rate", "Learning rate", "float", "Training"),
        Param("training.training_frac", "Training fraction", "float", "Training"),
        Param("training.validation_frac", "Validation fraction", "float", "Training"),
        Param("training.memory", "Cache dataset in RAM", "bool", "Training"),
        Param("training.num_workers", "Data-loader workers", "int", "Training"),
        Param("training.model_kwargs.kernel_size", "Kernel size", "int", "Training"),
    ),
    artifact_paths={
        "manifest_folder": "manifests",
        "rebalancing.out": "dataset/balanced.parquet",
        "manifest": "dataset/balanced.parquet",
        "artifact_dir": "model",
    },
)


TUNE = Pipeline(
    key="tune",
    label="Hyperparameter tuning pipeline",
    base_config="config.yaml",
    stages=(
        Stage(
            "merge_raw",
            "Merge predictions (no thresholding)",
            "displacement_tracker.h_merge_geojsons",
        ),
        Stage(
            "scan",
            "Scan validation & optimise thresholds",
            "displacement_tracker.g1_scan_validation",
        ),
        Stage(
            "merge_tuned",
            "Re-merge with tuned thresholds",
            "displacement_tracker.h2_merge_tuned",
        ),
    ),
    params=(
        Param(
            "merge.input_folder",
            "Prediction GeoJSONs folder",
            "path",
            "Inputs",
            help="Predictions to tune on — typically the preds/ folder of an "
            "earlier prediction pipeline run.",
        ),
        Param(
            "tuning.master_grid",
            "Master grid raster",
            "path",
            "Inputs",
            help="Grid the predictions and reference data are resolved onto.",
        ),
        Param(
            "tuning.reference.type",
            "Reference type",
            "str",
            "Reference",
            help="vector (point annotations: GeoJSON/GPKG/SHP), unosat "
            "(export file or directory + date), or raster (counts "
            "already on the master grid).",
        ),
        Param("tuning.reference.path", "Reference data path", "path", "Reference"),
        Param(
            "tuning.reference.date",
            "UNOSAT export date",
            "str",
            "Reference",
            optional=True,
            help="YYYY-MM-DD; pins one export when the path is a directory. "
            "Empty: the export closest to the dates stamped on the "
            "prediction files is auto-discovered (with a warning).",
        ),
        Param(
            "tuning.reference.layer",
            "Layer",
            "str",
            "Reference",
            optional=True,
            help="Layer to read from multi-layer files (GPKG/GDB).",
        ),
        Param(
            "tuning.reference.where",
            "Attribute filter",
            "str",
            "Reference",
            optional=True,
            help="OGR SQL filter applied to the reference features.",
        ),
        Param(
            "tuning.metric",
            "Evaluation metric",
            "str",
            "Tuning",
            help="The optimum of this metric becomes the tuned "
            "(min_adj_peak, adjustment_factor) used by the final merge. "
            "Choices: rms, mae, rmsle, abs_total_diff, abs_total_pdiff, "
            "spearman.",
        ),
        Param(
            "tuning.metrics",
            "Metrics to track",
            "list",
            "Tuning",
            optional=True,
            help="One metric per line; the evaluation metric is always tracked.",
        ),
        Param("tuning.factor_min", "Factor min", "float", "Tuning"),
        Param("tuning.factor_max", "Factor max", "float", "Tuning"),
        Param("tuning.cutoff_min", "Cutoff min", "float", "Tuning"),
        Param("tuning.cutoff_max", "Cutoff max", "float", "Tuning"),
        Param(
            "tuning.ridge_probes",
            "Ridge probes",
            "int",
            "Tuning",
            help="Number of factors at which the cutoff ridge is probed.",
        ),
        Param(
            "tuning.refine_maxiter",
            "Refinement iterations",
            "int",
            "Tuning",
            help="Max Nelder-Mead iterations for the 2-D refinement (0 disables).",
        ),
        Param(
            "tuning.exclusion_zones",
            "Scan clip zones",
            "path",
            "Tuning",
            optional=True,
            help="Optional gpkg; predictions are clipped to its union before the scan.",
        ),
        Param("merge.min_distance_m", "Merge distance (m)", "float", "Merge"),
        Param("merge.agreement", "Min cluster size", "int", "Merge"),
        Param(
            "merge.exclusion_zones_gpkg",
            "Exclusion zones",
            "path",
            "Merge",
            optional=True,
        ),
        Param("merge.inclusion_zone", "Inclusion zone", "path", "Merge", optional=True),
    ),
    artifact_paths={
        "merge.output": "merged_raw/merged_raw.gpkg",
        "tuning.input": "merged_raw/merged_raw.gpkg",
        "tuning.out_dir": "tuning",
        "tuning.best_params": "tuning/best_params.yaml",
        "tuning.final_output": "merged/merged_tuned.gpkg",
    },
    forced_values={
        # The scan must see every candidate point (the thresholds are
        # exactly what is being tuned), and per-file thresholds would
        # shadow the tuned global threshold — the raw pass is always
        # unthresholded, whatever the base config or overrides say.
        "merge.min_adj_peak": 0.0,
        "merge.adjustment_factor": 1.0,
        "merge.thresholds_config": None,
    },
    extra_defaults={
        "merge": {
            "min_distance_m": 3.0,
            "agreement": 1,
            "exclusion_zones_gpkg": None,
            "inclusion_zone": None,
        },
        "tuning": {
            "reference": {"type": "vector"},
            "metric": "rms",
            "metrics": ["rms", "mae", "abs_total_diff"],
            "factor_min": 0.0,
            "factor_max": 10.0,
            "cutoff_min": 0.0001,
            "cutoff_max": 0.01,
            "ridge_probes": 5,
            "xtol_factor": 1.0e-3,
            "xtol_cutoff": 1.0e-6,
            "refine_maxiter": 60,
            "exclusion_zones": None,
        },
    },
)


PIPELINES: dict[str, Pipeline] = {p.key: p for p in (PREDICT, TRAIN, TUNE)}

# The runner and UI resolve each pipeline's config section by its key.
assert set(PIPELINES) <= set(FLOWS), "pipeline keys must be config flow names"
