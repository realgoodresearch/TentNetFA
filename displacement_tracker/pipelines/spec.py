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

    By default a stage is run as ``python -m <module> <resolved_config>``.
    Stages with a different CLI surface (e.g. h_merge_geojsons) get a
    custom argv builder in ``runner.py``, keyed by ``key``.
    """

    key: str
    label: str
    module: str
    default_enabled: bool = True


@dataclass(frozen=True)
class Pipeline:
    key: str
    label: str
    base_config: str  # default base YAML, relative to the repo root
    stages: tuple[Stage, ...]
    params: tuple[Param, ...]
    # Dotted config path -> path relative to the run directory. These are
    # forced into the run directory so every run is self-contained.
    artifact_paths: dict[str, str] = field(default_factory=dict)
    # Config sections injected if absent from the base YAML (e.g. the merge
    # settings, which historically lived in CLI flags rather than config).
    extra_defaults: dict = field(default_factory=dict)

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
    base_config="predict_config.yaml",
    stages=(
        Stage(
            "download", "Download GeoTIFFs from Drive", "displacement_tracker.a_tif_loader",
            default_enabled=False,
        ),
        Stage("scan", "Scan imagery & build manifests", "displacement_tracker.b2_image_scanner"),
        Stage("predict", "Run model inference", "displacement_tracker.e_predict_json"),
        Stage("merge", "Merge & deduplicate predictions", "displacement_tracker.h_merge_geojsons"),
    ),
    params=(
        Param("geotiff_dir", "GeoTIFF directory", "path", "Inputs"),
        Param(
            "loading.files", "GeoTIFF files / search strings", "list", "Inputs",
            optional=True,
            help="One entry per line. Used as Drive search strings by the "
                 "download stage and as filename filters when scanning; "
                 "leave empty to scan every .tif in the GeoTIFF directory.",
        ),
        Param("boundaries", "Municipal boundaries (.shp)", "path", "Inputs"),
        Param("prewar_gaza", "Pre-war reference raster", "path", "Inputs"),
        Param("prediction.model", "Model checkpoint (.pth)", "path", "Inputs"),
        Param("processing.core_metres", "Tile core size (m)", "int", "Processing"),
        Param("processing.margin_metres", "Tile margin (m)", "int", "Processing"),
        Param(
            "processing.quality_thresholds.min_valid_fraction",
            "Min valid fraction", "float", "Processing",
            help="Minimum non-black/NaN fraction for a tile to be kept.",
        ),
        Param("prediction.batch_size", "Batch size", "int", "Prediction"),
        Param("prediction.num_workers", "Data-loader workers", "int", "Prediction"),
        Param("prediction.per_tile_standardisation", "Per-tile standardisation", "bool", "Prediction"),
        Param("prediction.sample.enable", "Sample tiles only", "bool", "Prediction"),
        Param("prediction.sample.size", "Sample size", "int", "Prediction"),
        Param("prediction.sample.seed", "Sample seed", "int", "Prediction"),
        Param("prediction.selection.threshold", "Selection threshold", "float", "Selection"),
        Param("prediction.selection.factor", "Selection factor", "float", "Selection"),
        Param("prediction.selection.min_area", "Min blob area (px)", "int", "Selection"),
        Param("prediction.selection.min_distance_m", "Min peak distance (m)", "float", "Selection"),
        Param("merge.min_distance_m", "Merge distance (m)", "float", "Merge"),
        Param("merge.agreement", "Min cluster size", "int", "Merge"),
        Param("merge.min_adj_peak", "Min adjusted peak", "float", "Merge"),
        Param("merge.adjustment_factor", "Adjustment factor", "float", "Merge"),
        Param(
            "merge.thresholds_config", "Per-file thresholds YAML", "path", "Merge",
            optional=True,
        ),
        Param("merge.exclusion_zones_gpkg", "Exclusion zones", "path", "Merge", optional=True),
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
            "download", "Download GeoTIFFs from Drive", "displacement_tracker.a_tif_loader",
            default_enabled=False,
        ),
        Stage("scan", "Scan annotated imagery", "displacement_tracker.b1_annotated_scanner"),
        Stage("rebalance", "Rebalance manifest", "displacement_tracker.c_resample_manifest"),
        Stage("train", "Train CNN", "displacement_tracker.d_train_cnn"),
    ),
    params=(
        Param("geotiff_dir", "GeoTIFF directory", "path", "Inputs"),
        Param(
            "loading.files", "GeoTIFF files / search strings", "list", "Inputs",
            optional=True,
            help="One entry per line. Used as Drive search strings by the "
                 "download stage and as filename filters when scanning; "
                 "leave empty to scan every .tif in the GeoTIFF directory.",
        ),
        Param("geojson", "Tent annotations (.geojson)", "path", "Inputs"),
        Param("boundaries", "Municipal boundaries (.shp)", "path", "Inputs"),
        Param("prewar_gaza", "Pre-war reference raster", "path", "Inputs"),
        Param("processing.core_metres", "Tile core size (m)", "int", "Processing"),
        Param("processing.margin_metres", "Tile margin (m)", "int", "Processing"),
        Param("processing.max_workers", "Scan workers", "int", "Processing"),
        Param(
            "processing.quality_thresholds.min_valid_fraction",
            "Min valid fraction", "float", "Processing",
        ),
        Param("rebalancing.rng_seed", "Rebalancing seed", "int", "Rebalancing"),
        Param("rebalancing.null_keep_fraction", "Null keep fraction", "float", "Rebalancing"),
        Param(
            "training.checkpoint", "Resume checkpoint (.pth)", "path", "Training",
            optional=True, help="Leave empty to train from scratch.",
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


PIPELINES: dict[str, Pipeline] = {p.key: p for p in (PREDICT, TRAIN)}
