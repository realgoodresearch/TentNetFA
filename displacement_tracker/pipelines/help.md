# TentNetFA pipelines

TentNetFA detects tents in Planet satellite imagery of the Gaza Strip with a
pixel-wise CNN. Three end-to-end pipelines cover the full workflow:
**training** builds a model from annotated imagery, **prediction** applies a
trained model to new imagery and produces a deduplicated GeoPackage of tent
locations, and **hyperparameter tuning** finds the merge thresholds
(`min_adj_peak`, `adjustment_factor`) that best match a reference dataset
and re-merges the predictions with them.

Every run is self-contained: the runner resolves your configuration, writes it
into a fresh run directory with fixed subfolder names, and executes the
selected stages against it.

```
<run root>/<pipeline>/<run name>/
    config.yaml     fully resolved config used by every stage
    logs/           one log file per stage
    manifests/ preds/ merged/     (prediction)
    manifests/ dataset/ model/    (training)
    merged_raw/ tuning/ merged/   (hyperparameter tuning)
```

## Pipeline diagrams

### Training pipeline (`train` section of `config.yaml`)

```mermaid
flowchart TB
    drive[("Google Drive")]
    ann["Tent annotations<br/>(geojson)"]
    prewar["Pre-war raster<br/>(prewar_gaza)"]
    tifs["GeoTIFFs<br/>(geotiff_dir)"]
    scan["<b>scan</b> — b1_annotated_scanner<br/>tiles imagery inside boundaries, pairs it with<br/>pre-war tiles and rasterised tent-label masks"]
    rebalance["<b>rebalance</b> — c_resample_manifest<br/>merges manifests, downsamples empty tiles<br/>(null_keep_fraction)"]
    train["<b>train</b> — d_train_cnn<br/>trains SimpleCNN on (image, pre-war, diff)<br/>stacks vs. label density maps"]
    model(["model/&lt;timestamp&gt;/best_model.pth"])

    drive -. "download (optional)<br/>a_tif_loader" .-> tifs
    tifs --> scan
    ann --> scan
    prewar --> scan
    scan -- "manifests/*.parquet" --> rebalance
    rebalance -- "dataset/balanced.parquet" --> train
    train --> model
```

### Prediction pipeline (`predict` section of `config.yaml`)

```mermaid
flowchart TB
    drive[("Google Drive")]
    tifs["GeoTIFFs<br/>(geotiff_dir)"]
    prewar["Pre-war raster<br/>(prewar_gaza)"]
    ckpt["Trained checkpoint<br/>(prediction.model)"]
    scan["<b>scan</b> — b2_image_scanner<br/>tiles new imagery inside boundaries,<br/>pairs it with pre-war tiles (no labels needed)"]
    predict["<b>predict</b> — e_predict_json<br/>runs inference per tile, extracts tent points<br/>(NMS / centroids) above selection thresholds"]
    merge["<b>merge</b> — h_merge_geojsons<br/>filters by adjusted peak, drops excluded zones,<br/>merges points within merge.min_distance_m"]
    out(["merged/merged.gpkg"])

    drive -. "download (optional)<br/>a_tif_loader" .-> tifs
    tifs --> scan
    prewar --> scan
    scan -- "manifests/*.parquet" --> predict
    ckpt --> predict
    predict -- "preds/*.geojson<br/>(overlapping tiles!)" --> merge
    merge --> out
```

### Hyperparameter tuning pipeline (`tune` section of `config.yaml`)

```mermaid
flowchart TB
    preds["Prediction GeoJSONs<br/>(merge.input_folder — the preds/ folder<br/>of an earlier prediction run)"]
    refdata["Reference data<br/>(tuning.reference:<br/>vector | unosat | raster)"]
    grid["Master grid raster<br/>(tuning.master_grid)"]
    merge_raw["<b>merge_raw</b> — h_merge_geojsons<br/>merges & deduplicates WITHOUT thresholding<br/>(min_adj_peak 0, adjustment_factor 1)"]
    scan["<b>scan</b> — g1_scan_validation<br/>rasterizes predictions vs. reference on the master grid,<br/>ridge-aware search over (adjustment_factor, min_adj_peak)<br/>optimising tuning.metric"]
    merge_tuned["<b>merge_tuned</b> — h2_merge_tuned<br/>re-merges the raw predictions<br/>with the tuned thresholds"]
    out(["merged/merged_tuned.gpkg"])

    preds --> merge_raw
    merge_raw -- "merged_raw/merged_raw.gpkg" --> scan
    refdata --> scan
    grid --> scan
    scan -- "tuning/best_params.yaml" --> merge_tuned
    preds --> merge_tuned
    merge_tuned --> out
```

Stages can be toggled individually in the sidebar — e.g. re-run only
*predict* + *merge* against manifests from an earlier run by pointing the
advanced YAML overrides at that run's folders, or skip *merge* while tuning
selection parameters.

## Configuration reference

All pipelines are configured through the single `config.yaml`, which has
four top-level sections: `shared` (single source of truth for values used
by more than one flow), and `train`, `predict` and `tune` (everything
specific to one flow). Before a run, the selected pipeline's section is
deep-merged over `shared` (the flow section wins), producing the flat key
layout documented below — so e.g. `boundaries` lives under `shared:` while
`prediction.model` lives under `predict:`. Stage CLIs resolve their natural
flow by default and accept `--flow train|predict|tune` to override; the
resolved config written into each run directory is already flat.

Paths may reference environment variables as `${VAR}` (resolved from `.env`);
`${DATA_DIR}` is the conventional data root. Keys marked **runner-managed**
are overwritten by the pipeline runner so all artifacts land in the run
directory — setting them yourself only matters when running stage CLIs
manually.

### Shared inputs (both pipelines)

| Key | Role |
|---|---|
| `geotiff_dir` | Directory containing the input GeoTIFF satellite images. The download stage writes here; the scan stages read from here. |
| `loading.files` | List of filename substrings. The download stage uses them as Google Drive search strings (empty = download nothing); the scan stages use them as filters (empty = scan **all** `.tif` files). |
| `boundaries` | Gaza municipal boundaries shapefile; tiles outside it are skipped. |
| `prewar_gaza` | Pre-war reference raster. Each tile is paired with the matching pre-war crop; the model sees (current, pre-war, difference). |
| `manifest_folder` | Where per-image tile manifests are written/read. **Runner-managed** → `manifests/`. |

### Processing (tiling)

| Key | Role |
|---|---|
| `processing.core_metres` | Side length (m) of a tile's core area — the region whose predictions/labels count. |
| `processing.margin_metres` | Extra context (m) around the core; tiles overlap by this margin. Also drives the prediction crop (`crop_pixels`) and NMS sigma. |
| `processing.quality_thresholds.min_valid_fraction` | Minimum non-black/NaN fraction for a tile to be kept (train: strict ~0.9; predict: loose ~0.1). |
| `processing.max_workers`, `max_tasks_per_child`, `max_pool_restarts` | Scan parallelism (training scanner). |
| `processing.complete` | Filenames processed in full, ignoring quality gates. |

### Training pipeline

| Key | Role |
|---|---|
| `geojson` | Tent-annotation snapshot (Forensic Architecture labels) rasterised into label masks. |
| `rebalancing.null_keep_fraction` | Fraction of empty (tent-free) tiles kept when balancing the dataset. |
| `rebalancing.rng_seed` | Seed for that downsampling. |
| `rebalancing.out`, `manifest` | Balanced dataset location. **Runner-managed** → `dataset/balanced.parquet`. |
| `training.checkpoint` | Optional `.pth` to resume from (empty = train from scratch; `model_kwargs` are then ignored). |
| `training.device` | `cuda` or `cpu`. |
| `training.epochs`, `batch_size`, `learning_rate` | Optimisation parameters. |
| `training.training_frac`, `validation_frac` | Dataset split fractions (remainder becomes a held-out test set). |
| `training.memory` | Cache the dataset in RAM for faster epochs. |
| `training.num_workers` | Data-loader workers. |
| `training.model_kwargs.kernel_size` | CNN kernel size (fresh models only). |
| `artifact_dir` | Where `<timestamp>/best_model.pth` + split info land. **Runner-managed** → `model/`. |
| `metadata_embedding.*` | Config for the separate `train-embedding` CLI; not part of this pipeline. |

### Prediction pipeline

| Key | Role |
|---|---|
| `prediction.model` | Trained checkpoint (`best_model.pth`) to run inference with. |
| `prediction.batch_size`, `num_workers` | Inference batching / data-loader workers. |
| `prediction.per_tile_standardisation` | Standardise each tile individually instead of per-raster. |
| `prediction.sample.enable`, `size`, `seed` | Predict only a random sample of tiles (for quick checks). |
| `prediction.validation_tifs` | Also write prediction rasters for visual validation (keep off for full runs). |
| `prediction.selection.method` | Point extraction: `nms` (local maxima) or `centroid` (blob centroids). |
| `prediction.selection.threshold` | Minimum score for a detection. |
| `prediction.selection.factor` | Weight of the blurred score added during NMS (adjusted peak). |
| `prediction.selection.min_area` | Minimum blob area in pixels (`centroid` method only; setting it with `nms` aborts the run). |
| `prediction.selection.nms_kernel_size` | Max-pool kernel size in pixels for NMS peak picking (`nms` method only). |
| `prediction.selection.min_distance_m` | Minimum distance between extracted points. |
| `prediction.input_folder`, `output_folder` | Manifest input (defaults to `manifest_folder`) / GeoJSON output. **Runner-managed** → `manifests/`, `preds/`. |
| `merge.min_distance_m` | Points closer than this (m) are merged into one tent. |
| `merge.agreement` | Minimum cluster size to keep after merging (overlapping tiles vote). |
| `merge.min_adj_peak` | Global adjusted-peak threshold applied before merging. |
| `merge.adjustment_factor` | Factor applied to (adjusted_peak − peak) when filtering. |
| `merge.thresholds_config` | Optional YAML with per-file adjusted-peak thresholds. |
| `merge.exclusion_zones_gpkg`, `inclusion_zone` | Drop points inside / outside these geometries. |
| `merge.input_folder` | Prediction GeoJSONs to merge; defaults to `prediction.output_folder`. |
| `merge.output` | Final GeoPackage. **Runner-managed** → `merged/merged.gpkg`. |

### Hyperparameter tuning pipeline

The tune flow reuses the `merge` section for both merge passes (the raw pass
forces `min_adj_peak: 0`, `adjustment_factor: 1` so the scan sees every
candidate point) and adds a `tuning` section:

| Key | Role |
|---|---|
| `merge.input_folder` | Prediction GeoJSONs to tune on — typically the `preds/` folder of an earlier prediction run. |
| `merge.output` | Unthresholded merged predictions the scan runs on. **Runner-managed** → `merged_raw/merged_raw.gpkg`. |
| `merge.min_distance_m`, `agreement`, zones | Same semantics as in the prediction pipeline; applied to both merge passes. |
| `tuning.master_grid` | Grid raster the predictions and the reference data are resolved onto. |
| `tuning.reference.type` | Reference source: `vector` (point annotations in any OGR-readable file), `unosat` (an export file, or a directory of exports + explicit `date`), or `raster` (counts already resolved on the master grid). |
| `tuning.reference.path` | The annotation file / export / directory / raster. |
| `tuning.reference.date` | Picks one export when the path is a directory (`unosat`); the reference is always selected explicitly, never inferred from prediction timestamps. |
| `tuning.reference.layer`, `where` | Optional layer name and OGR SQL attribute filter for vector sources. |
| `tuning.metric` | Evaluation metric whose optimum becomes the tuned `(min_adj_peak, adjustment_factor)`: `rms`, `mae`, `rmsle`, `abs_total_diff`, `abs_total_pdiff` or `spearman`. |
| `tuning.metrics` | Metrics tracked during the scan (the evaluation metric is always included). |
| `tuning.factor_min/max`, `cutoff_min/max` | Search bounds for `adjustment_factor` / `min_adj_peak`. |
| `tuning.ridge_probes`, `xtol_factor`, `xtol_cutoff`, `refine_maxiter` | Search effort and precision of the ridge-aware optimizer. |
| `tuning.exclusion_zones` | Optional gpkg; predictions are clipped to its union before the scan. |
| `tuning.input` | Merged raw predictions; defaults to `merge.output`. **Runner-managed** → `merged_raw/merged_raw.gpkg`. |
| `tuning.out_dir`, `best_params` | Scan artifacts (search trace, summary, rasters) and the tuned-parameter YAML. **Runner-managed** → `tuning/`. |
| `tuning.final_output` | Final tuned GeoPackage. **Runner-managed** → `merged/merged_tuned.gpkg`. |

### Anything else

Every config key can be overridden — parameters not exposed as form fields go
in **Advanced: extra YAML overrides** (Configuration tab), which is
deep-merged over the base config last.
