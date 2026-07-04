# Gaza Strip Tent Detection (TentNetFA)

> A fork from https://github.com/algorithmicgovernance/TentNetFA

This project processes high-resolution Planet satellite images of the Gaza Strip in combination with historic tent locations identified by **Forensic Architecture** — a multidisciplinary research group based at Goldsmiths, University of London.

---

## Overview

The goal of this work is to develop a convolutional neural network (CNN) that can predict, at the pixel level, the locations of tents in the Gaza Strip from satellite imagery. These predictions use Gaussian densities to create highly granular maps of displacement patterns over time.

This automated detection supports population nowcasting in the Gaza strip in collaboration with the United Nations, Acted, IMPACT, and other partners integrating privacy-preserving telecommunications data and other geospatial data sources related to real-time population movements (see [Gaza NowPop](https://github.com/realgoodresearch/GazaNowPop)). 

---

## Key Features

-   **Data Ingestion**: Processes Planet GeoTIFF satellite images and GeoJSON files containing labeled tent locations.
-   **Data Processing**:
    -   Scans satellite imagery based on geographic coordinates to extract image tiles.
    -   Generates paired datasets of image patches and corresponding label masks indicating tent locations.
    -   Creates HDF5 datasets for efficient handling during training.
-   **Model Training**: Trains a custom CNN (`SimpleCNN`) for pixel-wise semantic segmentation to predict tent presence as a density map.
-   **Prediction & Evaluation**:
    -   Generates GeoJSON point clouds of predicted tent locations from new satellite imagery.
    -   Includes tools for evaluating prediction accuracy against ground truth data and for performing spatial validation analysis.

---

## Installation

This project uses Poetry for dependency management. Ensure you have Python 3.10+ installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/realgoodresearch/TentNetFA.git
    cd TentNetFA
    ```

2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
    Alternatively, you can install from the `requirements.txt` file, although this is not the recommended method:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the project root to store necessary credentials, such as your `GOOGLE_API_KEY` and the Google Drive folder ID (`GDRIVE_ID`) for downloading satellite imagery.

    ```
    GOOGLE_API_KEY="your_api_key_here"
    GDRIVE_ID="your_folder_id_here"
    DATA_DIR="/path/to/your/data"

### Updating requirements.txt

If you modify dependencies in `pyproject.toml`, you must regenerate the `requirements.txt` file:
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

on a regular basis.


## Interactive Pipelines (recommended)

The individual stages below can also be run end-to-end through the pipeline runner, which handles config resolution and artifact layout for you. Every pipeline run creates a self-contained directory with fixed subfolder names:

```
<run root>/<pipeline>/<run name>/
    config.yaml     # fully resolved config used by all stages
    logs/           # one log file per stage
    manifests/ preds/ merged/    # prediction pipeline
    manifests/ dataset/ model/   # training pipeline
```

The run root defaults to `${DATA_DIR}/results/TentNetFA/pipeline_runs`; the run name defaults to a timestamp.

Both pipelines include an optional (default-off) download stage that fetches newly arrived GeoTIFFs from Google Drive into `geotiff_dir` before scanning, using the `loading.files` entries as search strings (requires `GOOGLE_API_KEY` and `GDRIVE_ID` in `.env`).

### Pipeline stages

Training pipeline (`config.yaml`):

```
   Google Drive                    tent annotations           pre-war raster
        │                            (geojson)                (prewar_gaza)
        ▼                                │                         │
  ┌─[download]─┐   GeoTIFFs             │                         │
  │ a_tif_loader├──▶ geotiff_dir ────────┼─────────────────────────┤
  └─(optional)──┘        │               ▼                         ▼
                         └────▶ ┌─────[scan]──────────────────────────┐
                                │ b1_annotated_scanner                │
                                │ tiles imagery inside `boundaries`,  │
                                │ pairs it with pre-war tiles and     │
                                │ rasterised tent-label masks         │
                                └──────────────┬──────────────────────┘
                                               ▼  manifests/*.parquet
                                ┌─────[rebalance]─────────────────────┐
                                │ c_resample_manifest                 │
                                │ merges manifests, downsamples       │
                                │ empty tiles (null_keep_fraction)    │
                                └──────────────┬──────────────────────┘
                                               ▼  dataset/balanced.parquet
                                ┌─────[train]─────────────────────────┐
                                │ d_train_cnn                         │
                                │ trains SimpleCNN on (image, prewar, │
                                │ diff) stacks vs. label density maps │
                                └──────────────┬──────────────────────┘
                                               ▼
                                  model/<timestamp>/best_model.pth
```

Prediction pipeline (`predict_config.yaml`):

```
   Google Drive                                          trained checkpoint
        │                                                (prediction.model)
        ▼                                                        │
  ┌─[download]─┐   GeoTIFFs                                      │
  │ a_tif_loader├──▶ geotiff_dir                                 │
  └─(optional)──┘        │                                       │
                         ▼                                       │
              ┌─────[scan]──────────────────────────┐            │
              │ b2_image_scanner                    │            │
              │ tiles new imagery inside            │            │
              │ `boundaries` (no labels needed)     │            │
              └──────────────┬──────────────────────┘            │
                             ▼  manifests/*.parquet              │
              ┌─────[predict]───────────────────────┐            │
              │ e_predict_json                      │◀───────────┘
              │ runs inference per tile, extracts   │
              │ tent points (NMS / centroids) above │
              │ `selection` thresholds              │
              └──────────────┬──────────────────────┘
                             ▼  preds/*.geojson  (overlapping tiles!)
              ┌─────[merge]─────────────────────────┐
              │ h_merge_geojsons                    │
              │ filters by adjusted peak, drops     │
              │ excluded zones, merges points       │
              │ within `merge.min_distance_m`       │
              └──────────────┬──────────────────────┘
                             ▼
                    merged/merged.gpkg
```

The same diagrams, together with a full reference of every config key, are available in the UI's **Help** tab (sourced from [`displacement_tracker/pipelines/help.md`](displacement_tracker/pipelines/help.md)).

### Browser UI

```bash
poetry install --with ui   # installs streamlit
poetry run pipeline-ui
```

This starts a local web server and opens a browser session where you pick the pipeline (training or prediction), override any parameter from `config.yaml` / `predict_config.yaml` in a form (plus a free-form YAML box for anything not exposed), toggle individual stages, and watch live logs while the run executes.

A run is tied to its browser session: switching pipeline, refreshing or closing the page cancels the running stage and terminates all of its child processes. Completed artifacts and per-stage logs remain in the run directory, so you can resume by re-running with only the remaining stages enabled. For long unattended runs prefer the headless CLI below inside tmux/screen.

Extra arguments are passed through to `streamlit run` (e.g. `--server.port 8501`).

#### Running on a remote machine (SSH tunnel)

Pipelines will typically run on a remote GPU machine. The UI is just a web server on that machine, so start it there and forward the port to your local browser:

```bash
# on the remote, inside the repo (tmux/screen recommended: the pipeline
# stages are children of the UI process and die with your SSH session)
poetry run pipeline-ui --remote

# it prints the matching tunnel command to run on your local machine, e.g.
#     ssh -L 8501:localhost:8501 user@remote
# then open http://localhost:8501 locally
```

`--remote` is shorthand for `--server.headless true --server.address localhost`: no browser is opened on the remote, and the UI is reachable only through the tunnel — recommended, since the UI has no authentication and can launch jobs and write to `DATA_DIR`. Individual flags can still be overridden (e.g. `--remote --server.port 8600`), and whenever an SSH session is detected the tunnel command is printed even without `--remote`. On a trusted LAN you can drop `--remote` and use the "Network URL" streamlit prints instead of a tunnel, but never expose the port to untrusted networks.

After pulling new code (or when a port is stuck), stop any stale backends before restarting:

```bash
poetry run pipeline-ui-stop            # add --dry-run to only list them
```

This gracefully terminates every running `pipeline-ui` server including the pipeline stages it spawned, plus stage processes orphaned by an earlier hard kill. Deliberate headless `pipeline-run` sessions are left untouched.

### Headless CLI

The same orchestration is scriptable without a browser:

```bash
poetry run pipeline-run predict --set prediction.batch_size=16 --name 2026-02-rerun
poetry run pipeline-run train --skip download --set training.epochs=500
poetry run pipeline-run predict --dry-run   # print the plan without executing
```

`--set` takes dotted config paths (`section.key=value`, YAML-parsed); `--only`/`--skip` select stages. Pipeline definitions (stages, exposed parameters, artifact layout) live in `displacement_tracker/pipelines/spec.py`.

## Workflow and CLI Usage

The core workflow is managed through a series of command-line scripts. Most scripts require a `config.yaml` file to specify paths and parameters.

### 1. Download Satellite Imagery
Download GeoTIFF files from Google Drive based on the filenames specified in your configuration file.

```bash
poetry run tif-loader config.yaml
```

### 2. Prepare Training Data
Scan the downloaded GeoTIFFs and corresponding GeoJSON labels to create an HDF5 dataset for training.

```bash
poetry run coordinate-scanner config.yaml
```

### 3. Train the Model
Train the CNN using the generated HDF5 dataset.

```bash
poetry run train-cnn config.yaml
```
Model checkpoints and training logs will be saved to a timestamped directory inside `runs/`.

### 4. Predict on New Imagery
Use a trained model to predict tent locations on new satellite images. This script typically uses a separate `predict_config.yaml`.

```bash
poetry run predict-json predict_config.yaml
```
This will generate GeoJSON files containing the coordinates of predicted tents.

### 5. Evaluate and Validate
The repository includes several scripts for analyzing the results:

-   `evaluate-geojson`: Compare a prediction GeoJSON against a ground truth GeoJSON to compute metrics like precision, recall, and F1-score.
-   `validate-geojson`: Perform spatial validation by comparing rasterized prediction counts against validation counts on a master grid.
-   `merge-geojsons`: Merge multiple prediction GeoJSONs into a single, deduplicated GeoPackage file.

---
## Configuration File (config.yaml)

The pipeline is primarily configured via YAML files (`config.yaml`, `predict_config.yaml`). Below is a typical structure for `config.yaml`.

```yaml
# --- Paths ---
geotiff_dir: ${DATA_DIR}/results/TentNetFA/2026-02/tiffs
boundaries: gaza_boundaries/GazaStrip_MunicipalBoundaries.shp
prewar_gaza: ${DATA_DIR}/results/TentNetFA/2026-02/prewar_gaza.tif
geojson: ${DATA_DIR}/results/TentNetFA/2026-02
hdf5: train_data_labelling_balanced.h5
artifact_dir: runs

# --- Data Loading ---
loading:
  files:
    - deir_el_balah_20250315_121122_ssc10_u0002_visual_clip_file_format.tif
    - khan_yunis_20250315_065509_ssc13_u0001_visual_file_format.tif
processing:
  individual: false # when true, ignore hdf5 and write one HDF5 per TIFF into hdf5_folder
  step: 0.0005 # step size for each tile in degress lat and long
  quality_thresholds:
    min_valid_fraction: 0.9  # minimum fraction of the image that needs to be not black / NaN
training:
  checkpoint: null  # checkpoint to restart from, e.g. path/to/model.pth
  epochs: 10000
  batch_size: 8
  learning_rate: 0.0005
  training_frac: 0.7
  validation_frac: 0.15

# --- Data Processing ---
processing:
  prediction_only: false
  step: 0.0005 # Step size for tiling in degrees latitude/longitude
  quality_thresholds:
    start_threshold: 0.2
    max_missing_end: 1.0
    min_valid_fraction: 0.9
  complete: # List of TIFFs to process entirely, ignoring quality gates
    - deir_el_balah_20250315_121122_ssc10_u0002_visual_clip_file_format.tif

# --- Training ---
training:
  checkpoint: v1.0.pth # Optional path to a model checkpoint to resume training
  device: cuda
  epochs: 10000
  batch_size: 32
  learning_rate: 0.005
  training_frac: 0.4
  validation_frac: 0.1
  memory: True # Cache dataset in RAM for faster training
  model_kwargs:
    kernel_size: 3
```

## Prediction Pipeline

Running predictions on new satellite imagery consists of three stages:

1. Process GeoTIFF imagery and generate manifests
2. Run model inference
3. Merge prediction outputs

Example [Collab Notebook](https://colab.research.google.com/drive/1pFXIwNghzgMpnOJng027GLW6LRaRgV91?usp=sharing) to get started.

### Prerequisites

Before running predictions, ensure the following files are available locally:

* GeoTIFF files to be processed
* Gaza Strip boundaries folder including [`GazaStrip_MunicipalBoundaries.shp`](https://drive.google.com/drive/folders/1JXj-MK33lG4RQLnZLZVElYc2cQjPgdjJ?usp=sharing)
* [`prewar_gaza.tif`](https://drive.google.com/file/d/1NyOgmIBv2NqwaG5quXaASuTa3P3pO5qL/view?usp=sharing)

The GeoTIFF files can be stored in any folder structure, provided the correct paths are specified in the configuration files.

---

### Step 1: Process GeoTIFF Imagery

1. Download or copy the GeoTIFF files you want to run predictions on.
2. Store them anywhere within or outside the repository.
3. Update `config.yaml` with:

   * `geotiff_dir`: directory containing the GeoTIFF files
   * `manifest_folder`: directory where manifests should be written
   * `boundaries`: path to `GazaStrip_MunicipalBoundaries.shp`
   * `prewar_gaza`: path to `prewar_gaza.tif`

Run:

```bash
poetry run python -m displacement_tracker.b2_image_scanner config.yaml
```

This command scans all GeoTIFFs in `geotiff_dir` and generates manifests for files that do not already have a corresponding manifest in `manifest_folder`.

---

### Step 2: Run Predictions

Update `predict_config.yaml` with the correct paths:

* `geotiff_dir`
* `input_folder`
* `output_folder`

Recommended selection parameters:

```yaml
selection:
  threshold: 0.0001
  factor: 1.0
  min_area: 7
  min_distance_m: 3.0
```

Run:

```bash
poetry run python -m displacement_tracker.e_predict_json predict_config.yaml
```

This command runs inference on all imagery that has a manifest available in the configured manifest folder.

Output files are written as GeoJSON/JSON prediction files in the configured output directory.

---

### Step 3: Merge Prediction Outputs

Predictions are generated as overlapping tiles and must be merged before analysis.

Run:

```bash
poetry run -m displacement_tracker.h_merge_geojsons tif_files/historic/predictions merged.gpkg \
  --min-adj-peak 0.003 \
  --adjustment-factor 10
```

This deduplicates overlapping predictions and produces a consolidated output. Note: This will merge everything in the input folder into a single gpkg file. Only do this if the predictions in the input folder are intended to be merged and dedeuplicated into one file. To merge by date instead, see below.

#### Group predictions by date

To merge predictions separately for each acquisition date, add:

```bash
--process-by-date
```

Example:

```bash
poetry run -m displacement_tracker.h_merge_geojsons tif_files/historic/predictions ignored.gpkg \
  --process-by-date \
  --min-adj-peak 0.003 \
  --adjustment-factor 10
```

With `--process-by-date`, outputs are grouped and merged independently for each date.

Without `--process-by-date`, all predictions in the input directory are merged into a single combined output containing all detected tents.

## Output

-   **HDF5 Datasets**: The `coordinate-scanner` script produces HDF5 files containing `feature`, `prewar`, `label`, and `meta` datasets for training and prediction.
-   **Model Checkpoints**: The training script saves the best-performing model (`best_model.pth`) and dataset split information in the `runs/<timestamp>/` directory.
-   **Prediction GeoJSONs**: The prediction script generates GeoJSON files with point coordinates for each detected tent, including a `peak_value` property.
-   **Evaluation Reports**: Validation and evaluation scripts produce CSV reports and difference rasters summarizing model performance.
---

## Acknowledgments

* Developed in collaboration with Forensic Architecture, Goldsmiths, University of London.
* Satellite data provided by Planet Labs.

---

## License

[The MIT License (MIT)](./LICENSE)

---

If you have any questions or want to contribute, please open an issue or submit a pull request.

