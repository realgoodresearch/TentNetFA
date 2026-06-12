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

### Prerequisites

Before running predictions, ensure the following files are available locally:

* GeoTIFF files to be processed
* `GazaStrip_MunicipalBoundaries.shp`
* `prewar_gaza.tif`

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
poetry run merge-geojsons tif_files/historic/predictions ignored.gpkg \
  --min-adj-peak 0.003 \
  --adjustment-factor 10
```

This deduplicates overlapping predictions and produces a consolidated output.

#### Group predictions by date

To merge predictions separately for each acquisition date, add:

```bash
--process-by-date
```

Example:

```bash
poetry run merge-geojsons tif_files/historic/predictions ignored.gpkg \
  --min-adj-peak 0.003 \
  --adjustment-factor 10 \
  --process-by-date
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

