# Master-grid tiling migration — plan

Status: proposal. Planned against `Karim` with **#28** merged and **#31** landed.
There is no test suite and no CI that runs project code
(`.github/workflows/claude-code-review.yml` is a manual review bot), so every PR
below carries explicit manual verification, and every PR leaves the train,
predict, tune and evaluation flows runnable.

## Goal

Move **all** tiling — the training scan (`b1_annotated_scanner`), the prediction
scan (`b2_image_scanner`), annotation binning, manifests and dataset splits —
onto the master grid already used by validation, tuning and evaluation. Delete
the orphaned coordinate-scanning workflow and the dead weight around it. End
with one grid abstraction, one tiling convention, and one cell-identity
convention shared by training tiles, predicted points, manual annotations and
validation counts.

## Where things actually stand

The migration is further along than it looks, which is what makes it tractable:

* `displacement_tracker/b_coordinate_scanner.py` (1,111 lines: degree-stepped
  `floor(lon/step)*step` tiling in EPSG:4326, the 3×5 neighbourhood hack at
  `:50-56`, HDF5 output) is **fully orphaned** — no importer anywhere, no
  `[tool.poetry.scripts]` entry (`pyproject.toml:48-66`), no pipeline stage
  (`pipelines/spec.py`), and it reads a flat config it can no longer parse
  (`:995-1004` wants top-level `geotiff_dir` and `processing.step`, neither of
  which exists in the sectioned `config.yaml`). Deleting it is consequence-free.
* The live scanners already tile on a **metric** lattice: centres at
  `(i·core_m, j·core_m)` anchored at each source raster's CRS origin, span
  `core_m + 2·margin_m` = 100 m, pixel-snapped per image
  (`b1_annotated_scanner.py:110-124`, `b2_image_scanner.py:189-197`,
  `util/tile_builder.py:87-108`). They are metric but **not master-grid
  aligned**, and tile identity is per-image
  (`tile_id = blake2b8(raster_path|r0|c0)`, `util/manifest_writer.py:49-54`).
* Four tiling conventions coexist: the dead degree grid, the live 70 m-stride
  metric grid, the master grid (`master_grid_100m.tif`), and the manual-eval
  random 100 m boxes. The refactor collapses these to one (plus the frozen
  legacy annotation rows, which stay honoured).

## End-state architecture

**Grid authority.** The external `master_grid_100m.tif` remains the runtime
authority for `(CRS, origin, cell size)`, mediated by a new
`displacement_tracker/util/master_grid.py`: a frozen `MasterGrid` dataclass with
`from_raster` validation, exposing `crs`, `transform`, `shape`, `cell_size`,
`cell_of_point`, `cell_centre`, `cell_bounds`, `cells_intersecting`,
`window_for_bounds`, `profile()` and `open_dataset()`. A `master-grid` CLI adds
`create` (float32 zeros, origin snapped to cell multiples) and `check`
(paste-ready YAML spec; `--against` asserts exact `(crs, transform, shape)`
equality).

Rationale: every geographically pinned artifact — the 1,248 manual annotations'
cell assignments, tuning outputs, `annotation-reference` and zonal rasters — is
implicitly pinned to the existing file. With no test suite, transcribing the
spec into config risks a silent one-cell shift. The loader plus `check` closes
the real gap (today there is **no** generator and **no** validation anywhere in
the repo) without moving authority.

**Tile geometry.** Tile core = one master-grid cell (100 m); span = cell +
2·`margin_metres` = 130 m; stride = 100 m; 260×260 px at 0.5 m/px (was
200×200 at a 70 m stride). `core_metres` is deleted — core size comes from the
grid. `margin_metres` survives as the single geometry knob.

**Emission geometry — document this, don't change it.** Because
`crop_pixels = round(margin_metres / pixel_metres)` equals the margin in pixels
(`e_predict_json.py:29-30, :458-466`), every peak inside the margin band is
discarded before emission (`:47-51, :99-103`). Emission regions therefore tile
the map *exactly*, with **zero** overlap — the 70 m core today, the 100 m cell
after migration.

Be careful to distinguish the two `agreement` knobs when writing this down, they
are not the same thing:

* `prediction.selection.agreement` (`config.yaml:121`) is the *within-raster*
  one. It is structurally inert both before and after the migration, because
  tiles of a single raster never emit into each other's territory — unless
  `selection.crop_pixels` is explicitly overridden below the margin.
* `merge.agreement` (`config.yaml:134`) is the *cross-file* one and stays
  meaningful. `merge_close_points_global` clusters points pooled from **all**
  prediction files (`h_merge_geojsons.py:260`) and drops clusters with fewer
  than `agreement` members (`util/deduplication.py:117-118`). Different
  acquisitions genuinely overlap on the ground, so `merge.agreement ≥ 2` means
  "seen in ≥ 2 overlapping *images*". Zero intra-raster tile overlap does not
  make it inert.

The shipped config uses `agreement: 1` at both sites, so nothing changes in
practice — but the docs must not claim the merge-level knob is a no-op.

**Identity.** Manifest schema v2 adds `cell_row`/`cell_col` (int32, the master
grid raster's own `rowcol()` convention — the same one already used at
`validation_core.py:80`, `i_zonal_point_sums.py:147`,
`reference_data.py:300`), plus `pixel_size` (float32) and `tile_px` (int32).
`tile_id = blake2b8(f"{origin_image}|{cell_row}|{cell_col}")` — geographic and
machine-portable, fixing the absolute-path coupling at
`manifest_writer.py:49-54`, while staying unique per row of the merged
multi-date parquet. The **cell key** `(cell_row, cell_col)` is the
cross-image/cross-date identity used by splits and evaluation joins.

**Snapping and multi-CRS.** Per-image pixel snapping is kept: cells are
enumerated in grid CRS, each centre reprojected into the raster CRS, and the
existing `world_window` (`tile_builder.py:87-108`) snaps to the nearest pixel
(≤ ¼ m offset — today's tolerance). No imagery is ever warped; the grid CRS is
an indexing CRS, not a resampling CRS. Uniform batching (which nothing enforces
today — both `custom_collate` at `d_train_cnn.py:41-54` and the default collate
in `e_predict_json` just `torch.stack`) becomes an explicit two-layer contract:
a scanner-side resolution guard, plus a `PairedImageDataset` assertion that all
rows share one `tile_px`.

## Headline decisions

1. **External file authoritative; validated loader + generator/check CLI** —
   no one-cell-shift risk to pinned artifacts; reproducibility gained without
   moving authority.
2. **Cell = core, 130 m span, 100 m stride** — one lattice for tiles and cells;
   context/border-crop semantics preserved; ~0.49× the tile count.
3. **`tile_id = blake2b8(origin_image|cell_row|cell_col)`; splits key on the
   bare cell** — row uniqueness kept, cross-date spatial leakage killed.
4. **Predict migrates before train** — predict is independently checkable
   against UNOSAT and the manual annotations before training tiling moves.
5. **Splits redesign lands before the train retile** — closes the silent
   split-corruption window rather than guarding it. Be precise about what lands
   when: PR 7 delivers *versioned, id-keyed, fail-loud* splits immediately,
   which is the corruption fix; the *leakage-free unique-cell* grouping only
   becomes effective for training once PR 8 puts cell indices on train rows.
6. **All position-affecting numeric fixes land together in one affine PR at
   retrain time** — so earlier drift gates are never confounded.
7. **Dead code deleted first; live-but-superseded code deleted only after its
   replacement is proven**, and never smuggled into a "dead code" PR.

## Regeneration story

| After PR | Invalidated | Required re-runs | Flows usable |
|---|---|---|---|
| 1–5 | nothing | none (byte-identical gates) | yes |
| 6 | predict manifests, prediction JSONs, merged gpkgs, new `best_params.yaml` | one predict + one tune run on the **existing** checkpoint (fully convolutional, loads at 260 px) | yes; train untouched |
| 7 | legacy positional `splits.csv` (refused loudly) | next `train-cnn` writes v2 splits | yes |
| 8 | train manifests, labels JSONs, `balanced.parquet` | rescan `b1` → `resample-manifest`; retrain **recommended, not required** | yes |
| 9–11 | nothing | none | yes |
| 12 | labels + decoded positions shift ≤ ~1 px; predictions, tuning | the one paid cycle: rescan → retrain → re-predict → re-tune → optional re-embed | yes |
| 13–15 | nothing | none | yes |

Never invalidated: `manual_annotation_results.csv`, both checkpoints, the
annotation GeoJSON, the prewar raster, boundaries, UNOSAT exports, and — since
the grid file itself never changes — every master-grid-side raster.

The metadata-embedding checkpoint's contract (`origin_image`, `origin_date`,
WGS84 bbox keys, `gaza_bbox` normalisation —
`util/train_metadata_embedding.py:38-47, :87-96`) is preserved throughout; PR 12
only *adds* meta keys. Bbox centres shift ≤ ~65 m, negligible against the
normalisation range (`config.yaml:82`). `SimpleCNN` is fully convolutional with
the global branch deliberately disabled (`simple_cnn.py:115-118`), so retiling
never breaks checkpoint *loadability* — only the input distribution.

---

## PR 1: Purge the dead coordinate-scanning workflow and dead code

**Scope** (~−2,100 LOC, +40; pure deletion plus trivial fixes):

* Delete `displacement_tracker/b_coordinate_scanner.py` (1,111 lines),
  `tile_labels.csv` (header-only, zero references),
  `visualization/visualise_predictions.py` (reads an HDF5 layout nothing
  produces), `visualization/visualise_training_predictions.py` (broken:
  `visualize_training_subset(hdf5_path=…)` vs the `manifest_path` parameter at
  `:28`, and a 6-channel feed into a 9-channel conv1 at `:46`),
  `visualization/visualise_geojson.py` (consumes a polygon format nothing
  emits), and `util/tiff_predictions.py`.
* Strip the `validation_tifs` plumbing from `e_predict_json.py`. **The full set
  of lines is `:23-25` (the `from …util.tiff_predictions import …` statement),
  `:123`, `:413`, `:427`, `:477`, `:493`, `:500-503`** — deleting
  `util/tiff_predictions.py` without removing the import at `:23-25` breaks the
  live predict flow on import.
* Remove keys `prediction.validation_tifs` (`config.yaml:127`) and
  `prediction.save_bounds` (`config.yaml:109`). Note `tiff_output_dir` /
  `tiff_mosaic_output` exist only as code-side defaults
  (`e_predict_json.py:501-502`), not in `config.yaml`.
* Drop the `hdf5_folder` fallbacks (`b1:366`, `b2:356`,
  `c_resample_manifest.py:35`); remove `h5py` from `pyproject.toml:23` **and
  regenerate `requirements.txt`** (`h5py==3.15.1` at `requirements.txt:26`;
  README.md:58-62 mandates
  `poetry export -f requirements.txt --output requirements.txt --without-hashes`).
* Fix the broken `train-embedding` entry point (`pyproject.toml:63` points at
  the nonexistent `util.train_embedding`) to
  `displacement_tracker.util.train_metadata_embedding:cli` — the embedding
  checkpoint is irreplaceable, so fix rather than delete.
* Docs: `README.md:21`, `:23`, `:490` (coordinate-scanner/HDF5 claims);
  **`pipelines/help.md:160`** (the `prediction.validation_tifs` row — help.md is
  rendered live by the UI at `app.py:385`); stale docstrings at
  `paired_image_dataset.py:32` and `manifest_writer.py:66`.

**Behavior change:** none for any live flow — everything removed is unreachable,
broken, or a no-op (`save_prediction_tiff` has no caller, so `validation_tifs`
only ever merged an empty directory using the fossil `/9` divisor at
`tiff_predictions.py:63-65`). `f_evaluate_geojson`'s polygon mode is *live* code
and is deliberately **not** touched here (see PR 9).

**Verification:** `grep -rn "b_coordinate_scanner\|h5py\|tiff_predictions\|save_bounds\|validation_tifs\|visualise_geojson"`
→ zero hits including docs; `poetry lock && poetry install`; a fresh
`pip install -r requirements.txt`; `annotated-scanner` and `image-scanner` each
on one TIFF with manifest row counts matching a pre-PR run; `predict-json`
end-to-end on one manifest; `poetry run train-embedding --help` imports cleanly.

**Dependencies:** none. **Size:** large deletion, trivial review.

## PR 2: Make the `manual_eval` reference type usable

**Scope** (~70 LOC): `evaluation/annotation_reference.py`,
`util/reference_data.py`, `g2_validate_geojson.py`.

* Fix the registration to the tuple contract:
  `SOURCE_TYPES["manual_eval"] = (ManualAnnotationReferenceSource, frozenset({"date","count_column","lat_column","lon_column","date_column"}))`.
  Today `annotation_reference.py:157-158` assigns a **bare class** while
  `build_reference_source` unpacks `factory, allowed = SOURCE_TYPES[t]`
  (`reference_data.py:356`) → `TypeError`.
* Delete the obsolete `ImportError` guard (`annotation_reference.py:54-61`) and
  update the docstring at `:16-30` plus `README.md:463-481`, which document
  exactly the limitation being removed.
* In `build_reference_source`, on an unknown type, lazily
  `import displacement_tracker.evaluation.annotation_reference` and retry — this
  is what makes the type reachable from config-driven flows at all.
* Add `.csv` → `manual_eval` suffix inference (`reference_data.py:319-328`).
* **Make `g2`'s `--reference-type` choices lazy.** `g2_validate_geojson.py:78`
  binds `click.Choice(sorted(SOURCE_TYPES))` at import time, i.e. before
  registration, so `--reference-type manual_eval` is rejected by Click
  regardless of the registry fix. Resolve the choice list at invocation.

**Verification:**
`validate-geojson --pred-dir <merged preds> --reference evaluation/manual_eval/manual_annotation_results.csv --reference-type manual_eval --reference-date <date> --master-grid $DATA_DIR/data/master_grid_100m.tif`
runs. Acceptance is **not** a raw raster diff against the `annotation-reference`
export: g2's `val_count` is written on a hull-cropped window
(`validation_core.py:54-61`) while the CLI writes the full grid. Compare
overlapping cells (window the export by `out_transform`) or compare summed
counts inside the hull.

**Dependencies:** none. **Size:** tiny.
**Unblocks:** manual annotations as an independent reference for PR 6's gate.

## PR 3: Config hygiene — pool keys and the inverted `exclusion_zones`

**Scope** (~90 LOC):

* Move `processing.{max_workers,max_tasks_per_child,max_pool_restarts}` from
  `train` to `predict` (`config.yaml:41-43`). They are read only by `b2`
  (`b2_image_scanner.py:349-354`); `b1`'s CLI reads none of them
  (`b1:357-361`), so today `b2` always runs on defaults. Also drop
  `Param("processing.max_workers", …)` from the TRAIN param list
  (`spec.py:181`) — `app.py:403-405` renders every Param unconditionally, so
  leaving it re-injects `processing.max_workers: 0` into every train run.
* Rename g1/g2 `exclusion_zones` → `inclusion_zones`. The current name is the
  inverse of its behaviour: g1 does `pred_gdf.clip(union)` — **keeps** points
  inside (`g1_scan_validation.py:424-426`), as does g2 (`:152-168`), whereas
  `merge.exclusion_zones_gpkg` **drops** points inside
  (`h_merge_geojsons.py:235-236`). Rename **all three** sites or tune breaks:
  `config.yaml:183`, the `Param("tuning.exclusion_zones", …)` at
  `spec.py:286`, **and `spec.py:331`'s TUNE `extra_defaults`**, which
  `runner.py:131-136` deep-merges into every prepared tune config.
* Hard-error on the old config key naming the new one. Scope the promise
  honestly: on the **config** path (`g1_scan_validation.py:363-407`) a custom
  message is implementable; renaming g2's CLI flag yields Click's stock
  "No such option".
* Docs rendered live by the UI: `help.md:131` (pool keys, currently mislabelled
  "training scanner") and `help.md:199` (`tuning.exclusion_zones`). Fix the
  `config.yaml:17` comment, which claims tiles outside boundaries are "skipped
  by every scan stage" — only `b1` crops; predict clips at point emission
  (`e_predict_json.py:323-346`). PR 10 makes it true for predict.

**Verification:** a g1 run with `inclusion_zones` reproduces the prior output
byte-identically; the old key errors clearly; `pipeline-run tune --dry-run`
resolves; a b2 run log shows the configured `max_workers`.

**Dependencies:** none. **Size:** tiny.

## PR 4: `MasterGrid` abstraction, generator/check CLI, migrate grid consumers

**Scope** (~400 LOC):

* New `util/master_grid.py`. `from_raster` validates fail-fast: **float dtype**
  (an int grid crashes `mask.mask(..., nodata=np.nan)` at
  `validation_core.py:56-61`), square/axis-aligned/north-up transform, CRS
  present and metric (warn otherwise), and warns if `cell_size != 100` while the
  `_100m` output naming is in play (`validation_core.py:227`). Expose
  `open_dataset()` explicitly — migrated call sites need a rasterio handle until
  PR 13 removes that need.
* `master-grid` CLI (new `pyproject` entry): `create --boundaries --crs
  --cell-size --output`; `check --grid [--against]`.
* Migrate grid opening at `g1_scan_validation.py:531`,
  `g2_validate_geojson.py:158`, `i_zonal_point_sums.py:140-142`,
  `annotation_reference.py:205`. `prepare_grouped_cell_inputs` keeps its dataset
  parameter; `mask.mask` internals are untouched here (PR 13 replaces them), so
  "no behavior change" stays honest. **Leave `i_zonal`'s hand-rolled int32
  `rowcol` + `np.add.at` and its int32/nodata-0 profile alone** — swapping in
  `rasterize_point_counts` returns float32 (`reference_data.py:300-307`) and
  would break the byte-identical gate.
* Config: add `shared.master_grid` (value from `config.yaml:158`); **delete**
  `tune.tuning.master_grid` with a hard error naming the new key;
  `spec.py:235-238` becomes a shared input on all three pipelines. Update
  `help.md:190`, `README.md:302`, **and the mermaid node labels naming the old
  key at `help.md:76` and `README.md:137`**.

**Behavior change:** none numerically on valid inputs. Note honestly that
fail-fast only *replaces* a crash for g1/g2; `i_zonal` and
`annotation-reference` never NaN-fill and work fine on int grids today, so for
them this is a new (correct, but new) restriction.

**Verification:** `master-grid check --grid …` prints the expected spec;
`master-grid create` + `check --against` reports identical — **and if it does
not, that is the finding**: the shipped grid is obtained out-of-band and nothing
establishes its origin is a cell multiple, so treat a mismatch as information
about the file, not a blocker. Rerun g1 on an existing merged gpkg →
`scan_summary.csv` byte-identical; rerun `annotation-reference` and
`zonal-point-sums` → **arrays** identical via `np.array_equal` (the added GDAL
provenance tags change file bytes, so checksum equality cannot hold). Confirm
the grid extent covers all current train and predict rasters; log any that
escape it.

**Dependencies:** PR 3 (spec.py churn ordering). **Size:** ~400 LOC.
**Unblocks:** PRs 5, 6, 8, 11, 13.

## PR 5: `pixel_metres` plumbing and the resolution guard

**Scope** (~90 LOC):

* Add `shared.processing.pixel_metres: 0.5` to `config.yaml`; replace the
  hardcoded `PIXEL_METRES` at `e_predict_json.py:29-30, :458-466` with the
  config value, defaulting to 0.5 when absent so legacy flat run-dir configs
  keep working (`util/config.py:79-80`). `crop_pixels = 30`,
  `nms_sigma = 22.5` are numerically unchanged.
* Guard: warn-and-skip rasters whose `|transform.a|` deviates from
  `pixel_metres`. **Tolerance must be ±0.25 %, not 1 %** — `tile_px =
  round(span_m/|transform.a|)` (`tile_builder.py:38-41`), so at a 100 m span
  only ~[0.4988, 0.5013] yields exactly 200 px; a 1 % band admits 198–202 px and
  lets the batching failure through. **Apply the guard to the scanned TIFF
  only, never to `prewar_gaza`** (`b1:243`, `b2:181`) — `_read_prewar_tile`
  deliberately sizes its window from the prewar raster's own resolution
  (`tile_builder.py:142-176`).
* Surface the knob in the UI: add a `Param` for `processing.pixel_metres` and a
  `help.md:128-132` row.

**Verification:** a predict log shows `crop_pixels=30 nms_sigma=22.50`; a
pre-PR flat run-dir config re-executes unchanged; a synthetically resampled 1 m
TIFF triggers warn-and-skip.

**Dependencies:** none strictly (PR 4 for review adjacency). **Size:** tiny.

## PR 6: Predict flow tiles on the master grid; manifest schema v2

The first behavior-change PR, deliberately predict-only.

**Scope** (~330 LOC):

* `util/manifest_writer.py`: schema v2 adds `cell_row`/`cell_col` (int32),
  `pixel_size` (float32), `tile_px` (int32) at `:22-42`; `compute_tile_id`
  becomes `blake2b8(f"{origin_image}|{cell_row}|{cell_col}")` (`:49-54`);
  Parquet metadata gains a `master_grid` provenance key with a read-time
  mismatch warning.
* **Give the new columns writer-side defaults.** `MANIFEST_COLUMNS` derives from
  the schema (`:44`) and `add_row` raises `KeyError` on any missing column
  (`:77-80`), while `b1._build_row` still emits exactly the 17 v1 columns
  (`b1:68-86`) until PR 8 — without defaults, **PR 6 breaks the train scan**.
* `util/manifest_reader.py:75-79`: treat the new columns as optional. Use
  **null**, not `-1`, for absent cell indices, and have PR 7's splitter refuse
  to group on null cells — a `-1` sentinel would collapse every v1 training tile
  into one cell group during the PR 6→8 window.
* `b2_image_scanner.py:189-197`: replace the `floor(bounds/core_m)` lattice with
  `grid.cells_intersecting(src.bounds, src.crs)` (bounds reprojected with
  `transform_bounds(..., densify_pts=21)`); per cell, centre → source CRS →
  existing `compute_tile_window(src, x, y, core_m=grid.cell_size, margin_m)`
  (`tile_builder.py:44-108` untouched). Rows carry cell indices; log the maximum
  centre-offset per raster.
* **Edge cells:** `world_window` clamps inward at raster edges
  (`tile_builder.py:104-106`), so a cell whose centre sits within ~65 m of the
  edge yields a window that no longer covers its own cell, and two adjacent edge
  cells can clamp to an identical pixel window under distinct ids. Decide
  explicitly — recommended: **drop** cells whose full span does not fit, which
  also keeps `tile_px` uniform.
* `b2` CLI (`:336-358`) requires `shared.master_grid` (added by PR 4);
  `core_metres` is accepted-and-ignored with a deprecation log, removed in PR 9.
* `PairedImageDataset.__init__`: assert all rows share one `tile_px`, naming
  offending rasters (harmless on v1 manifests where the column is absent).

**Behavior change:** predict tiles become 130 m span / 100 m stride,
grid-aligned, 260×260 px; ~0.49× the tile count. Prewar gate, valid-fraction
gate, standardisation and NMS decode are mechanically unchanged. Train flow
untouched.

**Verification** — agree the acceptance band before starting:

1. **Clear `manifests/` and `preds/` first.** Folder-mode predict globs every
   `*.parquet` (`e_predict_json.py:387-396`), merge consumes every prediction
   JSON, and `run_scans` never cleans stale files
   (`util/scan_orchestrator.py:36-63`) — stale artifacts would silently
   contaminate the go/no-go comparison.
2. Full predict pipeline twice, pre- and post-PR, same config and checkpoint.
3. Manifest spot-check: all windows 260 px square; one cell's bounds recomputed
   via `master-grid check`/gdal brackets the stored WGS84 bbox within margin +
   ½ px; two overlapping-date rasters yield identical `(cell_row, cell_col)` for
   the same ground cell.
4. `validate-geojson` on both merged outputs against UNOSAT **and**
   `manual_eval` (PR 2). **Go/no-go: rms/mae within ~10 % on both.** Rollback is
   a clean revert.
5. QGIS visual diff over 2–3 dense camps — systematic gaps along 100 m lines
   indicate an enumeration or crop bug.
6. Tune end-to-end on the new preds; `h2_merge_tuned` consumes the fresh
   `best_params.yaml`.

**Dependencies:** PRs 2, 4, 5. **Size:** ~330 LOC.
**Unblocks:** PRs 7, 8, 10; change detection becomes structurally possible.

## PR 7: Versioned, id-keyed, leakage-free splits

Lands **before** the train retile, so no window exists in which regenerated
manifests silently corrupt positional splits.

**Scope** (~200 LOC): `paired_image_dataset.create_subsets` (`:191-247`),
`d_train_cnn.py:101-116, :163-167`. Note `e_predict_json.py:166` also calls
`create_subsets` for sampled prediction — keep that path working (and note its
trap: `sample_cfg.get("enable", True)` defaults **on** when a `sample` block
exists without the key).

* `splits.csv` v2: header `version=2,fracs=…,seed=…`; subsequent lines are
  membership keys, not positional indices. Key = `tile_id`; when cell columns
  are present and non-null, assignment operates on **unique cells** so every
  date/image of a `(cell_row, cell_col)` lands in one split.
* Resolve membership by id at load; hard-error listing mismatches — **in both
  directions**: ids in splits but not the manifest, and manifest rows in no
  split (otherwise adding a date silently shrinks the training set).
* Reject legacy positional files (no version header) with a "regenerate splits"
  error. Seed the shuffle (`training.split_seed`, default 42).

**Behavior change:** splits survive manifest regeneration and reordering. State
the v1 caveat precisely — this is the only case the train flow has until PR 8:
v1 `tile_id` hashes `raster_path` (`manifest_writer.py:49-54`), so membership
survives a rescan only when the path strings are byte-identical and tile
geometry is unchanged.

**Verification:** a fresh `train-cnn` writes v2 splits; delete manifest →
rescan → resume gives identical membership; on a schema-v2 manifest assert no
cell appears in two splits; an old positional file errors cleanly.

**Dependencies:** PR 6. **Size:** ~200 LOC.

## PR 8: Train flow tiles on the master grid; delete `group_coords`

**Scope** (~250 LOC):

* `b1_annotated_scanner.py`: `_scan_complete_raster` / `_scan_grouped_tiles`
  (`:91-211`) iterate grid cells, same pattern as PR 6; `_build_row` (`:57-88`)
  emits the v2 columns. The boundaries crop (`:230-234`) is **unchanged** here —
  windows are world-anchored so it stays safe, and replacing it changes
  standardisation stats (they are computed on the cropped handle at `:238`), so
  it is deferred to the retrain cycle.
* Replace `annotations.group_coords` (`annotations.py:51-82`) with
  `MasterGrid.group_features(features, span_m=cell + 2·margin)` — the same exact
  *containment* semantics, computed once in grid CRS. Delete `group_coords`.
  Date filtering (`:27-48`), the positional `label_feature_ids` contract and
  `create_label_from_feats` are untouched.
* **Do not port `annotations.py:75-78` literally — it is a half-cell and
  axis-sign trap.** Those lines bin around lattice points at *multiples of*
  `core_m` (corner-anchored, `j` increasing northward), whereas a master-grid
  cell is *centred* at `origin + (col+0.5)·cell` and `rowcol()` rows increase
  **southward**. Derive the candidate cell range from `cell_bounds` /
  `cells_intersecting` on the feature's ± half-span box instead, and make
  `group_features` return the same `(cell_row, cell_col)` keys the scan loop
  looks up. Getting this wrong is silent in `_scan_complete_raster`, which does
  `grouped.get((i, j), [])` (`b1:121`) — a key-convention mismatch yields empty
  label lists, not an error.
* **Guard that a grouped cell actually intersects the raster.** `world_window`
  never rejects an out-of-raster centre, it clamps (`tile_builder.py:105-106`),
  and `group_coords` bins every date-filtered feature (`b1:282`) including ones
  outside this TIFF. Today that is wasted work; with canonical cell identity it
  becomes *wrong* identity. Reject cells whose centre falls outside the raster.
* Emit the v2 columns from `_build_row` (`:57-88`). Note the writer is fail-loud
  in one direction and fail-silent in the other: `add_row` raises `KeyError` on
  a missing column (`manifest_writer.py:77-81`) but
  `pa.Table.from_pylist(..., schema=MANIFEST_SCHEMA)` (`:111`) silently ignores
  extras.
* `b1` CLI requires `shared.master_grid` (added by PR 4); `core_metres`
  deprecation-logged.

**Behavior change:** training tiles become 130 m / 100 m-stride grid-aligned;
feature replication falls from ~2.05 to ~1.69 tiles per feature (the inclusion
range shrinks from 1.43 to 1.3 cells per axis; ≤ 2×2 in both worlds). All train
manifests, labels JSONs and `balanced.parquet` are invalidated — regenerable,
and scans are always full recomputes anyway (there is no resume:
`scan_orchestrator.py:36-63`). Splits are already safe (PR 7).

**Verification:**

1. **Unit-test `group_features` directly** (a scratch script, not via the
   manifest): plant features at known coordinates, assert the returned cell keys
   match `grid.cell_of_point` and that the ± half-span box membership is exact.
   A manifest-level `_idx` count is *not* a `group_features` check — a feature
   also has to survive `compute_tile_window`'s `min_valid_fraction: 0.9`
   (`config.yaml:45`, `tile_builder.py:57-67`) and the prewar gate, so the
   distinct-`_idx` totals legitimately differ pre/post even when the binning is
   correct. Cover the `complete` TIFFs too: `_scan_complete_raster` is the one
   path where a key mismatch is silent.
2. **Clear the manifest folder before rescanning.** `run_scans` rewrites only
   currently collected TIFFs and never prunes (`scan_orchestrator.py:51-53`),
   and `c_resample_manifest.py:114` calls `pa.concat_tables` **without**
   `promote_options` (unlike `manifest_reader.py:48`), so a stale v1 parquet
   beside new v2 ones fails with an opaque schema error.
3. `dataset_viewer.py` shows tents under their 3×3 stamps; `resample-manifest`
   gives sane positive/negative counts; a small CPU `train-cnn` smoke run
   decreases loss, writes a checkpoint and v2 splits, and confirms 260 px
   batching.

**Also note:** enlarging the span from 100 m to 130 m slightly *degrades* label
placement until PR 12 lands, because `create_label_from_feats` maps lon/lat
linearly across a bbox built from only the UL/LR corners
(`tile_builder.py:70-72, :123-131`), so UTM grid-convergence skew grows with
tile size. This is expected and is precisely what PR 12 fixes; call it out in
the PR description rather than discovering it in review.

**Dependencies:** PRs 6, 7. **Size:** ~250 LOC.

## PR 9: Remove the shims and the superseded evaluation mode

**Scope** (~60 added, ~250 deleted):

* Delete `shared.processing.core_metres` (`config.yaml:24`), the b1/b2
  deprecation shims, **`spec.py:101` and `:179` only** (`:102`/`:180` are
  `margin_metres`, which the plan keeps), **`README.md:275` only** (`:276` is
  `margin_metres`), and **the `help.md:128` row only** (lines 124–133 are the
  whole "Processing (tiling)" section, including the surviving `margin_metres`,
  `min_valid_fraction`, `max_workers` and `complete` rows).
* Delete `f_evaluate_geojson`'s polygon-tile mode and the `Bounds.width/height`
  back-compat parsing (`:9-17`). **Do not delete `:47-109` wholesale** — it
  contains `collect_points()` at `:90-96`, which the surviving global mode calls
  at `:257-258`; deleting the range NameErrors every run. Remove
  `collect_bounds`, the per-tile matcher, the `per_tile` report branch and the
  mode selection, keeping `collect_points`. **[BEHAVIOR — flagged; this is live
  code, which is why it gets its own PR rather than riding along with PR 1.]**

**Verification:** *before* deleting, run `evaluate-geojson` on the current GT and
confirm from the log that the polygon branch is not taken, empirically
validating the "always falls back to global" claim. After: `evaluate-geojson`
output is identical on the same inputs; `pipeline-ui` renders both pipelines.
Note the grep gate must be specific — `per_tile` substring-matches the live and
kept `prediction.per_tile_standardisation` (`config.yaml:128`,
`paired_image_dataset.py:43`, …) and `match_points_per_tile_lonlat`.

**Dependencies:** PR 8. **Size:** small.

## PR 10: `b2` skips out-of-boundary cells

Separate from PR 6 so tiler-swap drift and boundary-population change are
individually attributable.

**Scope** (~80 LOC): `b2` iterates only cells intersecting the boundaries union
(prepared geometry in grid CRS), **buffered by the tile span** so that clamped
edge windows cannot lose coverage: `world_window` clamps windows into the raster
(`tile_builder.py:105-106`), so a cell whose nominal bounds miss the union can
still emit points inside it. Update the `config.yaml:17`, README and `help.md`
boundary claims — but only for predict (see below).

**Behavior change:** fewer cells scanned. **Merged outputs are *near*-identical,
not identical** — do not promise identity. Global dedup runs **before** the
boundary clip: `merge_close_points_global` at `e_predict_json.py:316` operates
on all points, and only then does `save_geojson` drop out-of-boundary ones
(`:341-346`). Removing tiles therefore changes cluster composition near the
boundary and can move surviving points by up to `min_distance_m` (3 m). Expect
small differences in the boundary band and nothing elsewhere.

Guard the newly reachable empty case: a TIFF whose cells all miss the union now
yields a zero-row parquet (`ManifestWriter.close()` writes unconditionally), so
skip empty manifests in predict rather than dividing by zero in the sampling
path (`e_predict_json.py:161-168`).

**Verification:** an order-insensitive geometric diff, not a row diff — the
prediction DataLoader uses `shuffle=True` (`e_predict_json.py:191`), so point
*order* differs between two runs of unmodified code. Compare sorted rounded
coordinates, assert the symmetric difference lies within one `min_distance_m`
buffer of the boundary, and confirm interior points are unchanged. Disable
`prediction.sample` for the comparison. Log the tile-count reduction.

**Docs caveat:** after this PR `config.yaml:17`'s "skipped by every scan stage"
is *still* wrong for train — `b1` never skips a cell, it crops the source raster
in place. Word it per-stage.

**Dependencies:** PR 6. **Size:** tiny.

## PR 11: Evaluation and annotation workflow on the grid

The 1,248 legacy rows stay untouched and fully honoured — every consumer treats
a row as centroid + counts, and `ManualAnnotationReferenceSource` already bins by
centroid containment (`annotation_reference.py:108-128`): approximate before,
approximate after.

**Scope** (~350 LOC):

* `evaluation/manual_eval/manual_eval.py`: sample seeded random **cells** from
  grid ∩ boundary ∩ raster instead of `random.uniform` boxes (`:115-140`).
  **Adding columns is not append-compatible** — `:379-384` appends with
  `mode="a", header=False`, so 11-field rows would land under the committed
  9-field header and corrupt the file. Migrate the CSV explicitly in this PR
  (rewrite it once with the two new columns blank for the 1,248 legacy rows) and
  make the writer schema-aware. Note the cell polygon must be converted to the
  **raster** CRS before use — every downstream use in that file is raster-CRS
  (`from_bounds(*tile_geom.bounds, transform=src.transform)` at `:2xx`,
  `preds.within(tile_geom)` at `:348`). Dedup on `(cell_row, cell_col, date,
  tif_name)`, **not** the bare cell — a bare-cell key would forbid annotating
  the same cell on another date, which is exactly the cross-date comparison this
  identity scheme exists to enable. Delete the dead
  `count_predictions_in_tile` (`:143-181`, undefined global → `NameError`).
* `evaluation/scripts/add_new_model_results.py`: derive the box from
  `grid.cell_bounds(row, col)` when cell columns are present; reconstruct legacy
  rows in grid CRS (centroid ± 50 m). **`sample_tif` cannot simply be renamed**:
  it appears in `PATH_KEYS` (`run_all_analyses.py:49-59`) *and* in
  `NEW_MODEL_KEYS` (`:61`), which is both the trigger for the add-new-model
  branch (`if any(cfg.get(key) for key in NEW_MODEL_KEYS)`, `:126`) and a
  `require`d key (`:128-129`), and it is passed positionally at `:133`. Rework
  the trigger to `prediction_dir`/`new_model_column` and take the CRS from the
  master grid. There is no `--sample-tif` CLI flag today
  (`add_new_model_results` has no entry point) — if the escape hatch is wanted,
  it must be a new `analysis_config.json` key, not a flag.
* `total_error.py:16-17`: `evaluate_total_error` (`:20-28`) takes no grid
  argument and its call site passes none, so deriving `TILE_SIZE_M` from the
  grid means threading a parameter through `run_all_analyses` — include that in
  the scope or keep the constant with an assertion.
* Consider aligning membership semantics: once boxes are exactly cells,
  `sjoin(..., predicate="within")` (`add_new_model_results.py:67`) is
  closed-exclusive while every other master-grid consumer uses GDAL
  point-in-pixel floor semantics — a point on a shared edge is counted
  differently.

**Verification:** `run-evaluation` on the unmodified committed CSV before/after
→ identical outputs. Note this gate does **not** exercise the riskiest change:
the shipped `analysis_config.json:3-5` leaves `prediction_dir`/`sample_tif`/
`new_model_column` null, so `add_new_model_results` never runs. Add a second run
with those keys populated. Then a live ~5-cell annotation session →
`annotation-reference` → each new row's count lands in exactly its
`(cell_row, cell_col)` pixel; and a mixed old+new CSV through
`add_new_model_results`, cross-checked against the pre-PR reconstruction for the
legacy rows.

**Dependencies:** PRs 2, 4. **Size:** ~400 LOC.

## PR 12: Affine geo↔pixel exactness (lands with the retrain cycle)

All deferred position-affecting fixes land together, immediately before the one
paid retrain, so earlier drift gates were never confounded.

**Scope** (~300 LOC):

* `create_label_from_feats` (`tile_builder.py:113-139`): rasterise via
  WGS84 → raster CRS → inverse window transform, dropping the lon/lat-linear
  `(w−1)`/`(h−1)` map at `:130-131`.
* `paired_image_dataset.py`: meta gains **additive** keys `tile_transform`
  (6 floats) and `tile_crs`; all existing WGS84 keys preserved verbatim, so the
  embedding contract is intact.
* `e_predict_json.py`: decode NMS/centroid positions through `tile_transform` +
  pyproj; delete `util/distance.py:21-39`. **Pin the pixel-centre convention
  explicitly** — there are *two* independent half-pixel errors here, and naming
  only the `(h−1)` vs `h` divisor mismatch fixes the smaller one. Both the
  encode and decode paths must agree that integer pixel `(r, c)` means the
  **centre** of that pixel, i.e. go through `transform * (c + 0.5, r + 0.5)`.
  State it in the code as a comment; it is the kind of thing that silently
  regresses.
* `_read_prewar_tile` (`tile_builder.py:142-204`) is the last WGS84-mediated
  geometry in the pipeline: it centres the prewar window on the midpoint of the
  *stored WGS84 bbox*. Migrate it in this PR too, or the retrain bakes the
  survivor in and the claim that the systematic errors "die together" is false
  for the training input.
* `crop_pixels`/`nms_sigma` from the manifest `pixel_size`: note this cannot
  happen where it happens today. The derivation runs in `cli()`
  (`e_predict_json.py:457-467`), *before* `resolve_prediction_jobs` (`:484`) and
  before any manifest is opened, so the restructuring — moving the derivation
  into the per-job path — is part of this PR's scope, not a one-line change.
* Cache the pyproj transformer and vectorise: the decode conversion would
  otherwise be constructed inside the per-peak loop (`:98-110`), and
  `create_label_from_feats` is called once per sample.
* Update `help.md:129`, which states `margin_metres` "also drives the prediction
  crop (`crop_pixels`) and NMS sigma" — after this PR it is margin plus the
  manifest `pixel_size`.

**Keep the standardisation decision out of this PR.** Aligning train per-raster
(`d_train_cnn.py:112`, default False at `paired_image_dataset.py:43`) with the
shipped predict per-tile (`config.yaml:128`) is a real and worthwhile fix — the
current checkpoint genuinely sees differently normalised inputs at inference
than in training — but bundling it here confounds this PR's own drift gate,
which is the exact mistake headline decision 6 exists to prevent. Land it as its
own change inside the same retrain cycle, measured separately. Same reasoning
applies to the `crop_src_to_boundaries` replacement (Open decision 3).

**Verification:** a synthetic round-trip in a scratch script — feature at a known
pixel centre → rasterise → decode → assert < 0.05 px error. **Two guards must be
neutralised for this check to be meaningful**, or it returns zero peaks and a
broken implementation looks like a pass: `extract_tile_nms` discards peaks
within `crop_pixels` of the border (`:99-105`), so place the synthetic feature
well inside the core, and set the selection threshold low enough that the
synthetic delta survives. Then: predict on one TIFF shows sub-metre, spatially
unbiased shifts; `validate-geojson` before/after against both references;
retrained val metrics on the v2 leakage-free splits.

**Dependencies:** PRs 5, 6, 7, 8 (this PR consumes both the `pixel_size`
manifest column from PR 6 and `processing.pixel_metres` from PR 5 — neither
exists in the base tree). **Size:** ~350 LOC plus the operational retrain.

## PR 13: Replace `mask.mask` hull-windowing with affine arithmetic

**Scope** (~150 LOC): add `MasterGrid.window_for_bounds`; use it in
`prepare_grouped_cell_inputs` instead of
`mask.mask(src_grid, [hull], crop=True, nodata=np.nan)`
(`validation_core.py:56-61`). **Caller edits are mandatory, not optional:**
`write_output_rasters` is called with the keyword `src_grid=src_grid`
(`g1_scan_validation.py:452`, and the g2 equivalent), so both entry points
change signature with it — include them in the scope.

Three things the replacement must reproduce exactly, or "no numeric change" is
false:

* **Clamp to the grid extent.** `mask.mask` goes through rasterio's
  `geometry_window`, which intersects the rounded window with the dataset
  window. A hull extending past the grid edge therefore yields a *clamped*
  shape today; unclamped affine math would return a larger array.
* **Keep raising when the hull misses the grid.** Both callers depend on it:
  `g1_scan_validation.py:534-540` wraps the call and re-raises as
  `ClickException("Could not resolve … onto the master grid: …")`. An affine
  version that silently returns a zero-sized window turns a clear error into
  garbage metrics.
* **Derive the window with the full inverse affine**, not floor/ceil on
  `(minx, miny)` — `mask.mask` uses `dataset.window(left, bottom, right, top)`,
  which is correct for either row-direction sign. Hand-rolled arithmetic that
  assumes north-up silently inverts rows on a south-up grid.

Keep `write_output_rasters` on `src_grid.meta` semantics. The earlier rationale
here was backwards: rasterio's `.meta` is only
`{driver, dtype, nodata, width, height, count, crs, transform}` — it does *not*
carry creation options, so there is no "stray compression/blocksize" to shed;
switching to `.profile()` would *introduce* them.

**Behavior change:** none numerically, given the three points above. The real
benefit is deleting the float-dtype landmine outright rather than validating
around it, and dropping the grid file's pixel payload from the read path — not
speed: `prepare_grouped_cell_inputs` already runs once per invocation
(`g1_scan_validation.py:534`, outside the search loop), so this is not a hot
path.

**Verification:** rerun g1/g2 → `scan_summary.csv` and `validation_summary.csv`
byte-identical; numpy-diff one `diff_100m` raster (array, transform, nodata);
**plus two edge cases** — a prediction set whose hull crosses the grid edge
(shape must match the clamped `mask.mask` result) and one entirely off-grid
(must still raise the same `ClickException`).

**Dependencies:** PR 4. **Size:** small–medium.

## PR 14: Residual sweep

Items surfaced by the coupling audit that belong to no other PR:

* Unify `b1`/`b2`: both duplicate the row-dict construction (`b1:57-88` vs
  `b2:88-108`) and the prewar gate; `b2` is multiprocess with pool-restart
  machinery while `b1` is serial. After PRs 6 and 8 they differ only in label
  attachment — extract the shared cell→row path.
* Collapse the duplicate date parsers: `annotations.extract_date_from_filename`
  (`:19-24`, 8-digit token, returns `str`) vs
  `reference_data.extract_date_from_filename` (`:59-66`, regex, returns
  `datetime`).
* Fix `compute_standardisation_stats` dropping the last band unconditionally
  ("Remove last element, alpha channel", `raster_processing.py:142-143`) — a
  3-band source silently loses the blue channel's stats while `read_rgb` reads
  3 bands.
* Make `sample_cfg.get("enable", True)` (`e_predict_json.py:161`) default off,
  so a `sample:` block without `enable` cannot silently subsample a production
  run.
* Document `merge.thresholds_config`'s filename-keyed shape
  (`h_merge_geojsons.py:78-90`) — another contract that tiling changes must not
  rename.
* `b2` drops a batch's tiles when a worker raises a normal exception
  (`b2:297-302`, counted but not retried) — make it fail loudly or requeue.

**Dependencies:** PRs 8, 10. **Size:** small–medium.

## PR 15: Docs sweep and the regeneration ladder

* `README.md`: mermaid captions `:95`, `:117`, `:139`; workflow steps
  `:206-256`; shared-geometry prose `:263-276`; prediction pipeline `:334-351`;
  master-grid prose `:152`, `:302`; scope or delete the "legacy flat configs are
  still accepted by every script" overreach at `:310`.
* `pipelines/help.md`: captions `:34`, `:56`, `:78`; tiling table `:124-133`
  ("tiles = master-grid cells + margin, 130 m span, 100 m stride");
  shared-inputs table `:114-123`; `:190`.
* Document the **emission geometry** result exactly as stated in the
  architecture section above — including the `prediction.selection.agreement`
  vs `merge.agreement` distinction. Also fix the diagram captions that assert
  the old overlap premise, notably `README.md:127`
  (`"preds/*.geojson<br/>(overlapping tiles!)"`).
* Fix `help.md:131`, which labels the pool keys "Scan parallelism (training
  scanner)" — they are read only by `b2`, and PR 3 moves them to `predict`.
* Commit the regeneration-ladder table with refreshed post-PR-12 `best_params`
  reference values.
* **Decide, don't defer, the `_100m` suffix** (`validation_core.py:227`, which
  names both a directory and the file). Recommended: keep the literal string and
  document that it is a fixed label, not a resolution claim — deriving it from
  `cell_size` silently renames externally-visible artifacts.
* Replace the hardcoded personal paths in the checked-in config —
  `config.yaml:112` (`model: /home/karim/TentNetFA/runs/v2.0/best_model.pth`)
  and `config.yaml:60` — with `${DATA_DIR}`-relative values, otherwise the
  verification below is not achievable by anyone else.

**Verification:** `pipeline-run predict` and `pipeline-run tune` from a clean
run dir on the checked-in config; `pipeline-ui` renders; fresh-clone
`poetry install`. Scope the grep gate to tracked source and docs — a bare
repo-wide grep hits `poetry.lock` (`hdf5` extras strings) and this very
document, so it can never return zero.

**Dependencies:** all prior. **Size:** docs.

---

## Open decisions

1. **Promote the grid spec into config and retire the `.tif` at runtime?**
   Recommended: yes, as a follow-up after PR 15 — by then `master-grid check`
   has proven the generator reproduces the blessed file, which is exactly the
   mitigation that makes transcription safe. Keep the file in `$DATA_DIR`
   regardless, for external tooling.
2. **Grid extent vs future imagery.** Cells exist only inside the file's
   extent; imagery outside is warn-and-skipped. If extension is ever needed,
   regenerate with the same origin and cell size, extending right/down only, so
   existing `(cell_row, cell_col)` identities stay stable. Document the rule;
   do nothing until it bites.
3. **Replace `b1`'s destructive `crop_src_to_boundaries`
   (`raster_processing.py:172-301`, which overwrites the source GeoTIFF in
   place) with cell ∩ boundary filtering?** This removes a real hazard —
   re-downloading a TIFF silently invalidates every manifest for it — but
   changes per-raster standardisation stats (computed on the cropped handle,
   `b1:230-234` → `:238`) and the pixel content of boundary-straddling tiles.
   Recommended: do it **inside the PR 12 retrain cycle**, never as a standalone
   mid-sequence PR.
4. **Post-migration model policy.** After PR 6: accept drift and re-tune on the
   existing checkpoint (cheap, gated at ±10 %); retrain at PR 12 (recommended —
   260 px tiles, moved labels, and the chance to align standardisation
   regimes); re-embed metadata only if the embedding is actively used. If
   retrained recall drops (fewer training tiles, ~1.69× vs ~2.05× feature
   replication), the lever is raising `margin_metres` — decided after the first
   retrain, not preemptively.
5. **Incremental scans.** Rescans stay full recomputes; stable cell ids make
   skip-if-fresh easy later. Out of scope here.
6. **Legacy quality gates.** The old scanner's `start_threshold` /
   `max_missing_end` date-distribution gating (`b_coordinate_scanner.py:405-459`)
   never made it into `b1` and dies with PR 1. Confirm the loss is intentional —
   it has been absent from every live run already.
