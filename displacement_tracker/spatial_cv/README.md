# Spatial CV (SimpleCNN)

Latitude / longitude strips or governorates as **train/test geography**.
Scoring is snapshot-label density MSE, not UNOSAT. Point matching lives in
`displacement_tracker.point_match`.

`python -m displacement_tracker.d_train_cnn config.yaml` is unchanged: random
80/19 and the v2.2 checkpoint. Spatial runs go through this package and
**always train from scratch**.

## One command

Uses `train.spatial_cv` in `config.yaml` (scheme, epoch budget, patience).
Assigns folds if missing, writes the strip map, then trains. Finished folds
(`cv_fold_summary.json`) are skipped.

```bash
# Hold out the middle north–south strip only
python -m displacement_tracker.spatial_cv run config.yaml --folds 3

# Full 5-fold at the same budget (fold 3 is skipped if already done)
python -m displacement_tracker.spatial_cv run config.yaml
```

Poetry: `poetry run spatial-cv run config.yaml --folds 3`

## Locked settings (fold-3 pilot, 2026-09-22)

| | |
|---|---|
| Data | `$DATA_DIR/data/training_data/balanced.parquet` (all positives, 25% empty tiles). This step does **not** resample. |
| Labels | FA snapshot tents → Gaussian σ=3 density maps |
| Scheme | `north_south`: fold 1 = north … 5 = south; cuts = 20/40/60/80% lat quantiles |
| Buffer | 0.01° (~1 km) on holdout boundaries; those tiles dropped from **train** only |
| Inner val | 10% random of the train strips (not the test strip), seed 42 |
| Model | SimpleCNN 9→1, `kernel_size` 3, no dropout, Adam lr 1e-4, `weight_decay` 0 |
| Loss | `1e6 * MSE` |
| Batch / cache | 64, `memory: True` |
| Init | from scratch (`training.checkpoint` ignored) |
| Budget | max **83** epochs, early-stop **patience 30** on inner val (best was epoch 53) |
| Device | CUDA |

Fold 3 took ~3 h 40 min, including ~12 min RAM cache (RTX 2000 Ada). Expect a similar cost per remaining fold.

## Steps (if you do not use `run`)

```bash
python -m displacement_tracker.spatial_cv assign-folds config.yaml   # --scheme north_south
python -m displacement_tracker.spatial_cv plot-strips config.yaml
python -m displacement_tracker.spatial_cv train-folds config.yaml --folds 3
# or one fold via the trainer:
python -m displacement_tracker.d_train_cnn config.yaml \
  --holdout-fold 3 \
  --spatial-folds-path "$DATA_DIR/results/TentNetFA/evaluation/spatial_cv/north_south/tiles_with_folds.parquet" \
  --epochs 83 --early-stop-patience 30
```

`--force` retrains a fold that already has a summary. An incomplete dir
(no `cv_fold_summary.json`) is trained from scratch.

Other schemes: `--scheme east_west` or `--scheme admin`.

## Artifacts

Folds (do not reshuffle once written):

`$DATA_DIR/results/TentNetFA/evaluation/spatial_cv/<scheme>/tiles_with_folds.parquet`

Each train: `runs/spatial_cv/fold{K}/` (`north_south`) or
`runs/spatial_cv/<scheme>/fold{K}/`

- `best_model.pth` — inner-val best
- `epoch_log.csv`, `epoch_curves.png`
- `cv_fold_summary.json` — holdout MSE on the best checkpoint
- `splits.csv`, `fold_info.json`

After `train-folds` / `run`: `<scheme>/cv_summary.csv`.

## Point score

Density MSE stays the training score. Once a fold has `best_model.pth`, this
matches detected points to the snapshot tents (not UNOSAT):

```bash
python -m displacement_tracker.spatial_cv score-points config.yaml --folds 3
```

Overlapping tiles are merged first. The match gate is 5 m. Tents in the 15 m
tile margin are left out, on both the snapshot and the predicted peaks.
Detection uses the predict-flow NMS settings and the merge peak filter
(`min_adj_peak`, `adjustment_factor`).

Figures under `fold{K}/score_points/`:

- `inspect_matches.png` — example tiles. Counts on each tile are for that tile only. A cyan ring is a miss, a red cross is an extra, a lime line is one accepted pair.
- `TP.png`, `TN.png`, `FP.png`, `FN.png` — 25 tiles that are only that one outcome, before tiles are merged. Mixed tiles are left out. Rows are the 0–20, 20–40, 40–60, 60–80, and 80–100 percent ranks of that class's count, sparsest at the top. Empty tiles have no count range, so those 25 are only spread across the map.
- `offset_hist.png` — nearest-neighbour distances, the match bars, and the confusion matrix for the whole fold.
- `gate_5m.png` — pairs within 0.5 m of the 5 m gate. Each panel is a 40 m window. The dashed circle is 5 m around the snapshot tent. The top row is inside the gate, so the score counts the pair. The bottom row is just past it, so those two points are a miss and an extra.

Also writes `holdout_point_summary.json`, `holdout_pred_points.gpkg`,
`holdout_label_points.gpkg`, and `<scheme>/cv_point_summary.csv`.
A fold with no `best_model.pth` is an error.
`--n-tiles` 9, 16, or 25 changes the example grid (default 16). `0` skips it.
`--min-adj-peak 0.0038` writes `score_points_0.0038/` and leaves the default score in place.

## Tile figure

```bash
python -m displacement_tracker.spatial_cv inspect-tiles config.yaml --fold 3
```

Four holdout tiles that contain tents but are not the densest (snapshot count
between the 40th and 70th percentile, then spread apart). Columns, left to
right:

1. Origin RGB, before standardisation.
2. Snapshot stamp, a 3×3 block at each tent. This column still shows margin tents.
3. Gaussian target, that stamp blurred with σ=3. This is the training label.
4. Predicted density, on the same colour scale.
5. Origin RGB with snapshot tents as cyan rings and predictions as red crosses. Both skip the tile margin, same as the point score.

Saved to `runs/spatial_cv/fold{K}/inspect_tiles.png`.

`inspect-matches` redraws the example grid without scoring again:

```bash
python -m displacement_tracker.spatial_cv inspect-matches config.yaml --fold 3 --n-tiles 9
```

`inspect-gate` redraws the 5 m gate crops from the points already written:

```bash
python -m displacement_tracker.spatial_cv inspect-gate config.yaml --fold 3
python -m displacement_tracker.spatial_cv inspect-gate config.yaml --fold 3 --min-adj-peak 0.0038
```

## 100 m grid count

`score-grid` bins the points already written by `score-points`. It does not
reload the checkpoint. `score-points` writes this folder itself, so run the
command only to redo it.

```bash
python -m displacement_tracker.spatial_cv score-grid config.yaml --folds 3
```

A **core** is the inner 70 m of a tile. The 15 m margin is not counted.
A **cell** is a 100 m square that touches a holdout core, including squares
where both counts are zero.

Writes `fold{K}/score_grid/` (`cells.csv`, `holdout_grid_summary.json`).
`--min-adj-peak 0.0038` reads `score_points_0.0038/` and writes `score_grid_0.0038/`.

- `count_scatter.png` — predicted vs snapshot count, with a 1:1 line. Panel A is every cell. Panel B is cells where both counts are above zero, zoomed to 0–20.
- `residual_hist.png` — predicted count minus snapshot count, zoomed to −5…30 on x and 0…250 cells on y. Each bar is labelled with its cell count. Bars taller than 250 are cut off and the count is written inside the bar.
- `residual_map.png` — cells coloured by that residual, centred at 0.

In `cells.csv`, `label_count` is the snapshot, `pred_count` is the model, and `residual` is predicted minus snapshot.

| Field | Meaning |
|---|---|
| `n_label`, `n_pred` | Snapshot tents and model tents inside the cells. |
| `bias` | `n_pred` minus `n_label`. Positive means the model counted more tents in total. That is not a false-positive count. 0 means the totals match. |
| `ratio` | `n_pred` divided by `n_label`. 1 means the totals match. |
| `mae`, `median_ae` | Average and middle of \|predicted − snapshot\| per cell, in tents. 0 means a cell's two counts match. Lower means the typical cell is nearer the snapshot count. |
| `n_cells_either_positive` | Cells where the snapshot or the model counted at least one tent. |
| `mae_either_positive`, `median_ae_either_positive` | Same absolute miss, only on those cells. |
| `pearson_r` | Whether the two counts rise and fall together (−1 to 1). 1 is perfect. |
| `concordance` | Whether the pairs sit on predicted = snapshot (−1 to 1). 1 is perfect. This can be lower than `pearson_r` when the counts move together but are not the same number. |
| `presence.both_zero` | Empty in the snapshot and in the model. |
| `presence.both_positive` | Both counted at least one tent. |
| `presence.pred_only` | Model counted tents, snapshot did not. |
| `presence.label_only` | Snapshot counted tents, model did not. |
