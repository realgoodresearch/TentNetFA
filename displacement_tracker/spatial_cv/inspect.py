"""Four holdout tiles drawn through the label and detection steps."""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import ImageGrid

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.simple_cnn import SimpleCNN
from displacement_tracker.spatial_cv.points import (
    detect_tile_points,
    match_lonlat,
    row_features,
    snapshot_core_hits,
)
from displacement_tracker.util.distance import haversine_m
from displacement_tracker.util.tile_builder import create_label_from_feats

COLUMN_TITLES = (
    "Origin RGB",
    "Snapshot stamp",
    "Gaussian target",
    "Predicted density",
    "Predicted vs real label",
)

REAL_COLOUR = "#00d4ff"
PRED_COLOUR = "#ff1744"
MATCH_COLOUR = "#c6f135"


def _centroid(row: dict) -> tuple[float, float]:
    lat = 0.5 * (float(row["lat_min"]) + float(row["lat_max"]))
    lon = 0.5 * (float(row["lon_min"]) + float(row["lon_max"]))
    return lat, lon


def select_moderate_tiles(
    dataset: PairedImageDataset,
    row_indices: list[int],
    crop_pixels: int,
    n: int = 4,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Holdout tiles with tents, away from the densest ones, spread apart.

    Returns (manifest row index, core tent count). Positive tiles whose count
    sits between the 40th and 70th percentile are preferred. If that band has
    fewer than ``n`` tiles, every positive tile is eligible.
    """
    counts = []
    for idx in row_indices:
        row = dataset._rows[idx]
        n_pts = len(snapshot_core_hits(row, row_features(dataset, row), crop_pixels))
        if n_pts > 0:
            counts.append((idx, n_pts))
    if not counts:
        raise click.ClickException(
            "Holdout fold has no tiles with snapshot tents in the core."
        )
    values = np.array([c for _, c in counts], dtype=float)
    lo = float(np.percentile(values, 40))
    hi = float(np.percentile(values, 70))
    band = [(idx, c) for idx, c in counts if lo <= c <= hi]
    if len(band) >= n:
        pool = band
    else:
        click.echo(
            f"Only {len(band)} tiles between the 40th and 70th percentile "
            f"of tent counts ({lo:.0f}–{hi:.0f}); using every tile that has tents."
        )
        pool = counts
    return _farthest(dataset, pool, n, seed)


def _farthest(
    dataset: PairedImageDataset,
    pool: list[tuple[int, int]],
    n: int,
    seed: int,
) -> list[tuple[int, int]]:
    if len(pool) <= n:
        return list(pool)
    rng = random.Random(seed)
    remaining = list(pool)
    chosen = [remaining.pop(rng.randrange(len(remaining)))]
    centroids = {idx: _centroid(dataset._rows[idx]) for idx, _ in pool}
    while len(chosen) < n and remaining:
        best_i = 0
        best_d = -1.0
        for i, (idx, _) in enumerate(remaining):
            lat, lon = centroids[idx]
            dist = min(
                haversine_m(lat, lon, *centroids[other]) for other, _ in chosen
            )
            if dist > best_d:
                best_d = dist
                best_i = i
        chosen.append(remaining.pop(best_i))
    return chosen


OUTCOMES = ("TP", "TN", "FP", "FN")
DENSITY_BANDS = (0, 20, 40, 60, 80, 100)
TILES_PER_BAND = 5

OUTCOME_RULES = {
    "TP": "every snapshot tent matched and no extra detections",
    "TN": "no snapshot tents and no detections",
    "FP": "detections on a tile with no snapshot tents",
    "FN": "snapshot tents and no detections",
}

OUTCOME_TITLES = {
    "TP": "true positives",
    "TN": "true negatives",
    "FP": "false positives",
    "FN": "false negatives",
}


def tile_match_records(
    dataset: PairedImageDataset,
    tiles: list[tuple[int, list]],
    settings: dict,
    max_dist: float,
) -> list[dict]:
    """Per-tile match counts, before overlapping tiles are merged."""
    crop = int(settings["crop_pixels"])
    records = []
    for idx, detected in tiles:
        row = dataset._rows[idx]
        hits = snapshot_core_hits(row, row_features(dataset, row), crop)
        matched = match_lonlat(
            np.array([p[0] for p in detected], dtype=np.float64),
            np.array([p[1] for p in detected], dtype=np.float64),
            np.array([lon for _r, _c, lon, _lat in hits], dtype=np.float64),
            np.array([lat for _r, _c, _lon, lat in hits], dtype=np.float64),
            max_dist,
        )
        records.append(
            {
                "row_index": int(idx),
                "n_label": len(hits),
                "n_pred": len(detected),
                "tp": int(matched["tp"]),
                "fp": int(matched["fp"]),
                "fn": int(matched["fn"]),
            }
        )
    return records


def outcome_pools(records: list[dict]) -> dict[str, list[tuple[int, int]]]:
    """Tiles that are only one outcome. Mixed tiles stay out of every pool.

    The count stored with each tile is matched tents, extra detections, or
    missed tents. True negatives store 0.
    """
    pools: dict[str, list[tuple[int, int]]] = {kind: [] for kind in OUTCOMES}
    for rec in records:
        idx = int(rec["row_index"])
        n_label = int(rec["n_label"])
        n_pred = int(rec["n_pred"])
        tp = int(rec["tp"])
        fp = int(rec["fp"])
        fn = int(rec["fn"])
        if n_label == 0 and n_pred == 0:
            pools["TN"].append((idx, 0))
        elif n_label > 0 and fp == 0 and fn == 0:
            pools["TP"].append((idx, tp))
        elif n_label == 0 and fp > 0:
            pools["FP"].append((idx, fp))
        elif n_pred == 0 and fn > 0:
            pools["FN"].append((idx, fn))
    return pools


def select_density_grid(
    dataset: PairedImageDataset,
    pool: list[tuple[int, int]],
    seed: int = 42,
) -> tuple[list[tuple[int, int]], list[str] | None]:
    """25 tiles, five bands from the sparsest fifth to the densest.

    Rows run top to bottom through the 0–20, 20–40, 40–60, 60–80, and
    80–100 percent ranks. Within a row, left is sparser and right is denser.
    A pool whose counts are all the same, such as empty tiles, has no density
    scale, so the 25 tiles are only spread apart on the map.
    """
    n_bands = len(DENSITY_BANDS) - 1
    n = n_bands * TILES_PER_BAND
    if not pool:
        return [], None
    ordered = sorted(pool, key=lambda item: (item[1], item[0]))
    if ordered[0][1] == ordered[-1][1]:
        return _farthest(dataset, ordered, min(n, len(ordered)), seed), None
    chosen: list[tuple[int, int]] = []
    labels: list[str] = []
    for band in range(n_bands):
        start = int(round(band * len(ordered) / n_bands))
        end = int(round((band + 1) * len(ordered) / n_bands))
        group = ordered[start:end]
        labels.append(f"{DENSITY_BANDS[band]}–{DENSITY_BANDS[band + 1]}%")
        if len(group) <= TILES_PER_BAND:
            chosen.extend(group)
            continue
        indexes = [
            int(round(i * (len(group) - 1) / (TILES_PER_BAND - 1)))
            for i in range(TILES_PER_BAND)
        ]
        chosen.extend(group[i] for i in indexes)
    return chosen, labels


def _rgb_display(rgb: np.ndarray) -> np.ndarray:
    image = np.moveaxis(np.asarray(rgb), 0, -1).astype(np.float32)
    image = image[..., :3]
    lo = float(np.percentile(image, 2))
    hi = float(np.percentile(image, 98))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _read_raw_rgb(dataset: PairedImageDataset, row: dict) -> np.ndarray:
    raster = dataset._get_handle(row["raster_path"])
    window = ((row["r0"], row["r1"]), (row["c0"], row["c1"]))
    return raster.read([1, 2, 3], window=window)


def _predict_density(sample: dict, model: SimpleCNN, device: torch.device) -> np.ndarray:
    feature = sample["feature"].unsqueeze(0)
    prewar = sample["prewar"].unsqueeze(0)
    feats = torch.cat((feature, prewar, feature - prewar), dim=1).to(device)
    with torch.no_grad():
        return model(feats)[0, 0].detach().cpu().numpy()


def _detected_pixels(points: list[tuple], bounds: dict, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of ``interpolate_centroid`` so markers sit on the density peaks."""
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    height, width = shape
    lat_span = float(bounds["lat_max"]) - float(bounds["lat_min"])
    lon_span = float(bounds["lon_max"]) - float(bounds["lon_min"])
    rows = []
    for lat, lon, _peak, _adj in points:
        y = (float(bounds["lat_max"]) - lat) / lat_span * height
        x = (lon - float(bounds["lon_min"])) / lon_span * width
        rows.append((y, x))
    return np.asarray(rows, dtype=np.float64)


def build_panels(
    dataset: PairedImageDataset,
    chosen: list[tuple[int, int]],
    model: SimpleCNN,
    device: torch.device,
    settings: dict,
) -> list[dict]:
    panels = []
    crop = int(settings["crop_pixels"])
    for idx, n_tents in chosen:
        row = dataset._rows[idx]
        sample = dataset._materialise_row(idx)
        bounds = json.loads(sample["meta"])
        h = int(row["r1"]) - int(row["r0"])
        w = int(row["c1"]) - int(row["c0"])
        features = row_features(dataset, row)
        stamp = create_label_from_feats(
            row["lon_min"],
            row["lat_min"],
            row["lon_max"],
            row["lat_max"],
            features,
            (h, w),
        )
        gauss = sample["label"][0].numpy()
        pred = _predict_density(sample, model, device)
        detected = detect_tile_points(pred, bounds, settings)
        hits = snapshot_core_hits(row, features, crop)
        gt_rc = np.asarray([(r, c) for r, c, _lon, _lat in hits], dtype=np.float64)
        if gt_rc.size == 0:
            gt_rc = np.zeros((0, 2), dtype=np.float64)
        vmax = max(float(np.max(gauss)), float(np.max(pred)), 1e-6)
        panels.append(
            {
                "rgb": _rgb_display(_read_raw_rgb(dataset, row)),
                "stamp": stamp,
                "gauss": gauss,
                "pred": pred,
                "vmax": vmax,
                "gt_rc": gt_rc,
                "pred_rc": _detected_pixels(detected, bounds, pred.shape),
                "n_tents": n_tents,
                "row_index": idx,
            }
        )
    return panels


def _overlay_markers(ax, gt_rc: np.ndarray, pred_rc: np.ndarray) -> None:
    """Real labels as rings, predictions as crosses, each with a white halo."""
    if len(gt_rc):
        ax.scatter(
            gt_rc[:, 1],
            gt_rc[:, 0],
            s=120,
            facecolors="none",
            edgecolors="white",
            linewidths=3.0,
            zorder=3,
        )
        ax.scatter(
            gt_rc[:, 1],
            gt_rc[:, 0],
            s=86,
            facecolors="none",
            edgecolors=REAL_COLOUR,
            linewidths=1.8,
            zorder=4,
        )
    if len(pred_rc):
        ax.scatter(
            pred_rc[:, 1],
            pred_rc[:, 0],
            s=140,
            c="white",
            marker="x",
            linewidths=3.4,
            zorder=5,
        )
        ax.scatter(
            pred_rc[:, 1],
            pred_rc[:, 0],
            s=96,
            c=PRED_COLOUR,
            marker="x",
            linewidths=1.9,
            zorder=6,
        )


def save_inspect_figure(panels: list[dict], out_path: Path, fold_id: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(panels)
    fig, axes = plt.subplots(
        n_rows,
        5,
        figsize=(16.0, 3.3 * n_rows),
        squeeze=False,
    )
    for r, panel in enumerate(panels):
        axes[r, 0].imshow(panel["rgb"])
        axes[r, 1].imshow(panel["stamp"], cmap="gray", vmin=0, vmax=255)
        axes[r, 2].imshow(panel["gauss"], cmap="magma", vmin=0, vmax=panel["vmax"])
        axes[r, 3].imshow(panel["pred"], cmap="magma", vmin=0, vmax=panel["vmax"])
        axes[r, 4].imshow(panel["rgb"])
        _overlay_markers(axes[r, 4], panel["gt_rc"], panel["pred_rc"])
        axes[r, 0].set_ylabel(
            f"{panel['n_tents']} tents",
            fontsize=9,
        )
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(COLUMN_TITLES[c], fontsize=11)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=REAL_COLOUR,
            markeredgewidth=1.8,
            linestyle="None",
            markersize=10,
            label="Real label",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=PRED_COLOUR,
            markeredgewidth=2.0,
            linestyle="None",
            markersize=10,
            label="Predicted",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.0),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        f"Holdout fold {fold_id}: image, label, then detected tents",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_match_panels(
    dataset: PairedImageDataset,
    chosen: list[tuple[int, int]],
    model: SimpleCNN,
    device: torch.device,
    settings: dict,
    max_dist: float,
) -> list[dict]:
    """One panel per tile, with the same 1-to-1 match the point score uses."""
    panels = []
    crop = int(settings["crop_pixels"])
    for idx, n_tents in chosen:
        row = dataset._rows[idx]
        sample = dataset._materialise_row(idx)
        bounds = json.loads(sample["meta"])
        features = row_features(dataset, row)
        pred = _predict_density(sample, model, device)
        detected = detect_tile_points(pred, bounds, settings)
        hits = snapshot_core_hits(row, features, crop)
        gt_rc = np.asarray([(r, c) for r, c, _lon, _lat in hits], dtype=np.float64)
        if gt_rc.size == 0:
            gt_rc = np.zeros((0, 2), dtype=np.float64)
        pred_rc = _detected_pixels(detected, bounds, pred.shape)
        matched = match_lonlat(
            np.array([p[0] for p in detected], dtype=np.float64),
            np.array([p[1] for p in detected], dtype=np.float64),
            np.array([lon for _r, _c, lon, _lat in hits], dtype=np.float64),
            np.array([lat for _r, _c, _lon, lat in hits], dtype=np.float64),
            max_dist,
        )
        pairs = [(int(p), int(g)) for p, g, _dist in matched["matched_pairs"]]
        panels.append(
            {
                "rgb": _rgb_display(_read_raw_rgb(dataset, row)),
                "gt_rc": gt_rc,
                "pred_rc": pred_rc,
                "pairs": pairs,
                "tp": int(matched["tp"]),
                "fp": int(matched["fp"]),
                "fn": int(matched["fn"]),
                "n_tents": n_tents,
                "row_index": idx,
            }
        )
    return panels


def _rings(ax, rc: np.ndarray, *, halo: str, scale: float) -> None:
    if len(rc) == 0:
        return
    ax.scatter(
        rc[:, 1],
        rc[:, 0],
        s=108 * scale,
        facecolors="none",
        edgecolors=halo,
        linewidths=2.6,
        zorder=4,
    )
    ax.scatter(
        rc[:, 1],
        rc[:, 0],
        s=74 * scale,
        facecolors="none",
        edgecolors=REAL_COLOUR,
        linewidths=1.6,
        zorder=5,
    )


def _crosses(ax, rc: np.ndarray, *, halo: str, scale: float) -> None:
    if len(rc) == 0:
        return
    ax.scatter(
        rc[:, 1],
        rc[:, 0],
        s=124 * scale,
        c=halo,
        marker="x",
        linewidths=3.0,
        zorder=6,
    )
    ax.scatter(
        rc[:, 1],
        rc[:, 0],
        s=82 * scale,
        c=PRED_COLOUR,
        marker="x",
        linewidths=1.7,
        zorder=7,
    )


def _split_matched(rc: np.ndarray, matched: set[int]) -> tuple[np.ndarray, np.ndarray]:
    if len(rc) == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty
    keep = np.array([i in matched for i in range(len(rc))], dtype=bool)
    return rc[keep], rc[~keep]


def save_match_figure(
    panels: list[dict],
    out_path: Path,
    fold_id: int,
    max_dist: float,
    title: str | None = None,
    note: str | None = None,
    show_counts: bool = True,
    row_labels: list[str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    side = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / side))
    cell = 3.15 if side <= 3 else 2.85 if side == 4 else 2.5
    scale = 1.0 if side <= 3 else 0.82 if side == 4 else 0.68
    # A 2×2 grid is short, so the title and the caption need their own band.
    # Without per-tile counts the rows can sit close together.
    extra_h = 0.85 if n_rows <= 2 else 0.0
    fig = plt.figure(figsize=(cell * side, cell * n_rows + extra_h))
    if not show_counts:
        grid_rect = (0.10, 0.08, 0.88, 0.86) if row_labels else (0.012, 0.06, 0.976, 0.90)
        axes_pad = (0.05, 0.08) if row_labels else (0.05, 0.06)
    elif n_rows <= 2:
        grid_rect = (0.04, 0.14, 0.92, 0.72)
        axes_pad = (0.10, 0.36)
    else:
        grid_rect = (0.012, 0.07, 0.976, 0.88)
        axes_pad = (0.06, 0.28)
    grid = ImageGrid(
        fig,
        rect=grid_rect,
        nrows_ncols=(n_rows, side),
        ngrids=n,
        axes_pad=axes_pad,
        share_all=False,
        aspect=True,
        label_mode="1",
    )
    for i, panel in enumerate(panels):
        ax = grid[i]
        ax.imshow(panel["rgb"])
        matched_pred = {p for p, _g in panel["pairs"]}
        matched_gt = {g for _p, g in panel["pairs"]}
        for pred_i, gt_i in panel["pairs"]:
            y0, x0 = panel["gt_rc"][gt_i]
            y1, x1 = panel["pred_rc"][pred_i]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="white",
                linewidth=2.8,
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=MATCH_COLOUR,
                linewidth=1.5,
                solid_capstyle="round",
                zorder=3,
            )
        gt_hit, gt_miss = _split_matched(panel["gt_rc"], matched_gt)
        pred_hit, pred_extra = _split_matched(panel["pred_rc"], matched_pred)
        _rings(ax, gt_miss, halo="white", scale=scale)
        _crosses(ax, pred_extra, halo="white", scale=scale)
        _rings(ax, gt_hit, halo=MATCH_COLOUR, scale=scale)
        _crosses(ax, pred_hit, halo=MATCH_COLOUR, scale=scale)
        if show_counts:
            ax.set_title(
                f"{panel['tp']} matched   {panel['fn']} missed   {panel['fp']} extra",
                fontsize=8 if side >= 4 else 9,
                pad=2,
            )
        ax.set_xticks([])
        ax.set_yticks([])
    if row_labels:
        fig.canvas.draw()
        for r, label in enumerate(row_labels[:n_rows]):
            pos = grid[r * side].get_position()
            fig.text(
                pos.x0 - 0.008,
                pos.y0 + pos.height / 2,
                label,
                ha="right",
                va="center",
                fontsize=8,
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=REAL_COLOUR,
            markeredgewidth=1.8,
            linestyle="None",
            markersize=9,
            label="Missed label",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=PRED_COLOUR,
            markeredgewidth=2.0,
            linestyle="None",
            markersize=9,
            label="Extra detection",
        ),
        Line2D(
            [0],
            [0],
            color=MATCH_COLOUR,
            linewidth=2.0,
            label="Matched pair",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        title
        or f"Holdout fold {fold_id}: point match on each tile ({max_dist:g} m gate)",
        fontsize=13,
    )
    if show_counts or note:
        note_text = note or (
            "A line joins one prediction to one real label. "
            "A lime outline means that point was used in a pair. "
            "Counts are for that tile; the fold score merges overlapping tiles first."
        )
        wrap_at = 62 if side <= 2 else 140
        fig.text(
            0.5,
            0.055 if n_rows <= 2 else 0.02,
            "\n".join(textwrap.wrap(note_text, width=wrap_at)),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_class_match_figures(
    dataset: PairedImageDataset,
    tiles: list[tuple[int, list]],
    model: SimpleCNN,
    device: torch.device,
    settings: dict,
    max_dist: float,
    out_dir: Path,
    fold_id: int,
    n: int = TILES_PER_BAND * (len(DENSITY_BANDS) - 1),
) -> dict:
    """Four pure-outcome grids beside the mixed match figure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = tile_match_records(dataset, tiles, settings, max_dist)
    pools = outcome_pools(records)
    selection: dict = {
        "fold": fold_id,
        "max_dist_m": float(max_dist),
        "min_adj_peak": settings["min_adj_peak"],
        "n_holdout": len(records),
        "classes": {},
    }
    for kind in OUTCOMES:
        pool = pools[kind]
        chosen, row_labels = select_density_grid(dataset, pool)
        if n > 0 and len(chosen) > n:
            chosen = chosen[:n]
        click.echo(
            f"{kind}: {len(pool)} tiles ({OUTCOME_RULES[kind]}); drawing {len(chosen)}"
        )
        if chosen:
            click.echo(
                "  " + ", ".join(f"{idx}:{count}" for idx, count in chosen)
            )
            panels = build_match_panels(
                dataset, chosen, model, device, settings, max_dist
            )
            out_path = out_dir / f"{kind}.png"
            save_match_figure(
                panels,
                out_path,
                fold_id,
                max_dist,
                title=f"Holdout fold {fold_id}: {OUTCOME_TITLES[kind]} ({max_dist:g} m)",
                note=(
                    "Rows run from the sparsest fifth of this class at the top to the densest at the bottom. "
                    "Left to right is sparser to denser within that fifth."
                    if row_labels
                    else "Every tile in this class has the same count, so these 25 are spread across the map."
                ),
                show_counts=False,
                row_labels=row_labels,
            )
            click.echo(f"Wrote {out_path}")
        selection["classes"][kind] = {
            "rule": OUTCOME_RULES[kind],
            "n_tiles": len(pool),
            "shown": [
                {"row_index": idx, "count": count} for idx, count in chosen
            ],
        }
    note_path = out_dir / "class_tiles.json"
    note_path.write_text(json.dumps(selection, indent=2) + "\n")
    click.echo(f"Wrote {note_path}")
    return selection
