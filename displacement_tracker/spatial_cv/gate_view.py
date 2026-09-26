"""Zoomed crops of tent pairs that sit on the match gate."""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib import patheffects as pe
from rasterio.windows import Window

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.spatial_cv.inspect import (
    MATCH_COLOUR,
    PRED_COLOUR,
    REAL_COLOUR,
    _crosses,
    _rgb_display,
    _rings,
)
from displacement_tracker.spatial_cv.points import match_lonlat
from displacement_tracker.util.distance import haversine_m

GATE_COLOUR = "#c0392b"
OUTSIDE_COLOUR = "#e67e22"
HALO = [pe.withStroke(linewidth=3, foreground="black")]


def border_pairs(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    lab_lon: np.ndarray,
    lab_lat: np.ndarray,
    gate_m: float,
    band_m: float,
) -> tuple[list[dict], list[dict]]:
    """Greedy pairs in ``(gate - band, gate]`` and ``(gate, gate + band]``.

    The wider match sorts closest-first, so every pair at or under the gate
    is the same pair the score counts. A pair just over the gate is two
    points the score leaves unmatched.
    """
    if band_m <= 0:
        raise click.ClickException("--band must be positive")
    wide = match_lonlat(pred_lat, pred_lon, lab_lon, lab_lat, gate_m + band_m)
    inside: list[dict] = []
    outside: list[dict] = []
    lo = gate_m - band_m
    for pred_i, lab_i, dist in wide["matched_pairs"]:
        dist = float(dist)
        record = {
            "pred_i": int(pred_i),
            "lab_i": int(lab_i),
            "dist_m": dist,
            "pred_lat": float(pred_lat[pred_i]),
            "pred_lon": float(pred_lon[pred_i]),
            "lab_lat": float(lab_lat[lab_i]),
            "lab_lon": float(lab_lon[lab_i]),
            "lat": float(0.5 * (pred_lat[pred_i] + lab_lat[lab_i])),
            "lon": float(0.5 * (pred_lon[pred_i] + lab_lon[lab_i])),
        }
        if lo <= dist <= gate_m:
            inside.append(record)
        elif gate_m < dist <= gate_m + band_m:
            outside.append(record)
    return inside, outside


def select_spread(
    pairs: list[dict],
    n: int,
    gate_m: float,
    min_sep_m: float = 400.0,
) -> list[dict]:
    """Pairs closest to the gate, kept far enough apart to be different places.

    The separation shrinks until ``n`` pairs fit. The result is ordered by
    distance, shorter first.
    """
    ranked = sorted(pairs, key=lambda item: (abs(item["dist_m"] - gate_m), item["dist_m"]))
    chosen: list[dict] = []
    sep = float(min_sep_m)
    while True:
        chosen = []
        for item in ranked:
            if all(
                haversine_m(item["lat"], item["lon"], other["lat"], other["lon"]) >= sep
                for other in chosen
            ):
                chosen.append(item)
            if len(chosen) == n:
                break
        if len(chosen) == n or sep <= 0:
            break
        sep *= 0.5
    chosen.sort(key=lambda item: item["dist_m"])
    return chosen


def _covering_tile(rows: list[dict], lon: float, lat: float) -> dict | None:
    best = None
    best_margin = -1.0
    for row in rows:
        lon_min = float(row["lon_min"])
        lon_max = float(row["lon_max"])
        lat_min = float(row["lat_min"])
        lat_max = float(row["lat_max"])
        if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
            continue
        margin = min(lon - lon_min, lon_max - lon, lat - lat_min, lat_max - lat)
        if margin > best_margin:
            best = row
            best_margin = margin
    return best


def _global_rc(row: dict, lon: float, lat: float) -> tuple[float, float]:
    """Continuous pixel in the source raster. Row increases south."""
    h = int(row["r1"]) - int(row["r0"])
    w = int(row["c1"]) - int(row["c0"])
    lat_span = float(row["lat_max"]) - float(row["lat_min"])
    lon_span = float(row["lon_max"]) - float(row["lon_min"])
    y = (float(row["lat_max"]) - lat) / lat_span * h
    x = (lon - float(row["lon_min"])) / lon_span * w
    return float(row["r0"]) + y, float(row["c0"]) + x


def _metres_per_pixel(row: dict, lon: float, lat: float) -> tuple[float, float]:
    h = int(row["r1"]) - int(row["r0"])
    w = int(row["c1"]) - int(row["c0"])
    north = haversine_m(float(row["lat_min"]), lon, float(row["lat_max"]), lon) / h
    east = haversine_m(lat, float(row["lon_min"]), lat, float(row["lon_max"])) / w
    return east, north


def _read_crop(
    dataset: PairedImageDataset,
    row: dict,
    lon: float,
    lat: float,
    crop_m: float,
) -> tuple[np.ndarray, int, int, float, float]:
    east_m, north_m = _metres_per_pixel(row, lon, lat)
    half_e = (crop_m / 2.0) / east_m
    half_n = (crop_m / 2.0) / north_m
    width = int(round(2 * half_e))
    height = int(round(2 * half_n))
    gr, gc = _global_rc(row, lon, lat)
    r0 = int(round(gr - height / 2.0))
    c0 = int(round(gc - width / 2.0))
    raster = dataset._get_handle(row["raster_path"])
    rgb = raster.read(
        [1, 2, 3],
        window=Window(c0, r0, width, height),
        boundless=True,
        fill_value=0,
    )
    return rgb, r0, c0, east_m, north_m


def _crop_rc_many(
    row: dict,
    lon: np.ndarray,
    lat: np.ndarray,
    r0: int,
    c0: int,
) -> np.ndarray:
    if len(lon) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    h = int(row["r1"]) - int(row["r0"])
    w = int(row["c1"]) - int(row["c0"])
    lat_span = float(row["lat_max"]) - float(row["lat_min"])
    lon_span = float(row["lon_max"]) - float(row["lon_min"])
    y = (float(row["lat_max"]) - lat) / lat_span * h + float(row["r0"]) - r0
    x = (lon - float(row["lon_min"])) / lon_span * w + float(row["c0"]) - c0
    return np.column_stack([y, x])


def _line(ax, y0, x0, y1, x1, colour: str, width: float, dashed: bool) -> None:
    style = (0, (2.2, 1.6)) if dashed else "solid"
    ax.plot(
        [x0, x1],
        [y0, y1],
        color="white",
        linewidth=width + 1.6,
        linestyle=style,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        [x0, x1],
        [y0, y1],
        color=colour,
        linewidth=width,
        linestyle=style,
        solid_capstyle="round",
        zorder=3,
    )


def _gate_circle(ax, y: float, x: float, east_m: float, north_m: float, gate_m: float) -> None:
    width = 2 * gate_m / east_m
    height = 2 * gate_m / north_m
    ax.add_patch(
        Ellipse(
            (x, y),
            width=width,
            height=height,
            fill=False,
            edgecolor="white",
            linewidth=2.4,
            linestyle=(0, (1.4, 1.5)),
            zorder=3,
        )
    )
    ax.add_patch(
        Ellipse(
            (x, y),
            width=width,
            height=height,
            fill=False,
            edgecolor=GATE_COLOUR,
            linewidth=1.3,
            linestyle=(0, (1.4, 1.5)),
            zorder=4,
        )
    )


def _save_gate_figure(
    panels: list[dict],
    out_path: Path,
    fold_id: int,
    gate_m: float,
    band_m: float,
    crop_m: float,
    n_each: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        n_each,
        figsize=(3.2 * n_each, 7.8),
        squeeze=False,
    )
    row_names = ("Inside the gate", "Outside the gate")
    for row_i, name in enumerate(row_names):
        group = [panel for panel in panels if panel["row"] == row_i]
        for col, panel in enumerate(group):
            ax = axes[row_i, col]
            ax.imshow(panel["rgb"])
            accepted = panel["accepted_rc"]
            for pred_rc, lab_rc in accepted:
                _line(ax, lab_rc[0], lab_rc[1], pred_rc[0], pred_rc[1], MATCH_COLOUR, 1.3, False)
            focal = panel["focal"]
            outside = not focal["inside"]
            _line(
                ax,
                focal["lab_rc"][0],
                focal["lab_rc"][1],
                focal["pred_rc"][0],
                focal["pred_rc"][1],
                OUTSIDE_COLOUR if outside else MATCH_COLOUR,
                2.2,
                outside,
            )
            _rings(ax, panel["gt_miss"], halo="white", scale=0.72)
            _crosses(ax, panel["pred_extra"], halo="white", scale=0.72)
            _rings(ax, panel["gt_hit"], halo=MATCH_COLOUR, scale=0.72)
            _crosses(ax, panel["pred_hit"], halo=MATCH_COLOUR, scale=0.72)
            _gate_circle(
                ax,
                focal["lab_rc"][0],
                focal["lab_rc"][1],
                panel["east_m"],
                panel["north_m"],
                gate_m,
            )
            _rings(ax, focal["lab_rc"][None, :], halo="white" if outside else MATCH_COLOUR, scale=1.45)
            _crosses(
                ax,
                focal["pred_rc"][None, :],
                halo="white" if outside else MATCH_COLOUR,
                scale=1.45,
            )
            ax.set_title(
                f"{focal['dist_m']:.3f} m",
                fontsize=11,
                pad=4,
                color="#1e8449" if focal["inside"] else GATE_COLOUR,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        for col in range(len(group), n_each):
            axes[row_i, col].axis("off")
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.15, hspace=0.28, wspace=0.05)
    fig.canvas.draw()
    for row_i, name in enumerate(row_names):
        pos = axes[row_i, 0].get_position()
        fig.text(
            pos.x0 - 0.012,
            pos.y0 + pos.height / 2,
            name.replace(" ", "\n"),
            ha="right",
            va="center",
            fontsize=10,
        )
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=REAL_COLOUR,
            markeredgewidth=1.8,
            linestyle="None",
            markersize=9,
            label="Snapshot tent",
        ),
        Line2D(
            [0], [0],
            marker="x",
            color=PRED_COLOUR,
            markeredgewidth=2.0,
            linestyle="None",
            markersize=9,
            label="Detection",
        ),
        Line2D([0], [0], color=MATCH_COLOUR, linewidth=2.0, label="Counted pair"),
        Line2D(
            [0], [0],
            color=OUTSIDE_COLOUR,
            linewidth=2.0,
            linestyle=(0, (2.2, 1.6)),
            label="Just past the gate",
        ),
        Line2D(
            [0], [0],
            color=GATE_COLOUR,
            linewidth=1.3,
            linestyle=(0, (1.4, 1.5)),
            label=f"{gate_m:g} m gate",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"Holdout fold {fold_id}: pairs within {band_m:g} m of the {gate_m:g} m gate",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.045,
        (
            f"Each panel is a {crop_m:g} m window. The circle is {gate_m:g} m around that snapshot tent. "
            "A lime line is a pair the score counts. An amber line is the nearest unused partner, "
            "just too far, so those two points are a miss and an extra."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_gate_figure(
    dataset: PairedImageDataset,
    row_indices: list[int],
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    lab_lon: np.ndarray,
    lab_lat: np.ndarray,
    out_path: Path,
    fold_id: int,
    gate_m: float = 5.0,
    band_m: float = 0.5,
    n_each: int = 4,
    crop_m: float = 40.0,
) -> Path:
    """Write the gate figure and a JSON note of the pairs it shows."""
    if n_each < 1:
        raise click.ClickException("--n-each must be at least 1")
    if crop_m <= gate_m:
        raise click.ClickException("--crop-m must be larger than the gate")
    pred_lat = np.asarray(pred_lat, dtype=np.float64).reshape(-1)
    pred_lon = np.asarray(pred_lon, dtype=np.float64).reshape(-1)
    lab_lon = np.asarray(lab_lon, dtype=np.float64).reshape(-1)
    lab_lat = np.asarray(lab_lat, dtype=np.float64).reshape(-1)
    inside, outside = border_pairs(pred_lat, pred_lon, lab_lon, lab_lat, gate_m, band_m)
    click.echo(
        f"Pairs within {band_m:g} m of the {gate_m:g} m gate: "
        f"{len(inside)} inside, {len(outside)} outside"
    )
    if len(inside) < n_each or len(outside) < n_each:
        raise click.ClickException(
            f"Need {n_each} pairs on each side of the gate; "
            f"found {len(inside)} inside and {len(outside)} outside."
        )
    rows = [dataset._rows[idx] for idx in row_indices]
    chosen_in = select_spread(_with_tiles(inside, rows), n_each, gate_m)
    chosen_out = select_spread(_with_tiles(outside, rows), n_each, gate_m)
    if len(chosen_in) < n_each or len(chosen_out) < n_each:
        raise click.ClickException(
            f"Need {n_each} pairs on each side of the gate inside a holdout tile; "
            f"found {len(chosen_in)} inside and {len(chosen_out)} outside."
        )
    counted = match_lonlat(pred_lat, pred_lon, lab_lon, lab_lat, gate_m)
    panels = []
    shown = {"inside": [], "outside": []}
    for row_i, group, key in ((0, chosen_in, "inside"), (1, chosen_out, "outside")):
        for item in group:
            panel = _build_panel(
                dataset,
                item,
                row_i,
                rows_tile=item["tile"],
                pred_lon=pred_lon,
                pred_lat=pred_lat,
                lab_lon=lab_lon,
                lab_lat=lab_lat,
                counted=counted["matched_pairs"],
                crop_m=crop_m,
            )
            panels.append(panel)
            shown[key].append(
                {
                    "dist_m": item["dist_m"],
                    "label_lon": item["lab_lon"],
                    "label_lat": item["lab_lat"],
                    "pred_lon": item["pred_lon"],
                    "pred_lat": item["pred_lat"],
                }
            )
            click.echo(f"  {key} {item['dist_m']:.3f} m")
    _save_gate_figure(panels, out_path, fold_id, gate_m, band_m, crop_m, n_each)
    note = {
        "fold": fold_id,
        "gate_m": gate_m,
        "band_m": band_m,
        "crop_m": crop_m,
        "n_inside_band": len(inside),
        "n_outside_band": len(outside),
        "shown": shown,
    }
    note_path = out_path.with_suffix(".json")
    note_path.write_text(json.dumps(note, indent=2) + "\n")
    return out_path


def _with_tiles(pairs: list[dict], rows: list[dict]) -> list[dict]:
    chosen = []
    for item in pairs:
        tile = _covering_tile(rows, item["lon"], item["lat"])
        if tile is None:
            continue
        picked = dict(item)
        picked["tile"] = tile
        chosen.append(picked)
    if not chosen:
        raise click.ClickException("No pair near the gate falls inside a holdout tile.")
    return chosen


def _build_panel(
    dataset: PairedImageDataset,
    item: dict,
    row_i: int,
    rows_tile: dict,
    pred_lon: np.ndarray,
    pred_lat: np.ndarray,
    lab_lon: np.ndarray,
    lab_lat: np.ndarray,
    counted: list,
    crop_m: float,
) -> dict:
    rgb, r0, c0, east_m, north_m = _read_crop(
        dataset, rows_tile, item["lon"], item["lat"], crop_m
    )
    height = rgb.shape[1]
    width = rgb.shape[2]
    gt_rc = _crop_rc_many(rows_tile, lab_lon, lab_lat, r0, c0)
    pred_rc = _crop_rc_many(rows_tile, pred_lon, pred_lat, r0, c0)
    inside_gt = _in_frame(gt_rc, height, width)
    inside_pred = _in_frame(pred_rc, height, width)
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    accepted_rc = []
    for pred_i, lab_i, _dist in counted:
        if pred_i == item["pred_i"] and lab_i == item["lab_i"]:
            continue
        if not (inside_pred[pred_i] and inside_gt[lab_i]):
            continue
        matched_pred.add(int(pred_i))
        matched_gt.add(int(lab_i))
        accepted_rc.append((pred_rc[pred_i], gt_rc[lab_i]))
    focal_inside = row_i == 0
    if focal_inside:
        matched_gt.add(int(item["lab_i"]))
        matched_pred.add(int(item["pred_i"]))
    gt_hit, gt_miss = _split(gt_rc, inside_gt, matched_gt)
    pred_hit, pred_extra = _split(pred_rc, inside_pred, matched_pred)
    return {
        "row": row_i,
        "rgb": _rgb_display(rgb),
        "east_m": east_m,
        "north_m": north_m,
        "accepted_rc": accepted_rc,
        "gt_hit": gt_hit,
        "gt_miss": gt_miss,
        "pred_hit": pred_hit,
        "pred_extra": pred_extra,
        "focal": {
            "inside": focal_inside,
            "dist_m": item["dist_m"],
            "lab_rc": gt_rc[item["lab_i"]],
            "pred_rc": pred_rc[item["pred_i"]],
        },
    }


def _in_frame(rc: np.ndarray, height: int, width: int) -> np.ndarray:
    if len(rc) == 0:
        return np.zeros(0, dtype=bool)
    return (rc[:, 0] >= 0) & (rc[:, 0] < height) & (rc[:, 1] >= 0) & (rc[:, 1] < width)


def _split(rc: np.ndarray, visible: np.ndarray, matched: set[int]) -> tuple[np.ndarray, np.ndarray]:
    if len(rc) == 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty
    is_hit = np.array([i in matched and visible[i] for i in range(len(rc))], dtype=bool)
    is_miss = visible & ~np.array([i in matched for i in range(len(rc))], dtype=bool)
    return rc[is_hit], rc[is_miss]
