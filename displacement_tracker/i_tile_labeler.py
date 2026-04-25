"""Interactive tile labeler: show prewar/current images from an HDF5 file
and collect a user tent count per tile, storing counts and geo bounds to CSV.

Usage:
    poetry run python -m displacement_tracker.i_tile_labeler \
        path/to/tiles.hdf5 [--output labels.csv]
"""
from __future__ import annotations

import csv
import json
import os
import sys

import click

os.environ.setdefault("MPLBACKEND", "WebAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, TextBox
import h5py
import numpy as np


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) float array to uint8 (H, W, 3) via percentile clipping."""
    rgb = np.transpose(arr[:3].astype(np.float32), (1, 2, 0))
    p2 = np.percentile(rgb, 2)
    p98 = np.percentile(rgb, 98)
    span = p98 - p2
    if span < 1e-6:
        span = 1.0
    rgb = np.clip((rgb - p2) / span, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _show_tile(
    prewar: np.ndarray,
    current: np.ndarray,
    meta: dict,
    tile_idx: int,
    total: int,
) -> dict:
    """Display tile pair with interactive widgets. Returns {"action": "submit"|"skip"|"quit", "value": int|None}."""
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[8, 1], hspace=0.15)

    ax_pre    = fig.add_subplot(gs[0, :2])
    ax_cur    = fig.add_subplot(gs[0, 2:])
    ax_text   = fig.add_subplot(gs[1, 0])
    ax_submit = fig.add_subplot(gs[1, 1])
    ax_skip   = fig.add_subplot(gs[1, 2])
    ax_quit   = fig.add_subplot(gs[1, 3])

    ax_pre.imshow(_normalize_for_display(prewar))
    ax_pre.set_title("Pre-war", fontsize=12)
    ax_pre.axis("off")

    ax_cur.imshow(_normalize_for_display(current))
    ax_cur.set_title(f"Current ({meta.get('origin_date', 'unknown')})", fontsize=12)
    ax_cur.axis("off")

    lon_min = meta.get("lon_min", 0)
    lon_max = meta.get("lon_max", 0)
    lat_min = meta.get("lat_min", 0)
    lat_max = meta.get("lat_max", 0)
    fig.suptitle(
        f"Tile {tile_idx + 1} / {total}  |  {meta.get('origin_image', '')}  |  "
        f"lon [{lon_min:.4f}, {lon_max:.4f}]  lat [{lat_min:.4f}, {lat_max:.4f}]",
        fontsize=9,
    )

    textbox    = TextBox(ax_text,   "Count: ")
    btn_submit = Button(ax_submit, "Submit")
    btn_skip   = Button(ax_skip,   "Skip")
    btn_quit   = Button(ax_quit,   "Quit")

    status = fig.text(0.5, 0.005, "", ha="center", va="bottom", fontsize=9, color="red")

    result: dict = {"action": None, "value": None}

    def _try_submit(text: str) -> None:
        try:
            result["value"] = int(text.strip())
            result["action"] = "submit"
        except ValueError:
            status.set_text("Please enter a whole number.")
            fig.canvas.draw_idle()

    textbox.on_submit(_try_submit)
    btn_submit.on_clicked(lambda _: _try_submit(textbox.text))
    btn_skip.on_clicked(lambda _: result.update(action="skip"))
    btn_quit.on_clicked(lambda _: result.update(action="quit"))
    fig.canvas.mpl_connect("close_event", lambda _: result.update(action="quit"))

    fig.canvas.draw()
    plt.show(block=False)

    while result["action"] is None:
        plt.pause(0.05)

    plt.close(fig)
    return result


def _load_existing_indices(csv_path: str) -> set[int]:
    if not os.path.exists(csv_path):
        return set()
    indices: set[int] = set()
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                indices.add(int(row["tile_index"]))
            except (KeyError, ValueError):
                pass
    return indices


CSV_FIELDS = [
    "tile_index",
    "origin_image",
    "origin_date",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
    "user_count",
]


@click.command()
@click.argument("hdf5", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", default="tile_labels.csv", show_default=True, help="Output CSV path")
@click.option("--start", default=0, show_default=True, metavar="N", help="Start from tile index N")
def cli(hdf5: str, output: str, start: int) -> None:
    """Interactively label tiles from an HDF5 file, saving counts and geo bounds to CSV."""
    already_labeled = _load_existing_indices(output)
    if already_labeled:
        click.echo(f"Resuming: {len(already_labeled)} tile(s) already labeled in {output}.")

    csv_exists = os.path.exists(output)
    csv_fh = open(output, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not csv_exists:
        writer.writeheader()
        csv_fh.flush()

    try:
        with h5py.File(hdf5, "r") as hf:
            n_tiles = int(hf["feature"].shape[0])
            click.echo(f"HDF5 contains {n_tiles} tile(s).\n")

            for i in range(start, n_tiles):
                if i in already_labeled:
                    continue

                meta: dict = json.loads(hf["meta"][i])
                current: np.ndarray = hf["feature"][i]   # (3, H, W)
                prewar: np.ndarray  = hf["prewar"][i]    # (3, H, W)

                result = _show_tile(prewar, current, meta, i, n_tiles)

                if result["action"] == "quit":
                    click.echo("\nSaved and exited.")
                    sys.exit(0)

                if result["action"] == "submit":
                    writer.writerow(
                        {
                            "tile_index": i,
                            "origin_image": meta.get("origin_image", ""),
                            "origin_date": meta.get("origin_date", ""),
                            "lon_min": meta["lon_min"],
                            "lon_max": meta["lon_max"],
                            "lat_min": meta["lat_min"],
                            "lat_max": meta["lat_max"],
                            "user_count": result["value"],
                        }
                    )
                    csv_fh.flush()

    finally:
        csv_fh.close()

    click.echo(f"\nDone. Results saved to {output}.")


if __name__ == "__main__":
    cli()
