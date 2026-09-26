"""Map spatial-CV folds on the Gaza training footprint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.warp import transform_bounds
from shapely.geometry import box

from displacement_tracker.spatial_cv.folds import ADMIN_CRS, ADMIN_NAME_FIELD, fold_summary

FOLD_COLOURS = {
    1: "#1a5276",
    2: "#2874a6",
    3: "#17a589",
    4: "#e67e22",
    5: "#c0392b",
}
VIEW_LON = (34.18, 34.62)
VIEW_LAT = (31.20, 31.62)


def _tif_footprints(tif_dir: Path) -> gpd.GeoDataFrame:
    recs = []
    if not tif_dir.is_dir():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    for path in sorted(tif_dir.glob("*.tif")):
        with rasterio.open(path) as src:
            west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        recs.append({"geometry": box(west, south, east, north), "file": path.name})
    return gpd.GeoDataFrame(recs, crs="EPSG:4326")


def _tile_points(folds: pd.DataFrame) -> gpd.GeoDataFrame:
    lon = folds["lon_c"] if "lon_c" in folds.columns else (folds["lon_min"] + folds["lon_max"]) / 2.0
    lat = folds["lat_c"]
    return gpd.GeoDataFrame(
        folds,
        geometry=gpd.points_from_xy(lon, lat),
        crs="EPSG:4326",
    )


def _style_ax(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(*VIEW_LON)
    ax.set_ylim(*VIEW_LAT)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)


def _save(fig, out_png: Path, folds: pd.DataFrame) -> None:
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(fold_summary(folds).to_string(index=False))
    print(f"Wrote {out_png}")


def plot_folds(
    folds: pd.DataFrame,
    meta: dict[str, Any],
    out_png: Path,
    boundaries: Path,
    geotiff_dir: Path | None = None,
) -> None:
    scheme = meta.get("scheme", "north_south")
    if scheme == "admin":
        _plot_admin(folds, meta, out_png, geotiff_dir)
        return
    axis = meta.get("axis", "lat")
    cuts = meta.get("cuts") or meta.get("cuts_lat") or meta.get("cuts_lon") or []
    _plot_strips(folds, list(cuts), out_png, boundaries, geotiff_dir, axis=axis)


def plot_strips(
    folds: pd.DataFrame,
    cuts_lat: list[float],
    out_png: Path,
    boundaries: Path,
    geotiff_dir: Path | None = None,
) -> None:
    """Research-area map of latitude strips (fold 1 = north)."""
    _plot_strips(folds, cuts_lat, out_png, boundaries, geotiff_dir, axis="lat")


def _plot_strips(
    folds: pd.DataFrame,
    cuts: list[float],
    out_png: Path,
    boundaries: Path,
    geotiff_dir: Path | None,
    *,
    axis: str,
) -> None:
    gaza = gpd.read_file(boundaries).to_crs("EPSG:4326")
    pts = _tile_points(folds)
    fig, ax = plt.subplots(figsize=(8.2, 10.5))
    east_west = axis == "lon"
    coord_col = "lon_c" if east_west else "lat_c"
    if east_west and coord_col not in folds.columns:
        folds = folds.copy()
        folds[coord_col] = (folds["lon_min"] + folds["lon_max"]) / 2.0

    for fold_id, colour in FOLD_COLOURS.items():
        sub = folds.loc[folds["fold_id"] == fold_id, coord_col]
        if sub.empty:
            continue
        if east_west:
            ax.axvspan(float(sub.min()), float(sub.max()), color=colour, alpha=0.08, zorder=1)
        else:
            ax.axhspan(float(sub.min()), float(sub.max()), color=colour, alpha=0.08, zorder=1)

    for cut in cuts:
        if east_west:
            ax.axvline(cut, color="#4a4a4a", linestyle="--", linewidth=0.7, zorder=3)
        else:
            ax.axhline(cut, color="#4a4a4a", linestyle="--", linewidth=0.7, zorder=3)

    gaza.boundary.plot(ax=ax, color="#2c3e50", linewidth=0.7, zorder=4)
    _plot_footprints(ax, geotiff_dir)
    _plot_points(ax, pts)

    counts = _fold_counts(folds)
    if east_west:
        title = "Spatial CV strips (1 = west, 5 = east)\nTraining tiles by centroid longitude"
        edge = {1: "west", 5: "east"}
    else:
        title = "Spatial CV strips (1 = north, 5 = south)\nTraining tiles by centroid latitude"
        edge = {1: "north", 5: "south"}
    _style_ax(ax, title)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FOLD_COLOURS[i],
            markersize=8,
            label=_fold_legend(i, counts, edge.get(i)),
        )
        for i in range(1, 6)
    ]
    handles.append(Line2D([0], [0], color="#4a4a4a", linestyle="--", linewidth=0.8, label="Quantile cuts"))
    handles.append(Patch(facecolor="#95a5a6", edgecolor="#7f8c8d", alpha=0.4, label="SkySat training footprints"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    _save(fig, out_png, folds)


def _plot_admin(
    folds: pd.DataFrame,
    meta: dict[str, Any],
    out_png: Path,
    geotiff_dir: Path | None,
) -> None:
    admin = gpd.read_file(meta["boundaries"])
    if admin.crs is None:
        admin = admin.set_crs(ADMIN_CRS)
    admin = admin.to_crs("EPSG:4326")
    names = list(meta["fold_names"])
    rank = {name: i + 1 for i, name in enumerate(names)}
    name_col = meta.get("name_field", ADMIN_NAME_FIELD)
    admin["fold_id"] = admin[name_col].map(rank)
    admin = admin.dropna(subset=["fold_id"])

    pts = _tile_points(folds)
    fig, ax = plt.subplots(figsize=(8.2, 10.5))
    _plot_footprints(ax, geotiff_dir)
    for fold_id, colour in FOLD_COLOURS.items():
        chunk = admin[admin["fold_id"] == fold_id]
        if chunk.empty:
            continue
        chunk.plot(ax=ax, facecolor=colour, edgecolor="#2c3e50", alpha=0.18, linewidth=0.8, zorder=3)
    _plot_points(ax, pts)
    _style_ax(ax, "Spatial CV by governorate\nOCHA admin boundaries, fold 1 = north")

    labels = meta.get("fold_labels") or {}
    counts = _fold_counts(folds)
    handles = [
        Patch(
            facecolor=FOLD_COLOURS[i],
            edgecolor="#2c3e50",
            alpha=0.7,
            label=_fold_legend(i, counts, labels.get(str(i)) or None),
        )
        for i in range(1, len(names) + 1)
        if i in FOLD_COLOURS
    ]
    handles.append(Patch(facecolor="#95a5a6", edgecolor="#7f8c8d", alpha=0.4, label="SkySat training footprints"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    _save(fig, out_png, folds)


def _plot_footprints(ax, geotiff_dir: Path | None) -> None:
    if geotiff_dir is None:
        return
    fp = _tif_footprints(Path(geotiff_dir))
    if len(fp):
        fp.plot(
            ax=ax,
            facecolor="#95a5a6",
            edgecolor="#7f8c8d",
            alpha=0.18,
            linewidth=0.5,
            zorder=2,
        )


def _fold_counts(folds: pd.DataFrame) -> dict[int, int]:
    return {int(fold_id): int(n) for fold_id, n in folds.groupby("fold_id").size().items()}


def plot_epoch_curves(epoch_log: Path, summary: dict[str, Any], out_png: Path) -> None:
    """Train vs inner-val scaled MSE; holdout is a single post-hoc line."""
    df = pd.read_csv(epoch_log)
    if df.empty:
        return
    best = summary.get("best_epoch")
    holdout = summary.get("holdout_loss")
    best_val = summary.get("inner_val_loss")
    late = df[df["epoch"] >= 15] if (df["epoch"].max() >= 15) else df

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.2))
    fig.patch.set_facecolor("white")
    fold = summary.get("holdout_fold")
    title = f"Fold {fold} — train vs inner-val loss" if fold else "Train vs inner-val loss"

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], color="#1a5276", lw=1.6, label="Train")
    ax.plot(df["epoch"], df["val_loss"], color="#e67e22", lw=1.6, label="Inner val")
    if best is not None:
        ax.axvline(int(best), color="#17a589", ls="--", lw=1.0, label=f"Best ckpt (epoch {int(best)})")
    ax.set_yscale("log")
    ax.set_xlim(1, float(df["epoch"].max()))
    ax.set_ylabel("Scaled MSE (log)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(late["epoch"], late["train_loss"], color="#1a5276", lw=1.6, label="Train")
    ax.plot(late["epoch"], late["val_loss"], color="#e67e22", lw=1.6, label="Inner val")
    if best is not None:
        ax.axvline(int(best), color="#17a589", ls="--", lw=1.0)
    if best_val is not None:
        ax.axhline(float(best_val), color="#17a589", ls=":", lw=1.0, label=f"Best inner val {float(best_val):.2f}")
    if holdout is not None:
        ax.axhline(float(holdout), color="#7d3c98", ls="-.", lw=1.0, label=f"Holdout (best ckpt) {float(holdout):.2f}")
    ax.set_xlim(float(late["epoch"].min()), float(df["epoch"].max()))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled MSE")
    ax.set_title(f"Epochs {int(late['epoch'].min())}–{int(df['epoch'].max())} (linear)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fold_legend(fold_id: int, counts: dict[int, int], name: str | None) -> str:
    n = f"{counts.get(fold_id, 0):,}"
    if name:
        return f"Fold {fold_id} ({name}, {n})"
    return f"Fold {fold_id} ({n})"


def _plot_points(ax, pts: gpd.GeoDataFrame) -> None:
    for fold_id, colour in FOLD_COLOURS.items():
        chunk = pts[pts["fold_id"] == fold_id]
        if chunk.empty:
            continue
        chunk.plot(ax=ax, color=colour, markersize=1.2, alpha=0.45, linewidth=0, zorder=5)
