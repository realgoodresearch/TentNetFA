"""Assign locked spatial folds: north–south strips, east–west strips, or governorates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import Point

N_FOLDS = 5
BUFFER_DEG = 0.01  # ~1.1 km; 100 m tiles on a cut must not sit in train and test
BUFFER_M = 1110.0  # same gap, in metres, for polygon boundaries
CUT_QUANTILES = (0.2, 0.4, 0.6, 0.8)
SCHEMES = ("north_south", "east_west", "admin")
ADMIN_CRS = "EPSG:32636"
ADMIN_NAME_FIELD = "ADM2_EN"


def _centroid_lat(frame: pd.DataFrame) -> pd.Series:
    return (frame["lat_min"].astype(float) + frame["lat_max"].astype(float)) / 2.0


def _centroid_lon(frame: pd.DataFrame) -> pd.Series:
    return (frame["lon_min"].astype(float) + frame["lon_max"].astype(float)) / 2.0


def assign_fold_ids(
    frame: pd.DataFrame,
    n_folds: int = N_FOLDS,
    buffer_deg: float = BUFFER_DEG,
    *,
    scheme: str = "north_south",
    boundaries: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add fold_id, centroids, and in_buffer.

    ``north_south``: five latitude strips, fold 1 = north.
    ``east_west``: five longitude strips, fold 1 = west.
    ``admin``: one fold per OCHA governorate, fold 1 = northernmost.
    """
    scheme = _check_scheme(scheme)
    if scheme == "admin":
        if boundaries is None:
            raise ValueError("scheme 'admin' requires a governorate boundaries file")
        return _assign_admin(frame, Path(boundaries), buffer_deg)
    if n_folds != 5:
        raise ValueError("Strip splits are defined as five quantile bands.")
    axis = "lat" if scheme == "north_south" else "lon"
    return _assign_strips(frame, scheme, axis, n_folds, buffer_deg)


def _assign_strips(
    frame: pd.DataFrame,
    scheme: str,
    axis: str,
    n_folds: int,
    buffer_deg: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    out["lat_c"] = _centroid_lat(out)
    out["lon_c"] = _centroid_lon(out)
    values = out["lat_c" if axis == "lat" else "lon_c"].to_numpy()
    cuts = np.quantile(values, CUT_QUANTILES)
    bin_idx = np.searchsorted(cuts, values, side="right")
    if axis == "lat":
        out["fold_id"] = (n_folds - bin_idx).astype(np.int32)
        labels = {1: "north", 5: "south"}
    else:
        out["fold_id"] = (bin_idx + 1).astype(np.int32)
        labels = {
            1: "west",
            2: "west-central",
            3: "central",
            4: "east-central",
            5: "east",
        }
    dist = np.min(np.abs(values[:, None] - cuts[None, :]), axis=1)
    out["in_buffer"] = dist < float(buffer_deg)
    out["fold_name"] = out["fold_id"].map(labels).fillna("")
    cuts_list = [float(c) for c in cuts]
    meta: dict[str, Any] = {
        "scheme": scheme,
        "n_folds": n_folds,
        "buffer_deg": float(buffer_deg),
        "axis": axis,
        "cut_quantiles": list(CUT_QUANTILES),
        "cuts": cuts_list,
        "fold_labels": {str(k): v for k, v in labels.items()},
        "fold_1": labels[1],
        "fold_5": labels[n_folds],
    }
    if axis == "lat":
        meta["cuts_lat"] = cuts_list
    else:
        meta["cuts_lon"] = cuts_list
    return out, meta


def _assign_admin(
    frame: pd.DataFrame,
    boundaries: Path,
    buffer_deg: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not boundaries.exists():
        raise FileNotFoundError(f"Missing admin boundaries: {boundaries}")
    admin = _load_admin_polygons(boundaries)
    out = frame.copy()
    out["lat_c"] = _centroid_lat(out)
    out["lon_c"] = _centroid_lon(out)
    points = gpd.GeoDataFrame(
        {"_row": np.arange(len(out))},
        geometry=[Point(xy) for xy in zip(out["lon_c"], out["lat_c"])],
        crs="EPSG:4326",
    )
    admin_ll = admin.to_crs("EPSG:4326")
    joined = gpd.sjoin(
        points,
        admin_ll[["fold_id", "fold_name", "geometry"]],
        how="left",
        predicate="within",
    )
    joined = joined.sort_values(["_row", "fold_id"]).drop_duplicates("_row", keep="first")
    joined = joined.set_index("_row").reindex(points["_row"].to_numpy())

    missing = joined["fold_id"].isna()
    n_nearest = int(missing.sum())
    if n_nearest:
        nearest = gpd.sjoin_nearest(
            points.loc[missing.to_numpy()].to_crs(ADMIN_CRS),
            admin.to_crs(ADMIN_CRS)[["fold_id", "fold_name", "geometry"]],
            how="left",
        )
        nearest = nearest.sort_values(["_row", "fold_id"]).drop_duplicates("_row", keep="first")
        nearest = nearest.set_index("_row")
        joined.loc[nearest.index, "fold_id"] = nearest["fold_id"]
        joined.loc[nearest.index, "fold_name"] = nearest["fold_name"]

    out["fold_id"] = joined["fold_id"].astype(np.int32).to_numpy()
    out["fold_name"] = joined["fold_name"].astype(str).to_numpy()

    points_m = points.to_crs(ADMIN_CRS)
    admin_m = admin.to_crs(ADMIN_CRS)
    boundary_of = {
        int(row.fold_id): row.geometry.boundary for row in admin_m.itertuples(index=False)
    }
    in_buffer = np.zeros(len(out), dtype=bool)
    for fold_id, boundary in boundary_of.items():
        sel = out["fold_id"].to_numpy() == fold_id
        if not sel.any():
            continue
        dist = points_m.loc[sel].geometry.distance(boundary)
        in_buffer[np.flatnonzero(sel)] = dist.to_numpy() < BUFFER_M
    out["in_buffer"] = in_buffer

    meta = {
        "scheme": "admin",
        "admin_level": "governorate",
        "n_folds": int(len(admin)),
        "buffer_deg": float(buffer_deg),
        "buffer_m": BUFFER_M,
        "boundaries": str(boundaries),
        "name_field": ADMIN_NAME_FIELD,
        "fold_names": admin.sort_values("fold_id")["fold_name"].tolist(),
        "fold_labels": {
            str(int(row.fold_id)): str(row.fold_name) for row in admin.itertuples(index=False)
        },
        "n_nearest": n_nearest,
    }
    return out, meta


def _load_admin_polygons(boundaries: Path, fold_names: list[str] | None = None) -> gpd.GeoDataFrame:
    """Governorate polygons with fold_id 1 = northernmost, unless names are locked."""
    gdf = gpd.read_file(boundaries)
    if ADMIN_NAME_FIELD not in gdf.columns:
        raise ValueError(
            f"{boundaries} has no {ADMIN_NAME_FIELD} column; expected OCHA governorates"
        )
    gdf = gdf.dissolve(by=ADMIN_NAME_FIELD, as_index=False)
    if gdf.crs is None:
        gdf = gdf.set_crs(ADMIN_CRS)
    projected = gdf.to_crs(ADMIN_CRS)
    if fold_names is None:
        order = projected.geometry.centroid.y.sort_values(ascending=False).index
        gdf = gdf.loc[order].reset_index(drop=True)
        gdf["fold_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    else:
        rank = {name: i for i, name in enumerate(fold_names)}
        unknown = sorted(set(gdf[ADMIN_NAME_FIELD]) - set(fold_names))
        if unknown or len(gdf) != len(fold_names):
            raise ValueError(
                f"Governorate names in {boundaries} do not match locked folds {fold_names}"
            )
        gdf = gdf.copy()
        gdf["_ord"] = gdf[ADMIN_NAME_FIELD].map(rank)
        gdf = gdf.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
        gdf["fold_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    gdf = gdf.rename(columns={ADMIN_NAME_FIELD: "fold_name"})
    return gdf


def _adjacent_cut_indices(holdout_fold: int, n_folds: int, *, high_is_fold_1: bool) -> list[int]:
    if holdout_fold < 1 or holdout_fold > n_folds:
        raise ValueError(f"holdout_fold must be 1..{n_folds}, got {holdout_fold}")
    bin_idx = (n_folds - holdout_fold) if high_is_fold_1 else (holdout_fold - 1)
    indices = []
    if bin_idx > 0:
        indices.append(bin_idx - 1)
    if bin_idx < n_folds - 1:
        indices.append(bin_idx)
    return indices


def adjacent_cuts(holdout_fold: int, cuts_lat: list[float]) -> np.ndarray:
    """Latitude cuts that bound the holdout strip (1=north … 5=south)."""
    cuts = np.asarray(cuts_lat, dtype=float)
    return cuts[_adjacent_cut_indices(holdout_fold, len(cuts) + 1, high_is_fold_1=True)]


def train_test_masks(
    frame: pd.DataFrame,
    holdout_fold: int,
    cuts_or_meta: list[float] | dict[str, Any],
    buffer_deg: float = BUFFER_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks: train (other folds, outside the holdout buffer) and test."""
    if isinstance(cuts_or_meta, dict):
        meta = cuts_or_meta
        scheme = meta.get("scheme", "north_south")
        buf = float(meta.get("buffer_deg", buffer_deg))
        if scheme == "admin":
            return _admin_masks(frame, holdout_fold, meta)
        cuts = meta.get("cuts") or meta.get("cuts_lat")
        if cuts is None:
            raise ValueError("Strip fold meta is missing cuts")
        axis = meta.get("axis", "lat")
        return _strip_masks(frame, holdout_fold, cuts, buf, axis)
    return _strip_masks(frame, holdout_fold, cuts_or_meta, buffer_deg, "lat")


def _strip_masks(
    frame: pd.DataFrame,
    holdout_fold: int,
    cuts: list[float],
    buffer_deg: float,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    if axis == "lat":
        coord = frame["lat_c"].to_numpy() if "lat_c" in frame.columns else _centroid_lat(frame).to_numpy()
        high_is_fold_1 = True
    else:
        coord = frame["lon_c"].to_numpy() if "lon_c" in frame.columns else _centroid_lon(frame).to_numpy()
        high_is_fold_1 = False
    fold = frame["fold_id"].to_numpy()
    test = fold == int(holdout_fold)
    adj = np.asarray(cuts, dtype=float)[
        _adjacent_cut_indices(int(holdout_fold), len(cuts) + 1, high_is_fold_1=high_is_fold_1)
    ]
    near = np.min(np.abs(coord[:, None] - adj[None, :]), axis=1) < float(buffer_deg)
    train = (~test) & (~near)
    return train, test


def _admin_masks(
    frame: pd.DataFrame,
    holdout_fold: int,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    boundaries = Path(meta["boundaries"])
    admin = _load_admin_polygons(boundaries, fold_names=list(meta["fold_names"])).to_crs(ADMIN_CRS)
    holdout_rows = admin.loc[admin["fold_id"] == int(holdout_fold)]
    if holdout_rows.empty:
        raise ValueError(f"holdout_fold {holdout_fold} is not in {meta['fold_names']}")
    holdout = holdout_rows.geometry.union_all()
    lon = frame["lon_c"].to_numpy() if "lon_c" in frame.columns else _centroid_lon(frame).to_numpy()
    lat = frame["lat_c"].to_numpy() if "lat_c" in frame.columns else _centroid_lat(frame).to_numpy()
    points = gpd.GeoSeries([Point(xy) for xy in zip(lon, lat)], crs="EPSG:4326").to_crs(ADMIN_CRS)
    dist = points.distance(holdout.boundary)
    buffer_m = float(meta.get("buffer_m", BUFFER_M))
    fold = frame["fold_id"].to_numpy()
    test = fold == int(holdout_fold)
    train = (~test) & (dist.to_numpy() >= buffer_m)
    return train, test


def fold_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, grp in frame.groupby("fold_id"):
        images = (
            grp["origin_image"].astype(str).value_counts().head(3).index.tolist()
            if "origin_image" in grp.columns
            else []
        )
        name = ""
        if "fold_name" in grp.columns:
            names = grp["fold_name"].dropna().astype(str)
            names = names[names != ""]
            name = names.mode().iloc[0] if len(names) else ""
        rows.append(
            {
                "fold_id": int(fold_id),
                "fold_name": name,
                "n_tiles": int(len(grp)),
                "n_buffer": int(grp["in_buffer"].sum()) if "in_buffer" in grp.columns else 0,
                "lat_min": float(grp["lat_c"].min()),
                "lat_max": float(grp["lat_c"].max()),
                "lon_min": float(grp["lon_c"].min()) if "lon_c" in grp.columns else float("nan"),
                "lon_max": float(grp["lon_c"].max()) if "lon_c" in grp.columns else float("nan"),
                "origin_image": "; ".join(images),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_id")


def load_manifest_frame(path: str | Path) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


def write_folds(
    frame: pd.DataFrame,
    meta: dict[str, Any],
    out_parquet: Path,
) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        c
        for c in (
            "tile_id",
            "origin_image",
            "origin_date",
            "lon_min",
            "lon_max",
            "lat_min",
            "lat_max",
            "lat_c",
            "lon_c",
            "fold_id",
            "fold_name",
            "in_buffer",
        )
        if c in frame.columns
    ]
    frame[keep].to_parquet(out_parquet, index=False)
    meta_path = out_parquet.with_name("folds_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _check_scheme(scheme: str) -> str:
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    return scheme
