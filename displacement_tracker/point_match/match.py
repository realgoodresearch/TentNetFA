"""Greedy 1-to-1 matching of predicted points to ground truth with a metre gate."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def match_points(
    pred_xy: np.ndarray,
    ground_truth_xy: np.ndarray,
    max_dist: float = 5.0,
) -> dict:
    """Match each prediction to at most one GT within ``max_dist`` metres.

    Coordinates must already be projected (e.g. UTM), not lon/lat degrees.
    Candidates farther than ``max_dist`` are ignored. Remaining pairs are
    claimed closest-first so each pred and each GT is used at most once.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    ground_truth_xy = np.asarray(ground_truth_xy, dtype=np.float64)
    if pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
        raise ValueError("pred_xy must have shape (N, 2)")
    if ground_truth_xy.ndim != 2 or ground_truth_xy.shape[1] != 2:
        raise ValueError("ground_truth_xy must have shape (M, 2)")

    n_pred = len(pred_xy)
    n_gt = len(ground_truth_xy)
    if n_pred == 0 or n_gt == 0:
        return _metrics(
            tp=0,
            fp=n_pred,
            fn=n_gt,
            pair_distances=np.array([], dtype=np.float64),
            matched_pairs=[],
            max_dist=max_dist,
        )

    gt_tree = cKDTree(ground_truth_xy)
    pred_tree = cKDTree(pred_xy)
    sparse = pred_tree.sparse_distance_matrix(gt_tree, max_dist)
    edges = [(dist, p_idx, g_idx) for (p_idx, g_idx), dist in sparse.items()]
    edges.sort(key=lambda x: x[0])

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    pair_distances: list[float] = []
    matched_pairs: list[tuple[int, int, float]] = []

    for dist, p_idx, g_idx in edges:
        if p_idx not in matched_preds and g_idx not in matched_gts:
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)
            matched_pairs.append((p_idx, g_idx, dist))
            pair_distances.append(dist)

    return _metrics(
        tp=len(matched_pairs),
        fp=n_pred - len(matched_pairs),
        fn=n_gt - len(matched_pairs),
        pair_distances=np.asarray(pair_distances, dtype=np.float64),
        matched_pairs=matched_pairs,
        max_dist=max_dist,
    )


def _metrics(tp, fp, fn, pair_distances, matched_pairs, max_dist) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    if pair_distances.size:
        mean_pos_error = float(np.mean(pair_distances))
        rmse_pos_error = float(np.sqrt(np.mean(pair_distances**2)))
    else:
        mean_pos_error = 0.0
        rmse_pos_error = 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_pos_error_m": mean_pos_error,
        "rmse_pos_error_m": rmse_pos_error,
        "max_dist_m": max_dist,
        "pair_distances": pair_distances,
        "matched_pairs": matched_pairs,
    }


def nearest_distances(src_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    """Distance from each ``src`` point to its nearest ``target`` (metres)."""
    src_xy = np.asarray(src_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if len(src_xy) == 0:
        return np.array([], dtype=np.float64)
    if len(target_xy) == 0:
        return np.full(len(src_xy), np.inf, dtype=np.float64)
    tree = cKDTree(target_xy)
    dist, _ = tree.query(src_xy, k=1)
    return np.asarray(dist, dtype=np.float64)
