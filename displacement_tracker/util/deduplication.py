import math
import time

import numpy as np
from scipy.spatial import cKDTree

from displacement_tracker.util.distance import haversine_m
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("deduplication")


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_close_points_global(flat, min_distance_m=2.0, agreement: int = 1):
    """
    Deduplicate points across all spatial clusters.

    Points have the form ``(lat, lon, peak, adjusted_peak, *provenance)``.
    Each retained cluster is represented by the original member with the
    highest adjusted peak, preserving its exact location and any provenance.
    Exact adjusted-peak ties use provenance and coordinates for a stable
    deterministic choice.
    """
    n = len(flat)
    if n == 0:
        LOGGER.info("No points provided for global deduplication.")
        return []

    start_t = time.perf_counter()
    LOGGER.info(
        "Starting global deduplication: points=%d, min_distance_m=%.3f, agreement=%d",
        n,
        float(min_distance_m),
        int(agreement),
    )

    uf = UnionFind(n)

    # Build a local metric projection so spatial indexing can cheaply find
    # candidate neighbors within min_distance_m.
    lats = np.array([pt[0] for pt in flat], dtype=np.float64)
    lons = np.array([pt[1] for pt in flat], dtype=np.float64)

    lat0_rad = math.radians(float(lats.mean()))
    meters_per_deg = (math.pi / 180.0) * 6371000.0
    x = lons * meters_per_deg * max(math.cos(lat0_rad), 1e-12)
    y = lats * meters_per_deg

    tree = cKDTree(np.column_stack((x, y)))
    LOGGER.info("Spatial index built for %d points.", n)

    # Candidate pairs from the index; verify with haversine for exactness.
    candidate_pairs = tree.query_pairs(r=max(0.0, float(min_distance_m)))
    candidate_count = len(candidate_pairs)
    LOGGER.info("Candidate neighbor pairs from KD-tree: %d", candidate_count)

    unions = 0
    if candidate_count > 0:
        pair_log_every = max(10_000, candidate_count // 10)
    else:
        pair_log_every = 10_000

    for idx, (i, j) in enumerate(candidate_pairs, start=1):
        lat_i, lon_i = flat[i][:2]
        lat_j, lon_j = flat[j][:2]
        if haversine_m(lat_i, lon_i, lat_j, lon_j) <= min_distance_m:
            uf.union(i, j)
            unions += 1

        if idx % pair_log_every == 0 or idx == candidate_count:
            LOGGER.info(
                "Pair check progress: %d/%d (%.1f%%), unions=%d",
                idx,
                candidate_count,
                (100.0 * idx / candidate_count) if candidate_count else 100.0,
                unions,
            )

    # collect clusters
    clusters = {}
    cluster_log_every = max(10_000, n // 10)
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

        checked = idx + 1
        if checked % cluster_log_every == 0 or checked == n:
            LOGGER.info(
                "Cluster assignment progress: %d/%d (%.1f%%)",
                checked,
                n,
                100.0 * checked / n,
            )

    LOGGER.info("Clusters formed: %d", len(clusters))

    # Retain a real detection rather than averaging locations across a cluster.
    merged = []
    for members in clusters.values():
        if len(members) < agreement:
            continue
        representative = min(
            members,
            key=lambda member: (
                -float(flat[member][3]),
                *(str(value) for value in flat[member][4:]),
                float(flat[member][0]),
                float(flat[member][1]),
                float(flat[member][2]),
            ),
        )
        merged.append(flat[representative])

    elapsed_s = time.perf_counter() - start_t
    LOGGER.info(
        "Deduplication complete: input=%d, output=%d, elapsed=%.2fs",
        n,
        len(merged),
        elapsed_s,
    )
    return merged
