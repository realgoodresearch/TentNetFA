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


MERGE_LOGICS = {"average_location", "max_adjusted_peak", "priority_image"}


def merge_close_points_global(
    flat,
    min_distance_m=2.0,
    agreement: int = 1,
    logic: str = "average_location",
    source_image_priorities: list[str] | None = None,
):
    """Deduplicate spatial clusters using the configured representative logic."""
    if logic not in MERGE_LOGICS:
        raise ValueError(f"Unknown merge logic: {logic}")
    priorities = list(source_image_priorities or [])
    priority_rank = {name: rank for rank, name in reversed(list(enumerate(priorities)))}
    n = len(flat)
    if n == 0:
        LOGGER.info("No points provided for global deduplication.")
        return []

    start_t = time.perf_counter()
    LOGGER.info(
        "Starting global deduplication: points=%d, min_distance_m=%.3f, agreement=%d",
        n, float(min_distance_m), int(agreement),
    )
    uf = UnionFind(n)
    lats = np.array([pt[0] for pt in flat], dtype=np.float64)
    lons = np.array([pt[1] for pt in flat], dtype=np.float64)
    lat0_rad = math.radians(float(lats.mean()))
    meters_per_deg = (math.pi / 180.0) * 6371000.0
    x = lons * meters_per_deg * max(math.cos(lat0_rad), 1e-12)
    y = lats * meters_per_deg
    tree = cKDTree(np.column_stack((x, y)))
    candidate_pairs = tree.query_pairs(r=max(0.0, float(min_distance_m)))
    for i, j in candidate_pairs:
        if haversine_m(*flat[i][:2], *flat[j][:2]) <= min_distance_m:
            uf.union(i, j)

    clusters = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)

    def stable_key(member):
        point = flat[member]
        return (*(str(value) for value in point[4:]), float(point[0]), float(point[1]), float(point[2]))

    def max_adjusted_member(members):
        return min(members, key=lambda member: (-float(flat[member][3]), *stable_key(member)))

    def max_peak_member(members):
        return min(members, key=lambda member: (-float(flat[member][2]), *stable_key(member)))

    merged = []
    for members in clusters.values():
        if len(members) < agreement:
            continue
        if logic == "average_location":
            representative = max_peak_member(members)
            count = len(members)
            merged.append((
                sum(flat[member][0] for member in members) / count,
                sum(flat[member][1] for member in members) / count,
                flat[representative][2], flat[representative][3],
                flat[representative][4] if len(flat[representative]) > 4 else None,
            ))
        elif logic == "priority_image":
            representative = min(
                members,
                key=lambda member: (
                    priority_rank.get(str(flat[member][4]), len(priorities)),
                    -float(flat[member][3]), *stable_key(member),
                ),
            )
            merged.append(flat[representative])
        else:
            merged.append(flat[max_adjusted_member(members)])

    LOGGER.info("Deduplication complete: input=%d, output=%d, elapsed=%.2fs", n, len(merged), time.perf_counter() - start_t)
    return merged
