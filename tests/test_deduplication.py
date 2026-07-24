"""Tests for displacement_tracker/util/deduplication.py and util/distance.py.

merge_close_points_global semantics under test:
  - points are unioned when haversine distance <= min_distance_m (inclusive),
  - merging is transitive via union-find (chains collapse into one cluster),
  - the merged point is the arithmetic mean of member lats/lons,
  - the merged peak is the max peak in the cluster and the merged adjusted
    peak is the adjusted peak of that max-peak member (not the max adjusted
    peak),
  - clusters with fewer than `agreement` members are dropped.

Geometry helper: along a meridian (same longitude) the haversine distance is
exactly R * dphi, so a latitude offset of d meters is d * 180 / (pi * R)
degrees. At base latitude 0 these constructions are float-exact.
"""

import math

import pytest

from displacement_tracker.util.deduplication import UnionFind, merge_close_points_global
from displacement_tracker.util.distance import haversine_m, interpolate_centroid

EARTH_RADIUS_M = 6371000.0
DEG_PER_M = 180.0 / (math.pi * EARTH_RADIUS_M)


def test_merge_empty_input_returns_empty_list():
    # Given: no input points
    flat = []

    # When: merge_close_points_global runs
    result = merge_close_points_global(flat, min_distance_m=2.0)

    # Then: it returns an empty list
    assert result == []


def test_merge_single_point_is_returned_unchanged():
    # Given: a single point and the default agreement of 1
    flat = [(12.5, 34.5, 0.8, 0.6)]

    # When: merge_close_points_global runs
    result = merge_close_points_global(flat, min_distance_m=2.0)

    # Then: the point comes back as its own cluster with identical values
    assert len(result) == 1
    lat, lon, peak, adj = result[0]
    assert lat == pytest.approx(12.5, abs=1e-12)
    assert lon == pytest.approx(34.5, abs=1e-12)
    assert peak == pytest.approx(0.8, abs=1e-12)
    assert adj == pytest.approx(0.6, abs=1e-12)


def test_merge_two_points_within_threshold_averages_position():
    # Given: two points 1 m apart along a meridian, min_distance_m=2
    dlat = 1.0 * DEG_PER_M
    flat = [(0.0, 10.0, 0.5, 0.7), (dlat, 10.0, 0.9, 0.3)]

    # When: merge_close_points_global runs
    result = merge_close_points_global(flat, min_distance_m=2.0)

    # Then: they collapse to one point at the mean latitude and shared longitude
    assert len(result) == 1
    lat, lon, _, _ = result[0]
    assert lat == pytest.approx(dlat / 2.0, abs=1e-15)
    assert lon == pytest.approx(10.0, abs=1e-12)


def test_merged_adjusted_peak_follows_the_max_peak_member():
    # Given: two mergeable points where the max-peak member (peak=0.9) carries
    #        the SMALLER adjusted peak (0.3) and the other carries adj=0.7
    dlat = 1.0 * DEG_PER_M
    flat = [(0.0, 10.0, 0.5, 0.7), (dlat, 10.0, 0.9, 0.3)]

    # When: merge_close_points_global runs with min_distance_m=2
    result = merge_close_points_global(flat, min_distance_m=2.0)

    # Then: merged peak is 0.9 and merged adjusted peak is 0.3 — the adjusted
    #       peak of the max-peak member, NOT the max adjusted peak 0.7
    assert len(result) == 1
    _, _, peak, adj = result[0]
    assert peak == pytest.approx(0.9, abs=1e-12)
    assert adj == pytest.approx(0.3, abs=1e-12)


def test_points_beyond_threshold_stay_separate():
    # Given: two points 3 m apart along a meridian, min_distance_m=2
    dlat = 3.0 * DEG_PER_M
    flat = [(0.0, 10.0, 0.5, 0.5), (dlat, 10.0, 0.6, 0.6)]

    # When: merge_close_points_global runs
    result = merge_close_points_global(flat, min_distance_m=2.0)

    # Then: both points survive as separate clusters
    assert len(result) == 2
    lats = sorted(p[0] for p in result)
    assert lats[0] == pytest.approx(0.0, abs=1e-15)
    assert lats[1] == pytest.approx(dlat, abs=1e-15)


def test_merge_radius_separates_just_inside_from_just_outside():
    # Given: two pairs of points along a meridian at the equator, one
    #        separated by a micrometre less than the 2 m radius and one by a
    #        micrometre more. The bracket is deliberately wider than any
    #        rounding in the distance path: merge_close_points_global gates
    #        on a KD-tree prefilter and then on haversine_m, and haversine_m
    #        goes through sin/asin, which are not correctly rounded and
    #        differ by an ulp or two between platform libms. Pinning
    #        behavior at exactly 2.0 m would make the result depend on the
    #        libm the tests happen to run against.
    inside = 2.0 - 1e-6
    outside = 2.0 + 1e-6

    # When: each pair is merged with a 2 m radius
    merged_inside = merge_close_points_global(
        [(0.0, 10.0, 0.5, 0.5), (inside * DEG_PER_M, 10.0, 0.6, 0.6)],
        min_distance_m=2.0,
    )
    merged_outside = merge_close_points_global(
        [(0.0, 10.0, 0.5, 0.5), (outside * DEG_PER_M, 10.0, 0.6, 0.6)],
        min_distance_m=2.0,
    )

    # Then: the closer pair collapses to one point and the farther pair
    #       stays as two
    assert len(merged_inside) == 1
    assert len(merged_outside) == 2


def test_merging_is_transitive_through_chains():
    # Given: three collinear points at 0 m, 1.5 m and 3.0 m along a meridian —
    #        endpoints are 3 m apart (beyond the 2 m threshold) but each is
    #        within 1.5 m of the middle point
    d = DEG_PER_M
    flat = [
        (0.0, 10.0, 0.1, 0.1),
        (1.5 * d, 10.0, 0.2, 0.2),
        (3.0 * d, 10.0, 0.3, 0.3),
    ]

    # When: merge_close_points_global runs with min_distance_m=2
    result = merge_close_points_global(flat, min_distance_m=2.0)

    # Then: union-find chains all three into ONE cluster whose centroid is the
    #       arithmetic mean latitude (1.5 m equivalent)
    assert len(result) == 1
    lat, lon, peak, _ = result[0]
    assert lat == pytest.approx(1.5 * d, abs=1e-15)
    assert lon == pytest.approx(10.0, abs=1e-12)
    assert peak == pytest.approx(0.3, abs=1e-12)


def test_union_find_find_follows_parent_chain_deeper_than_one_level():
    # Given: UnionFind(4) unioned in DESCENDING order — union(2, 3), then
    #        union(1, 2), then union(0, 1) — which links each new root under
    #        the next one, so the parent array ends as [0, 0, 1, 2] and
    #        element 3 sits at the end of a chain 3 -> 2 -> 1 -> 0 that is
    #        three levels deep
    uf = UnionFind(4)
    uf.union(2, 3)
    uf.union(1, 2)
    uf.union(0, 1)

    # When: find is called on every element
    roots = {uf.find(i) for i in range(4)}

    # Then: all four resolve to the single true root 0 — find must FOLLOW the
    #       chain to the root, not just return the immediate parent (a
    #       one-level lookup would report the three distinct "roots" 0, 1, 2)
    assert roots == {0}


def test_union_find_ascending_union_order_also_collapses_to_one_root():
    # Given: UnionFind(5) unioned in the symmetric ASCENDING order —
    #        union(0, 1), union(1, 2), union(2, 3) — with element 4 untouched
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(2, 3)

    # When: find is called on the four unioned elements
    roots = {uf.find(i) for i in range(4)}

    # Then: elements 0-3 all share the single root 0 regardless of the order
    #       the unions arrived in
    assert roots == {0}

    # When: find is called on the untouched element 4
    root_of_4 = uf.find(4)

    # Then: it remains its own root, separate from the merged cluster
    assert root_of_4 == 4


def test_agreement_two_drops_singletons_but_keeps_pairs():
    # Given: a pair of points 1 m apart plus a lone point 100 m away
    d = DEG_PER_M
    flat = [
        (0.0, 10.0, 0.5, 0.5),
        (1.0 * d, 10.0, 0.7, 0.6),
        (100.0 * d, 10.0, 0.9, 0.9),
    ]

    # When: merge_close_points_global runs with min_distance_m=2, agreement=2
    result = merge_close_points_global(flat, min_distance_m=2.0, agreement=2)

    # Then: only the pair survives (cluster size 2 >= 2); the singleton is
    #       dropped (cluster size 1 < 2)
    assert len(result) == 1
    lat, _, peak, _ = result[0]
    assert lat == pytest.approx(0.5 * d, abs=1e-15)
    assert peak == pytest.approx(0.7, abs=1e-12)


def test_agreement_larger_than_every_cluster_returns_empty():
    # Given: two points forming one cluster of size 2
    d = DEG_PER_M
    flat = [(0.0, 10.0, 0.5, 0.5), (1.0 * d, 10.0, 0.7, 0.6)]

    # When: merge_close_points_global runs with agreement=3
    result = merge_close_points_global(flat, min_distance_m=2.0, agreement=3)

    # Then: nothing survives
    assert result == []


def test_exact_duplicates_merge_even_with_zero_distance():
    # Given: two identical points and min_distance_m=0
    flat = [(5.0, 6.0, 0.4, 0.4), (5.0, 6.0, 0.8, 0.7)]

    # When: merge_close_points_global runs
    result = merge_close_points_global(flat, min_distance_m=0.0)

    # Then: the distance-0 pair still merges (0 <= 0) into a single point
    assert len(result) == 1
    lat, lon, peak, adj = result[0]
    assert lat == pytest.approx(5.0, abs=1e-12)
    assert lon == pytest.approx(6.0, abs=1e-12)
    assert peak == pytest.approx(0.8, abs=1e-12)
    assert adj == pytest.approx(0.7, abs=1e-12)


def test_two_distinct_clusters_each_merge_independently():
    # Given: two pairs of points, each pair 1 m wide, pairs 100 m apart
    d = DEG_PER_M
    flat = [
        (0.0, 10.0, 0.1, 0.1),
        (1.0 * d, 10.0, 0.2, 0.2),
        (100.0 * d, 10.0, 0.3, 0.3),
        (101.0 * d, 10.0, 0.4, 0.4),
    ]

    # When: merge_close_points_global runs with min_distance_m=2
    result = merge_close_points_global(flat, min_distance_m=2.0)

    # Then: exactly two merged points remain, each at its pair's mean latitude
    assert len(result) == 2
    lats = sorted(p[0] for p in result)
    assert lats[0] == pytest.approx(0.5 * d, abs=1e-15)
    assert lats[1] == pytest.approx(100.5 * d, abs=1e-15)


# ---------------------------------------------------------------------------
# distance.py
# ---------------------------------------------------------------------------


def test_haversine_zero_for_identical_points():
    # Given: two identical (lat, lon) pairs
    lat, lon = 31.5, 34.4

    # When: the distance between them is computed
    distance = haversine_m(lat, lon, lat, lon)

    # Then: the distance is exactly 0
    assert distance == 0.0


def test_haversine_one_degree_of_latitude():
    # Given: two points 1 degree of latitude apart on the same meridian
    lat1, lon1, lat2, lon2 = 10.0, 25.0, 11.0, 25.0

    # When: the distance between them is computed
    distance = haversine_m(lat1, lon1, lat2, lon2)

    # Then: the distance is R * 1deg in radians = 6371000 * pi/180
    #       = 111194.9266... m
    expected = EARTH_RADIUS_M * math.radians(1.0)
    assert distance == pytest.approx(expected, rel=1e-12)


def test_haversine_quarter_circumference_along_equator():
    # Given: (0, 0) and (0, 90) — a quarter of the great circle at the equator
    lat1, lon1, lat2, lon2 = 0.0, 0.0, 0.0, 90.0

    # When: the distance between them is computed
    distance = haversine_m(lat1, lon1, lat2, lon2)

    # Then: the distance is pi/2 * R = 10007543.398... m
    expected = math.pi / 2.0 * EARTH_RADIUS_M
    assert distance == pytest.approx(expected, rel=1e-12)


def test_haversine_longitude_shrinks_with_cosine_of_latitude():
    # Given: two points 0.001 deg of longitude apart at latitude 60
    lat, lon1, lon2 = 60.0, 20.0, 20.001

    # When: the distance between them is computed
    distance = haversine_m(lat, lon1, lat, lon2)

    # Then: the distance is 2R*asin(cos(60deg)*sin(0.0005deg)) — for this small
    #       separation about half the equatorial value, ~55.597 m
    expected = (
        2.0
        * EARTH_RADIUS_M
        * math.asin(math.cos(math.radians(60.0)) * math.sin(math.radians(0.0005)))
    )
    assert distance == pytest.approx(expected, rel=1e-9)
    assert expected == pytest.approx(55.5975, abs=1e-3)


def test_haversine_general_position_uses_both_latitudes():
    # Given: points (30, 40) and (31, 41) — both latitude AND longitude differ,
    #        so the cross term cos(phi1)*cos(phi2) with phi1 != phi2 is exercised
    lat1, lon1, lat2, lon2 = 30.0, 40.0, 31.0, 41.0

    # When: the distance between them is computed
    distance = haversine_m(lat1, lon1, lat2, lon2)

    # Then: the distance is 146775.681227 m, derived independently via both the
    #       spherical law of cosines and the Vincenty sphere formula (which
    #       agree to 1e-13 relative); a cos(phi1)**2 bug would give 147098.46 m
    assert distance == pytest.approx(146775.681227, rel=1e-9)


def test_interpolate_centroid_top_left_pixel_origin():
    # Given: bounds lat [10, 20], lon [30, 40] over a (10, 10) grid
    bounds = {"lat_min": 10.0, "lat_max": 20.0, "lon_min": 30.0, "lon_max": 40.0}

    # When: pixel (0, 0) is interpolated
    lat, lon = interpolate_centroid((0.0, 0.0), bounds, (10, 10))

    # Then: the origin maps to (lat_max, lon_min) = (20, 30) — y grows downward
    assert lat == pytest.approx(20.0, abs=1e-12)
    assert lon == pytest.approx(30.0, abs=1e-12)


def test_interpolate_centroid_center_pixel_maps_to_midpoint():
    # Given: bounds lat [10, 20], lon [30, 40] over a (10, 10) grid
    bounds = {"lat_min": 10.0, "lat_max": 20.0, "lon_min": 30.0, "lon_max": 40.0}

    # When: the center pixel (5, 5) is interpolated
    lat, lon = interpolate_centroid((5.0, 5.0), bounds, (10, 10))

    # Then: lat = 20 - 10*(5/10) = 15 and lon = 30 + 10*(5/10) = 35
    assert lat == pytest.approx(15.0, abs=1e-12)
    assert lon == pytest.approx(35.0, abs=1e-12)


def test_interpolate_centroid_non_square_asymmetric_position():
    # Given: bounds lat [0, 4], lon [100, 108] over a non-square grid
    #        (height=4, width=8)
    bounds = {"lat_min": 0.0, "lat_max": 4.0, "lon_min": 100.0, "lon_max": 108.0}

    # When: pixel (y=1, x=6) is interpolated
    lat, lon = interpolate_centroid((1.0, 6.0), bounds, (4, 8))

    # Then: lat = 4 - 4*(1/4) = 3 and lon = 100 + 8*(6/8) = 106
    assert lat == pytest.approx(3.0, abs=1e-12)
    assert lon == pytest.approx(106.0, abs=1e-12)


def test_interpolate_centroid_missing_bound_raises_value_error():
    # Given: bounds missing the lat_min key
    bounds = {"lat_max": 20.0, "lon_min": 30.0, "lon_max": 40.0}

    # When: a centroid is interpolated against those bounds
    # Then: it raises ValueError about missing bounds
    with pytest.raises(ValueError, match="Missing bounds"):
        interpolate_centroid((1.0, 1.0), bounds, (10, 10))


def test_interpolate_centroid_none_bound_value_raises_value_error():
    # Given: bounds where lon_max is present but None
    bounds = {"lat_min": 10.0, "lat_max": 20.0, "lon_min": 30.0, "lon_max": None}

    # When: a centroid is interpolated against those bounds
    # Then: it raises ValueError (None values are treated as missing)
    with pytest.raises(ValueError, match="Missing bounds"):
        interpolate_centroid((1.0, 1.0), bounds, (10, 10))
