"""Tests for displacement_tracker/util/thresholding.py.

Semantics under test (single source of truth for adjusted-peak thresholding):

    rescaled = peak_value + factor * (adjusted_peak - peak_value)
    keep iff rescaled >= threshold
"""

import numpy as np
import pytest

from displacement_tracker.util.thresholding import (
    filter_points_by_adjusted_peak,
    passes_threshold,
    rescale_adjusted_peak,
)


def test_rescale_factor_zero_collapses_to_peak():
    # Given: peak=0.4, adjusted_peak=0.8 and factor=0
    peak, adjusted_peak, factor = 0.4, 0.8, 0.0

    # When: rescale_adjusted_peak runs
    rescaled = rescale_adjusted_peak(peak, adjusted_peak, factor)

    # Then: the result is exactly the raw peak, 0.4
    assert rescaled == pytest.approx(0.4, abs=1e-12)


def test_rescale_factor_one_is_identity_on_adjusted_peak():
    # Given: peak=0.4, adjusted_peak=0.8 and factor=1
    peak, adjusted_peak, factor = 0.4, 0.8, 1.0

    # When: rescale_adjusted_peak runs
    rescaled = rescale_adjusted_peak(peak, adjusted_peak, factor)

    # Then: the result is exactly the adjusted peak, 0.8
    assert rescaled == pytest.approx(0.8, abs=1e-12)


def test_rescale_factor_half_is_midpoint():
    # Given: peak=0.4, adjusted_peak=0.8 and factor=0.5
    peak, adjusted_peak, factor = 0.4, 0.8, 0.5

    # When: rescale_adjusted_peak runs
    rescaled = rescale_adjusted_peak(peak, adjusted_peak, factor)

    # Then: the result is the midpoint 0.4 + 0.5*(0.8-0.4) = 0.6
    assert rescaled == pytest.approx(0.6, abs=1e-12)


def test_rescale_factor_above_one_extrapolates_beyond_adjusted_peak():
    # Given: peak=0.4, adjusted_peak=0.6 and factor=2
    peak, adjusted_peak, factor = 0.4, 0.6, 2.0

    # When: rescale_adjusted_peak runs
    rescaled = rescale_adjusted_peak(peak, adjusted_peak, factor)

    # Then: the result extrapolates past the adjusted peak to 0.4 + 2*0.2 = 0.8
    assert rescaled == pytest.approx(0.8, abs=1e-12)


def test_rescale_when_adjusted_below_peak_moves_down():
    # Given: adjusted_peak=0.2 below peak=0.6 and factor=0.5
    peak, adjusted_peak, factor = 0.6, 0.2, 0.5

    # When: rescale_adjusted_peak runs
    rescaled = rescale_adjusted_peak(peak, adjusted_peak, factor)

    # Then: the result moves down toward the adjusted peak: 0.6 + 0.5*(-0.4) = 0.4
    assert rescaled == pytest.approx(0.4, abs=1e-12)


def test_rescale_is_elementwise_on_numpy_arrays():
    # Given: numpy arrays peak=[0.2, 0.5] and adjusted=[0.6, 0.1] with factor=0.5
    peak = np.array([0.2, 0.5])
    adj = np.array([0.6, 0.1])

    # When: rescale_adjusted_peak runs
    result = rescale_adjusted_peak(peak, adj, 0.5)

    # Then: each element is rescaled independently to [0.4, 0.3]
    assert result == pytest.approx([0.4, 0.3], abs=1e-12)


def test_passes_threshold_is_inclusive_at_the_boundary():
    # Given: a value exactly equal to the threshold
    value, threshold = 0.5, 0.5

    # When: passes_threshold runs
    kept = passes_threshold(value, threshold)

    # Then: the value is kept (>= semantics, not strict >)
    assert kept


def test_passes_threshold_rejects_below_and_accepts_above():
    # Given: values just below and just above a 0.5 threshold
    threshold = 0.5

    # When: passes_threshold runs on the value just below it
    below = passes_threshold(0.4999, threshold)

    # Then: the value is rejected
    assert not below

    # When: passes_threshold runs on the value just above it
    above = passes_threshold(0.5001, threshold)

    # Then: the value is accepted
    assert above


def test_passes_threshold_elementwise_on_arrays():
    # Given: an array [0.2, 0.5, 0.8] and threshold 0.5
    values = np.array([0.2, 0.5, 0.8])

    # When: passes_threshold runs
    mask = passes_threshold(values, 0.5)

    # Then: the elementwise mask is [False, True, True] (boundary inclusive)
    assert mask.tolist() == [False, True, True]


def test_filter_replaces_adjusted_peak_with_rescaled_value():
    # Given: one point (lat=1, lon=2, peak=0.5, adj=0.9) with factor=0.5
    points = [(1.0, 2.0, 0.5, 0.9)]

    # When: filter_points_by_adjusted_peak runs with threshold 0.65
    kept = filter_points_by_adjusted_peak(points, threshold=0.65, adjustment_factor=0.5)

    # Then: it is kept as (1, 2, 0.5, 0.7) — raw peak preserved, adjusted peak
    #       replaced by the rescaled value 0.5 + 0.5*(0.9-0.5) = 0.7
    assert len(kept) == 1
    lat, lon, peak, rescaled = kept[0]
    assert (lat, lon, peak) == (1.0, 2.0, 0.5)
    assert rescaled == pytest.approx(0.7, abs=1e-12)


def test_filter_keeps_point_exactly_at_threshold():
    # Given: peak=0.2, adj=0.8, factor=0.5 so rescaled = 0.2 + 0.5*0.6 = 0.5
    points = [(0.0, 0.0, 0.2, 0.8)]

    # When: filter_points_by_adjusted_peak runs with threshold exactly 0.5
    kept = filter_points_by_adjusted_peak(points, threshold=0.5, adjustment_factor=0.5)

    # Then: the point is kept (>= is inclusive)
    assert len(kept) == 1


def test_filter_thresholds_the_rescaled_value_not_the_raw_adjusted_peak():
    # Given: peak=0.1, adj=0.9 (raw adjusted peak is above the 0.6 threshold)
    #        and factor=0.5 so rescaled = 0.1 + 0.5*0.8 = 0.5
    points = [(0.0, 0.0, 0.1, 0.9)]

    # When: filter_points_by_adjusted_peak runs with threshold 0.6
    kept = filter_points_by_adjusted_peak(points, threshold=0.6, adjustment_factor=0.5)

    # Then: the point is dropped because the RESCALED value 0.5 < 0.6
    assert kept == []


def test_filter_factor_zero_thresholds_on_raw_peak_only():
    # Given: factor=0 and two points: (peak=0.9, adj=0.1) and (peak=0.1, adj=0.9)
    points = [(1.0, 1.0, 0.9, 0.1), (2.0, 2.0, 0.1, 0.9)]

    # When: filter_points_by_adjusted_peak runs with threshold 0.5
    kept = filter_points_by_adjusted_peak(points, threshold=0.5, adjustment_factor=0.0)

    # Then: only the high-raw-peak point survives (rescaled collapses to peak),
    #       and its stored value is the raw peak 0.9
    assert len(kept) == 1
    assert kept[0][:2] == (1.0, 1.0)
    assert kept[0][3] == pytest.approx(0.9, abs=1e-12)


def test_filter_default_factor_thresholds_adjusted_peak_unchanged():
    # Given: two points with adjusted peaks 0.7 and 0.3, default factor (1.0)
    points = [(1.0, 1.0, 0.2, 0.7), (2.0, 2.0, 0.9, 0.3)]

    # When: filter_points_by_adjusted_peak runs with threshold 0.5
    kept = filter_points_by_adjusted_peak(points, threshold=0.5)

    # Then: only the adj=0.7 point survives and its value stays 0.7
    assert len(kept) == 1
    assert kept[0][:2] == (1.0, 1.0)
    assert kept[0][3] == pytest.approx(0.7, abs=1e-12)


def test_filter_preserves_input_order_of_survivors():
    # Given: four points where the 1st and 3rd pass threshold 0.5 at factor=1
    points = [
        (1.0, 1.0, 0.5, 0.8),
        (2.0, 2.0, 0.5, 0.2),
        (3.0, 3.0, 0.5, 0.6),
        (4.0, 4.0, 0.5, 0.1),
    ]

    # When: filter_points_by_adjusted_peak runs
    kept = filter_points_by_adjusted_peak(points, threshold=0.5)

    # Then: survivors appear in original input order
    assert [(p[0], p[1]) for p in kept] == [(1.0, 1.0), (3.0, 3.0)]


def test_filter_empty_input_returns_empty_list():
    # Given: no points
    points = []

    # When: filter_points_by_adjusted_peak runs
    kept = filter_points_by_adjusted_peak(points, threshold=0.5)

    # Then: it returns an empty list
    assert kept == []
