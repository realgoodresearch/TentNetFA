"""Tests for displacement_tracker/util/thresholding.py.

Semantics under test — every point carries three values, related by the
invariant on PredictedPoint::

    adjusted_peak == peak_value + prediction factor * adjustment_signal

A later stage adjusts the raw peak by its own factor times the raw signal::

    rescaled = peak_value + factor * adjustment_signal
    keep iff rescaled >= threshold

and leaves the recorded values alone, so no stage's factor is ever folded
into a number a later stage reads.
"""

import json

import numpy as np
import pytest

from displacement_tracker.util.thresholding import (
    PredictedPoint,
    adjusted_peak_from_signal,
    adjustment_signal_from_peaks,
    filter_points_by_rescaled_peak,
    passes_threshold,
)


def test_adjust_factor_zero_collapses_to_peak():
    # Given: peak=0.4, signal=0.4 and factor=0
    peak, signal, factor = 0.4, 0.4, 0.0

    # When: adjusted_peak_from_signal runs
    adjusted = adjusted_peak_from_signal(peak, signal, factor)

    # Then: the result is exactly the raw peak, 0.4
    assert adjusted == pytest.approx(0.4, abs=1e-12)


def test_adjust_factor_one_adds_the_whole_signal():
    # Given: peak=0.4, signal=0.4 and factor=1
    peak, signal, factor = 0.4, 0.4, 1.0

    # When: adjusted_peak_from_signal runs
    adjusted = adjusted_peak_from_signal(peak, signal, factor)

    # Then: the whole signal is added, giving 0.8
    assert adjusted == pytest.approx(0.8, abs=1e-12)


def test_adjust_factor_half_adds_half_the_signal():
    # Given: peak=0.4, signal=0.4 and factor=0.5
    peak, signal, factor = 0.4, 0.4, 0.5

    # When: adjusted_peak_from_signal runs
    adjusted = adjusted_peak_from_signal(peak, signal, factor)

    # Then: half the signal is added, giving 0.6
    assert adjusted == pytest.approx(0.6, abs=1e-12)


def test_adjust_factor_above_one_scales_the_signal_up():
    # Given: peak=0.4, signal=0.2 and factor=2
    peak, signal, factor = 0.4, 0.2, 2.0

    # When: adjusted_peak_from_signal runs
    adjusted = adjusted_peak_from_signal(peak, signal, factor)

    # Then: the signal is doubled, giving 0.4 + 2*0.2 = 0.8
    assert adjusted == pytest.approx(0.8, abs=1e-12)


def test_adjust_negative_signal_moves_the_peak_down():
    # Given: a negative signal -0.4 against peak=0.6 with factor=0.5
    peak, signal, factor = 0.6, -0.4, 0.5

    # When: adjusted_peak_from_signal runs
    adjusted = adjusted_peak_from_signal(peak, signal, factor)

    # Then: the adjusted peak moves down: 0.6 + 0.5*(-0.4) = 0.4
    assert adjusted == pytest.approx(0.4, abs=1e-12)


def test_adjust_is_elementwise_on_numpy_arrays():
    # Given: numpy arrays peak=[0.2, 0.5] and signal=[0.4, -0.4] with factor=0.5
    peak = np.array([0.2, 0.5])
    signal = np.array([0.4, -0.4])

    # When: adjusted_peak_from_signal runs
    result = adjusted_peak_from_signal(peak, signal, 0.5)

    # Then: each element is adjusted independently to [0.4, 0.3]
    assert result == pytest.approx([0.4, 0.3], abs=1e-12)


def test_signal_from_peaks_inverts_the_invariant_at_factor_one():
    # Given: a point recorded at a prediction-time factor of 1.0, so that
    #        adjusted_peak = peak + signal = 0.2 + 0.05
    peak, adjusted_peak = 0.2, 0.25

    # When: adjustment_signal_from_peaks recovers the signal
    signal = adjustment_signal_from_peaks(peak, adjusted_peak)

    # Then: it returns the signal that was folded in, 0.05
    assert signal == pytest.approx(0.05, abs=1e-12)


def test_signal_from_peaks_is_elementwise_on_numpy_arrays():
    # Given: numpy arrays of peaks and adjusted peaks
    peak = np.array([0.2, 0.5])
    adjusted = np.array([0.25, 0.4])

    # When: adjustment_signal_from_peaks runs
    result = adjustment_signal_from_peaks(peak, adjusted)

    # Then: each element is inverted independently to [0.05, -0.1]
    assert result == pytest.approx([0.05, -0.1], abs=1e-12)


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


def test_unadjusted_point_has_a_zero_signal_and_satisfies_the_invariant():
    # Given: a detection method that applies no adjustment, at peak 0.3
    lat, lon, peak = 1.0, 2.0, 0.3

    # When: PredictedPoint.unadjusted builds the point
    built = PredictedPoint.unadjusted(lat, lon, peak)

    # Then: its signal is zero and its adjusted peak is the raw peak, which is
    #       the invariant at any factor
    assert built == (1.0, 2.0, 0.3, 0.3, 0.0)
    assert built.adjustment_signal == 0.0
    assert built.adjusted_peak == pytest.approx(built.peak_value, abs=1e-12)


def test_predicted_point_round_trips_through_json_as_a_flat_array():
    # Given: a point, as the prediction stage streams them to NDJSON
    original = PredictedPoint(1.0, 2.0, 0.5, 0.7, 0.2)

    # When: it is dumped and re-read positionally
    restored = PredictedPoint(*json.loads(json.dumps(original)))

    # Then: it survives the round trip, because a NamedTuple serialises as a
    #       plain JSON array
    assert json.dumps(original) == "[1.0, 2.0, 0.5, 0.7, 0.2]"
    assert restored == original
    assert restored.adjustment_signal == pytest.approx(0.2, abs=1e-12)


def test_filter_returns_survivors_unmodified():
    # Given: one point (lat=1, lon=2, peak=0.5, adj=0.9, signal=0.4) filtered
    #        at factor 0.5, so the rescaled peak is 0.5 + 0.5*0.4 = 0.7
    points = [PredictedPoint(1.0, 2.0, 0.5, 0.9, 0.4)]

    # When: filter_points_by_rescaled_peak runs with threshold 0.65
    kept = filter_points_by_rescaled_peak(points, threshold=0.65, adjustment_factor=0.5)

    # Then: it is kept exactly as it went in — the rescaled 0.7 decided the
    #       keep but is not written over the recorded adjusted peak, so the
    #       filter's factor cannot leak into a later reader's arithmetic
    assert kept == points
    assert kept[0].adjusted_peak == pytest.approx(0.9, abs=1e-12)
    assert kept[0].adjustment_signal == pytest.approx(0.4, abs=1e-12)


def test_filter_keeps_point_exactly_at_threshold():
    # Given: peak=0.2, signal=0.6, factor=0.5 so rescaled = 0.2 + 0.3 = 0.5
    points = [PredictedPoint(0.0, 0.0, 0.2, 0.8, 0.6)]

    # When: filter_points_by_rescaled_peak runs with threshold exactly 0.5
    kept = filter_points_by_rescaled_peak(points, threshold=0.5, adjustment_factor=0.5)

    # Then: the point is kept (>= is inclusive)
    assert len(kept) == 1


def test_filter_thresholds_the_rescaled_value_not_the_recorded_adjusted_peak():
    # Given: peak=0.1, adj=0.9 (the recorded adjusted peak is above the 0.6
    #        threshold) and signal=0.8 with factor=0.5, so the rescaled value
    #        is 0.1 + 0.4 = 0.5
    points = [PredictedPoint(0.0, 0.0, 0.1, 0.9, 0.8)]

    # When: filter_points_by_rescaled_peak runs with threshold 0.6
    kept = filter_points_by_rescaled_peak(points, threshold=0.6, adjustment_factor=0.5)

    # Then: the point is dropped because the RESCALED value 0.5 < 0.6
    assert kept == []


def test_filter_factor_zero_thresholds_on_raw_peak_only():
    # Given: factor=0 and two points whose signals would swap the outcome:
    #        (peak=0.9, signal=-0.8) and (peak=0.1, signal=0.8)
    points = [
        PredictedPoint(1.0, 1.0, 0.9, 0.1, -0.8),
        PredictedPoint(2.0, 2.0, 0.1, 0.9, 0.8),
    ]

    # When: filter_points_by_rescaled_peak runs with threshold 0.5
    kept = filter_points_by_rescaled_peak(points, threshold=0.5, adjustment_factor=0.0)

    # Then: only the high-raw-peak point survives, because the signal is
    #       weighted to nothing
    assert len(kept) == 1
    assert (kept[0].lat, kept[0].lon) == (1.0, 1.0)


def test_filter_default_factor_weights_the_signal_once():
    # Given: two points with signals 0.5 and -0.6 against peak 0.2 and 0.9,
    #        under the default factor of 1.0
    points = [
        PredictedPoint(1.0, 1.0, 0.2, 0.7, 0.5),
        PredictedPoint(2.0, 2.0, 0.9, 0.3, -0.6),
    ]

    # When: filter_points_by_rescaled_peak runs with threshold 0.5
    kept = filter_points_by_rescaled_peak(points, threshold=0.5)

    # Then: only the first survives — 0.2 + 0.5 = 0.7 clears it while
    #       0.9 - 0.6 = 0.3 does not
    assert len(kept) == 1
    assert (kept[0].lat, kept[0].lon) == (1.0, 1.0)


def test_filter_preserves_input_order_of_survivors():
    # Given: four points where the 1st and 3rd pass threshold 0.5 at factor=1
    points = [
        PredictedPoint(1.0, 1.0, 0.5, 0.8, 0.3),
        PredictedPoint(2.0, 2.0, 0.5, 0.2, -0.3),
        PredictedPoint(3.0, 3.0, 0.5, 0.6, 0.1),
        PredictedPoint(4.0, 4.0, 0.5, 0.1, -0.4),
    ]

    # When: filter_points_by_rescaled_peak runs
    kept = filter_points_by_rescaled_peak(points, threshold=0.5)

    # Then: survivors appear in original input order
    assert [(p.lat, p.lon) for p in kept] == [(1.0, 1.0), (3.0, 3.0)]


def test_filter_empty_input_returns_empty_list():
    # Given: no points
    points = []

    # When: filter_points_by_rescaled_peak runs
    kept = filter_points_by_rescaled_peak(points, threshold=0.5)

    # Then: it returns an empty list
    assert kept == []


@pytest.mark.parametrize("merge_factor", [0.0, 0.5, 1.0, 10.0])
def test_filtering_at_any_factor_leaves_the_invariant_intact(merge_factor):
    # Given: a point satisfying the invariant at a prediction-time factor of 1
    points = [PredictedPoint(0.0, 0.0, 0.2, 0.25, 0.05)]

    # When: it is filtered at a range of merge factors, with a threshold low
    #       enough that it always survives
    kept = filter_points_by_rescaled_peak(
        points, threshold=-1.0, adjustment_factor=merge_factor
    )

    # Then: the survivor still satisfies the invariant whatever the factor —
    #       this is what lets a reader recover the raw signal from a merged
    #       file by subtraction, instead of recovering factor * signal
    assert len(kept) == 1
    survivor = kept[0]
    assert survivor.adjusted_peak == pytest.approx(
        survivor.peak_value + survivor.adjustment_signal, abs=1e-12
    )
