"""
Single source of truth for the predicted-point contract and its thresholding.

The prediction flow (e_predict_json), the merge flow (h_merge_geojsons) and the
validation flow (util/validation_core) all read and write ``PredictedPoint``,
whose docstring states the invariant every point in this pipeline satisfies.
"""

from typing import NamedTuple


class PredictedPoint(NamedTuple):
    """One predicted tent and the three values describing its score.

    ``adjustment_signal`` is the raw blurred neighbourhood score at the peak,
    before any factor is applied to it, and ``adjusted_peak`` is that signal
    folded into the raw peak at prediction time::

        adjusted_peak == peak_value + prediction factor * adjustment_signal

    That identity holds for every point every stage of this pipeline writes.
    Later stages threshold on their own adjusted peak, derived from the raw
    signal via ``adjusted_peak_from_signal``, and leave ``adjusted_peak``
    alone — so a downstream factor never compounds with the prediction-time
    one, and the written values always reconcile with each other.

    Reading a point back: a missing ``adjustment_signal`` is derived with
    ``adjustment_signal_from_peaks``, and a missing ``adjusted_peak`` reads as
    unadjusted (equal to ``peak_value``, hence a zero signal). Both readers
    apply exactly these rules.
    """

    lat: float
    lon: float
    peak_value: float
    adjusted_peak: float
    adjustment_signal: float

    @classmethod
    def unadjusted(cls, lat, lon, peak_value) -> "PredictedPoint":
        """Build a point from a method that applies no adjustment at all."""
        return cls(lat, lon, peak_value, peak_value, 0.0)


def adjustment_signal_from_peaks(peak_value, adjusted_peak):
    """Recover the raw adjustment signal from a point that does not carry it.

    Points written before ``adjustment_signal`` was propagated only store the
    already-folded adjusted peak, from which the signal is recoverable when the
    prediction-time factor was 1.0 (the default). That assumption is not
    checkable from the file, so both readers log when they fall back to this.

    Works elementwise on scalars, numpy arrays, pandas Series and torch tensors.
    """
    return adjusted_peak - peak_value


def adjusted_peak_from_signal(peak_value, adjustment_signal, factor):
    """Adjust the raw peak by ``factor`` times the raw adjustment signal.

    Works elementwise on scalars, numpy arrays, pandas Series and torch tensors.
    factor=0 collapses to peak_value.
    """
    return peak_value + factor * adjustment_signal


def passes_threshold(value, threshold):
    """Keep iff ``value >= threshold``. Works elementwise on arrays/tensors."""
    return value >= threshold


def filter_points_by_rescaled_peak(points, threshold, adjustment_factor=1.0):
    """Keep points whose adjusted peak at ``adjustment_factor`` clears ``threshold``.

    Points are returned unmodified: the peak this filters on is derived from
    the raw signal for this stage's factor, and recording it in place of the
    prediction-time ``adjusted_peak`` would break the invariant above. Any
    consumer that wants the thresholded value can recompute it.
    """
    return [
        point
        for point in points
        if passes_threshold(
            adjusted_peak_from_signal(
                point.peak_value, point.adjustment_signal, adjustment_factor
            ),
            threshold,
        )
    ]
