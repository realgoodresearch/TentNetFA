"""
Single source of truth for adjusted-peak thresholding.

Every predicted point carries three values:

    peak_value          raw model probability at the detected peak
    adjustment_signal   raw (un-multiplied) blurred neighbourhood score at that peak
    adjusted_peak       peak_value + prediction factor * adjustment_signal

The prediction flow (e_predict_json), the merge flow (h_merge_geojsons) and the
validation flow (util/validation_core) must all apply identical semantics:

    rescaled = peak_value + factor * adjustment_signal
    keep iff rescaled >= threshold

Carrying the raw signal rather than only the already-multiplied adjusted peak
keeps the factor used at prediction time from leaking into later rescaling, and
lets the signal itself be inspected to sanity check the data.
"""


def adjustment_signal_from_peaks(peak_value, adjusted_peak):
    """Recover the raw adjustment signal from a point that does not carry it.

    Points written before ``adjustment_signal`` was propagated only store the
    already-multiplied adjusted peak, from which the signal is recoverable when
    the prediction-time factor was 1.0 (the default).

    Works elementwise on scalars, numpy arrays, pandas Series and torch tensors.
    """
    return adjusted_peak - peak_value


def rescale_adjusted_peak(peak_value, adjustment_signal, factor):
    """Adjust the raw peak by ``factor`` times the raw adjustment signal.

    Works elementwise on scalars, numpy arrays, pandas Series and torch tensors.
    factor=0 collapses to peak_value.
    """
    return peak_value + factor * adjustment_signal


def passes_threshold(value, threshold):
    """Keep iff ``value >= threshold``. Works elementwise on arrays/tensors."""
    return value >= threshold


def filter_points_by_adjusted_peak(points, threshold, adjustment_factor=1.0):
    """Rescale and threshold (lat, lon, peak, adjusted_peak, signal) points.

    Returns the kept points with adjusted_peak replaced by its rescaled value,
    so downstream consumers see the same value that was thresholded. The raw
    adjustment signal is passed through untouched.
    """
    kept = []
    for lat, lon, peak, _adj_peak, signal in points:
        rescaled = rescale_adjusted_peak(peak, signal, adjustment_factor)
        if passes_threshold(rescaled, threshold):
            kept.append((lat, lon, peak, rescaled, signal))
    return kept
