"""
Single source of truth for adjusted-peak thresholding.

The prediction flow (e_predict_json), the merge flow (h_merge_geojsons) and the
validation flow (util/validation_core) must all apply identical semantics:

    rescaled = peak_value + factor * (adjusted_peak - peak_value)
    keep iff rescaled >= threshold
"""


def rescale_adjusted_peak(peak_value, adjusted_peak, factor):
    """Rescale the adjusted peak around the raw peak by ``factor``.

    Works elementwise on scalars, numpy arrays, pandas Series and torch tensors.
    factor=0 collapses to peak_value, factor=1 leaves adjusted_peak unchanged.
    """
    return peak_value + factor * (adjusted_peak - peak_value)


def passes_threshold(value, threshold):
    """Keep iff ``value >= threshold``. Works elementwise on arrays/tensors."""
    return value >= threshold


def filter_points_by_adjusted_peak(points, threshold, adjustment_factor=1.0):
    """Rescale and threshold (lat, lon, peak_value, adjusted_peak) points.

    Keeps any trailing provenance fields and replaces adjusted_peak with its
    rescaled value, so downstream consumers see the value that was thresholded.
    """
    kept = []
    for point in points:
        lat, lon, peak, adj_peak, *provenance = point
        rescaled = rescale_adjusted_peak(peak, adj_peak, adjustment_factor)
        if passes_threshold(rescaled, threshold):
            kept.append((lat, lon, peak, rescaled, *provenance))
    return kept
