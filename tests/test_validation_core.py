"""Unit tests for displacement_tracker/util/validation_core.py.

Covers the per-tile error metrics computed inside the analysis mask — the
formulas themselves, the totals they derive from, and the degenerate
branches (an empty mask, a zero reference total, and inputs too constant
to rank) — plus the direction comparison every metric is optimized
through.

These pin ``compute_metrics`` and ``is_better`` directly. The tuning scan
reaches both through a closure over one tile's grouped cell inputs; the
scan's own tests pin that composition — which cells a ``(factor, cutoff)``
pair selects — rather than restating these formulas.

Also covered: how ``prepare_grouped_cell_inputs`` resolves the three score
columns off a prediction file, since which of them are optional decides
whether a malformed file is skipped loudly or scored silently.
"""

import math

import geopandas as gpd
import numpy as np
import pytest
from _helpers import CRS_WGS84
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point

from displacement_tracker.util.validation_core import (
    compute_metrics,
    is_better,
    prepare_grouped_cell_inputs,
)

LN2, LN3 = math.log(2.0), math.log(3.0)


def _masked_tile():
    """A 2x2 tile whose bottom-right cell is outside the analysis mask.

    In-mask cells (row, col) -> (pred, val):
        (0,0) -> (1, 2)    (0,1) -> (1, 0)    (1,0) -> (0, 0)

    The masked-out cell holds pred == val == 9, large enough that every
    metric below would move if it were counted. ``error_raster`` carries
    log1p(pred) - log1p(val) per cell, which is what the caller supplies.
    """
    pred = np.array([[1.0, 1.0], [0.0, 9.0]], dtype=np.float32)
    val = np.array([[2.0, 0.0], [0.0, 9.0]], dtype=np.float32)
    error = np.array([[LN2 - LN3, LN2], [0.0, 0.0]], dtype=np.float64)
    mask = np.array([[True, True], [True, False]])
    return pred, val, error, mask


def test_compute_metrics_error_formulas_over_the_masked_cells():
    # Given: in-mask pred [1, 1, 0] against val [2, 0, 0] — differences
    #        [-1, 1, 0] and log errors [ln2-ln3, ln2, 0] — plus a masked-out
    #        cell holding 9s
    pred, val, error, mask = _masked_tile()

    # When: compute_metrics scores the tile
    m = compute_metrics(pred, val, error, mask)

    # Then: only the three in-mask cells contribute, and each formula matches
    #       the value derived by hand: rms the root mean square of the
    #       differences, mae their mean magnitude, rmsle the root mean square
    #       of the supplied log errors
    assert m["n_cells"] == 3
    assert m["rms"] == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-6)
    assert m["mae"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert m["rmsle"] == pytest.approx(
        math.sqrt(((LN2 - LN3) ** 2 + LN2**2) / 3.0), rel=1e-6
    )

    # Then: spearman is the Pearson correlation of the ranks — pred ranks
    #       [2.5, 2.5, 1] against val ranks [3, 1.5, 1.5] give exactly 0.5
    assert m["spearman"] == pytest.approx(0.5, abs=1e-12)


def test_compute_metrics_totals_and_percentage_difference():
    # Given: an under-predicting tile — in-mask predictions summing to 1
    #        against a reference summing to 2 — so the signed and absolute
    #        totals genuinely differ
    pred = np.array([[1.0, 0.0]], dtype=np.float32)
    val = np.array([[2.0, 0.0]], dtype=np.float32)

    # When: compute_metrics scores the tile
    m = compute_metrics(pred, val, np.zeros((1, 2)), np.ones((1, 2), dtype=bool))

    # Then: the signed fields carry the shortfall while the abs_ fields drop
    #       the sign. abs_total_diff is one of the default scan objectives, so
    #       an unsigned one would improve without bound as the scan
    #       under-predicts and export a cutoff that discards every detection.
    assert m["total_pred"] == 1.0
    assert m["total_val"] == 2.0
    assert m["total_diff"] == pytest.approx(-1.0)
    assert m["abs_total_diff"] == pytest.approx(1.0)

    # Then: the percentage difference is taken against the reference total,
    #       not the predicted one — -1/2, not -1/1
    assert m["total_pdiff"] == pytest.approx(-0.5)
    assert m["abs_total_pdiff"] == pytest.approx(0.5)


def test_compute_metrics_zero_reference_total_gives_zero_percentage():
    # Given: an all-zero reference, so the percentage difference has no
    #        denominator
    pred = np.array([[1.0, 2.0]], dtype=np.float32)
    val = np.zeros((1, 2), dtype=np.float32)

    # When: compute_metrics scores the tile
    m = compute_metrics(pred, val, np.zeros((1, 2)), np.ones((1, 2), dtype=bool))

    # Then: the percentage difference is 0 rather than an infinity or a
    #       ZeroDivisionError, while the absolute difference still reports
    #       the 3 predictions that have no reference behind them
    assert m["total_diff"] == pytest.approx(3.0)
    assert m["total_pdiff"] == 0.0
    assert m["abs_total_pdiff"] == 0.0


def test_compute_metrics_empty_mask_returns_infinite_error_metrics():
    # Given: a mask that excludes every cell of the tile
    pred = np.ones((2, 2), dtype=np.float32)
    val = np.zeros((2, 2), dtype=np.float32)

    # When: compute_metrics scores it
    m = compute_metrics(pred, val, np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))

    # Then: the error metrics are +inf rather than the NaN a mean over an
    #       empty slice would give — a tile with nothing to score must never
    #       compare as better than one that scored badly
    assert m["n_cells"] == 0
    assert m["rms"] == float("inf")
    assert m["mae"] == float("inf")
    assert m["rmsle"] == float("inf")
    assert math.isnan(m["spearman"])


def test_is_better_compares_in_each_metric_direction():
    # Given: rms, which METRIC_DIRECTIONS minimizes, and spearman, the only
    #        metric it maximizes
    incumbent = 0.5

    # When: a candidate below the incumbent is offered to each
    # Then: it improves the minimized metric and not the maximized one
    assert is_better("rms", 0.4, incumbent)
    assert not is_better("spearman", 0.4, incumbent)

    # When: a candidate above the incumbent is offered to each
    # Then: the verdicts reverse. A scan tuned on spearman that compared in
    #       the minimizing direction would export the worst correlation it
    #       found rather than the best.
    assert not is_better("rms", 0.6, incumbent)
    assert is_better("spearman", 0.6, incumbent)


def test_is_better_rejects_a_non_finite_candidate():
    # Given: the sentinel each direction starts from, which any real
    #        measurement beats
    worst_for_min = float("inf")
    worst_for_max = -float("inf")

    # When: a NaN candidate is offered against those sentinels
    # Then: neither direction accepts it. A tile that could not be scored
    #       must never become the incumbent, and NaN compares false against
    #       everything anyway — so without this guard the max branch would
    #       depend on comparison order rather than rejecting outright.
    assert not is_better("rms", float("nan"), worst_for_min)
    assert not is_better("spearman", float("nan"), worst_for_max)

    # When: an infinite candidate is offered in the direction that would
    #       otherwise welcome it
    # Then: it is rejected too — only finite measurements can win
    assert not is_better("spearman", float("inf"), worst_for_max)


def test_compute_metrics_constant_reference_leaves_spearman_undefined():
    # Given: predictions that vary against a reference that is constant, so
    #        the reference has no ranking to correlate against
    pred = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    val = np.full((1, 3), 4.0, dtype=np.float32)

    # When: compute_metrics scores the tile
    m = compute_metrics(pred, val, np.zeros((1, 3)), np.ones((1, 3), dtype=bool))

    # Then: spearman is NaN, while the error metrics stay finite — an
    #       undefined correlation must not take the rest of the row with it
    assert math.isnan(m["spearman"])
    assert math.isfinite(m["rms"])
    assert math.isfinite(m["mae"])


# ---------------------------------------------------------------------------
# prepare_grouped_cell_inputs — score-column resolution
#
# peak_value is required; adjusted_peak and adjustment_signal are optional and
# follow the missing-field rules documented on PredictedPoint. The distinction
# matters because both callers wrap this in try/except and skip the tile: a
# file that stops raising starts being scored against an all-NaN column.
# ---------------------------------------------------------------------------


class _EmptyReference:
    """A reference source that contributes no counts to any cell.

    prepare_grouped_cell_inputs only asks a reference for its counts raster;
    what those counts are is the reference layer's business, tested in
    test_reference_data.py.
    """

    def counts_on_grid(self, grid_shape, transform, crs, clip_geom=None):
        return np.zeros(grid_shape, dtype=np.float32)


# A square of predictions inside the grid below, so the convex hull is a
# polygon rather than a degenerate line.
_PRED_GEOMETRY = [
    Point(34.401, 31.511),
    Point(34.404, 31.511),
    Point(34.404, 31.514),
    Point(34.401, 31.514),
]


def _prepped(columns, source=None):
    """Run prepare_grouped_cell_inputs over an in-memory 20x20 master grid."""
    pred_gdf = gpd.GeoDataFrame(columns, geometry=_PRED_GEOMETRY, crs=CRS_WGS84)
    profile = {
        "driver": "GTiff",
        "height": 20,
        "width": 20,
        "count": 1,
        "dtype": "float32",
        "crs": CRS_WGS84,
        "transform": from_origin(34.40, 31.52, 0.0005, 0.0005),
    }
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(np.zeros((1, 20, 20), dtype="float32"))
        with mem.open() as src:
            grouped = prepare_grouped_cell_inputs(
                pred_gdf, _EmptyReference(), src, source=source
            )
    return grouped["pred_prepped"]


PEAKS = [0.2, 0.1, 0.3, 0.4]
ADJUSTED = [0.25, 0.11, 0.36, 0.44]
SIGNALS = [0.05, 0.01, 0.06, 0.04]


def test_prepare_reads_a_recorded_adjustment_signal_unchanged():
    # Given: a prediction file carrying all three score columns
    columns = {
        "peak_value": PEAKS,
        "adjusted_peak": ADJUSTED,
        "adjustment_signal": SIGNALS,
    }

    # When: prepare_grouped_cell_inputs resolves them
    prepped = _prepped(columns)

    # Then: the signal is taken from the file as recorded. Rows keep their
    #       index through the in-bounds selection, so the surviving rows'
    #       expected signals are looked up by it.
    assert len(prepped) > 0
    assert prepped["adjustment_signal"].tolist() == pytest.approx(
        [SIGNALS[i] for i in prepped.index]
    )

    # Then: the frame carries only what the keep-mask and the cell arithmetic
    #       consume — adjusted_peak has served its purpose once the signal is
    #       resolved, and nothing downstream reads it
    assert list(prepped.columns) == ["peak_value", "adjustment_signal", "row", "col"]


def test_prepare_derives_the_signal_when_the_column_is_absent():
    # Given: a file predating adjustment_signal, so only two columns exist
    with_signal = _prepped(
        {
            "peak_value": PEAKS,
            "adjusted_peak": ADJUSTED,
            "adjustment_signal": SIGNALS,
        }
    )

    # When: the same file is read without the signal column
    without_signal = _prepped({"peak_value": PEAKS, "adjusted_peak": ADJUSTED})

    # Then: the derived signal equals what the recorded one would have been,
    #       because these fixtures satisfy the invariant at factor 1.0
    assert without_signal["adjustment_signal"].tolist() == pytest.approx(
        with_signal["adjustment_signal"].tolist()
    )


def test_prepare_derives_the_signal_for_null_values_in_a_present_column():
    # Given: a file whose adjustment_signal column exists but is null in places
    columns = {
        "peak_value": PEAKS,
        "adjusted_peak": ADJUSTED,
        "adjustment_signal": [0.05, None, 0.06, None],
    }

    # When: prepare_grouped_cell_inputs resolves them
    prepped = _prepped(columns)

    # Then: nulls are filled the same way an absent column is, so "column
    #       missing" and "value missing" cannot drift apart
    reference = _prepped({"peak_value": PEAKS, "adjusted_peak": ADJUSTED})
    assert prepped["adjustment_signal"].tolist() == pytest.approx(
        reference["adjustment_signal"].tolist()
    )


def test_prepare_reads_a_point_with_no_adjusted_peak_as_unadjusted():
    # Given: a file carrying peaks only
    columns = {"peak_value": PEAKS}

    # When: prepare_grouped_cell_inputs resolves them
    prepped = _prepped(columns)

    # Then: the signal is zero rather than -peak_value, matching the rule the
    #       GeoJSON reader applies to the same absence
    assert prepped["adjustment_signal"].tolist() == pytest.approx([0.0] * len(prepped))


def test_prepare_raises_when_the_required_peak_value_column_is_missing():
    # Given: a file with the optional columns but no peak_value
    columns = {"adjusted_peak": ADJUSTED, "adjustment_signal": SIGNALS}

    # When/Then: resolving it raises, so both callers go on skipping the tile
    #            instead of scoring every row as NaN against an empty
    #            prediction raster
    with pytest.raises(KeyError):
        _prepped(columns)


def test_prepare_names_the_source_file_when_it_derives_the_signal(caplog):
    # Given: a file with no adjustment_signal column, read with a source label
    columns = {"peak_value": PEAKS, "adjusted_peak": ADJUSTED}

    # When: prepare_grouped_cell_inputs resolves it
    with caplog.at_level("WARNING"):
        _prepped(columns, source="tile_A.geojson")

    # Then: the warning names the file, so a sweep over many pairs can be
    #       traced back to the one whose numbers rest on the factor=1.0
    #       assumption
    assert "tile_A.geojson" in caplog.text
    assert "adjustment_signal" in caplog.text
