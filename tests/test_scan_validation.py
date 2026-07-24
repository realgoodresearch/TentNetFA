"""Unit tests for displacement_tracker/g1_scan_validation.py.

Covers settings resolution, metric resolution, the sign-adjusted objective,
the evaluator/bests bookkeeping, the ridge scan core, prediction-date
inference and best-params export.
"""

import math
import os
from datetime import datetime

import click
import numpy as np
import pandas as pd
import pytest
import yaml

from displacement_tracker.g1_scan_validation import (
    LARGE_PENALTY,
    ScanSettings,
    _make_evaluator,
    _objective,
    _optimize_metric,
    _prediction_date,
    _resolve_metrics,
    _scan_settings,
    _write_best_params,
    scan_tile,
)
from displacement_tracker.util.validation_core import initial_best_value, is_better


def _settings(**overrides) -> ScanSettings:
    base = dict(
        input_path="merged_raw.gpkg",
        pred_folder=None,
        master_grid="grid.tif",
        reference={"path": "ref.geojson"},
        out_dir="scan_results",
        best_params_path="scan_results/best_params.yaml",
        metric="rms",
        scan_metrics=["rms"],
        factor_bounds=(0.0, 10.0),
        cutoff_bounds=(0.0001, 0.01),
        ridge_probes=5,
        xtol_factor=1e-3,
        xtol_cutoff=1e-6,
        refine_maxiter=60,
        exclusion_zones=None,
    )
    base.update(overrides)
    return ScanSettings(**base)


def _fresh_bests(scan_metrics):
    return {
        m: {
            "value": initial_best_value(m),
            "factor": None,
            "cutoff": None,
            "keep": None,
        }
        for m in scan_metrics
    }


def _grouped(points, val_counts, masked_out=()):
    """Build the cell-input dict that prepare_grouped_cell_inputs produces.

    Keeping the schema in one place here means a new required key breaks every
    test at once instead of letting hand-built copies drift from the producer.

    ``points`` are (peak_value, adjusted_peak, row, col) prediction rows;
    ``val_counts`` is the reference-count raster, whose shape sets grid_shape;
    ``masked_out`` lists the (row, col) cells excluded from the analysis mask,
    which defaults to every cell included.
    """
    peak_value, adjusted_peak, rows, cols = (list(v) for v in zip(*points))
    val_raster = np.array(val_counts, dtype=np.float32)
    mask_array = np.ones(val_raster.shape, dtype=bool)
    for cell in masked_out:
        mask_array[cell] = False
    return {
        "pred_prepped": pd.DataFrame(
            {
                "peak_value": peak_value,
                "adjusted_peak": adjusted_peak,
                "row": rows,
                "col": cols,
            }
        ),
        "val_raster": val_raster,
        "mask_array": mask_array,
        "grid_shape": val_raster.shape,
    }


# ---------------------------------------------------------------------------
# _resolve_metrics
# ---------------------------------------------------------------------------


def test_resolve_metrics_defaults():
    # Given: an empty tuning config
    tuning = {}

    # When: _resolve_metrics runs
    metric, metrics = _resolve_metrics(tuning)

    # Then: the eval metric defaults to "rms" and the tracked metrics to the
    #       default triple (rms, mae, abs_total_diff), rms not duplicated
    assert metric == "rms"
    assert metrics == ["rms", "mae", "abs_total_diff"]


def test_resolve_metrics_inserts_eval_metric_first():
    # Given: metric=spearman which is absent from the default metric list
    tuning = {"metric": "spearman"}

    # When: _resolve_metrics runs
    metric, metrics = _resolve_metrics(tuning)

    # Then: spearman is prepended so the eval metric is always tracked
    assert metric == "spearman"
    assert metrics == ["spearman", "rms", "mae", "abs_total_diff"]


def test_resolve_metrics_respects_explicit_list_containing_metric():
    # Given: metric=mae and an explicit metrics list already containing it
    tuning = {"metric": "mae", "metrics": ["mae", "rmsle"]}

    # When: _resolve_metrics runs
    metric, metrics = _resolve_metrics(tuning)

    # Then: the explicit list is used verbatim (no insertion, no defaults)
    assert metric == "mae"
    assert metrics == ["mae", "rmsle"]


def test_resolve_metrics_unknown_metric_raises():
    # Given: an eval metric that is not in METRIC_DIRECTIONS
    tuning = {"metric": "bogus"}

    # When: _resolve_metrics runs
    # Then: a ClickException names the unknown metric
    with pytest.raises(click.ClickException, match="bogus"):
        _resolve_metrics(tuning)


# ---------------------------------------------------------------------------
# _scan_settings
# ---------------------------------------------------------------------------


def test_scan_settings_defaults_and_merge_output_fallback():
    # Given: a minimal config with no tuning.input but a merge.output
    params = {
        "tuning": {"master_grid": "grid.tif", "reference": {"path": "r.geojson"}},
        "merge": {"output": "merged_raw.gpkg"},
    }

    # When: _scan_settings resolves it
    s = _scan_settings(params)

    # Then: the input falls back to merge.output and every optional knob gets
    #       its documented default
    assert s.input_path == "merged_raw.gpkg"
    assert s.out_dir == "scan_results"
    assert s.best_params_path == os.path.join("scan_results", "best_params.yaml")
    assert s.metric == "rms"
    assert s.scan_metrics == ["rms", "mae", "abs_total_diff"]
    assert s.factor_bounds == (0.0, 10.0)
    assert s.cutoff_bounds == (0.0001, 0.01)
    assert s.ridge_probes == 5
    assert s.xtol_factor == pytest.approx(1e-3)
    assert s.xtol_cutoff == pytest.approx(1e-6)
    assert s.refine_maxiter == 60
    assert s.exclusion_zones is None
    assert s.pred_folder is None


def test_scan_settings_explicit_input_wins_and_pred_folder_from_merge():
    # Given: both tuning.input and merge.output set, plus merge.input_folder
    #        and a custom tuning.out_dir
    params = {
        "tuning": {
            "input": "explicit.gpkg",
            "master_grid": "grid.tif",
            "reference": "r.geojson",
            "out_dir": "custom_out",
        },
        "merge": {"output": "merged_raw.gpkg", "input_folder": "preds"},
    }

    # When: _scan_settings resolves it
    s = _scan_settings(params)

    # Then: tuning.input takes precedence and pred_folder comes from
    #       merge.input_folder; the custom out_dir also relocates best_params
    assert s.input_path == "explicit.gpkg"
    assert s.pred_folder == "preds"
    assert s.best_params_path == os.path.join("custom_out", "best_params.yaml")


def test_scan_settings_missing_input_raises():
    # Given: neither tuning.input nor merge.output present
    params = {"tuning": {"master_grid": "g.tif", "reference": "r.geojson"}}

    # When: _scan_settings resolves the config
    # Then: a ClickException points at tuning.input / merge.output
    with pytest.raises(click.ClickException, match="tuning.input"):
        _scan_settings(params)


def test_scan_settings_missing_required_keys_raise():
    # Given: one config missing master_grid, and one that supplies master_grid
    #        but no reference
    base = {"tuning": {"input": "m.gpkg"}, "merge": {}}
    params = {"tuning": {"input": "m.gpkg", "master_grid": "g.tif"}}

    # When: _scan_settings resolves the config missing master_grid
    # Then: a ClickException names that dotted missing key
    with pytest.raises(click.ClickException, match="tuning.master_grid"):
        _scan_settings(base)

    # When: it resolves the config missing reference
    # Then: a ClickException names that dotted missing key instead
    with pytest.raises(click.ClickException, match="tuning.reference"):
        _scan_settings(params)


def test_scan_settings_degenerate_bounds_raise():
    # Given: an otherwise valid tuning config to layer degenerate bounds onto
    tuning = {
        "input": "m.gpkg",
        "master_grid": "g.tif",
        "reference": "r.geojson",
    }

    # When: _scan_settings validates factor_min == factor_max
    # Then: the empty range is rejected with a ClickException
    with pytest.raises(click.ClickException, match="strictly less"):
        _scan_settings({"tuning": {**tuning, "factor_min": 2.0, "factor_max": 2.0}})

    # When: it validates cutoff_min > cutoff_max
    # Then: the inverted range is rejected the same way
    with pytest.raises(click.ClickException, match="strictly less"):
        _scan_settings({"tuning": {**tuning, "cutoff_min": 0.5, "cutoff_max": 0.1}})


def test_scan_settings_base_name_strips_dir_and_extension():
    # Given: an input path with directories and a .gpkg extension
    s = _settings(input_path="/data/runs/merged_raw.gpkg")

    # When: base_name is read
    base_name = s.base_name

    # Then: only the extension-free file stem remains
    assert base_name == "merged_raw"


# ---------------------------------------------------------------------------
# _objective (sign adjustment + penalties)
# ---------------------------------------------------------------------------


def test_objective_sign_and_penalties():
    # Given: evaluators returning a min-metric, a max-metric, a failure (None),
    #        a NaN and a dict missing the requested metric
    min_eval = lambda f, c: {"rms": 2.5}  # noqa: E731
    max_eval = lambda f, c: {"spearman": 0.7}  # noqa: E731
    failed_eval = lambda f, c: None  # noqa: E731
    nan_eval = lambda f, c: {"rms": float("nan")}  # noqa: E731
    missing_eval = lambda f, c: {"mae": 1.0}  # noqa: E731

    # When: _objective converts each to the always-minimized scalar
    # Then: min metrics pass through, max metrics are negated, and every
    #       failure mode maps to LARGE_PENALTY
    assert _objective(min_eval, "rms", 0.0, 0.0) == 2.5
    assert _objective(max_eval, "spearman", 0.0, 0.0) == -0.7
    assert _objective(failed_eval, "rms", 0.0, 0.0) == LARGE_PENALTY
    assert _objective(nan_eval, "rms", 0.0, 0.0) == LARGE_PENALTY
    assert _objective(missing_eval, "rms", 0.0, 0.0) == LARGE_PENALTY


# ---------------------------------------------------------------------------
# _make_evaluator: metric values, keep semantics, bests bookkeeping
# ---------------------------------------------------------------------------


def _evaluator_fixture():
    """2x2 grid, one masked-out cell, four candidate points.

    Cell layout (row, col) -> reference counts:
        (0,0)=2  (0,1)=0
        (1,0)=0  (1,1)=1 but (1,1) is OUTSIDE the analysis mask
    """
    return _grouped(
        points=[
            (0.9, 0.9, 0, 0),
            (0.1, 0.4, 0, 1),
            (0.2, 0.2, 0, 0),
            (0.05, 0.05, 1, 1),
        ],
        val_counts=[[2.0, 0.0], [0.0, 1.0]],
        masked_out=[(1, 1)],
    )


def test_evaluator_hand_derived_metrics():
    # Given: the 2x2 fixture wired to a fresh bests dict and an empty trace
    grouped = _evaluator_fixture()
    scan_metrics = ["rms", "mae", "abs_total_diff"]
    bests = _fresh_bests(scan_metrics)
    trace = []
    evaluate = _make_evaluator(grouped, scan_metrics, bests, trace)

    # When: the evaluator scores factor=1, cutoff=0.3, so only the two points
    #       with adjusted_peak >= 0.3 survive -> pred cells (0,0)=1, (0,1)=1;
    #       in-mask pred=[1,1,0] vs val=[2,0,0]
    m = evaluate(1.0, 0.3)

    # Then: rms=sqrt(2/3), mae=2/3, abs_total_diff=0, rmsle and spearman match
    #       the hand-derived values, and the masked cell (1,1) is excluded
    assert m["n_cells"] == 3
    assert m["total_pred"] == 2.0
    assert m["total_val"] == 2.0
    assert m["abs_total_diff"] == 0.0
    assert m["total_pdiff"] == 0.0
    assert m["rms"] == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-6)
    assert m["mae"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    ln2, ln3 = math.log(2.0), math.log(3.0)
    expected_rmsle = math.sqrt(((ln2 - ln3) ** 2 + ln2**2 + 0.0) / 3.0)
    assert m["rmsle"] == pytest.approx(expected_rmsle, rel=1e-6)
    # ranks: pred [2.5, 2.5, 1] vs val [3, 1.5, 1.5] -> Pearson of ranks = 0.5
    assert m["spearman"] == pytest.approx(0.5, abs=1e-12)

    # Then: the trace records that one evaluation with its scan metrics
    assert len(trace) == 1
    assert trace[0]["factor"] == 1.0
    assert trace[0]["cutoff"] == 0.3
    assert set(scan_metrics) <= set(trace[0])

    # Then: bests picks it up, carrying the keep mask of the two survivors
    assert bests["rms"]["value"] == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-6)
    assert bests["rms"]["factor"] == 1.0
    assert bests["rms"]["cutoff"] == 0.3
    np.testing.assert_array_equal(bests["rms"]["keep"], [True, True, False, False])


def test_evaluator_per_metric_bests_and_ties():
    # Given: the 2x2 fixture wired to a bests dict and trace shared across
    #        every evaluation below
    grouped = _evaluator_fixture()
    scan_metrics = ["rms", "mae", "abs_total_diff"]
    bests = _fresh_bests(scan_metrics)
    trace = []
    evaluate = _make_evaluator(grouped, scan_metrics, bests, trace)

    # When: (1.0, 0.3) is scored, then (1.0, 0.04) which improves rms/mae but
    #       worsens abs_total_diff
    evaluate(1.0, 0.3)
    m2 = evaluate(1.0, 0.04)  # keeps all 4 points: in-mask pred=[2,1,0]

    # Then: rms/mae move to cutoff=0.04 while abs_total_diff stays at cutoff=0.3
    assert m2["rms"] == pytest.approx(math.sqrt(1.0 / 3.0), rel=1e-6)
    assert m2["abs_total_diff"] == 1.0
    assert bests["rms"]["cutoff"] == 0.04
    assert bests["mae"]["cutoff"] == 0.04
    np.testing.assert_array_equal(bests["rms"]["keep"], [True, True, True, True])
    # abs_total_diff got worse (1 > 0) so its best is untouched
    assert bests["abs_total_diff"]["value"] == 0.0
    assert bests["abs_total_diff"]["cutoff"] == 0.3

    # When: (0.0, 0.3) is scored — factor=0 collapses rescaled peaks to
    #       peak_value, so only the 0.9 point survives 0.3 and in-mask
    #       pred=[1,0,0] gives rms=sqrt(1/3), an EXACT tie with the incumbent
    m3 = evaluate(0.0, 0.3)

    # Then: the tie does not displace the incumbent (strict improvement only),
    #       and all three evaluations are traced
    assert m3["total_pred"] == 1.0
    assert m3["rms"] == pytest.approx(math.sqrt(1.0 / 3.0), rel=1e-6)
    assert bests["rms"]["factor"] == 1.0
    assert bests["rms"]["cutoff"] == 0.04
    assert len(trace) == 3


def test_evaluator_abs_total_pdiff_value_and_objective_pass_through():
    # Given: a 1x2 fully-masked grid with reference counts [2, 0] and three
    #        prediction points (two in cell (0,0), one in (0,1)) that all
    #        survive factor=1, cutoff=0.5 -> pred=[2,1] vs val=[2,0], so
    #        total_pred=3 and total_val=2
    grouped = _grouped(
        points=[(0.9, 0.9, 0, 0), (0.8, 0.8, 0, 0), (0.7, 0.7, 0, 1)],
        val_counts=[[2.0, 0.0]],
    )
    scan_metrics = ["abs_total_pdiff"]
    bests = _fresh_bests(scan_metrics)
    trace = []
    evaluate = _make_evaluator(grouped, scan_metrics, bests, trace)

    # When: the evaluator computes the metrics at that point
    m = evaluate(1.0, 0.5)

    # Then: abs_total_pdiff = |(3 - 2) / 2| = 0.5, and it becomes the best
    assert m["total_pred"] == 3.0
    assert m["total_val"] == 2.0
    assert m["total_pdiff"] == pytest.approx(0.5)
    assert m["abs_total_pdiff"] == pytest.approx(0.5)
    assert bests["abs_total_pdiff"]["value"] == pytest.approx(0.5)

    # When: _objective consumes the abs_total_pdiff metric at the same point
    # Then: it returns the value unnegated (+0.5) because abs_total_pdiff is
    #       minimized
    assert _objective(evaluate, "abs_total_pdiff", 1.0, 0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _optimize_metric: ridge fit, along-ridge clipping, Nelder-Mead seeding
# ---------------------------------------------------------------------------


def test_optimize_metric_ridge_fit_recovers_linear_ridge():
    # Given: a synthetic objective (cutoff - (0.3*factor + 0.05))**2 whose
    #        best cutoff at each factor lies exactly on the line
    #        cutoff = 0.3*factor + 0.05, strictly inside the cutoff bounds at
    #        all three probe factors (0 -> 0.05, 1 -> 0.35, 2 -> 0.65)
    def evaluate(factor, cutoff):
        return {"rms": (cutoff - (0.3 * factor + 0.05)) ** 2}

    # When: _optimize_metric probes the ridge (Phase 1) and fits
    #       cutoff = a*factor + b through the probe optima (Phase 2)
    a, b = _optimize_metric(
        evaluate=evaluate,
        metric="rms",
        bests=_fresh_bests(["rms"]),
        factor_bounds=(0.0, 2.0),
        cutoff_bounds=(0.0, 1.0),
        n_probes=3,
        xtol_factor=1e-3,
        xtol_cutoff=1e-6,
        refine_maxiter=0,
    )

    # Then: the returned coefficients recover (a, b) = (0.3, 0.05) — the fit
    #       regresses cutoff on factor, not factor on cutoff
    assert a == pytest.approx(0.3, abs=1e-2)
    assert b == pytest.approx(0.05, abs=1e-2)


def test_optimize_metric_along_ridge_cutoff_clipped_to_cutoff_bounds():
    # Given: a tight cutoff box (0.2, 0.25) inside wide factor bounds (0, 10)
    #        and an objective whose per-factor best cutoff
    #        0.25 + 0.003*f - 0.0008*f**2 leaves the box at both ends, so the
    #        Phase-1 probe optima (f=0, 5, 10) pin near (0.25, 0.245, 0.2) and
    #        the Phase-2 least-squares line b ~ 0.257 overshoots cutoff_max
    #        near factor 0; the + 0.001*factor term pulls the Phase-3
    #        along-ridge search toward factor 0, exactly where the fitted
    #        line extrapolates out of the box
    calls = []
    bests = _fresh_bests(["rms"])

    def evaluate(factor, cutoff):
        calls.append((float(factor), float(cutoff)))
        ridge_cutoff = 0.25 + 0.003 * factor - 0.0008 * factor**2
        value = (cutoff - ridge_cutoff) ** 2 + 0.001 * factor
        if is_better("rms", value, bests["rms"]["value"]):
            bests["rms"] = {
                "value": float(value),
                "factor": float(factor),
                "cutoff": float(cutoff),
                "keep": None,
            }
        return {"rms": value}

    # When: _optimize_metric walks along the fitted ridge
    _optimize_metric(
        evaluate=evaluate,
        metric="rms",
        bests=bests,
        factor_bounds=(0.0, 10.0),
        cutoff_bounds=(0.2, 0.25),
        n_probes=3,
        xtol_factor=1e-3,
        xtol_cutoff=1e-6,
        refine_maxiter=0,
    )

    # Then: every evaluated cutoff — the Phase-3 ones included — is clipped
    #       into the CUTOFF bounds, and the best cutoff found lies in the box
    assert len(calls) > 3
    assert all(0.2 <= c <= 0.25 for _, c in calls)
    assert 0.2 <= bests["rms"]["cutoff"] <= 0.25


def test_optimize_metric_nelder_mead_seeded_at_incumbent():
    # Given: a runner over a flat objective (every evaluation returns rms=1.0,
    #        never a strict improvement, so the bests dict stays untouched)
    #        with an incumbent best pinned beforehand at the asymmetric point
    #        (factor=0.7, cutoff=0.2), which no Phase 1-3 evaluation hits
    #        (probe factors are 0 and 2; Brent visits golden-section points)
    def run(refine_maxiter):
        calls = []

        def evaluate(factor, cutoff):
            calls.append((float(factor), float(cutoff)))
            return {"rms": 1.0}

        bests = {"rms": {"value": 1.0, "factor": 0.7, "cutoff": 0.2, "keep": None}}
        _optimize_metric(
            evaluate=evaluate,
            metric="rms",
            bests=bests,
            factor_bounds=(0.0, 2.0),
            cutoff_bounds=(0.0, 1.0),
            n_probes=2,
            xtol_factor=1e-2,
            xtol_cutoff=1e-2,
            refine_maxiter=refine_maxiter,
        )
        return calls

    # When: _optimize_metric runs once with refine_maxiter=0 (Phases 1-3 only)
    #       and once with refine_maxiter=5 (adding Phase 4), both recording
    #       every evaluated point
    phases_1_to_3 = run(0)
    with_refine = run(5)

    # Then: the call sequences agree over Phases 1-3, and the first Phase-4
    #       evaluation is exactly the incumbent (factor, cutoff) — in that
    #       order — where Nelder-Mead must be seeded
    assert with_refine[: len(phases_1_to_3)] == phases_1_to_3
    assert len(with_refine) > len(phases_1_to_3)
    assert with_refine[len(phases_1_to_3)] == (0.7, 0.2)


# ---------------------------------------------------------------------------
# scan_tile: the ridge-scan core
# ---------------------------------------------------------------------------


def test_scan_tile_finds_separating_cutoff():
    # Given: one signal point (rescaled peak 0.5 at a cell with reference
    #        count 1) and one noise point (rescaled peak 0.0 at a count-0
    #        cell); any cutoff in (0, 0.5] yields a perfect rms of 0
    grouped = _grouped(
        points=[(0.5, 0.5, 0, 0), (0.0, 0.0, 0, 1)],
        val_counts=[[1.0, 0.0]],
    )

    # When: scan_tile searches factor in [0,2] and cutoff in [0.01, 1.0]
    bests, trace, ridges = scan_tile(
        grouped=grouped,
        factor_bounds=(0.0, 2.0),
        cutoff_bounds=(0.01, 1.0),
        scan_metrics=("rms",),
        n_probes=3,
        xtol_factor=1e-2,
        xtol_cutoff=1e-3,
        refine_maxiter=10,
    )

    # Then: the best rms is exactly 0, achieved with a cutoff in (0.01, 0.5]
    #       that keeps only the signal point
    best = bests["rms"]
    assert best["value"] == 0.0
    assert 0.01 <= best["cutoff"] <= 0.5
    assert 0.0 <= best["factor"] <= 2.0
    np.testing.assert_array_equal(best["keep"], [True, False])

    # Then: the fitted ridge cutoff = a*factor + b comes from a polyfit through
    #       the per-probe Brent optima, all of which lie in the cutoff bounds —
    #       so both coefficients must be finite
    a, b = ridges["rms"]
    assert math.isfinite(a) and math.isfinite(b)

    # Then: every visited point stays inside the bounds. Phase 1 alone runs
    #       n_probes bounded Brent searches, each evaluating at least once, so
    #       the trace cannot be shorter than n_probes.
    assert len(trace) >= 3
    for t in trace:
        assert 0.0 <= t["factor"] <= 2.0
        assert 0.01 - 1e-9 <= t["cutoff"] <= 1.0 + 1e-9


def test_scan_tile_metric_never_finite_leaves_best_unset():
    # Given: a reference raster of all zeros, so val never varies and
    #        spearman is NaN at every evaluation
    grouped = _grouped(
        points=[(0.5, 0.5, 0, 0), (0.1, 0.1, 0, 1)],
        val_counts=[[0.0, 0.0]],
    )

    # When: scan_tile optimizes the spearman metric alone
    bests, trace, _ = scan_tile(
        grouped=grouped,
        factor_bounds=(0.0, 1.0),
        cutoff_bounds=(0.01, 1.0),
        scan_metrics=("spearman",),
        n_probes=2,
        xtol_factor=0.1,
        xtol_cutoff=0.05,
        refine_maxiter=5,
    )

    # Then: no evaluation ever counts as an improvement — factor/cutoff stay
    #       None and the value stays at the -inf sentinel
    assert bests["spearman"]["factor"] is None
    assert bests["spearman"]["cutoff"] is None
    assert bests["spearman"]["value"] == -float("inf")
    # Guard against a vacuous all(): the two Phase-1 probes each evaluate.
    assert len(trace) >= 2
    assert all(math.isnan(t["spearman"]) for t in trace)


# ---------------------------------------------------------------------------
# _prediction_date: date parsing / median selection on real files
# ---------------------------------------------------------------------------


def test_prediction_date_median_ignores_non_point_files(tmp_path):
    # Given: three date-stamped point files (mixed YYYYMMDD / YYYY-MM-DD
    #        stamps and extensions) plus a .txt file with an earlier date
    for name in (
        "pred_20230101.geojson",
        "pred_2023-06-01.gpkg",
        "pred_20231215.json",
        "notes_20200101.txt",
    ):
        (tmp_path / name).touch()
    s = _settings(pred_folder=str(tmp_path))

    # When: _prediction_date infers the target date
    inferred = _prediction_date(s)

    # Then: it returns the median of the three point-file dates, ignoring
    #       the .txt file entirely
    assert inferred == datetime(2023, 6, 1)


def test_prediction_date_even_count_and_missing_folder(tmp_path):
    # Given: a folder holding two date-stamped files (an even count)
    (tmp_path / "a_20230101.geojson").touch()
    (tmp_path / "b_20230601.geojson").touch()

    # When: _prediction_date runs on that folder
    even_count = _prediction_date(_settings(pred_folder=str(tmp_path)))

    # Then: the even count picks the upper median (index len//2)
    assert even_count == datetime(2023, 6, 1)

    # When: it runs with no folder set and with a nonexistent folder
    unset = _prediction_date(_settings(pred_folder=None))
    missing = str(tmp_path / "does_not_exist")
    nonexistent = _prediction_date(_settings(pred_folder=missing))

    # Then: both return None
    assert unset is None
    assert nonexistent is None


# ---------------------------------------------------------------------------
# _write_best_params: serialization round-trip
# ---------------------------------------------------------------------------


def test_write_best_params_yaml_round_trip(tmp_path):
    # Given: bests for two metrics with mae chosen as the eval metric, and a
    #        best_params path inside a not-yet-existing subdirectory
    bests = {
        "mae": {"value": 0.25, "factor": 1.5, "cutoff": 0.004, "keep": None},
        "rms": {"value": 0.5, "factor": 2.0, "cutoff": 0.005, "keep": None},
    }
    path = tmp_path / "sub" / "best_params.yaml"
    s = _settings(
        input_path="/runs/merged_raw.gpkg",
        metric="mae",
        scan_metrics=["mae", "rms"],
        best_params_path=str(path),
    )

    # When: _write_best_params writes the YAML
    returned = _write_best_params(s, bests, bests["mae"])

    # Then: the file parses back with (factor, cutoff) renamed to the merge
    #       keys (adjustment_factor, min_adj_peak), the chosen metric at top
    #       level, and every tracked metric under "bests"
    assert path.is_file()
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded == returned
    assert loaded["metric"] == "mae"
    assert loaded["value"] == pytest.approx(0.25)
    assert loaded["adjustment_factor"] == pytest.approx(1.5)
    assert loaded["min_adj_peak"] == pytest.approx(0.004)
    assert loaded["input"] == "/runs/merged_raw.gpkg"
    assert loaded["bests"]["rms"] == {
        "value": 0.5,
        "adjustment_factor": 2.0,
        "min_adj_peak": 0.005,
    }
