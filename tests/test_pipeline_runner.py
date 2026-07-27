"""Tests for the pipeline specs (displacement_tracker.pipelines.spec) and
the run preparation logic (displacement_tracker.pipelines.runner).

prepare_run's config layering contract:
    extra_defaults < resolved base config < overrides < forced_values,
with artifact_paths forced into the run directory last.
"""

import os
import sys

import pytest
import yaml

from displacement_tracker.pipelines.runner import (
    default_run_root,
    prepare_run,
    stage_argv,
)
from displacement_tracker.pipelines.spec import PREDICT, TRAIN, TUNE, Pipeline


def _make_pipeline(**kwargs) -> Pipeline:
    defaults = dict(
        key="predict",
        label="Test pipeline",
        base_config="config.yaml",
        stages=(),
        params=(),
    )
    defaults.update(kwargs)
    return Pipeline(**defaults)


def test_subfolders_take_first_path_component_plus_logs():
    # Given: artifact paths "manifests", "dataset/balanced.parquet",
    #        "dataset/other.bin" and "deep/a/b"
    p = _make_pipeline(
        artifact_paths={
            "a": "manifests",
            "b": "dataset/balanced.parquet",
            "c": "dataset/other.bin",
            "d": "deep/a/b",
        }
    )

    # When: Pipeline.subfolders derives the fixed run-dir subfolders
    subfolders = p.subfolders

    # Then: each relative path contributes only its first component,
    #       duplicates collapse, "logs" is always included, sorted output
    assert subfolders == ["dataset", "deep", "logs", "manifests"]


def test_prepare_run_layering_defaults_base_overrides_forced(tmp_path):
    # Given: a pipeline with extra_defaults sec={a:1,b:2,d:4}, forced
    #        sec.b=99, and a flat base config sec={b:5,c:7} plus other=keep
    pipeline = _make_pipeline(
        extra_defaults={"sec": {"a": 1, "b": 2, "d": 4}},
        forced_values={"sec.b": 99},
    )
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"sec": {"b": 5, "c": 7}, "other": "keep"}))

    # When: prepare_run applies overrides {sec.c:8, sec.d:40, new.deep.key:True}
    ctx = prepare_run(
        pipeline,
        base,
        overrides={"sec.c": 8, "sec.d": 40, "new.deep.key": True},
        run_name="run1",
        run_root=tmp_path / "runs",
    )

    # Then: each key lands at its layer — a=1 (default only), b=99 (forced
    #       beats base's 5), c=8 (override beats base's 7), d=40 (override
    #       beats default's 4), other survives, and the dotted override
    #       created the nested "new" structure
    assert ctx.config["sec"] == {"a": 1, "b": 99, "c": 8, "d": 40}
    assert ctx.config["other"] == "keep"
    assert ctx.config["new"] == {"deep": {"key": True}}
    # The written config.yaml is the same resolved config, read back for real.
    with open(ctx.config_path) as f:
        assert yaml.safe_load(f) == ctx.config


def test_prepare_run_tune_invariants_survive_poisoned_config_and_overrides(tmp_path):
    # Given: a sectioned base config whose shared and tune sections poison
    #        the merge thresholds (min_adj_peak 0.9, adjustment_factor 5.0,
    #        thresholds_config set)
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "shared": {"merge": {"min_adj_peak": 0.9}},
                "predict": {"model": "m.pth"},
                "tune": {
                    "merge": {
                        "adjustment_factor": 5.0,
                        "thresholds_config": "/poison.yaml",
                        "agreement": 2,
                    },
                    "tuning": {"metric": "mae"},
                },
            }
        )
    )

    # When: prepare_run builds a TUNE run with overrides poisoning the same
    #       thresholds again
    ctx = prepare_run(
        TUNE,
        base,
        overrides={
            "merge.min_adj_peak": 0.5,
            "merge.adjustment_factor": 9.0,
            "merge.thresholds_config": "/still_poison.yaml",
            "merge.agreement": 4,
        },
        run_name="tune_run",
        run_root=tmp_path / "runs",
    )

    # Then: the config written to disk still carries the forced invariants —
    #       min_adj_peak 0.0, adjustment_factor 1.0, thresholds_config null —
    #       while non-forced merge keys keep their override/base values
    with open(ctx.config_path) as f:
        written = yaml.safe_load(f)
    assert written["merge"]["min_adj_peak"] == pytest.approx(0.0)
    assert written["merge"]["adjustment_factor"] == pytest.approx(1.0)
    assert written["merge"]["thresholds_config"] is None
    # Non-forced keys follow normal precedence: override wins over base.
    assert written["merge"]["agreement"] == 4
    # Sectioned resolution happened: flat config, no section keys, no
    # leakage from the predict section.
    assert "shared" not in written and "predict" not in written
    assert "model" not in written


def test_prepare_run_extra_defaults_fill_gaps_but_never_override(tmp_path):
    # Given: a flat tune-style base config that sets merge.min_distance_m=7.5
    #        and tuning.metric=mae but omits the other merge/tuning keys
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "merge": {"min_distance_m": 7.5},
                "tuning": {"metric": "mae"},
            }
        )
    )

    # When: prepare_run builds a TUNE run with no overrides
    ctx = prepare_run(TUNE, base, run_name="r", run_root=tmp_path / "runs")

    # Then: explicit base values survive (7.5, mae) while absent keys are
    #       filled from TUNE.extra_defaults (agreement 1, ridge_probes 5,
    #       cutoff_min 0.0001, reference.type "vector")
    cfg = ctx.config
    assert cfg["merge"]["min_distance_m"] == pytest.approx(7.5)
    assert cfg["merge"]["agreement"] == 1
    assert cfg["tuning"]["metric"] == "mae"
    assert cfg["tuning"]["ridge_probes"] == 5
    assert cfg["tuning"]["cutoff_min"] == pytest.approx(1.0e-4)
    assert cfg["tuning"]["reference"] == {"type": "vector"}


def test_prepare_run_forces_artifact_paths_into_run_dir(tmp_path):
    # Given: a TUNE base config and an override that points merge.output
    #        somewhere outside the run directory
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"tune": {"tuning": {"metric": "rms"}}}))
    run_root = tmp_path / "runs"

    # When: prepare_run builds the run
    ctx = prepare_run(
        TUNE,
        base,
        overrides={"merge.output": "/elsewhere/out.gpkg"},
        run_name="myrun",
        run_root=run_root,
    )

    # Then: every artifact path is rewritten to the fixed location inside
    #       <run_root>/tune/<run_name>, the fixed subfolders exist on disk,
    #       and the resolved config lands at <run_dir>/config.yaml
    run_dir = run_root / "tune" / "myrun"
    assert ctx.run_dir == run_dir
    assert ctx.config_path == run_dir / "config.yaml"
    assert ctx.config_path.is_file()
    assert ctx.config["merge"]["output"] == str(run_dir / "merged_raw/merged_raw.gpkg")
    assert ctx.config["tuning"]["out_dir"] == str(run_dir / "tuning")
    assert ctx.config["tuning"]["best_params"] == str(
        run_dir / "tuning/best_params.yaml"
    )
    assert ctx.config["tuning"]["final_output"] == str(
        run_dir / "merged/merged_tuned.gpkg"
    )
    for sub in ("logs", "merged", "merged_raw", "tuning"):
        assert (run_dir / sub).is_dir()


def test_prepare_run_does_not_mutate_the_pipeline_spec(tmp_path):
    # Given: the shared TUNE spec and a tune base config
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"tune": {"tuning": {"metric": "rms"}}}))

    # When: prepare_run applies overrides that write into paths whose values
    #       originate from extra_defaults (tuning.reference.type)
    prepare_run(
        TUNE,
        base,
        overrides={"tuning.reference.type": "raster", "merge.agreement": 12},
        run_name="first",
        run_root=tmp_path / "runs",
    )

    # Then: TUNE.extra_defaults is untouched
    assert TUNE.extra_defaults["tuning"]["reference"] == {"type": "vector"}
    assert TUNE.extra_defaults["merge"]["agreement"] == 1

    # When: a second run is prepared without overrides
    ctx2 = prepare_run(TUNE, base, run_name="second", run_root=tmp_path / "runs")

    # Then: it still gets the pristine defaults
    assert ctx2.config["tuning"]["reference"]["type"] == "vector"
    assert ctx2.config["merge"]["agreement"] == 1


def test_prepare_run_predict_fills_documented_merge_defaults(tmp_path):
    # Given: a minimal flat base config with no merge section at all
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"geotiff_dir": "/tifs"}))

    # When: prepare_run builds a PREDICT run with no overrides
    ctx = prepare_run(PREDICT, base, run_name="r", run_root=tmp_path / "runs")

    # Then: the resolved config carries exactly the documented merge
    #       defaults — min_distance_m 3.0, agreement 1 (single-model runs
    #       keep every point), min_adj_peak 0.0, adjustment_factor 1.0,
    #       and null thresholds_config / exclusion_zones_gpkg /
    #       inclusion_zone — plus merge.output, which is artifact-forced
    #       into the run directory rather than defaulted
    merge = dict(ctx.config["merge"])
    assert merge.pop("output") == str(ctx.run_dir / "merged/merged.gpkg")
    assert merge == {
        "min_distance_m": 3.0,
        "agreement": 1,
        "min_adj_peak": 0.0,
        "adjustment_factor": 1.0,
        "thresholds_config": None,
        "exclusion_zones_gpkg": None,
        "inclusion_zone": None,
    }


def test_prepare_run_predict_artifacts_chain_the_stages_through_run_dir(tmp_path):
    # Given: a flat predict base config pointing the stage folders at
    #        locations outside the run directory
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "manifest_folder": "/elsewhere/manifests",
                "prediction": {"input_folder": "/elsewhere/preds"},
            }
        )
    )

    # When: prepare_run builds a PREDICT run
    ctx = prepare_run(PREDICT, base, run_name="p", run_root=tmp_path / "runs")

    # Then: the scanner's manifest_folder and the predictor's input_folder
    #       both resolve to <run_dir>/manifests (predict reads exactly what
    #       scan wrote), predictions land in <run_dir>/preds, and the merge
    #       output at <run_dir>/merged/merged.gpkg — the base values lose
    run_dir = tmp_path / "runs" / "predict" / "p"
    assert ctx.run_dir == run_dir
    assert ctx.config["manifest_folder"] == str(run_dir / "manifests")
    assert ctx.config["prediction"]["input_folder"] == str(run_dir / "manifests")
    assert ctx.config["prediction"]["output_folder"] == str(run_dir / "preds")
    assert ctx.config["merge"]["output"] == str(run_dir / "merged/merged.gpkg")


def test_prepare_run_train_artifacts_chain_the_stages_through_run_dir(tmp_path):
    # Given: a flat train base config pointing the manifest elsewhere
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"manifest": "/elsewhere/old.parquet"}))

    # When: prepare_run builds a TRAIN run
    ctx = prepare_run(TRAIN, base, run_name="t", run_root=tmp_path / "runs")

    # Then: manifest_folder resolves to <run_dir>/manifests, the
    #       rebalancer's output and the trainer's manifest both to
    #       <run_dir>/dataset/balanced.parquet (train reads exactly what
    #       rebalance wrote), and model artifacts to <run_dir>/model
    run_dir = tmp_path / "runs" / "train" / "t"
    assert ctx.run_dir == run_dir
    assert ctx.config["manifest_folder"] == str(run_dir / "manifests")
    assert ctx.config["rebalancing"]["out"] == str(run_dir / "dataset/balanced.parquet")
    assert ctx.config["manifest"] == str(run_dir / "dataset/balanced.parquet")
    assert ctx.config["artifact_dir"] == str(run_dir / "model")


def test_stage_argv_runs_the_stage_module_on_the_resolved_config(tmp_path):
    # Given: a prepared PREDICT run context and its scan stage
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"geotiff_dir": "/tifs"}))
    ctx = prepare_run(PREDICT, base, run_name="r", run_root=tmp_path / "runs")
    scan = PREDICT.stages[1]
    assert scan.key == "scan"

    # When: stage_argv builds the stage command line
    argv = stage_argv(ctx, scan)

    # Then: it is exactly [current interpreter, -m, the stage's module,
    #       the resolved config written into the run dir] — every stage
    #       executes against the run's own config.yaml, nothing else
    assert argv == [
        sys.executable,
        "-m",
        "displacement_tracker.b2_image_scanner",
        str(ctx.run_dir / "config.yaml"),
    ]


def test_download_stage_is_opt_in_for_predict_and_train():
    # Given: the shipped PREDICT and TRAIN specs
    # When: a frontend selects the stages to run by their defaults
    predict_stages = [(s.key, s.default_enabled) for s in PREDICT.stages]
    train_stages = [(s.key, s.default_enabled) for s in TRAIN.stages]

    # Then: "download" is the only stage disabled by default in both
    #       pipelines — otherwise every default run would re-download
    #       GeoTIFFs from Drive, clobbering curated local imagery — and
    #       the remaining stages run in their documented order
    assert predict_stages == [
        ("download", False),
        ("scan", True),
        ("predict", True),
        ("merge", True),
    ]
    assert train_stages == [
        ("download", False),
        ("scan", True),
        ("rebalance", True),
        ("train", True),
    ]


def test_default_run_root_prefers_data_dir(monkeypatch):
    # Given: DATA_DIR set in the environment, and default_run_root's
    #        load_dotenv() call stubbed out so a developer's .env cannot
    #        contribute a value either way
    monkeypatch.setattr(
        "displacement_tracker.pipelines.runner.load_dotenv", lambda: None
    )
    monkeypatch.setenv("DATA_DIR", "/big/disk")

    # When: default_run_root resolves the run root
    root = default_run_root()

    # Then: it is <DATA_DIR>/results/TentNetFA/pipeline_runs
    assert root == os.path.join("/big/disk", "results", "TentNetFA", "pipeline_runs")


def test_default_run_root_falls_back_to_local_repo(monkeypatch):
    # Given: no DATA_DIR in the environment, with load_dotenv() stubbed so
    #        a local .env cannot put it back (clearing the variable alone
    #        would not survive the load_dotenv() inside default_run_root)
    monkeypatch.setattr(
        "displacement_tracker.pipelines.runner.load_dotenv", lambda: None
    )
    monkeypatch.delenv("DATA_DIR", raising=False)

    # When: default_run_root resolves the run root
    root = default_run_root()

    # Then: it falls back to the repo-local runs/pipelines
    assert root == os.path.join("runs", "pipelines")
