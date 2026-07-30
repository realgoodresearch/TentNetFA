"""Tests for displacement_tracker.util.config: dotted-path helpers, deep
merging, and sectioned shared/per-flow config resolution."""

import click
import pytest

from displacement_tracker.util.config import (
    deep_get,
    deep_merge,
    deep_set,
    forwarded,
    is_sectioned_config,
    load_flow_config,
    require,
    resolve_flow_config,
)


def test_deep_get_returns_nested_value():
    # Given: a three-level nested dict with value 7 at a.b.c
    cfg = {"a": {"b": {"c": 7}}}

    # When: deep_get is asked for the dotted path "a.b.c" and for "a.b"
    leaf = deep_get(cfg, "a.b.c")
    branch = deep_get(cfg, "a.b")

    # Then: it returns 7, and a one-level path returns the intermediate dict
    assert leaf == 7
    assert branch == {"c": 7}


def test_deep_get_missing_or_non_dict_returns_default():
    # Given: a dict where "a.b" is a scalar and "x" does not exist
    cfg = {"a": {"b": 5}, "lst": [1, 2]}

    # When: deep_get traverses a missing key or descends through a non-dict
    absent_root = deep_get(cfg, "x", default="dflt")
    through_scalar = deep_get(cfg, "a.b.c", default=-1)  # 5 is not a dict
    into_list = deep_get(cfg, "lst.0")  # lists are not traversed
    absent_leaf = deep_get(cfg, "a.missing", default=0)

    # Then: the supplied default is returned instead of raising
    assert absent_root == "dflt"
    assert through_scalar == -1
    assert into_list is None
    assert absent_leaf == 0


def test_require_rejects_missing_and_falsy_values():
    # Given: a config where a dotted path is set, another is empty and a
    #        third is absent entirely
    cfg = {"evaluation": {"annotation_csv": "ann.csv", "output_dir": ""}}

    # When: require is asked for the dotted path that is set
    value = require(cfg, "evaluation.annotation_csv")

    # Then: its value is returned
    assert value == "ann.csv"

    # When: require is asked for the empty value and for the absent key
    # Then: both raise ClickException naming the dotted path to fix
    with pytest.raises(click.ClickException, match="evaluation.output_dir"):
        require(cfg, "evaluation.output_dir")
    with pytest.raises(click.ClickException, match="boundaries"):
        require(cfg, "boundaries")


def test_require_with_fallbacks_returns_first_set_value():
    # Given: a config where the primary path is unset but a fallback is
    cfg = {"merge": {"output": "merged.gpkg"}}

    # When: require is asked for tuning.input, falling back to merge.output
    value = require(cfg, "tuning.input", "merge.output")

    # Then: the fallback's value is returned
    assert value == "merged.gpkg"


def test_require_with_fallbacks_raises_naming_all_of_them():
    # Given: a config where neither the primary path nor its fallback is set
    cfg = {}

    # When: require is asked for a primary path with two fallbacks
    # Then: it names the primary and lists every fallback, so the message
    #       tells the reader every place they could set the value
    with pytest.raises(
        click.ClickException,
        match=r"tuning\.input \(or merge\.output / legacy\.input as fallback\)",
    ):
        require(cfg, "tuning.input", "merge.output", "legacy.input")


def test_forwarded_returns_only_the_keys_the_config_sets():
    # Given: a config naming one key, leaving a second null, and carrying a
    #        third that is not asked for
    cfg = {
        "model_column": "model_beta2",
        "manual_column": None,
        "output_dir": "results",
    }

    # When: forwarded is asked for the two column keys
    kwargs = forwarded(cfg, "manual_column", "model_column")

    # Then: only the set one comes back, so the callee's own signature
    #       default supplies the rest instead of a second copy of it here
    assert kwargs == {"model_column": "model_beta2"}


def test_forwarded_with_no_keys_set_is_empty():
    # Given: a config setting neither of the two keys asked for
    cfg = {"manual_column": "manual_tent_count"}

    # When: forwarded is asked for a key that is absent
    # Then: nothing is forwarded, leaving every callee default in force
    assert forwarded(cfg, "hex_size_m") == {}


def test_deep_set_creates_intermediate_dicts():
    # Given: an empty config dict
    cfg = {}

    # When: deep_set writes value 1 at dotted path "a.b.c"
    deep_set(cfg, "a.b.c", 1)

    # Then: all intermediate dicts are created and the leaf holds 1
    assert cfg == {"a": {"b": {"c": 1}}}


def test_deep_set_replaces_non_dict_intermediates():
    # Given: a config where "a" is a scalar (5)
    cfg = {"a": 5, "keep": True}

    # When: deep_set writes at "a.b", which needs "a" to be a dict
    deep_set(cfg, "a.b", 2)

    # Then: the scalar is replaced by a fresh dict holding the leaf
    assert cfg == {"a": {"b": 2}, "keep": True}


def test_deep_merge_nested_extra_wins_and_mutates_base():
    # Given: base and extra sharing the nested key m.x with different values
    base = {"m": {"x": 1, "y": 2}, "only_base": "b"}
    extra = {"m": {"x": 10, "z": 3}, "only_extra": "e"}

    # When: deep_merge(base, extra) runs
    result = deep_merge(base, extra)

    # Then: extra's m.x wins, base-only keys survive at every level, and the
    #       returned object is the mutated base itself
    assert result is base
    assert base == {
        "m": {"x": 10, "y": 2, "z": 3},
        "only_base": "b",
        "only_extra": "e",
    }


def test_deep_merge_type_conflicts_extra_always_wins():
    # Given: base has a dict where extra has a scalar, and vice versa
    base = {"a": {"nested": 1}, "b": 2}
    extra = {"a": "scalar", "b": {"now": "dict"}}

    # When: deep_merge runs
    deep_merge(base, extra)

    # Then: extra's value replaces base's wholesale in both directions
    assert base == {"a": "scalar", "b": {"now": "dict"}}


def test_is_sectioned_config_detection():
    # Given: dicts with and without shared/train/predict/tune top-level keys,
    #        plus a value that is not a dict at all
    shared_marked = {"shared": {}}
    tune_marked = {"tune": {"a": 1}}
    flat = {"geotiff_dir": "/x", "merge": {}}

    # When: is_sectioned_config inspects each of them
    shared_result = is_sectioned_config(shared_marked)
    tune_result = is_sectioned_config(tune_marked)
    flat_result = is_sectioned_config(flat)
    non_dict_result = is_sectioned_config("not a dict")

    # Then: any section key marks it sectioned; flat configs and non-dicts
    #       are not sectioned
    assert shared_result is True
    assert tune_result is True
    assert flat_result is False
    assert non_dict_result is False


def test_resolve_flat_config_passes_through_unchanged():
    # Given: a legacy flat config (no section keys)
    cfg = {"geotiff_dir": "/data", "merge": {"agreement": 2}}

    # When: resolve_flow_config is called with a flow, and with None
    with_flow = resolve_flow_config(cfg, "predict")
    without_flow = resolve_flow_config(cfg, None)

    # Then: the very same object is returned untouched
    assert with_flow is cfg
    assert without_flow is cfg
    assert cfg == {"geotiff_dir": "/data", "merge": {"agreement": 2}}


def test_resolve_sectioned_flow_wins_over_shared():
    # Given: shared defines processing.core=100 and boundaries; the predict
    #        section overrides processing.core=50 and adds a model key
    cfg = {
        "shared": {
            "boundaries": "/b.shp",
            "processing": {"core": 100, "margin": 10},
        },
        "predict": {"model": "/m.pth", "processing": {"core": 50}},
        "train": {"epochs": 3},
    }

    # When: resolving flow "predict"
    resolved = resolve_flow_config(cfg, "predict")

    # Then: the flat result deep-merges predict over shared — core=50,
    #       shared-only keys (margin, boundaries) survive, and no section
    #       names appear in the output
    assert resolved == {
        "boundaries": "/b.shp",
        "processing": {"core": 50, "margin": 10},
        "model": "/m.pth",
    }
    assert "shared" not in resolved and "train" not in resolved


def test_resolve_sectioned_returns_deep_copies():
    # Given: a sectioned config with a nested dict in shared
    cfg = {
        "shared": {"processing": {"core": 100}},
        "tune": {"tuning": {"metric": "rms"}},
    }

    # When: the tune flow is resolved and the result is mutated afterwards
    resolved = resolve_flow_config(cfg, "tune")
    resolved["processing"]["core"] = -1
    resolved["tuning"]["metric"] = "mae"

    # Then: the original shared/flow sections are unaffected (deep copies)
    assert cfg["shared"]["processing"]["core"] == 100
    assert cfg["tune"]["tuning"]["metric"] == "rms"


def test_resolve_rejects_mixed_sectioned_and_stray_keys():
    # Given: a half-migrated config with a "shared" section plus stray
    #        top-level keys "zeta" and "alpha"
    cfg = {"shared": {}, "predict": {}, "zeta": 1, "alpha": 2}

    # When: resolving any flow
    # Then: click.UsageError names the stray keys in sorted order
    with pytest.raises(click.UsageError, match="alpha, zeta"):
        resolve_flow_config(cfg, "predict")


def test_resolve_sectioned_without_flow_raises():
    # Given: a valid sectioned config
    cfg = {"shared": {}, "predict": {"a": 1}}

    # When: resolve_flow_config is called with flow=None
    # Then: click.UsageError asks for --flow
    with pytest.raises(click.UsageError, match="--flow"):
        resolve_flow_config(cfg, None)


def test_resolve_unknown_flow_raises():
    # Given: a sectioned config and a flow name outside FLOWS
    cfg = {"shared": {}, "predict": {"a": 1}}

    # When: resolving flow "evaluate"
    # Then: click.UsageError reports the unknown flow name
    with pytest.raises(click.UsageError, match="evaluate"):
        resolve_flow_config(cfg, "evaluate")


def test_resolve_missing_flow_section_raises():
    # Given: a sectioned config that has predict but no train section
    cfg = {"shared": {"a": 1}, "predict": {"b": 2}}

    # When: resolving flow "train" (a valid flow name)
    # Then: click.UsageError reports the missing 'train' section
    with pytest.raises(click.UsageError, match="no 'train' section"):
        resolve_flow_config(cfg, "train")


def test_resolve_handles_null_shared_and_null_flow_sections():
    # Given: YAML-style empty sections parse to None — one config with
    #        shared: null, another with an empty tune: section
    null_shared = {"shared": None, "tune": {"a": 1}}
    null_flow = {"shared": {"b": 2}, "tune": None}

    # When: resolving the tune flow in each
    from_null_shared = resolve_flow_config(null_shared, "tune")
    from_null_flow = resolve_flow_config(null_flow, "tune")

    # Then: null shared resolves to just the flow section, and a null flow
    #       section resolves to just shared (no crash on None)
    assert from_null_shared == {"a": 1}
    assert from_null_flow == {"b": 2}


def test_load_flow_config_reads_yaml_substitutes_env_and_resolves(
    tmp_path, monkeypatch
):
    # Given: a real sectioned YAML file referencing ${TENTNET_TEST_DATA} in
    #        shared, with a train section overriding batch_size, and the env
    #        var set
    monkeypatch.setenv("TENTNET_TEST_DATA", "/mnt/data")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(
        "shared:\n"
        "  geotiff_dir: ${TENTNET_TEST_DATA}/tifs\n"
        "  training:\n"
        "    batch_size: 8\n"
        "train:\n"
        "  training:\n"
        "    batch_size: 32\n"
        "predict:\n"
        "  model: m.pth\n"
    )

    # When: load_flow_config loads it for flow "train"
    resolved = load_flow_config(str(cfg_file), "train")

    # Then: the env var is substituted into the value and the train section
    #       is deep-merged over shared (train's batch_size wins)
    assert resolved == {
        "geotiff_dir": "/mnt/data/tifs",
        "training": {"batch_size": 32},
    }
