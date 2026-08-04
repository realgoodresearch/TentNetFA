"""Tests that a Hugging Face model reference survives config loading.

``config.yaml`` goes through ``${ENV}`` substitution and shared/per-flow
merging before any stage sees it. Neither step may mangle an ``hf:``
reference — the ``@``, the ``#`` and the 40-character commit SHA all have to
arrive intact, or a pinned model silently stops being pinned.
"""

import pytest

from displacement_tracker.util.config import load_flow_config
from displacement_tracker.util.model_ref import HubRef, parse_model_ref

REPO = "realgoodresearch/tentnetfa"
SHA = "a" * 40
FILE = "best_model.safetensors"
REF = f"hf:{REPO}@{SHA}#{FILE}"

CONFIG = f"""
shared:
  boundaries: gaza_boundaries/GazaStrip_MunicipalBoundaries.shp
train:
  training:
    checkpoint: {REF}
predict:
  prediction:
    model: {REF}
"""


@pytest.fixture
def config_path(tmp_path):
    """A sectioned config pinning a model in both flows, written unquoted."""
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return str(path)


def test_a_hub_reference_survives_substitution_and_merging(config_path):
    # Given: a sectioned config pinning a model in the predict and train
    #        flows, written unquoted as a user would write it

    # When: the predict flow is resolved
    predict = load_flow_config(config_path, "predict")

    # Then: the reference arrives byte-for-byte — the bare "#" did not start
    #       a YAML comment and the "@" was not reinterpreted
    assert predict["prediction"]["model"] == REF

    # When: the train flow is resolved
    train = load_flow_config(config_path, "train")

    # Then: its own pinned checkpoint arrives intact too
    assert train["training"]["checkpoint"] == REF


def test_a_pinned_reference_from_a_config_parses_back(config_path):
    # Given: a config whose predict flow pins a model
    resolved = load_flow_config(config_path, "predict")

    # When: the stored reference is parsed as the resolver would
    ref = parse_model_ref(resolved["prediction"]["model"])

    # Then: the round trip through YAML yields the same three parts
    assert ref == HubRef(REPO, SHA, FILE)
