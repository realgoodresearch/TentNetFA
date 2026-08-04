"""Tests that a Hugging Face model reference survives config loading.

``config.yaml`` goes through ``${ENV}`` substitution and shared/per-flow
merging before any stage sees it. Neither step may mangle an ``hf:``
reference — the ``@``, the ``#`` and the 40-character commit SHA all have to
arrive intact, or a pinned model silently stops being pinned.
"""

import pytest

from displacement_tracker.util.config import load_flow_config
from displacement_tracker.util.model_ref import (
    HubRef,
    ModelRefError,
    parse_model_ref,
    resolve_model_ref,
)

REPO = "realgoodresearch/tentnetfa"
SHA = "a" * 40
FILE = "best_model.safetensors"
REF = f"hf:{REPO}@{SHA}#{FILE}"

# Unquoted, and every character an octal digit: YAML 1.1 resolves this to an
# int, not a string. Written the way a user would write it in config.yaml.
OCTAL_LOOKING_DIGEST = "01234567" * 8

CONFIG = f"""
shared:
  boundaries: gaza_boundaries/GazaStrip_MunicipalBoundaries.shp
train:
  training:
    checkpoint: {REF}
predict:
  prediction:
    model: {REF}
    model_sha256: {OCTAL_LOOKING_DIGEST}
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


def test_a_digest_yaml_read_as_a_number_names_the_quoting_fix(config_path, tmp_path):
    # Given: a config whose digest was written unquoted and is all octal
    #        digits, so YAML hands back an int rather than a string
    digest = load_flow_config(config_path, "predict")["prediction"]["model_sha256"]
    assert isinstance(digest, int)  # the hazard this test exists for

    # Given: a local checkpoint to resolve against
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"weights")

    # When: the checkpoint is resolved with that value as its expected digest
    with pytest.raises(ModelRefError) as err:
        resolve_model_ref(str(checkpoint), sha256=digest)

    # Then: the error names the real fix — quoting the value — rather than
    #       raising AttributeError from a string method on an int
    message = str(err.value)
    assert "not a 64-character hex digest" in message
    assert "quote it" in message

    # Then: it does not tell the user to delete a checkpoint that is fine
    assert "Delete it" not in message
