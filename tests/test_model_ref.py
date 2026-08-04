"""Tests for displacement_tracker.util.model_ref: parsing a model reference
into the Hub release or local checkpoint it names, and resolving a local one
to a verified file.

Parsing needs no network, and resolving a local checkpoint needs only real
files in tmp_path, so nothing here is stubbed.
"""

import hashlib
from pathlib import Path

import pytest

from displacement_tracker.util.model_ref import (
    HubRef,
    ModelRefError,
    parse_model_ref,
    resolve_model_ref,
)

SHA = "a" * 40
REPO = "realgoodresearch/tentnetfa"
FILE = "best_model.safetensors"
REF = f"hf:{REPO}@{SHA}#{FILE}"


# ---------------------------------------------------------------------------
# Hub references
# ---------------------------------------------------------------------------


def test_parses_a_fully_pinned_reference():
    # Given: a reference naming a repo, a commit SHA and a file
    # When: it is parsed
    ref = parse_model_ref(REF)

    # Then: all three parts are recovered, and it round-trips back to the
    #       text a config would hold
    assert ref == HubRef(REPO, SHA, FILE)
    assert str(ref) == REF


@pytest.mark.parametrize(
    "text, expected",
    [
        (f"hf:{REPO}", HubRef(REPO, None, None)),
        (f"hf:{REPO}@{SHA}", HubRef(REPO, SHA, None)),
        (f"hf:{REPO}#{FILE}", HubRef(REPO, None, FILE)),
        (f"hf:{REPO}@main#{FILE}", HubRef(REPO, "main", FILE)),
        (f"  hf:{REPO}@{SHA}#{FILE}  ", HubRef(REPO, SHA, FILE)),
    ],
)
def test_parses_references_with_parts_missing(text, expected):
    # Given: a reference missing a revision, a file, or both
    # When: it is parsed
    # Then: the absent parts read as None rather than raising, so whoever
    #       resolves it can report precisely which one is missing
    assert parse_model_ref(text) == expected


@pytest.mark.parametrize("kind", ["blob", "resolve"])
def test_parses_urls_copied_from_the_hub_ui(kind):
    # Given: a file URL of the kind the Hub UI offers, in either form
    url = f"https://huggingface.co/{REPO}/{kind}/{SHA}/{FILE}"

    # When: it is parsed
    # Then: it yields the same reference as the compact hf: syntax
    assert parse_model_ref(url) == HubRef(REPO, SHA, FILE)


def test_parses_a_url_naming_only_the_repository():
    # Given: a bare repository URL, with no revision or file
    # When: it is parsed
    # Then: the repo is recovered and the rest is left absent
    assert parse_model_ref(f"https://huggingface.co/{REPO}") == HubRef(REPO, None, None)


def test_parses_a_file_nested_in_a_subdirectory():
    # Given: a URL whose file sits below the repository root
    url = f"https://huggingface.co/{REPO}/blob/{SHA}/weights/{FILE}"

    # When: it is parsed
    # Then: the whole relative path is kept as the filename
    assert parse_model_ref(url).filename == f"weights/{FILE}"


def test_accepts_an_uppercase_sha():
    # Given: a SHA pasted in uppercase
    # When: the reference is parsed
    ref = parse_model_ref(f"hf:{REPO}@{SHA.upper()}#{FILE}")

    # Then: it is normalised to the Hub's lowercase form, so it still reads
    #       as a commit rather than as a branch name
    assert ref.revision == SHA


# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------


def test_local_checkpoint_paths_pass_through():
    # Given: the kind of path config.yaml has always held
    # When: it is parsed
    # Then: it comes back as a plain path, with Hub handling out of the way
    assert parse_model_ref("runs/20260511_104602/best_model.pth") == Path(
        "runs/20260511_104602/best_model.pth"
    )


@pytest.mark.parametrize(
    "path",
    ["runs/v2@1/best_model.pth", "runs/v2#1/best_model.pth"],
)
def test_a_path_containing_reference_punctuation_is_still_a_path(path):
    # Given: a local path whose directory name contains "@" or "#" — the
    #        characters that delimit a Hub reference
    # When: it is parsed
    # Then: it is still a path; only the hf: scheme and Hub URLs are special
    assert parse_model_ref(path) == Path(path)


def test_home_relative_paths_are_expanded():
    # Given: a path written relative to the home directory
    # When: it is parsed
    # Then: the tilde is expanded, so the file can actually be opened
    assert parse_model_ref("~/runs/best_model.pth").is_absolute()


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


def test_an_empty_reference_is_rejected():
    # Given: a config key left blank
    # When: it is parsed
    # Then: it raises rather than resolving to the current directory
    with pytest.raises(ModelRefError, match="empty"):
        parse_model_ref("   ")


@pytest.mark.parametrize("bad", [f"hf:{REPO}/extra@{SHA}", "hf:nameonly", "hf:/name"])
def test_malformed_repo_ids_are_rejected(bad):
    # Given: an hf: reference whose repo id is not <owner>/<name>
    # When: it is parsed
    # Then: it raises naming the expected syntax
    with pytest.raises(ModelRefError, match="repo id"):
        parse_model_ref(bad)


def test_a_hub_url_without_a_repository_name_is_rejected():
    # Given: a Hub URL naming only an owner
    # When: it is parsed
    # Then: it raises rather than inventing a repository
    with pytest.raises(ModelRefError, match="Hugging Face model URL"):
        parse_model_ref("https://huggingface.co/owner")


# ---------------------------------------------------------------------------
# Resolving a local checkpoint
# ---------------------------------------------------------------------------


def test_an_existing_local_checkpoint_resolves_to_itself(tmp_path):
    # Given: a checkpoint on disk
    path = tmp_path / "best_model.pth"
    path.write_bytes(b"weights")

    # When: it is resolved
    # Then: the same path comes back, unchanged from today's behaviour
    assert resolve_model_ref(str(path)) == path


def test_a_missing_local_checkpoint_is_reported(tmp_path):
    # Given: a path to a checkpoint that is not there
    # When: it is resolved
    # Then: it raises, naming the path, before any loader sees it
    with pytest.raises(ModelRefError, match="not found"):
        resolve_model_ref(str(tmp_path / "absent.pth"))


def test_a_hub_reference_is_refused_until_fetching_exists():
    # Given: a fully pinned Hub reference, which this build cannot fetch
    # When: it is resolved
    # Then: it says so plainly instead of failing as a missing file — the
    #       reference is well formed, the capability is what is absent
    with pytest.raises(ModelRefError, match="cannot fetch yet"):
        resolve_model_ref(REF)


def test_a_local_checkpoint_digest_is_verified(tmp_path):
    # Given: a checkpoint and the sha256 of its contents
    path = tmp_path / "best_model.pth"
    path.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()

    # When: it is resolved with the matching digest, in either accepted form
    # Then: it resolves
    assert resolve_model_ref(str(path), sha256=digest) == path
    assert resolve_model_ref(str(path), sha256=f"sha256:{digest.upper()}") == path

    # When: it is resolved with a digest that does not match
    # Then: it raises, quoting both digests, rather than loading the file
    with pytest.raises(ModelRefError, match=f"expected sha256 {'0' * 64}"):
        resolve_model_ref(str(path), sha256="0" * 64)


def test_a_malformed_expected_digest_is_reported_as_malformed(tmp_path):
    # Given: a checkpoint, and an expected digest one character short
    path = tmp_path / "best_model.pth"
    path.write_bytes(b"weights")

    # When: it is resolved with that digest
    # Then: the typo is named as a typo, rather than the file being reported
    #       as corrupt. (The config-level twin covers the YAML case that
    #       produces a non-string.)
    with pytest.raises(ModelRefError, match="not a 64-character hex digest"):
        resolve_model_ref(str(path), sha256="0" * 63)
