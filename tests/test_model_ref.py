"""Tests for displacement_tracker.util.model_ref: parsing a model reference,
resolving a local checkpoint, and fetching a pinned Hugging Face release.

Parsing needs no network, and resolving a local checkpoint needs only real
files in tmp_path. The Hub client's entry points are stubbed under the
network-boundary carve-out in ``docs/test-patterns.md`` §3: the stubs raise
what the Hub would raise, and the assertions are on the resulting behaviour —
the error a user acts on, or the file that comes back — never on the traffic
itself. The one exception the doc allows is the credential actually sent,
which is only observable here and has a security consequence.

The stubs live in this file rather than ``_helpers.py`` because it is their
only consumer, and they encode no on-disk format.
"""

import hashlib
from pathlib import Path

import pytest
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from displacement_tracker.util import model_ref
from displacement_tracker.util.model_ref import (
    HubRef,
    ModelRefError,
    parse_model_ref,
    resolve_model_ref,
)

SHA = "a" * 40
OTHER_SHA = "b" * 40
REPO = "realgoodresearch/tentnetfa"
FILE = "best_model.safetensors"
REF = f"hf:{REPO}@{SHA}#{FILE}"

# Must never appear in anything a user sees.
SENTINEL_TOKEN = "hf_sentineltokenvalue0123456789"


# ---------------------------------------------------------------------------
# Hub stubs
# ---------------------------------------------------------------------------


class StubResponse:
    """The few response fields huggingface_hub's HTTP errors read.

    Building these errors needs a response object. A stub keeps the suite off
    whichever HTTP client huggingface_hub happens to use this major version.
    """

    def __init__(self, status):
        self.status_code = status
        self.headers = {}
        self.text = ""
        self.request = type("StubRequest", (), {"method": "GET", "url": ""})()


def hub_error(cls, message, status=404):
    """Build a huggingface_hub HTTP error as the Hub would raise it."""
    return cls(message, response=StubResponse(status))


class FakeInfo:
    """Stands in for the ModelInfo returned by HfApi.model_info."""

    def __init__(self, sha=SHA, filenames=(FILE,)):
        self.sha = sha
        self.siblings = [
            type("Sibling", (), {"rfilename": name})() for name in filenames
        ]


class Calls:
    """Records what crossed the Hub boundary.

    Only the credential fields are asserted on; the rest exist so a stub can
    answer a call at all.
    """

    def __init__(self):
        self.api_tokens = []
        self.download_tokens = []


def stub_hub(
    monkeypatch,
    *,
    info=None,
    info_error=None,
    download=None,
    download_error=None,
):
    """Replace the Hub client's entry points; return the credential record.

    Also neutralises dotenv loading and HF_TOKEN, so a developer's real .env
    cannot decide the outcome of a test that asserts on the no-token path.
    """
    calls = Calls()

    class FakeApi:
        def __init__(self, token=None):
            calls.api_tokens.append(token)

        def model_info(self, repo_id, revision=None):
            if info_error is not None:
                raise info_error
            return info if info is not None else FakeInfo()

    def fake_download(*, token=None, **kwargs):
        calls.download_tokens.append(token)
        if download_error is not None:
            raise download_error
        return download

    monkeypatch.setattr(model_ref, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(model_ref, "HfApi", FakeApi)
    monkeypatch.setattr(model_ref, "hf_hub_download", fake_download)
    return calls


def write_cached_file(root, sha=SHA, name=FILE, content=b"weights"):
    """Write a file where the Hub cache would put it, and return its path."""
    path = root / f"models--{REPO.replace('/', '--')}" / "snapshots" / sha / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


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


# ---------------------------------------------------------------------------
# Pinning: only a commit SHA pins
# ---------------------------------------------------------------------------


def test_a_branch_is_rejected_with_the_sha_to_pin(monkeypatch):
    # Given: a reference naming a branch, which can move, and a Hub where
    #        that branch points at a known commit
    stub_hub(monkeypatch, info=FakeInfo(sha=SHA))

    # When: it is resolved
    # Then: it is refused, and the message carries the commit the branch
    #       points at, formatted ready to paste back into the config
    with pytest.raises(ModelRefError, match=f"branch or tag.*hf:{REPO}@{SHA}#{FILE}"):
        resolve_model_ref(f"hf:{REPO}@main#{FILE}")


def test_a_missing_revision_is_rejected_with_the_sha_to_pin(monkeypatch):
    # Given: a reference with no revision at all
    stub_hub(monkeypatch, info=FakeInfo(sha=SHA))

    # When: it is resolved
    # Then: it is refused rather than silently defaulting to main, and says
    #       what main currently points at
    with pytest.raises(
        ModelRefError, match=f"names no revision.*hf:{REPO}@{SHA}#{FILE}"
    ):
        resolve_model_ref(f"hf:{REPO}#{FILE}")


def test_a_missing_filename_is_rejected_with_the_candidates(monkeypatch):
    # Given: a pinned reference that names no file, in a repo holding
    #        weights alongside other artifacts
    stub_hub(
        monkeypatch,
        info=FakeInfo(filenames=(FILE, "training_log.csv", "old_model.pth")),
    )

    # When: it is resolved
    with pytest.raises(ModelRefError) as err:
        resolve_model_ref(f"hf:{REPO}@{SHA}")

    # Then: the message lists the weight files to choose from, and leaves out
    #       everything that could not be a checkpoint
    message = str(err.value)
    assert "names no file" in message
    assert FILE in message
    assert "old_model.pth" in message
    assert "training_log.csv" not in message


# ---------------------------------------------------------------------------
# Download and verification
# ---------------------------------------------------------------------------


def test_a_pinned_reference_resolves_to_the_cached_file(monkeypatch, tmp_path):
    # Given: a pinned reference whose file is in the cache
    path = write_cached_file(tmp_path)
    stub_hub(monkeypatch, download=str(path))

    # When: it is resolved
    # Then: that file comes back
    assert resolve_model_ref(REF, cache_dir=tmp_path) == path


def test_a_file_below_the_repository_root_resolves(monkeypatch, tmp_path):
    # Given: a reference to a file nested inside the repository, in a
    #        directory that happens to be called "snapshots" — the cache
    #        layout then contains that word twice
    nested = "weights/snapshots/best.safetensors"
    path = write_cached_file(tmp_path, name=nested)
    stub_hub(monkeypatch, download=str(path))

    # When: it is resolved
    # Then: it resolves, rather than mistaking part of the filename for the
    #       commit the cache recorded
    assert resolve_model_ref(f"hf:{REPO}@{SHA}#{nested}", cache_dir=tmp_path) == path


def test_a_file_served_from_another_commit_is_refused(monkeypatch, tmp_path):
    # Given: a cache entry recording a commit other than the pinned one
    served = write_cached_file(tmp_path, sha=OTHER_SHA)
    stub_hub(monkeypatch, download=str(served))

    # When: the pinned reference is resolved
    # Then: it refuses, naming the commit it was handed and the one it pins
    with pytest.raises(ModelRefError, match=f"{OTHER_SHA}.*pins commit {SHA}"):
        resolve_model_ref(REF, cache_dir=tmp_path)


def test_a_file_from_outside_the_cache_layout_is_refused(monkeypatch, tmp_path):
    # Given: a served path that does not match the cache layout at all
    stray = tmp_path / "somewhere-else" / FILE
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"weights")
    stub_hub(monkeypatch, download=str(stray))

    # When: the pinned reference is resolved
    # Then: it is refused rather than accepted unchecked — an unrecognised
    #       layout fails closed
    with pytest.raises(ModelRefError, match="Refusing to load"):
        resolve_model_ref(REF, cache_dir=tmp_path)


def test_a_downloaded_digest_is_verified(monkeypatch, tmp_path):
    # Given: a pinned reference whose contents are known
    path = write_cached_file(tmp_path, content=b"weights")
    stub_hub(monkeypatch, download=str(path))
    digest = hashlib.sha256(b"weights").hexdigest()

    # When: it is resolved with the matching digest
    # Then: it resolves
    assert resolve_model_ref(REF, sha256=digest, cache_dir=tmp_path) == path

    # When: it is resolved with a digest that does not match
    # Then: it raises — the check runs on cache hits too, so a swapped or
    #       corrupted cached file cannot load silently
    with pytest.raises(ModelRefError, match="Checksum mismatch"):
        resolve_model_ref(REF, sha256="0" * 64, cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# Credentials
#
# Which credential is sent is the one call-record assertion docs/
# test-patterns.md §3 allows: it is only observable at this boundary, and
# getting it wrong silently widens where credentials come from.
# ---------------------------------------------------------------------------


def test_stored_hub_credentials_are_not_used(monkeypatch):
    # Given: no HF_TOKEN, and a reference that forces a Hub call
    calls = stub_hub(monkeypatch, info=FakeInfo(sha=SHA))

    # When: it is resolved
    with pytest.raises(ModelRefError):
        resolve_model_ref(f"hf:{REPO}@main#{FILE}")

    # Then: the Hub was told to send no credential at all. Passing None would
    #       let it fall back to a token saved by `hf auth login`, so HF_TOKEN
    #       would quietly stop being the single source.
    assert calls.api_tokens == [False]


def test_the_token_is_read_from_the_environment(monkeypatch, tmp_path):
    # Given: HF_TOKEN set in the environment, padded with whitespace
    calls = stub_hub(monkeypatch, download=str(write_cached_file(tmp_path)))
    monkeypatch.setenv("HF_TOKEN", f"  {SENTINEL_TOKEN}  ")

    # When: a reference is resolved without an explicit token
    resolve_model_ref(REF, cache_dir=tmp_path)

    # Then: the environment's token is what reaches the Hub, trimmed
    assert calls.download_tokens == [SENTINEL_TOKEN]


# ---------------------------------------------------------------------------
# Failure diagnosis: a private repository answers 404, not 401
# ---------------------------------------------------------------------------


def test_a_missing_token_is_named_as_the_cause(monkeypatch):
    # Given: no HF_TOKEN, and a Hub that reports the private repo as absent
    stub_hub(
        monkeypatch, download_error=hub_error(RepositoryNotFoundError, "404 not found")
    )

    # When: a pinned reference is resolved
    with pytest.raises(ModelRefError) as err:
        resolve_model_ref(REF)

    # Then: the message says the repository is private and no token was
    #       found, rather than "repository not found" — which reads as a typo
    #       in the repo name
    message = str(err.value)
    assert "private" in message
    assert "HF_TOKEN" in message


def test_a_token_that_cannot_read_the_repository_is_named_as_the_cause(monkeypatch):
    # Given: an HF_TOKEN that is set, and the same 404 the Hub returns
    #        whether the token is expired or merely unscoped
    stub_hub(
        monkeypatch, download_error=hub_error(RepositoryNotFoundError, "404 not found")
    )

    # When: a pinned reference is resolved
    with pytest.raises(ModelRefError) as err:
        resolve_model_ref(REF, token=SENTINEL_TOKEN)

    # Then: the message distinguishes this from a missing token and names
    #       both remedies, since 404 cannot tell them apart
    message = str(err.value)
    assert "HF_TOKEN is set but cannot read" in message
    assert "expired" in message


def test_a_gated_repository_is_named_as_the_cause(monkeypatch):
    # Given: a repository whose access is gated
    stub_hub(monkeypatch, download_error=hub_error(GatedRepoError, "403 gated", 403))

    # When: a pinned reference is resolved
    # Then: the message says so, rather than blaming the token
    with pytest.raises(ModelRefError, match="gated"):
        resolve_model_ref(REF, token=SENTINEL_TOKEN)


def test_an_unknown_revision_is_named_as_the_cause(monkeypatch):
    # Given: a repository that does not contain the pinned revision
    stub_hub(
        monkeypatch, download_error=hub_error(RevisionNotFoundError, "404 revision")
    )

    # When: the reference is resolved
    # Then: the message names the revision that is missing
    with pytest.raises(ModelRefError, match=f"has no revision '{SHA}'"):
        resolve_model_ref(REF, token=SENTINEL_TOKEN)


def test_a_missing_file_lists_what_the_commit_holds(monkeypatch):
    # Given: a commit that exists but does not hold the named file
    stub_hub(
        monkeypatch,
        download_error=EntryNotFoundError("404 entry"),
        info=FakeInfo(filenames=("other_model.safetensors",)),
    )

    # When: the reference is resolved
    with pytest.raises(ModelRefError) as err:
        resolve_model_ref(REF, token=SENTINEL_TOKEN)

    # Then: the message names the missing file and what is there instead
    message = str(err.value)
    assert f"'{FILE}' is not in" in message
    assert "other_model.safetensors" in message


def test_a_failed_file_listing_does_not_mask_the_real_error(monkeypatch):
    # Given: a missing file, and a repository listing that also fails
    stub_hub(
        monkeypatch,
        download_error=EntryNotFoundError("404 entry"),
        info_error=hub_error(RepositoryNotFoundError, "404 not found"),
    )

    # When: the reference is resolved
    # Then: the original diagnosis survives — decorating an error must not
    #       replace it
    with pytest.raises(ModelRefError, match=f"'{FILE}' is not in"):
        resolve_model_ref(REF, token=SENTINEL_TOKEN)


def test_an_offline_cache_miss_names_the_expected_path(monkeypatch, tmp_path):
    # Given: no network, and nothing in the cache
    stub_hub(monkeypatch, download_error=LocalEntryNotFoundError("offline"))

    # When: a pinned reference is resolved
    with pytest.raises(ModelRefError) as err:
        resolve_model_ref(REF, token=SENTINEL_TOKEN, cache_dir=tmp_path)

    # Then: the message says where the file was expected, so it can be
    #       fetched or copied there from a networked machine
    message = str(err.value)
    assert "could not be reached" in message
    expected = (
        tmp_path / f"models--{REPO.replace('/', '--')}" / "snapshots" / SHA / FILE
    )
    assert str(expected) in message


def test_the_token_is_scrubbed_from_hub_error_text(monkeypatch):
    # Given: a Hub error whose text quotes the token back
    stub_hub(
        monkeypatch,
        download_error=hub_error(
            HfHubHTTPError, f"500 with token {SENTINEL_TOKEN} inside", 500
        ),
    )

    # When: a reference is resolved
    with pytest.raises(ModelRefError) as err:
        resolve_model_ref(REF, token=SENTINEL_TOKEN)

    # Then: the token is redacted from what the user sees
    message = str(err.value)
    assert SENTINEL_TOKEN not in message
    assert "***" in message
