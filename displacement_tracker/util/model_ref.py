"""Resolve a model reference to a local checkpoint file.

A model reference is either a filesystem path — the historic behaviour, kept
for local development checkpoints — or a pinned release on the Hugging Face
Hub::

    hf:<owner>/<name>@<40-char-commit-sha>#<filename>
    https://huggingface.co/<owner>/<name>/blob/<sha>/<filename>

Only a commit SHA pins. Branches and tags move, so a reference naming one is
rejected with the SHA it currently points at, ready to paste back into the
config.

The repository holding the weights is private, so every Hub call is
authenticated with ``HF_TOKEN`` taken from the environment (``.env`` is loaded
first, as elsewhere in this project). A private repository answers an
unauthorised request with 404 rather than 401 — it will not confirm that it
exists — so a bare "repository not found" is nearly always a missing or
under-scoped token rather than a typo, and the error says so.
"""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import click
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import constants as hf_constants
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

# Not re-exported at package level in 1.x, but the module is public. Worth the
# deeper import: _check_snapshot compares against the path this builds, so a
# hand-rolled copy of the Hub's folder naming would fail every resolution the
# day the Hub changed it.
from huggingface_hub.file_download import repo_folder_name

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("model_ref")

HF_SCHEME = "hf:"
HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})

# A Hub commit SHA: 40 lowercase hex characters. Nothing else pins.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

_SYNTAX = "hf:<owner>/<name>@<40-char-commit-sha>#<filename>"
_OWNER = "the TentNetFA repo owner"
_WEIGHT_SUFFIXES = (".safetensors", ".pth", ".pt", ".bin")
_HASH_CHUNK = 1024 * 1024


class ModelRefError(click.ClickException):
    """A model reference could not be resolved.

    Subclasses ClickException so stage CLIs print the message rather than a
    traceback; the message is the user-facing explanation and always carries
    the next action.
    """


@dataclass(frozen=True)
class HubRef:
    """A parsed Hugging Face Hub reference.

    ``revision`` and ``filename`` are optional at parse time so that
    resolution can report precisely which part is missing.
    """

    repo_id: str
    revision: str | None = None
    filename: str | None = None

    def __str__(self) -> str:
        text = f"{HF_SCHEME}{self.repo_id}"
        if self.revision:
            text += f"@{self.revision}"
        if self.filename:
            text += f"#{self.filename}"
        return text


def parse_model_ref(ref: str | Path) -> HubRef | Path:
    """Parse ``ref`` into a :class:`HubRef` or a local :class:`Path`.

    Performs no network access — a reference can be parsed offline; only
    resolving it needs the Hub.
    """
    text = str(ref).strip()
    if not text:
        raise ModelRefError(
            f"Model reference is empty; expected a checkpoint path or {_SYNTAX}."
        )
    if text.startswith(HF_SCHEME):
        return _parse_hub_ref(text[len(HF_SCHEME) :])
    if _is_hub_url(text):
        return _parse_hub_url(text)
    return Path(text).expanduser()


def resolve_model_ref(
    ref: str | Path,
    *,
    sha256: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve ``ref`` to a readable local file.

    Local paths are returned as-is once confirmed to exist. Hub references are
    downloaded into the Hub cache (or reused from it, without touching the
    network) and verified.

    ``sha256``, when given, is checked on every resolution — including cache
    hits — so a corrupted or swapped cached file cannot be loaded silently.
    """
    parsed = parse_model_ref(ref)
    if isinstance(parsed, Path):
        if not parsed.is_file():
            raise ModelRefError(f"Model checkpoint not found: {parsed}")
        _verify_digest(parsed, sha256)
        return parsed

    resolved_token = token if token is not None else _token_from_env()
    if not parsed.revision or not _COMMIT_SHA.match(parsed.revision):
        _raise_unpinned(parsed, resolved_token)
    if not parsed.filename:
        raise _missing_filename_error(parsed, resolved_token)

    path = _download(parsed, resolved_token, cache_dir)
    _check_snapshot(path, parsed, cache_dir)
    _verify_digest(path, sha256, ref=parsed)
    return path


# --------------------------------------------------------------------------
# parsing


def _is_hub_url(text: str) -> bool:
    if not text.startswith(("http://", "https://")):
        return False
    return urlparse(text).netloc.lower() in HF_HOSTS


def _parse_hub_ref(body: str) -> HubRef:
    body, _, filename = body.partition("#")
    repo_id, _, revision = body.partition("@")
    return _hub_ref(repo_id, revision, filename)


def _parse_hub_url(text: str) -> HubRef:
    parts = [part for part in urlparse(text).path.split("/") if part]
    if len(parts) < 2:
        raise ModelRefError(
            f"Not a Hugging Face model URL: {text}. Expected "
            f"https://huggingface.co/<owner>/<name>/blob/<sha>/<filename>, or {_SYNTAX}."
        )
    repo_id = "/".join(parts[:2])
    revision = filename = ""
    # .../blob/<rev>/<path> and .../resolve/<rev>/<path> both appear in URLs
    # copied out of the Hub UI.
    if len(parts) > 3 and parts[2] in ("blob", "resolve"):
        revision = parts[3]
        filename = "/".join(parts[4:])
    return _hub_ref(repo_id, revision, filename)


def _hub_ref(repo_id: str, revision: str, filename: str) -> HubRef:
    if not _REPO_ID.match(repo_id):
        raise ModelRefError(
            f"'{repo_id}' is not a Hugging Face repo id; expected <owner>/<name> in {_SYNTAX}."
        )
    revision = revision.strip()
    # Hub SHAs are lowercase; accept a pasted uppercase one rather than
    # mistaking it for a branch name.
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        revision = revision.lower()
    return HubRef(repo_id, revision or None, filename.strip() or None)


def _token_from_env() -> str | None:
    """Return ``HF_TOKEN`` from the environment, loading ``.env`` first."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    token = token.strip() if token else ""
    return token or None


# --------------------------------------------------------------------------
# Hub access, with every failure mapped onto its actual cause


def _auth(token: str | None) -> str | bool:
    """Hub credential to send: the token, or an explicit "no credential".

    ``token=None`` would let huggingface_hub fall back to a credential saved by
    ``hf auth login``. HF_TOKEN is the single supported source, so say no
    explicitly — otherwise a stored credential could quietly decide whether a
    run works, and "no HF_TOKEN found" would be a lie.
    """
    return token or False


@contextmanager
def _hub_errors(repo_id: str, revision: str | None, token: str | None, what: str):
    """Translate the Hub's HTTP failures into ModelRefError.

    Covers every failure both metadata lookups and downloads can hit. The two
    that only downloads produce — an absent file, and an offline cache miss —
    are not HTTP errors and pass through to the caller that can describe them.
    """
    try:
        yield
    # GatedRepoError subclasses RepositoryNotFoundError, so it comes first.
    except GatedRepoError as exc:
        raise ModelRefError(
            f"Access to '{repo_id}' is gated and has not been granted for this "
            f"token. Ask {_OWNER} for access."
        ) from exc
    except RepositoryNotFoundError as exc:
        raise _access_error(repo_id, token) from exc
    except RevisionNotFoundError as exc:
        # Reached for a mistyped branch name as well as an unknown commit, so
        # this says "revision" rather than claiming it was a commit.
        raise ModelRefError(f"'{repo_id}' has no revision '{revision}'.") from exc
    except HfHubHTTPError as exc:
        raise ModelRefError(
            f"Hugging Face Hub error while {what}: {_scrub(exc, token)}"
        ) from exc


def _api_call(repo_id: str, revision: str, token: str | None):
    """Fetch repo metadata."""
    with _hub_errors(repo_id, revision, token, f"reading '{repo_id}'"):
        return HfApi(token=_auth(token)).model_info(repo_id, revision=revision)


def _download(ref: HubRef, token: str | None, cache_dir: str | Path | None) -> Path:
    LOGGER.info(f"🔹 Resolving {ref} from the Hugging Face Hub.")
    try:
        with _hub_errors(ref.repo_id, ref.revision, token, f"downloading {ref}"):
            return Path(
                hf_hub_download(
                    repo_id=ref.repo_id,
                    filename=ref.filename,
                    revision=ref.revision,
                    token=_auth(token),
                    cache_dir=cache_dir,
                )
            )
    # LocalEntryNotFoundError subclasses EntryNotFoundError, so it comes first.
    except LocalEntryNotFoundError as exc:
        raise ModelRefError(
            f"{ref} is not in the local Hugging Face cache and the Hub could not be "
            f"reached. Expected at: {_expected_cache_path(ref, cache_dir)}. Run this "
            "on a machine with network access to populate the cache, or copy that "
            "file across."
        ) from exc
    except EntryNotFoundError as exc:
        raise ModelRefError(
            f"File '{ref.filename}' is not in '{ref.repo_id}' at commit "
            f"{ref.revision}.{_available_files(ref, token)}"
        ) from exc


def _access_error(repo_id: str, token: str | None) -> ModelRefError:
    """Turn a 404 into the reason it actually happened.

    A private repository 404s rather than 403s, so "not found" on its own is
    misleading. The distinction that changes what the user does is whether
    they have a token at all: without one there is a local fix, with one the
    token has to be replaced or granted access by its owner.
    """
    if not token:
        return ModelRefError(
            f"Cannot reach '{repo_id}': the repository is private and no HF_TOKEN was "
            "found in the environment or .env. Add HF_TOKEN=<read token> to .env, or "
            f"ask {_OWNER} for access if you do not have a token."
        )
    return ModelRefError(
        f"HF_TOKEN is set but cannot read '{repo_id}' — a private repository answers "
        "404 rather than 403, so the token is expired, revoked, or its repository "
        f"scope does not cover this repo. Ask {_OWNER} for access."
    )


def _raise_unpinned(ref: HubRef, token: str | None) -> None:
    """Reject a missing or movable revision, quoting the SHA to pin instead.

    Always raises: either the Hub lookup fails, or the reference is unpinned
    and that is itself the error.
    """
    lookup = ref.revision or "main"
    info = _api_call(ref.repo_id, lookup, token)
    sha = getattr(info, "sha", None) or "<unknown>"
    pinned = HubRef(ref.repo_id, sha, ref.filename)
    if not ref.revision:
        raise ModelRefError(
            f"{ref} names no revision, and only a commit SHA pins a release. "
            f"'{lookup}' currently points at {sha} — pin it with: {pinned}"
        )
    raise ModelRefError(
        f"'{ref.revision}' is a branch or tag, not a commit SHA. Branches and tags "
        f"move, so they cannot pin a release; '{ref.revision}' currently points at "
        f"{sha} — pin it with: {pinned}"
    )


def _missing_filename_error(ref: HubRef, token: str | None) -> ModelRefError:
    return ModelRefError(
        f"{ref} names no file. Append the checkpoint to load, e.g. "
        f"{HubRef(ref.repo_id, ref.revision, 'best_model.safetensors')}."
        f"{_available_files(ref, token)}"
    )


def _available_files(ref: HubRef, token: str | None) -> str:
    """Best-effort listing of candidate weight files, for error messages.

    Never raises: this decorates an error that has already been diagnosed, and
    a second failure here must not mask the first.
    """
    try:
        info = _api_call(ref.repo_id, ref.revision or "main", token)
        names = [sibling.rfilename for sibling in getattr(info, "siblings", None) or []]
    except Exception:
        return ""
    weights = [name for name in names if name.endswith(_WEIGHT_SUFFIXES)]
    listed = weights or names
    if not listed:
        return ""
    shown = ", ".join(sorted(listed)[:10])
    return f" Available: {shown}."


# --------------------------------------------------------------------------
# verification


def _check_snapshot(path: Path, ref: HubRef, cache_dir: str | Path | None) -> None:
    """Confirm the Hub served the commit the reference pins.

    The cache stores a download under ``snapshots/<sha>/<filename>``, so the
    path a correct download lands on is fully determined by the reference.
    Comparing against it fails closed: any path we cannot account for is
    refused rather than accepted unchecked.
    """
    expected = _expected_cache_path(ref, cache_dir)
    if path != expected:
        raise ModelRefError(
            f"Hub served {path} for {ref}, which pins commit {ref.revision} at "
            f"{expected}. Refusing to load."
        )


def _verify_digest(path: Path, expected: str | None, ref: HubRef | None = None) -> None:
    """Check the file against an expected sha256, if one was configured."""
    if not expected:
        return
    want = str(expected).strip().lower().removeprefix("sha256:")
    if not _SHA256.fullmatch(want):
        # Reached when YAML read an unquoted all-digit digest as a number, and
        # for an ordinary typo. Reporting a checksum mismatch here would blame
        # the checkpoint for a mistake in the config.
        raise ModelRefError(
            f"Expected sha256 is not a 64-character hex digest: {want}. If it was "
            "written unquoted in config.yaml, YAML read it as a number — quote it."
        )
    got = _sha256(path)
    if got == want:
        return
    source = f"{ref} at {path}" if ref else str(path)
    raise ModelRefError(
        f"Checksum mismatch for {source}: expected sha256 {want}, got {got}. The "
        "file may be corrupt or the release may have been replaced. Delete it and "
        "retry."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_cache_path(ref: HubRef, cache_dir: str | Path | None) -> Path:
    root = Path(cache_dir or hf_constants.HF_HUB_CACHE)
    folder = repo_folder_name(repo_id=ref.repo_id, repo_type="model")
    return root / folder / "snapshots" / str(ref.revision) / str(ref.filename)


def _scrub(value: object, token: str | None) -> str:
    """Stringify Hub output with the token redacted, wherever it came from."""
    text = str(value)
    if token:
        text = text.replace(token, "***")
    return text
