"""Parse a model reference into the checkpoint it names.

A model reference is either a filesystem path — the historic behaviour, kept
for local development checkpoints — or a release on the Hugging Face Hub::

    hf:<owner>/<name>@<40-char-commit-sha>#<filename>
    https://huggingface.co/<owner>/<name>/blob/<sha>/<filename>

Parsing is separated from resolving them because it is a total, offline
function: any string is either a Hub reference, a path, or a syntax error,
and deciding which needs no filesystem and no network.

Only a commit SHA can pin a release — branches and tags move — but a
reference is parsed whether or not it names one, so that resolution can
report precisely which part is missing rather than failing on the syntax.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import click

HF_SCHEME = "hf:"
HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co"})

# A Hub commit SHA: 40 lowercase hex characters. Nothing else pins.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

_SYNTAX = "hf:<owner>/<name>@<40-char-commit-sha>#<filename>"


class ModelRefError(click.ClickException):
    """A model reference could not be parsed or resolved.

    Subclasses ClickException so stage CLIs print the message rather than a
    traceback; the message is the user-facing explanation and always carries
    the next action.
    """


@dataclass(frozen=True)
class HubRef:
    """A parsed Hugging Face Hub reference.

    ``revision`` and ``filename`` are optional so that a partial reference
    still parses, and whoever resolves it can say which part is missing.
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

    Performs no network or filesystem access. Only the ``hf:`` scheme and
    Hugging Face URLs are treated as Hub references, so a local path is never
    mistaken for one — including paths containing ``@`` or ``#``.
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
