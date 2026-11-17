"""Utilities for stable source-image identifiers."""

from pathlib import Path


def source_image_key(value: object | None) -> str:
    """Return the filename stem used by UI priority lists and source matching."""
    return Path(str(value)).stem if value is not None else ""
