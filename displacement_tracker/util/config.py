"""Single-config resolution: a shared section plus per-flow sections.

The repository ships one ``config.yaml`` with three top-level sections:

    shared:    values used by more than one flow (single source of truth)
    train:     the training flow    (scan -> rebalance -> train CNN)
    predict:   the prediction flow  (scan -> predict -> merge)

A stage resolves its configuration by deep-merging its flow section over
``shared`` — flow values win on conflicts. The result has the same flat
shape as the historic per-flow YAML files (``config.yaml`` /
``predict_config.yaml``), so legacy flat configs — including the resolved
configs the pipeline runner writes into run directories — pass through
unchanged.
"""

from __future__ import annotations

import copy

import click

from displacement_tracker.util.env_loader import load_yaml_with_env

FLOWS = ("train", "predict")
_SECTION_KEYS = ("shared", *FLOWS)


def deep_get(cfg: dict, dotted: str, default=None):
    """Look up a dotted path (``"a.b.c"``) in nested dicts."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def deep_set(cfg: dict, dotted: str, value) -> None:
    """Set a dotted path in nested dicts, creating intermediate dicts."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def deep_merge(base: dict, extra: dict) -> dict:
    """Recursively merge ``extra`` into ``base`` (extra wins). Returns base."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def is_sectioned_config(config: dict) -> bool:
    """True if the config uses the shared/train/predict layout."""
    return isinstance(config, dict) and any(key in config for key in _SECTION_KEYS)


def resolve_flow_config(config: dict, flow: str | None) -> dict:
    """Return the flat config for ``flow``.

    Sectioned configs resolve to ``deep_merge(shared, config[flow])`` on
    deep copies; legacy flat configs are returned unchanged.

    Raises click.UsageError whenever the flow cannot be resolved: no flow
    given for a sectioned config, an unknown flow name, a missing flow
    section, or a half-migrated config mixing sections with stray
    top-level keys (which would otherwise be silently dropped).
    """
    if not is_sectioned_config(config):
        return config
    stray = sorted(set(config) - set(_SECTION_KEYS))
    if stray:
        raise click.UsageError(
            f"Config mixes {'/'.join(_SECTION_KEYS)} sections with other "
            f"top-level keys: {', '.join(stray)}. Move them into shared "
            "or the relevant flow section."
        )
    if flow is None:
        raise click.UsageError(
            "This config uses shared/train/predict sections; pass "
            f"--flow to pick one of: {', '.join(FLOWS)}."
        )
    if flow not in FLOWS:
        raise click.UsageError(
            f"Unknown flow '{flow}' (expected one of: {', '.join(FLOWS)})."
        )
    if flow not in config:
        raise click.UsageError(f"Config has no '{flow}' section.")
    resolved = copy.deepcopy(config.get("shared") or {})
    deep_merge(resolved, copy.deepcopy(config[flow] or {}))
    return resolved


def load_flow_config(path: str, flow: str | None = None) -> dict:
    """Load a YAML config (with ``${ENV}`` substitution) and resolve ``flow``."""
    return resolve_flow_config(load_yaml_with_env(path), flow)


def flow_option(default: str | None):
    """The ``--flow`` option shared by every stage CLI."""
    return click.option(
        "--flow",
        type=click.Choice(list(FLOWS)),
        default=default,
        show_default=default is not None,
        help="Config section to resolve (shared/train/predict layout only; "
        "ignored for legacy flat configs).",
    )
