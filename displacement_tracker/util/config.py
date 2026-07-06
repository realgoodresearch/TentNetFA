"""Single-config resolution: a shared section plus per-flow sections.

The repository ships one ``config.yaml`` with four top-level sections:

    shared:    values used by more than one flow (single source of truth)
    train:     the training flow    (scan -> rebalance -> train CNN)
    predict:   the prediction flow  (scan -> predict -> merge)
    tune:      the tuning flow      (raw merge -> threshold scan -> tuned merge)

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

FLOWS = ("train", "predict", "tune")
_SECTION_KEYS = ("shared", *FLOWS)


def deep_merge(base: dict, extra: dict) -> dict:
    """Recursively merge ``extra`` into ``base`` (extra wins). Returns base."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def is_sectioned_config(config: dict) -> bool:
    """True if the config uses the sectioned shared/per-flow layout."""
    return isinstance(config, dict) and any(key in config for key in _SECTION_KEYS)


def resolve_flow_config(config: dict, flow: str | None) -> dict:
    """Return the flat config for ``flow``.

    Sectioned configs resolve to ``deep_merge(shared, config[flow])`` on
    deep copies; legacy flat configs are returned unchanged.

    Raises:
        click.UsageError: sectioned config but ``flow`` is missing/unknown.
        KeyError: the requested flow section is absent from the config.
    """
    if not is_sectioned_config(config):
        return config
    if flow is None:
        raise click.UsageError(
            "This config uses shared/per-flow sections; pass "
            f"--flow to pick one of: {', '.join(FLOWS)}."
        )
    if flow not in FLOWS:
        raise click.UsageError(
            f"Unknown flow '{flow}' (expected one of: {', '.join(FLOWS)})."
        )
    if flow not in config:
        raise KeyError(f"Config has no '{flow}' section.")
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
        help="Config section to resolve (sectioned shared/per-flow layout "
        "only; ignored for legacy flat configs).",
    )
