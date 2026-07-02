"""Frontend-agnostic pipeline execution.

Given a pipeline spec, a base YAML and a set of overrides, this module
creates a self-contained run directory with fixed subfolder names, writes
the fully resolved config into it, and executes the enabled stages as
subprocesses while streaming their output.

Run directory layout (fixed names, see ``Pipeline.artifact_paths``)::

    <run_root>/<pipeline>/<run_name>/
        config.yaml     resolved config used by every stage
        logs/           one log file per stage
        manifests/ preds/ merged/          (predict)
        manifests/ dataset/ model/         (train)
"""

from __future__ import annotations

import copy
import datetime
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import yaml
from dotenv import load_dotenv

from displacement_tracker.util.env_loader import load_yaml_with_env
from displacement_tracker.pipelines.spec import Pipeline, Stage


class StageFailedError(RuntimeError):
    def __init__(self, stage: Stage, returncode: int, log_path: Path):
        self.stage = stage
        self.returncode = returncode
        self.log_path = log_path
        super().__init__(
            f"Stage '{stage.key}' failed with exit code {returncode} (log: {log_path})"
        )


def deep_get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def deep_set(cfg: dict, dotted: str, value) -> None:
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


def default_run_root() -> str:
    """Prefer DATA_DIR (large artifacts) and fall back to the local repo."""
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        return os.path.join(data_dir, "results", "TentNetFA", "pipeline_runs")
    return os.path.join("runs", "pipelines")


@dataclass
class RunContext:
    pipeline: Pipeline
    run_dir: Path
    config_path: Path
    config: dict

    def log_path(self, stage: Stage) -> Path:
        return self.run_dir / "logs" / f"{stage.key}.log"


def prepare_run(
    pipeline: Pipeline,
    base_config_path: str | Path,
    overrides: dict | None = None,
    run_name: str | None = None,
    run_root: str | Path | None = None,
) -> RunContext:
    """Build the run directory and write the fully resolved config.

    ``overrides`` maps dotted config paths to values; artifact locations are
    then forced into the run directory regardless of base config/overrides.
    """
    config = load_yaml_with_env(str(base_config_path))

    for section, defaults in pipeline.extra_defaults.items():
        existing = config.get(section)
        merged = copy.deepcopy(defaults)
        if isinstance(existing, dict):
            deep_merge(merged, existing)
        config[section] = merged

    for dotted, value in (overrides or {}).items():
        deep_set(config, dotted, value)

    run_root = Path(run_root or default_run_root())
    run_name = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = run_root / pipeline.key / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in pipeline.subfolders:
        (run_dir / sub).mkdir(exist_ok=True)

    for dotted, rel in pipeline.artifact_paths.items():
        deep_set(config, dotted, str(run_dir / rel))

    config_path = run_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return RunContext(pipeline=pipeline, run_dir=run_dir, config_path=config_path, config=config)


def _default_argv(ctx: RunContext, stage: Stage) -> list[str]:
    return [sys.executable, "-m", stage.module, str(ctx.config_path)]


def _merge_argv(ctx: RunContext, stage: Stage) -> list[str]:
    """h_merge_geojsons takes positional/flag args instead of a config file."""
    merge_cfg = ctx.config.get("merge", {})
    input_folder = deep_get(ctx.config, "prediction.output_folder")
    output = merge_cfg.get("output", str(ctx.run_dir / "merged" / "merged.gpkg"))
    argv = [
        sys.executable, "-m", stage.module,
        str(input_folder), str(output),
        "--min-distance-m", str(merge_cfg.get("min_distance_m", 3.0)),
        "--agreement", str(merge_cfg.get("agreement", 1)),
        "--min-adj-peak", str(merge_cfg.get("min_adj_peak", 0.0)),
        "--adjustment-factor", str(merge_cfg.get("adjustment_factor", 1.0)),
    ]
    for key, flag in (
        ("thresholds_config", "--thresholds-config"),
        ("exclusion_zones_gpkg", "--exclusion-zones-gpkg"),
        ("inclusion_zone", "--inclusion-zone"),
    ):
        if merge_cfg.get(key):
            argv += [flag, str(merge_cfg[key])]
    return argv


ARGV_BUILDERS: dict[str, Callable[[RunContext, Stage], list[str]]] = {
    "merge": _merge_argv,
}


def stage_argv(ctx: RunContext, stage: Stage) -> list[str]:
    return ARGV_BUILDERS.get(stage.key, _default_argv)(ctx, stage)


def iter_stage_output(ctx: RunContext, stage: Stage) -> Iterator[str]:
    """Run a stage, yielding output lines and teeing them to the stage log.

    Raises StageFailedError on a non-zero exit code.
    """
    argv = stage_argv(ctx, stage)
    log_path = ctx.log_path(stage)
    with open(log_path, "w") as log:
        log.write(f"$ {' '.join(argv)}\n")
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            yield line
        returncode = proc.wait()
    if returncode != 0:
        raise StageFailedError(stage, returncode, log_path)
