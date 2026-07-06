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
        merged_raw/ tuning/ merged/        (tune)
"""

from __future__ import annotations

import copy
import datetime
import io
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml
from dotenv import load_dotenv

from displacement_tracker.util.config import deep_merge, load_flow_config
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
    # Set by iter_stage_output while a stage subprocess is running, so
    # frontends can observe it (e.g. resource monitoring).
    active_process: subprocess.Popen | None = None

    def log_path(self, stage: Stage) -> Path:
        return self.run_dir / "logs" / f"{stage.key}.log"


class StageLoadMonitor:
    """Samples total CPU and memory of a stage's whole process tree.

    psutil's cpu_percent() measures the interval since the previous call
    on the *same* Process object, so instances are cached across samples;
    the first sample after construction reports 0% CPU.
    """

    def __init__(self, proc: subprocess.Popen):
        import psutil

        self._psutil = psutil
        self._root_pid = proc.pid
        self._procs: dict[int, "psutil.Process"] = {}
        self.cpu_count: int = psutil.cpu_count() or 1
        self.total_memory: int = psutil.virtual_memory().total
        self.sample()  # prime the cpu_percent counters

    def sample(self) -> tuple[float, int]:
        """Return (total CPU percent, total RSS bytes) for the tree.

        CPU is summed over processes, so it can exceed 100 on multi-core.
        """
        ps = self._psutil
        try:
            root = ps.Process(self._root_pid)
            tree = [root, *root.children(recursive=True)]
        except ps.NoSuchProcess:
            tree = []
        cpu, rss, procs = 0.0, 0, {}
        for proc in tree:
            cached = self._procs.get(proc.pid, proc)
            procs[proc.pid] = cached
            try:
                cpu += cached.cpu_percent(interval=None)
                rss += cached.memory_info().rss
            except (ps.NoSuchProcess, ps.ZombieProcess, ps.AccessDenied):
                continue
        self._procs = procs
        return cpu, rss


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

    The pipeline's flow section (``train``/``predict``/``tune``, matching
    the pipeline key) is resolved against ``shared`` here, so the config written
    into the run directory — and read by every stage — is already flat.
    """
    config = load_flow_config(str(base_config_path), pipeline.key)

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


def stage_argv(ctx: RunContext, stage: Stage) -> list[str]:
    return [sys.executable, "-m", stage.module, str(ctx.config_path)]


def _terminate_group(proc: subprocess.Popen) -> None:
    """SIGTERM the stage's process group, escalating to SIGKILL."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()


def iter_stage_output(ctx: RunContext, stage: Stage) -> Iterator[str]:
    """Run a stage, yielding output segments and teeing them to the stage log.

    Segments keep their terminator: ``\\n`` means "append a line" while a
    bare ``\\r`` means "overwrite the current line" — the convention used
    by tqdm progress bars and ``print(..., end="\\r")``. Terminals handle
    this natively (the CLI just writes segments through); the streamlit
    frontend replays it onto its line buffer.

    Raises StageFailedError on a non-zero exit code. If the consumer stops
    iterating early (e.g. the driving UI session is rerun, refreshed or
    closed), the stage's whole process group — including any worker
    children it spawned — is terminated so no orphaned jobs keep running.
    """
    argv = stage_argv(ctx, stage)
    log_path = ctx.log_path(stage)
    env = dict(os.environ)
    # Live logs: don't let child Python block-buffer stdout, and cap tqdm's
    # refresh rate (callers that set their own values win).
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TQDM_MININTERVAL", "1")
    with open(log_path, "w", newline="") as log:
        log.write(f"$ {' '.join(argv)}\n")
        # New session = new process group, so cancellation can take down
        # dataloader workers / multiprocessing pools along with the stage.
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            start_new_session=True,
            env=env,
        )
        assert proc.stdout is not None
        ctx.active_process = proc
        # newline="" keeps \r visible (text mode would fold it into \n,
        # turning every progress-bar refresh into a new line).
        stdout = io.TextIOWrapper(proc.stdout, errors="replace", newline="")
        try:
            for segment in iter(stdout.readline, ""):
                log.write(segment)
                log.flush()
                yield segment
            returncode = proc.wait()
        finally:
            ctx.active_process = None
            if proc.poll() is None:
                _terminate_group(proc)
                log.write("\n[pipeline] stage cancelled: session interrupted\n")
    if returncode != 0:
        raise StageFailedError(stage, returncode, log_path)
