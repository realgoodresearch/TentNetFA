"""Non-interactive frontend for the pipelines (``pipeline-run``).

Useful for scripting and for testing the orchestration without a browser:

    poetry run pipeline-run predict \\
        --set prediction.batch_size=16 --skip merge --name smoke-test
"""

from __future__ import annotations

import sys

import click
import yaml

from displacement_tracker.pipelines import runner
from displacement_tracker.pipelines.spec import PIPELINES
from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("pipeline-run")


def _parse_override(raw: str) -> tuple[str, object]:
    if "=" not in raw:
        raise click.BadParameter(f"Expected key=value, got: {raw}")
    key, value = raw.split("=", 1)
    return key, yaml.safe_load(value)


@click.command()
@click.argument("pipeline_key", type=click.Choice(sorted(PIPELINES)))
@click.option(
    "--config",
    "base_config",
    default=None,
    help="Base YAML (defaults to the pipeline's standard config).",
)
@click.option(
    "--set",
    "assignments",
    multiple=True,
    help="Override a config value: dotted.path=value (YAML-parsed).",
)
@click.option(
    "--skip", "skipped", multiple=True, help="Stage key to skip (repeatable)."
)
@click.option(
    "--only", "only", multiple=True, help="Run only these stage keys (repeatable)."
)
@click.option(
    "--name", "run_name", default=None, help="Run name (defaults to a timestamp)."
)
@click.option(
    "--run-root",
    default=None,
    help=f"Artifact root (default: {runner.default_run_root()}).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Prepare the run directory and print the plan without executing.",
)
def cli(
    pipeline_key, base_config, assignments, skipped, only, run_name, run_root, dry_run
):
    pipeline = PIPELINES[pipeline_key]
    overrides = dict(_parse_override(a) for a in assignments)

    stages = [s for s in pipeline.stages if s.default_enabled or s.key in only]
    if only:
        stages = [s for s in stages if s.key in only]
    stages = [s for s in stages if s.key not in skipped]
    if not stages:
        raise click.ClickException("No stages left to run.")

    ctx = runner.prepare_run(
        pipeline,
        base_config or pipeline.base_config,
        overrides=overrides,
        run_name=run_name,
        run_root=run_root,
    )
    LOGGER.info("Run directory: %s", ctx.run_dir)

    for stage in stages:
        LOGGER.info("[%s] %s", stage.key, " ".join(runner.stage_argv(ctx, stage)))
    if dry_run:
        return

    for stage in stages:
        LOGGER.info("--- stage: %s ---", stage.key)
        try:
            for line in runner.iter_stage_output(ctx, stage):
                sys.stdout.write(line)
        except runner.StageFailedError as exc:
            raise click.ClickException(str(exc)) from exc

    LOGGER.info("Pipeline finished. Artifacts in %s", ctx.run_dir)


if __name__ == "__main__":
    cli()
