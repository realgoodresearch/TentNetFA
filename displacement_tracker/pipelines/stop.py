"""Find and gracefully terminate running pipeline-ui backends.

``pipeline-ui-stop`` terminates every streamlit server running the pipeline
app — including any pipeline stages they spawned — plus stage processes
orphaned by an earlier hard kill of their server. Run it after pulling new
code so no stale backend keeps serving (or holding a port with) the old
version.
"""

from __future__ import annotations

import os

import click
import psutil

from displacement_tracker.util.logging_config import setup_logging

LOGGER = setup_logging("pipeline-ui-stop")

_APP_MARKER = os.path.join("displacement_tracker", "pipelines", "app.py")
_STAGE_MARKER = "-m displacement_tracker."
# Deliberate headless runs (pipeline-run in nohup/tmux) are not "weird
# state" and must survive; their stages have a live parent anyway.
_HEADLESS_MARKER = "-m displacement_tracker.pipelines"


def _cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return ""


def find_targets() -> tuple[list[psutil.Process], list[psutil.Process]]:
    """Return (UI servers, orphaned stage processes)."""
    servers: list[psutil.Process] = []
    orphans: list[psutil.Process] = []
    me = psutil.Process(os.getpid())
    protected = {me.pid, *(p.pid for p in me.parents())}
    for proc in psutil.process_iter(["ppid"]):
        if proc.pid in protected:
            continue
        cmd = _cmdline(proc)
        if "streamlit" in cmd and _APP_MARKER in cmd:
            servers.append(proc)
        elif (
            proc.info["ppid"] == 1
            and _STAGE_MARKER in cmd
            and _HEADLESS_MARKER not in cmd
        ):
            orphans.append(proc)
    return servers, orphans


def _with_descendants(roots: list[psutil.Process]) -> dict[int, psutil.Process]:
    procs: dict[int, psutil.Process] = {}
    for root in roots:
        procs[root.pid] = root
        try:
            for child in root.children(recursive=True):
                procs[child.pid] = child
        except psutil.NoSuchProcess:
            continue
    return procs


@click.command()
@click.option("--dry-run", is_flag=True, help="List what would be terminated and exit.")
@click.option(
    "--timeout",
    default=10.0,
    show_default=True,
    help="Seconds to wait after SIGTERM before escalating to SIGKILL.",
)
def cli(dry_run: bool, timeout: float) -> None:
    servers, orphans = find_targets()
    for server in servers:
        LOGGER.info("UI server  pid %d: %s", server.pid, _cmdline(server))
    for orphan in orphans:
        LOGGER.info("orphaned stage  pid %d: %s", orphan.pid, _cmdline(orphan))

    targets = _with_descendants(servers + orphans)
    if not targets:
        LOGGER.info("No running pipeline-ui backends found.")
        return
    LOGGER.info(
        "%d process(es) to terminate (%d server(s), %d orphaned stage(s), "
        "rest are their children).",
        len(targets),
        len(servers),
        len(orphans),
    )
    if dry_run:
        return

    procs = list(targets.values())
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            continue
    gone, alive = psutil.wait_procs(procs, timeout=timeout)
    for proc in alive:
        LOGGER.warning("pid %d did not exit within %.0fs — killing", proc.pid, timeout)
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(alive, timeout=5)
    LOGGER.info("Terminated %d process(es).", len(procs))


if __name__ == "__main__":
    cli()
