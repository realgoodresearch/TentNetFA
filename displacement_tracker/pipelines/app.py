"""Streamlit frontend for the end-to-end pipelines.

Launch with ``poetry run pipeline-ui`` (or ``streamlit run`` on this file).
Forms are generated from the pipeline specs in ``spec.py``; anything not
exposed there can still be overridden via the advanced YAML box.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import streamlit as st
import yaml

from displacement_tracker.pipelines import runner
from displacement_tracker.pipelines.spec import PIPELINES, Param
from displacement_tracker.util.env_loader import load_yaml_with_env


def _widget(param: Param, default, key: str):
    if param.type == "bool":
        return st.checkbox(param.label, value=bool(default), key=key, help=param.help or None)
    if param.type == "int":
        return st.number_input(
            param.label, value=int(default if default is not None else 0),
            step=1, key=key, help=param.help or None,
        )
    if param.type == "float":
        return st.number_input(
            param.label, value=float(default if default is not None else 0.0),
            format="%g", key=key, help=param.help or None,
        )
    if param.type == "list":
        text = st.text_area(
            param.label, value="\n".join(default or []),
            key=key, help=param.help or None,
            placeholder="Empty: scan stage processes ALL .tif files in the "
            "GeoTIFF directory; download stage downloads nothing.",
        )
        entries = [line.strip() for line in text.splitlines() if line.strip()]
        return entries or None
    # str / path
    value = st.text_input(
        param.label, value="" if default is None else str(default),
        key=key, help=param.help or None,
    )
    if value == "":
        return None if param.optional else ""
    return value


def main() -> None:
    st.set_page_config(page_title="TentNetFA pipelines", layout="wide")
    st.title("TentNetFA pipeline runner")

    with st.sidebar:
        pipeline_key = st.radio(
            "Pipeline", list(PIPELINES), format_func=lambda k: PIPELINES[k].label
        )
        pipeline = PIPELINES[pipeline_key]

        base_config_path = st.text_input("Base config", value=pipeline.base_config)
        run_root = st.text_input("Run root", value=runner.default_run_root())
        run_name = st.text_input(
            "Run name", value="", placeholder="empty = timestamp",
            help="Artifacts land in <run root>/<pipeline>/<run name>/.",
        )
        st.caption("Artifacts will be written to:")
        st.code(
            str(Path(run_root) / pipeline.key / (run_name.strip() or "<timestamp>")),
            language=None,
            wrap_lines=True,
        )

        st.subheader("Stages")
        enabled_stages = [
            stage
            for stage in pipeline.stages
            if st.checkbox(
                stage.label, value=stage.default_enabled, key=f"{pipeline.key}:stage:{stage.key}"
            )
        ]

    try:
        base_config = load_yaml_with_env(base_config_path)
    except (FileNotFoundError, KeyError) as exc:
        st.error(f"Could not load base config: {exc}")
        st.stop()

    # merge extra defaults so widget defaults reflect what will actually run
    for section, defaults in pipeline.extra_defaults.items():
        merged = dict(defaults)
        if isinstance(base_config.get(section), dict):
            runner.deep_merge(merged, base_config[section])
        base_config[section] = merged

    st.caption(
        "Values below are prefilled from the base config; edit to override. "
        "Output locations are managed by the runner and land in the run directory."
    )

    groups: dict[str, list[Param]] = {}
    for param in pipeline.params:
        groups.setdefault(param.group, []).append(param)

    overrides: dict[str, object] = {}
    for group, params in groups.items():
        with st.expander(group, expanded=(group == "Inputs")):
            cols = st.columns(2)
            for i, param in enumerate(params):
                with cols[i % 2]:
                    default = runner.deep_get(base_config, param.path)
                    overrides[param.path] = _widget(
                        param, default, key=f"{pipeline.key}:{param.path}"
                    )

    with st.expander("Advanced: extra YAML overrides"):
        st.caption(
            "Deep-merged on top of everything above. Use the same structure "
            "as the base config, e.g. `prediction: {selection: {method: nms}}`."
        )
        raw_yaml = st.text_area("YAML", value="", height=160, label_visibility="collapsed")

    st.caption(
        ":warning: Keep this tab open while a run is in progress — switching "
        "pipeline, changing inputs, refreshing or closing the page cancels the "
        "running stage and terminates its processes. Completed artifacts and "
        "logs stay in the run directory; re-run with only the remaining stages "
        "enabled to resume. For long unattended runs use `pipeline-run` in tmux."
    )
    if not st.button("Run pipeline", type="primary", disabled=not enabled_stages):
        return

    extra: dict = {}
    if raw_yaml.strip():
        try:
            extra = yaml.safe_load(raw_yaml) or {}
        except yaml.YAMLError as exc:
            st.error(f"Invalid YAML in advanced overrides: {exc}")
            return
        if not isinstance(extra, dict):
            st.error("Advanced overrides must be a YAML mapping.")
            return

    ctx = runner.prepare_run(
        pipeline,
        base_config_path,
        overrides=overrides,
        run_name=run_name.strip() or None,
        run_root=run_root,
    )
    if extra:
        runner.deep_merge(ctx.config, extra)
        with open(ctx.config_path, "w") as f:
            yaml.safe_dump(ctx.config, f, sort_keys=False)

    st.info(f"Run directory: `{ctx.run_dir}`")
    with st.expander("Resolved config"):
        st.code(yaml.safe_dump(ctx.config, sort_keys=False), language="yaml")

    # Bottom bar: created before the stage loop so it can be updated while
    # logs stream, but placed after `stages_area` so it renders below them.
    stages_area = st.container()
    with st.expander("System load", expanded=True):
        load_box = st.empty()
        load_box.caption("Waiting for pipeline processes…")

    def render_load(cpu: float, rss: int) -> None:
        load_box.markdown(
            f"**CPU** {cpu:.0f} % &nbsp;·&nbsp; "
            f"**Memory** {rss / 2**30:.2f} GiB "
            f"<small>(current stage incl. workers; CPU is summed over "
            f"cores)</small>",
            unsafe_allow_html=True,
        )

    for stage in enabled_stages:
        with stages_area.status(stage.label, expanded=True) as status:
            box = st.empty()
            # Mini terminal emulation: segments ending in \n append a line,
            # segments ending in a bare \r overwrite the last one — so tqdm
            # bars update in place instead of flooding the tail.
            tail: deque[str] = deque(maxlen=30)
            overwrite = False
            monitor = None
            last_render = last_load = 0.0
            try:
                for segment in runner.iter_stage_output(ctx, stage):
                    text = segment.rstrip("\r\n")
                    if overwrite and tail:
                        tail[-1] = text
                    else:
                        tail.append(text)
                    overwrite = segment.endswith("\r") and not segment.endswith("\r\n")
                    now = time.monotonic()
                    if now - last_render > 0.2:
                        box.code("\n".join(tail))
                        last_render = now
                    if now - last_load > 1.0:
                        if monitor is None and ctx.active_process is not None:
                            monitor = runner.StageLoadMonitor(ctx.active_process)
                        elif monitor is not None:
                            render_load(*monitor.sample())
                        last_load = now
                box.code("\n".join(tail))
            except runner.StageFailedError as exc:
                box.code("\n".join(tail))
                status.update(state="error", expanded=True)
                load_box.caption("No active pipeline processes.")
                st.error(f"{exc} — full log at `{exc.log_path}`")
                return
            status.update(state="complete", expanded=False)

    load_box.caption("Pipeline finished — no active processes.")
    st.success(f"Pipeline finished. Artifacts in `{ctx.run_dir}`")


main()
