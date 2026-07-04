"""Streamlit frontend for the end-to-end pipelines.

Launch with ``poetry run pipeline-ui`` (or ``streamlit run`` on this file).
Forms are generated from the pipeline specs in ``spec.py``; anything not
exposed there can still be overridden via the advanced YAML box.

Layout: the sidebar holds run setup (pipeline, destination, stages, Run
button); the central panel splits into a Configuration tab and a Run logs
tab; a fixed bottom bar shows the system load of the running stage.
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

# Meter styling follows the reference dataviz palette: blue accent with a
# same-ramp track, yellow/red severity steps, text in text tokens, and
# explicitly selected dark-mode colors (not an automatic flip).
_LOADBAR_STYLE = """
<style>
div[data-testid="stMainBlockContainer"] { padding-bottom: 7rem; }
.tn-loadbar {
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
  background: #fcfcfb; border-top: 1px solid rgba(11,11,11,.10);
  padding: 0.65rem 1.5rem 0.85rem;
}
.tn-loadbar .tn-row { display: flex; gap: 2.5rem; align-items: flex-end; }
.tn-loadbar .tn-title {
  font-size: 0.72rem; letter-spacing: .06em; text-transform: uppercase;
  color: #52514e; white-space: nowrap; padding-bottom: 0.15rem;
}
.tn-meter { flex: 1; min-width: 8rem; }
.tn-meter .tn-lbl {
  display: flex; justify-content: space-between;
  font-size: 0.8rem; color: #52514e; margin-bottom: 0.25rem;
}
.tn-meter .tn-lbl b {
  color: #0b0b0b; font-weight: 600; font-variant-numeric: tabular-nums;
}
.tn-meter .tn-track {
  height: 8px; border-radius: 4px; overflow: hidden; background: #cde2fb;
}
.tn-meter .tn-fill { height: 100%; border-radius: 4px; background: #2a78d6; }
.tn-meter.warn .tn-track { background: #f7e3b6; }
.tn-meter.warn .tn-fill { background: #eda100; }
.tn-meter.crit .tn-track { background: #f8d2d2; }
.tn-meter.crit .tn-fill { background: #e34948; }
@media (prefers-color-scheme: dark) {
  .tn-loadbar { background: #1a1a19; border-top-color: rgba(255,255,255,.14); }
  .tn-loadbar .tn-title { color: #c3c2b7; }
  .tn-meter .tn-lbl { color: #c3c2b7; }
  .tn-meter .tn-lbl b { color: #ffffff; }
  .tn-meter .tn-track { background: #0d366b; }
  .tn-meter .tn-fill { background: #3987e5; }
  .tn-meter.warn .tn-track { background: #4a3404; }
  .tn-meter.warn .tn-fill { background: #c98500; }
  .tn-meter.crit .tn-track { background: #571f1f; }
  .tn-meter.crit .tn-fill { background: #e66767; }
}
</style>
"""


def _meter_html(label: str, value: str, pct: float | None) -> str:
    if pct is None:
        cls, width = "", 0.0
    else:
        width = max(0.0, min(100.0, pct))
        cls = "crit" if pct >= 90 else "warn" if pct >= 75 else ""
    return (
        f'<div class="tn-meter {cls}">'
        f'<div class="tn-lbl"><span>{label}</span><b>{value}</b></div>'
        f'<div class="tn-track"><div class="tn-fill" style="width:{width:.1f}%"></div></div>'
        f"</div>"
    )


def _render_load_bar(
    box,
    title: str,
    cpu_pct: float | None = None,
    cpu_text: str = "—",
    mem_pct: float | None = None,
    mem_text: str = "—",
) -> None:
    box.markdown(
        _LOADBAR_STYLE
        + '<div class="tn-loadbar"><div class="tn-row">'
        + f'<span class="tn-title">{title}</span>'
        + _meter_html("CPU (avg per core)", cpu_text, cpu_pct)
        + _meter_html("Memory (of total)", mem_text, mem_pct)
        + "</div></div>",
        unsafe_allow_html=True,
    )


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

        run_clicked = st.button(
            "Run pipeline", type="primary", disabled=not enabled_stages,
            use_container_width=True,
        )
        st.caption(
            ":warning: Keep this tab open while a run is in progress — "
            "switching pipeline, changing inputs, refreshing or closing the "
            "page cancels the running stage and terminates its processes. "
            "Completed artifacts and logs stay in the run directory. For "
            "long unattended runs use `pipeline-run` in tmux."
        )
        show_load = st.checkbox("Show system load bar", value=True)

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

    config_tab, logs_tab = st.tabs(["Configuration", "Run logs"])

    with config_tab:
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

    # Fixed bottom bar (position: fixed escapes the tab container).
    load_box = st.empty()
    if show_load:
        _render_load_bar(load_box, "System load — idle")

    if not run_clicked:
        with logs_tab:
            st.caption("No run in this session yet — configure and press *Run pipeline*.")
        return

    extra: dict = {}
    if raw_yaml.strip():
        try:
            extra = yaml.safe_load(raw_yaml) or {}
        except yaml.YAMLError as exc:
            with logs_tab:
                st.error(f"Invalid YAML in advanced overrides: {exc}")
            return
        if not isinstance(extra, dict):
            with logs_tab:
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

    with config_tab:
        with st.expander("Resolved config (as written to the run directory)"):
            st.code(yaml.safe_dump(ctx.config, sort_keys=False), language="yaml")

    with logs_tab:
        st.info(f"Run directory: `{ctx.run_dir}`")

        for stage in enabled_stages:
            with st.status(stage.label, expanded=True) as status:
                box = st.empty()
                # Mini terminal emulation: segments ending in \n append a
                # line, segments ending in a bare \r overwrite the last one —
                # so tqdm bars update in place instead of flooding the tail.
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
                        if show_load and now - last_load > 1.0:
                            if monitor is None and ctx.active_process is not None:
                                monitor = runner.StageLoadMonitor(ctx.active_process)
                            elif monitor is not None:
                                cpu_sum, rss = monitor.sample()
                                cpu_pct = cpu_sum / monitor.cpu_count
                                mem_pct = rss / monitor.total_memory * 100
                                _render_load_bar(
                                    load_box,
                                    f"System load — {stage.label}",
                                    cpu_pct, f"{cpu_pct:.0f} %",
                                    mem_pct,
                                    f"{rss / 2**30:.1f} / "
                                    f"{monitor.total_memory / 2**30:.0f} GiB",
                                )
                            last_load = now
                    box.code("\n".join(tail))
                except runner.StageFailedError as exc:
                    box.code("\n".join(tail))
                    status.update(state="error", expanded=True)
                    if show_load:
                        _render_load_bar(load_box, "System load — idle")
                    st.error(f"{exc} — full log at `{exc.log_path}`")
                    return
                status.update(state="complete", expanded=False)

        if show_load:
            _render_load_bar(load_box, "System load — idle")
        st.success(f"Pipeline finished. Artifacts in `{ctx.run_dir}`")


main()
