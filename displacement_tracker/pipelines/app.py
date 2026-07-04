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
import streamlit.components.v1 as components
import yaml

from displacement_tracker.pipelines import runner
from displacement_tracker.pipelines.spec import PIPELINES, Param
from displacement_tracker.util.env_loader import load_yaml_with_env

# The load bar is a components.html iframe rather than st.markdown: the
# same-origin script pins the frame to the viewport bottom, aligns its left
# edge with the sidebar (observing collapse/resize), and drives the chevron
# collapse toggle, persisting that state in localStorage across re-renders.
# Meter styling follows the reference dataviz palette: blue accent on a
# same-ramp track, yellow/red severity steps, text in text tokens, dark-mode
# colors selected explicitly.
_LOADBAR_TEMPLATE = """
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; overflow: hidden;
    font-family: "Source Sans Pro", "Source Sans", sans-serif;
  }
  .bar {
    display: flex; gap: 2.5rem; align-items: flex-end;
    height: 70px; padding: 0.7rem 1.5rem 0.95rem;
    background: #fcfcfb; border-top: 1px solid rgba(11,11,11,.10);
  }
  .title {
    font-size: 0.72rem; letter-spacing: .06em; text-transform: uppercase;
    color: #52514e; white-space: nowrap; padding-bottom: 0.15rem;
  }
  .meter { flex: 1; min-width: 8rem; }
  .lbl {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: #52514e; margin-bottom: 0.25rem;
  }
  .lbl b { color: #0b0b0b; font-weight: 600; font-variant-numeric: tabular-nums; }
  .track { height: 8px; border-radius: 4px; overflow: hidden; background: #cde2fb; }
  .fill { height: 100%; border-radius: 4px; background: #2a78d6; }
  .meter.warn .track { background: #f7e3b6; }
  .meter.warn .fill { background: #eda100; }
  .meter.crit .track { background: #f8d2d2; }
  .meter.crit .fill { background: #e34948; }
  .chev {
    border: 1px solid rgba(11,11,11,.15); border-radius: 50%;
    width: 26px; height: 26px; padding: 0; cursor: pointer;
    background: #fcfcfb; color: #52514e; font-size: 13px; line-height: 1;
    align-self: center;
  }
  .chev:hover { color: #0b0b0b; border-color: rgba(11,11,11,.4); }
  #tn-show { display: none; width: 34px; height: 34px; margin: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,.2); }
  body.collapsed .bar { display: none; }
  body.collapsed #tn-show { display: block; }
  @media (prefers-color-scheme: dark) {
    .bar { background: #1a1a19; border-top-color: rgba(255,255,255,.14); }
    .title, .lbl { color: #c3c2b7; }
    .lbl b { color: #ffffff; }
    .track { background: #0d366b; }
    .fill { background: #3987e5; }
    .meter.warn .track { background: #4a3404; }
    .meter.warn .fill { background: #c98500; }
    .meter.crit .track { background: #571f1f; }
    .meter.crit .fill { background: #e66767; }
    .chev { background: #1a1a19; color: #c3c2b7;
      border-color: rgba(255,255,255,.25); }
    .chev:hover { color: #ffffff; }
  }
</style>
<body>
  <div class="bar">
    <span class="title">__TITLE__</span>
    <div class="meter __CPU_CLS__">
      <div class="lbl"><span>CPU (avg per core)</span><b>__CPU_TEXT__</b></div>
      <div class="track"><div class="fill" style="width:__CPU_W__%"></div></div>
    </div>
    <div class="meter __MEM_CLS__">
      <div class="lbl"><span>Memory (of total)</span><b>__MEM_TEXT__</b></div>
      <div class="track"><div class="fill" style="width:__MEM_W__%"></div></div>
    </div>
    <button class="chev" id="tn-hide" title="Hide system load">&#x2304;</button>
  </div>
  <button class="chev" id="tn-show" title="Show system load">&#x2303;</button>
  <script>
    const KEY = "tnLoadbarCollapsed";
    const frame = window.frameElement;
    const pdoc = window.parent.document;
    const sidebar = pdoc.querySelector('section[data-testid="stSidebar"]');
    frame.style.position = "fixed";
    frame.style.border = "0";
    frame.style.zIndex = "999";

    function mainBlock() {
      return pdoc.querySelector('div[data-testid="stMainBlockContainer"]')
          || pdoc.querySelector('section[data-testid="stMain"]');
    }
    function layout() {
      const collapsed = localStorage.getItem(KEY) === "1";
      document.body.classList.toggle("collapsed", collapsed);
      if (collapsed) {
        frame.style.left = "auto";
        frame.style.right = "8px";
        frame.style.bottom = "8px";
        frame.style.width = "44px";
        frame.style.height = "44px";
      } else {
        const left = sidebar
          ? Math.max(0, sidebar.getBoundingClientRect().right) : 0;
        frame.style.right = "auto";
        frame.style.bottom = "0";
        frame.style.left = left + "px";
        frame.style.width = "calc(100vw - " + left + "px)";
        frame.style.height = "70px";
      }
      const mb = mainBlock();
      if (mb) mb.style.paddingBottom = collapsed ? "3rem" : "6.5rem";
    }
    document.getElementById("tn-hide").onclick = () => {
      localStorage.setItem(KEY, "1"); layout();
    };
    document.getElementById("tn-show").onclick = () => {
      localStorage.setItem(KEY, "0"); layout();
    };
    if (sidebar) {
      new ResizeObserver(layout).observe(sidebar);
      new MutationObserver(layout).observe(sidebar, { attributes: true });
    }
    window.addEventListener("resize", layout);
    setInterval(layout, 500);  // fallback for animated sidebar transitions
    layout();
  </script>
</body>
"""


def _severity(pct: float | None) -> str:
    if pct is None:
        return ""
    return "crit" if pct >= 90 else "warn" if pct >= 75 else ""


def _render_load_bar(
    box,
    title: str,
    cpu_pct: float | None = None,
    cpu_text: str = "—",
    mem_pct: float | None = None,
    mem_text: str = "—",
) -> None:
    def width(pct: float | None) -> str:
        return f"{max(0.0, min(100.0, pct or 0.0)):.1f}"

    html = (
        _LOADBAR_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__CPU_CLS__", _severity(cpu_pct))
        .replace("__CPU_TEXT__", cpu_text)
        .replace("__CPU_W__", width(cpu_pct))
        .replace("__MEM_CLS__", _severity(mem_pct))
        .replace("__MEM_TEXT__", mem_text)
        .replace("__MEM_W__", width(mem_pct))
    )
    with box:
        components.html(html, height=70)


def _update_load_bar(box, monitor, stage_label: str) -> None:
    """Best-effort load sampling — must never break a running pipeline."""
    try:
        cpu_sum, rss = monitor.sample()
        cores = getattr(monitor, "cpu_count", None) or 1
        total = getattr(monitor, "total_memory", None) or 0
        cpu_pct = cpu_sum / cores
        mem_pct = rss / total * 100 if total else None
        mem_text = (
            f"{rss / 2**30:.1f} / {total / 2**30:.0f} GiB" if total
            else f"{rss / 2**30:.1f} GiB"
        )
        _render_load_bar(
            box, f"System load — {stage_label}",
            cpu_pct, f"{cpu_pct:.0f} %", mem_pct, mem_text,
        )
    except Exception:
        pass


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
    st.set_page_config(page_title="TentNetFA", layout="wide")
    st.title("TentNetFA")

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

    # Fixed bottom bar (the iframe script escapes the tab flow entirely).
    load_box = st.empty()
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
                        if now - last_load > 1.0:
                            if monitor is None and ctx.active_process is not None:
                                try:
                                    monitor = runner.StageLoadMonitor(ctx.active_process)
                                except Exception:
                                    monitor = None
                            elif monitor is not None:
                                _update_load_bar(load_box, monitor, stage.label)
                            last_load = now
                    box.code("\n".join(tail))
                except runner.StageFailedError as exc:
                    box.code("\n".join(tail))
                    status.update(state="error", expanded=True)
                    _render_load_bar(load_box, "System load — idle")
                    st.error(f"{exc} — full log at `{exc.log_path}`")
                    return
                status.update(state="complete", expanded=False)

        _render_load_bar(load_box, "System load — idle")
        st.success(f"Pipeline finished. Artifacts in `{ctx.run_dir}`")


main()
