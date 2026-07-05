"""End-to-end pipeline orchestration for the training and prediction flows.

The package is split into three layers so frontends can be swapped out:

- ``spec``:   declarative description of each pipeline (stages, tunable
  parameters, artifact layout inside a run directory).
- ``runner``: frontend-agnostic execution — builds the run directory,
  resolves the config, and streams stage output.
- ``app``:    the Streamlit frontend (``pipeline-ui``).
- ``cli``:    a thin non-interactive frontend (``pipeline-run``) for
  scripting and debugging.
"""
