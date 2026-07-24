# Working in this repository

Guidance for coding agents (and humans) making changes to TentNetFA — a
pipeline that detects tents in Gaza Strip satellite imagery to support
displacement nowcasting. See `README.md` for what the project does and how to
run it end to end.

## Layout

| Path | Contents |
|---|---|
| `displacement_tracker/` | All source. Stage modules are ordered by pipeline position (`a_tif_loader`, `b2_image_scanner`, … `h_merge_geojsons`, `i_zonal_point_sums`). |
| `displacement_tracker/util/` | Shared logic: config resolution, thresholding, deduplication, reference data, validation. |
| `displacement_tracker/pipelines/` | Pipeline specs and the two frontends (Streamlit UI, `pipeline-run` CLI) that render from them. |
| `displacement_tracker/evaluation/` | The standalone analysis suite scoring predictions against manual annotations. |
| `tests/` | Unit tests. **See [`docs/test-patterns.md`](docs/test-patterns.md) before adding any.** |
| `config.yaml` | Single config for every flow, in `shared` / `train` / `predict` / `tune` sections. |

## Commands

```bash
poetry install                  # set up
poetry run pytest tests/ -q     # unit tests
poetry run ruff check .         # lint
poetry run ruff format .        # format
poetry run pipeline-run tune --dry-run   # resolve a flow without running it
```

CI runs the tests, `ruff check`, and `ruff format --check` on every pull
request into `main` or `Karim`, and all three must be green. That base list
lives in the `pull_request` trigger of `.github/workflows/ci.yml`; the
thermo-nuclear review workflow keeps its own copy in `ALLOWED_BASES`, so a
new base branch has to be added in both.

## Conventions

- **Tests: read [`docs/test-patterns.md`](docs/test-patterns.md) first.** It
  is binding. In short: the suite is a refactoring safety net for
  deterministic pipeline behavior, so test genuine business logic only,
  annotate every test with Given/When/Then comments split inline above the
  code each describes, use real files in `tmp_path` instead of mocks, and
  derive expected values by hand. A test that cannot fail for a real
  behavioral reason does not belong in the suite.
- **Config over flags.** Stages read their settings from a section of
  `config.yaml` resolved through `util/config.py`; `shared` holds anything
  more than one flow needs. Prefer adding a config key over adding a CLI
  flag, and keep function-signature defaults as the single source of truth
  for values the config does not set.
- **Pipelines are declarative.** `pipelines/spec.py` describes stages,
  parameters, and artifact paths; both frontends render from it. Adding a
  stage or a parameter means editing the spec, not the UI.
- **Match the surrounding code** in naming, structure, and comment density.
  Comment to record a constraint the code cannot express — not to narrate
  what the next line does.
- **Keep the repo small.** Imagery, model checkpoints, and generated results
  stay out of git; `.gitignore` covers the known output directories. Document
  where large inputs come from instead of committing them.
