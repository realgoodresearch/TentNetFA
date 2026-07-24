# Test patterns

Conventions for everything under `tests/`. These are requirements, not
suggestions: a test that breaks them should be rewritten or deleted rather
than merged.

## 0. What this suite is for

It is a refactoring safety net. Its job is to pin the **deterministic**
behavior of the pipelines — the input-to-output contracts that must survive
any future restructuring of the code that implements them.

That purpose sets the boundary in both directions:

- **Only deterministic behavior is constrained.** Given the same inputs, the
  behavior under test must produce the same result on every machine and every
  run. Anything whose output depends on wall-clock time, a random draw, model
  training, floating-point noise beyond a stated tolerance, or the contents of
  a data share is out of scope — a test that pins it will either be flaky or
  be silently weakened until it stops catching anything. Where a computation
  is seeded and reproducible, it is fair game; where it is genuinely random,
  test the surrounding logic that *is* deterministic.
- **Only behavior worth preserving is constrained.** Pin the contract, not
  the implementation. A test should survive a rewrite of the internals that
  keeps the observable behavior intact, and fail the moment that behavior
  changes. Tests that reach into private helpers or restate the source line
  by line make refactoring harder, which is the opposite of the point.

## 1. Test genuine business logic only

A test earns its place by failing when the pipeline would produce a wrong
answer for a real user. Business logic here means the code that decides
*what the numbers are*:

- algorithms and numeric computation — thresholding, deduplication,
  distances, error statistics, grid aggregation;
- selection and filtering semantics — which points survive a threshold,
  which reference export pairs with which predictions, which rows a date
  filter keeps;
- configuration resolution — precedence between defaults, config sections,
  overrides, and forced invariants;
- validation and error rules — what raises, and whether the message names
  the thing the user has to fix;
- parsing of names, dates, and paths that drive the above.

**Do not test:**

- logging output, progress reporting, or plot rendering;
- click flag plumbing — test the function the CLI calls instead. (Driving a
  CLI is fine when the behavior under test *only* exists there, such as
  config resolution across sections; the test must then assert the resulting
  behavior, not that a flag was parsed.)
- trivial pass-throughs, `__repr__`, or constants restated from the source;
- anything whose only failure mode is "the mock wasn't called".

Ask of every test: **what bug does this catch?** If the honest answer is
"none that could plausibly be written", delete it. A smaller suite of sharp
tests beats a large one that reports green through real regressions.

## 2. Given / When / Then, split inline

Every test is annotated with three comments, each sitting **directly above
the code it describes**, separated by a blank line:

```python
def test_points_closer_than_the_threshold_merge_into_one():
    # Given: two detections 2 m apart, inside the 3 m merge radius
    points = [(31.5000000, 34.5, 0.9, 0.9), (31.5000180, 34.5, 0.4, 0.4)]

    # When: they are merged with a 3 m radius and no agreement requirement
    merged = merge_close_points_global(points, min_distance_m=3.0, agreement=1)

    # Then: they collapse to a single point at the midpoint, carrying the
    #       higher peak
    assert len(merged) == 1
    assert merged[0][0] == pytest.approx(31.5000090)
    assert merged[0][2] == pytest.approx(0.9)
```

Rules:

- **Given** describes the fixture and the properties that matter — the
  distances, counts, and CRSs the expectation depends on — not a restatement
  of the code beneath it.
- **When** names the call under test. Exactly one action; if a test needs two,
  see the multi-phase form below.
- **Then** states the expected outcome *and the reasoning behind a
  non-obvious number*, so a reader can check the expectation without
  re-deriving it.
- Continuation lines align under the first word after the label.
- The comments must stay **truthful**: if the code changes, the prose changes
  with it. A stale Given/When/Then is a defect.

For a test whose action is a single expression, bind it to a name so the
three phases remain distinct:

```python
def test_compact_date_in_a_file_name_is_parsed():
    # Given: an export file name carrying a compact YYYYMMDD date
    path = "/data/exports/UNOSAT_20240103_shelters.gpkg"

    # When: the date is extracted from the name
    parsed = extract_date_from_filename(path)

    # Then: it resolves to that calendar date
    assert parsed == datetime(2024, 1, 3)
```

A test that exercises the same behavior across several inputs keeps one set
of labels and loops inside the **When/Then** section. A test that genuinely
needs two actions — typically a contrasting case that would otherwise be a
near-duplicate — repeats the **When/Then** pair under the shared **Given**,
each above its own block.

When the action is expected to raise, `pytest.raises` fuses the call and the
assertion into one statement, so the two labels sit together above the
`with` line — When naming the call, Then describing the failure and the part
of the message that must appear:

```python
def test_a_two_date_csv_requires_an_explicit_date(tmp_path):
    # Given: a CSV holding annotations for two acquisition dates
    csv = write_csv(tmp_path / "ann.csv", ["2024-10-14,...", "2024-11-01,..."])

    # When: a reference source is built without naming a date
    # Then: it refuses, listing both dates the caller can choose from
    with pytest.raises(ValueError, match=r"2024-10-14.*2024-11-01"):
        ManualAnnotationReferenceSource(csv)
```

Do not contort a test to avoid this — binding `excinfo` and asserting
below it only to separate the labels adds an assertion without adding
coverage.

## 3. No mocks

`unittest.mock`, `MagicMock`, and patching the code under test are not used.

`monkeypatch` is permitted for one purpose: **cutting the test off from the
ambient environment**, so the result does not depend on the machine it runs
on. Setting or clearing an environment variable qualifies; so does stubbing
`load_dotenv` in a function that would otherwise read a developer's local
`.env`. The line is that you may neutralise the environment a behavior reads
from — never the behavior itself.

Where code touches the filesystem, build **real files in `tmp_path`** —
GeoJSON, GPKG, GeoTIFF, YAML, CSV, Parquet. Reading a real file back is real
behavior; a mocked reader only tests the mock. Fixtures stay small enough to
reason about: a handful of points, a 3×3 grid, two dates.

Passing a plain function or lambda where the code under test takes a callable
is not mocking — that is the function's real parameter contract.

## 4. Derive expectations by hand

Compute the expected value yourself from the definition of the behavior, on
inputs small enough to work through exactly. **Never** paste the
implementation's own output back in as the expectation — that locks in
whatever the code does today, including its bugs.

Where a value is exactly computable, assert it with `pytest.approx` and a
tight tolerance. Where two independent derivations exist (say a closed form
and a geometric argument), agreeing on both is good evidence the expectation
is right.

## 5. Assert sharply

Every assertion should be able to fail for a real behavioral reason:

- `len(result) > 0`, `assert result is not None`, and bare `isinstance`
  checks are almost never enough — assert *which* elements survived and what
  they contain;
- pin the specific cell, coordinate, or key the logic is responsible for, not
  just the aggregate;
- a tolerance wide enough that any plausible answer passes is not a test;
- when asserting an error, match the part of the message that names what the
  user must fix.

A useful check while writing: perturb the expectation or an input and confirm
the test fails. If it still passes, it is not testing what its name claims.

## 6. Structure

- Plain `pytest` functions, named for the behavior they pin
  (`test_<subject>_<expected behavior>`), not the function they call.
- **Test logic stays in its own file.** Reading one file top to bottom should
  tell you what is being pinned and why, without chasing indirection.
- **Shared machinery does not.** Builders that encode an on-disk format or a
  domain schema — the GeoJSON/GPKG/GeoTIFF/YAML/CSV writers, the CRS
  constants and coordinate transformers — live in `tests/_helpers.py` and are
  imported explicitly:

  ```python
  from _helpers import CRS_UTM, write_geojson
  ```

  Copying such a builder into a second test file duplicates a contract, and
  the copies drift: an annotation-CSV header written out in two places is two
  places to update and one place to forget. Explicit imports keep the
  dependency visible, which implicit fixture injection would not.
- `tests/conftest.py` is for cross-cutting *fixtures* only — isolating global
  state so tests cannot leak into one another, and pinning the matplotlib
  backend. It is not a dumping ground for helpers, and nothing in it should
  encode what a test expects.
- Group related tests under a comment banner naming the area under test.
- Tests must pass in any order — no reliance on execution order or on state
  left by another test or file. Watch for module-level registries and
  caches that an import mutates: those leak across files, and the fix is to
  isolate the state in a fixture, never to loosen the assertion until it
  tolerates both orderings.

## 7. Running them

```bash
poetry run pytest tests/ -q     # the suite
poetry run ruff check .         # lint
poetry run ruff format .        # formatting
```

CI runs all three on every pull request into one of the base branches listed
in `.github/workflows/ci.yml`. Tests install from `requirements-test.txt`, a
pinned subset that deliberately excludes the torch/selenium/opencv stack —
if a test needs one of those, it is an integration test and does not belong
in this suite.
