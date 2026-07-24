"""Suite-wide environment isolation.

This file holds only what has to happen outside any single test file:
process-level environment setup, and the guard that keeps a module-level
registry from leaking between test files. Fixture *builders* live in
``_helpers.py`` and are imported explicitly by the tests that use them.
"""

import os

# Importing the evaluation plot modules must not need a display.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from displacement_tracker.util import reference_data  # noqa: E402

# ``reference_data.SOURCE_TYPES`` is a module-level dict that any importer
# may extend: importing ``evaluation.annotation_reference`` registers the
# "manual_eval" type into it. That happens at IMPORT time, i.e. during
# collection, before any fixture runs — so the built-in registry has to be
# captured here, in conftest, which pytest imports before it imports a
# single test module.
BUILTIN_SOURCE_TYPES = dict(reference_data.SOURCE_TYPES)


@pytest.fixture(autouse=True)
def source_types_registry():
    """Give every test the built-in reference registry, then put it back.

    Without this, whether ``_infer_type`` lists "manual_eval" among the
    valid types depends on which files pytest happened to collect first.
    Tests that need a registered extra type add it to the yielded dict.
    """
    registry = reference_data.SOURCE_TYPES
    collected = dict(registry)
    registry.clear()
    registry.update(BUILTIN_SOURCE_TYPES)

    yield registry

    registry.clear()
    registry.update(collected)
