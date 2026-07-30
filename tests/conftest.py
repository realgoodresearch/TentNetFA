"""Suite-wide environment isolation.

This file holds only what has to happen outside any single test file:
process-level environment setup. Fixture *builders* live in ``_helpers.py``
and are imported explicitly by the tests that use them.
"""

import os

# Importing the evaluation plot modules must not need a display.
os.environ.setdefault("MPLBACKEND", "Agg")
