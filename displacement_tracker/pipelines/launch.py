"""Console entry point that launches the Streamlit pipeline UI."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit(
            "streamlit is not installed. Install the UI extras first:\n"
            "    poetry install --with ui"
        )

    app_path = Path(__file__).with_name("app.py")
    extra = sys.argv[1:]
    # Hide streamlit's deploy button and developer menu unless the caller
    # explicitly asks for a different toolbar mode.
    if not any("client.toolbarMode" in arg for arg in extra):
        extra = ["--client.toolbarMode", "viewer", *extra]
    sys.argv = ["streamlit", "run", str(app_path), *extra]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
