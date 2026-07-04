"""Console entry point that launches the Streamlit pipeline UI.

``pipeline-ui`` forwards any extra arguments to ``streamlit run``. The
``--remote`` flag applies the defaults for running on a remote machine
behind an SSH tunnel: bind to localhost only and don't open a browser.
"""

from __future__ import annotations

import getpass
import os
import socket
import sys
from pathlib import Path


def _flag_value(args: list[str], flag: str, default: str) -> str:
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


def _print_tunnel_hint(port: str, remote: bool) -> None:
    user = getpass.getuser()
    host = socket.gethostname()
    print(
        f"\nTo open the UI from your local machine, run this on it:\n"
        f"\n    ssh -L {port}:localhost:{port} {user}@{host}\n"
        f"\nthen browse to http://localhost:{port}\n"
    )
    if not remote:
        print(
            "Tip: `pipeline-ui --remote` binds to localhost only and skips "
            "opening a browser — recommended over SSH.\n"
        )


def main() -> None:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit(
            "streamlit is not installed. Install the UI extras first:\n"
            "    poetry install --with ui"
        )

    extra = sys.argv[1:]
    remote = "--remote" in extra
    if remote:
        extra = [arg for arg in extra if arg != "--remote"]
        if not any("server.headless" in arg for arg in extra):
            extra = ["--server.headless", "true", *extra]
        if not any("server.address" in arg for arg in extra):
            extra = ["--server.address", "localhost", *extra]
    # Hide streamlit's deploy button and developer menu unless the caller
    # explicitly asks for a different toolbar mode.
    if not any("client.toolbarMode" in arg for arg in extra):
        extra = ["--client.toolbarMode", "viewer", *extra]

    if remote or os.getenv("SSH_CONNECTION"):
        port = _flag_value(extra, "--server.port", "8501")
        _print_tunnel_hint(port, remote)

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *extra]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
