"""Local production server launcher for the Word Forge workspace."""

from __future__ import annotations

import secrets
import socket
import threading
import time
import webbrowser
from pathlib import Path


def _open_when_ready(url: str, *, host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.1)


def serve(
    *,
    database_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    """Serve the local GUI until interrupted."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "The bundled GUI is local-only. Use a reviewed authenticated reverse "
            "proxy if remote access is required."
        )
    if not 1 <= port <= 65_535:
        raise ValueError("port must be between 1 and 65535")

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError(
            'GUI dependencies are missing. Install with `pip install "word-forge[gui]"`.'
        ) from exc

    from word_forge.web.app import create_app

    session_token = secrets.token_urlsafe(24)
    application = create_app(database_path=database_path, api_token=session_token)
    display_host = "127.0.0.1" if host in {"localhost", "::1"} else host
    url = f"http://{display_host}:{port}/#token={session_token}"
    if open_browser:
        threading.Thread(
            target=_open_when_ready,
            kwargs={"url": url, "host": display_host, "port": port},
            daemon=True,
        ).start()
    print(f"Word Forge workspace: {url}", flush=True)
    uvicorn.run(application, host=host, port=port, log_level="info")
    return 0


__all__ = ["serve"]
