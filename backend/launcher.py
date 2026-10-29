"""Local PC CPU Launcher for Ren.

Starts the local FastAPI + SQLite + CPU RAG service on http://127.0.0.1:8765
and automatically opens the Ren interface in the default desktop browser.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def open_browser_when_ready(url: str, host: str, port: int) -> None:
    for _ in range(30):
        if is_port_in_use(host, port):
            time.sleep(0.2)
            webbrowser.open(url)
            return
        time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Ren Local PC Knowledge Companion (CPU Mode)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    dist_index = Path(__file__).resolve().parent.parent / "frontend" / "dist" / "index.html"
    app_url = f"http://{args.host}:{args.port}" if dist_index.exists() else "http://localhost:5173"

    print("=" * 64)
    print("  Ren — Local-First Personal Knowledge Companion (CPU Mode)")
    print("=" * 64)
    print(f"  Local App URL : {app_url}")
    print(f"  API Docs      : http://{args.host}:{args.port}/docs")
    print("  Hardware Mode : 100% Local CPU (SQLite FTS5 + SimHash Vectors)")
    print("=" * 64)

    if is_port_in_use(args.host, args.port):
        print(f"[!] Ren is already running on {app_url}")
        if not args.no_browser:
            webbrowser.open(app_url)
        return

    if not args.no_browser:
        threading.Thread(
            target=open_browser_when_ready,
            args=(app_url, args.host, args.port),
            daemon=True,
        ).start()

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()

