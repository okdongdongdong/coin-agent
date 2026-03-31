from __future__ import annotations

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

from .api import DashboardAPI

LOGGER = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class DashboardHandler(BaseHTTPRequestHandler):
    api: DashboardAPI

    def log_message(self, format: str, *args) -> None:
        LOGGER.debug(format, *args)

    def _send_json(self, data: object, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        ct = "text/html; charset=utf-8"
        if path.suffix == ".css":
            ct = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            ct = "application/javascript; charset=utf-8"
        elif path.suffix == ".png":
            ct = "image/png"
        elif path.suffix == ".svg":
            ct = "image/svg+xml"
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        # API routes
        if path == "/api/overview":
            self._send_json(self.api.overview())
        elif path == "/api/leaderboard":
            self._send_json(self.api.leaderboard())
        elif path == "/api/sessions":
            self._send_json(self.api.sessions())
        elif path == "/api/signals":
            self._send_json(self.api.signals())
        elif path == "/api/orders":
            self._send_json(self.api.orders())
        elif path == "/api/meta_decisions":
            self._send_json(self.api.meta_decisions())
        elif path == "/api/session_decisions":
            self._send_json(self.api.session_decisions())
        elif path == "/api/session_timeline":
            self._send_json(self.api.session_timeline())
        elif path == "/api/equity":
            self._send_json(self.api.equity_history())
        elif path == "/api/allocations":
            self._send_json(self.api.allocations())
        elif path == "/api/risk":
            self._send_json(self.api.risk_status())
        elif path == "/api/settings":
            self._send_json(self.api.settings_info())
        elif path == "/api/report":
            self._send_json({"text": self.api.report_text()})
        elif path == "/" or path == "/index.html":
            self._send_html(STATIC_DIR / "index.html")
        else:
            # Try static file
            safe_path = path.lstrip("/")
            file_path = STATIC_DIR / safe_path
            if file_path.exists() and file_path.is_file():
                self._send_html(file_path)
            else:
                self._send_html(STATIC_DIR / "index.html")


def run_dashboard(root: Path, host: str = "0.0.0.0", port: int = 8080) -> None:
    api = DashboardAPI(root)
    DashboardHandler.api = api

    server = HTTPServer((host, port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()
