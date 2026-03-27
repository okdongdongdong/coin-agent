from __future__ import annotations

import json
import tempfile
import threading
import time
import urllib.request
from decimal import Decimal
from pathlib import Path

from coin_agent.config.settings import Settings
from coin_agent.dashboard.api import DashboardAPI
from coin_agent.dashboard.server import run_dashboard, DashboardHandler, HTTPServer
from coin_agent.storage.jsonl_store import JsonlStore
from coin_agent.storage.state_store import StateStore


def _make_root() -> Path:
    root = Path(tempfile.mkdtemp())
    # Write minimal .env
    (root / ".env").write_text(
        "BITHUMB_ACCESS_KEY=\n"
        "BITHUMB_SECRET_KEY=\n"
        "BOT_MARKETS=KRW-BTC\n"
        "BOT_DRY_RUN=true\n"
        "BOT_PAPER_KRW_BALANCE=300000\n"
    )
    data_dir = root / "data"
    data_dir.mkdir()
    # Write bot state
    store = JsonlStore(data_dir)
    store.write_json("bot_state", {"status": "idle", "tick_count": 5})
    store.write_json("latest_report", {"text": "Test report\nCurrent Price: 85000000 KRW", "tick": 5})
    # Write a signal
    store.append("decisions", {"agent_id": "sma_agent", "action": "buy", "confidence": 0.7, "reason": "test"})
    # Write an order
    store.append("orders", {"side": "bid", "volume": "0.001", "price": "85000000", "success": True, "agent_id": "sma_agent"})
    return root


class TestDashboardAPI:
    def test_overview(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.overview()
        assert result["bot_status"] == "idle"
        assert result["tick_count"] == 5
        assert result["mode"] == "Paper"
        assert result["initial_capital"] == 300000.0

    def test_leaderboard_default(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.leaderboard()
        assert isinstance(result, list)
        assert len(result) == 4  # Default 4 agents

    def test_signals(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.signals()
        assert len(result) >= 1
        assert result[0]["agent_id"] == "sma_agent"

    def test_orders(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.orders()
        assert len(result) >= 1
        assert result[0]["side"] == "bid"

    def test_risk_status(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.risk_status()
        assert result["kill_switch"] is False
        assert result["max_daily_loss"] == 30000.0

    def test_settings_info(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.settings_info()
        assert result["paper_krw_balance"] == 300000.0
        assert result["dry_run"] is True

    def test_report_text(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.report_text()
        assert "Test report" in result


class TestDashboardServer:
    def test_http_endpoints(self):
        root = _make_root()
        api = DashboardAPI(root)
        DashboardHandler.api = api

        server = HTTPServer(("127.0.0.1", 0), DashboardHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            base = f"http://127.0.0.1:{port}"

            # Test index page
            resp = urllib.request.urlopen(f"{base}/")
            assert resp.status == 200
            html = resp.read().decode()
            assert "Coin Agent Dashboard" in html

            # Test API endpoints
            for endpoint in ["/api/overview", "/api/leaderboard", "/api/signals",
                             "/api/orders", "/api/risk", "/api/settings", "/api/report"]:
                resp = urllib.request.urlopen(f"{base}{endpoint}")
                assert resp.status == 200
                data = json.loads(resp.read())
                assert data is not None
        finally:
            server.shutdown()
