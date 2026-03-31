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


def _write_sessions(root: Path) -> None:
    env = (root / ".env").read_text(encoding="utf-8")
    (root / ".env").write_text(env + "BOT_SESSION_ENABLED=true\n", encoding="utf-8")
    store = JsonlStore(root / "data")
    store.write_json("sessions", {
        "generation": 1,
        "sessions": {
            "session_sma_main": {
                "config": {
                    "session_id": "session_sma_main",
                    "provider_type": "technical_consensus",
                    "agent_ids": ["sma_agent", "momentum_agent", "mean_reversion_agent", "breakout_agent"],
                    "initial_capital_krw": "75000",
                    "created_at": "2026-03-31T00:00:00Z",
                    "main_agent_id": "sma_agent",
                    "vote_weights": {
                        "sma_agent": 0.4,
                        "momentum_agent": 0.2,
                        "mean_reversion_agent": 0.2,
                        "breakout_agent": 0.2,
                    },
                    "hyperparams": {"confidence_threshold": 0.3},
                },
                "is_active": True,
                "current_value_krw": "80000",
                "peak_value_krw": "82000",
                "total_pnl_krw": "5000",
                "return_pct": "6.67",
                "total_trades": 3,
                "winning_trades": 2,
                "max_drawdown_pct": "2.5",
                "latest_action": "buy",
                "latest_confidence": 0.36,
                "latest_reason": "buy (buy_w=0.360, sell_w=0.000)",
                "latest_buy_vote": "0.36",
                "latest_sell_vote": "0",
                "latest_leader_agent_id": "sma_agent",
                "last_tick": 5,
            }
        },
    })


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

    def test_signals_flatten_batched_decisions(self):
        root = _make_root()
        store = JsonlStore(root / "data")
        store.append("decisions", {
            "tick": 2,
            "market": "KRW-BTC",
            "price": "86000000",
            "signals": {
                "sma_agent": {"action": "buy", "confidence": 0.7, "reason": "sma"},
                "breakout_agent": {"action": "hold", "confidence": 0.4, "reason": "breakout"},
            },
        })
        api = DashboardAPI(root)
        result = api.signals()
        assert any(item["agent_id"] == "sma_agent" and item["action"] == "buy" for item in result)
        assert any(item["agent_id"] == "breakout_agent" and item["action"] == "hold" for item in result)

    def test_orders(self):
        root = _make_root()
        api = DashboardAPI(root)
        result = api.orders()
        assert len(result) >= 1
        assert result[0]["side"] == "bid"

    def test_sessions(self):
        root = _make_root()
        _write_sessions(root)
        api = DashboardAPI(root)
        result = api.sessions()
        assert len(result) == 1
        assert result[0]["session_id"] == "session_sma_main"
        assert result[0]["main_agent_id"] == "sma_agent"
        assert result[0]["latest_action"] == "buy"

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
            for endpoint in ["/api/overview", "/api/leaderboard", "/api/sessions", "/api/signals",
                             "/api/orders", "/api/risk", "/api/settings", "/api/report"]:
                resp = urllib.request.urlopen(f"{base}{endpoint}")
                assert resp.status == 200
                data = json.loads(resp.read())
                assert data is not None
        finally:
            server.shutdown()
