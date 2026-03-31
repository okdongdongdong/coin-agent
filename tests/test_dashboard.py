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
        "BOT_SESSION_CAPITAL_KRW=0\n"
    )
    data_dir = root / "data"
    data_dir.mkdir()
    # Write bot state
    store = JsonlStore(data_dir)
    store.write_json("bot_state", {"status": "idle", "tick_count": 5, "system_mode": "session_multi", "system_started_at": 1})
    store.write_json("latest_report", {"text": "Test report\nCurrent Price: 85000000 KRW", "tick": 5})
    # Write a signal
    store.append("decisions", {"agent_id": "sma_agent", "action": "buy", "confidence": 0.7, "reason": "test"})
    store.append("meta_decisions", {
        "tick": 5,
        "market": "KRW-BTC",
        "action": "buy",
        "confidence": 0.71,
        "agree_count": 6,
        "buy_count": 6,
        "sell_count": 1,
        "agreeing_session_ids": ["session_trend_core", "session_alpha_core"],
        "leader_session_id": "session_trend_core",
        "reason": "meta_consensus_buy",
    })
    # Write an order
    store.append("orders", {"side": "bid", "volume": "0.001", "price": "85000000", "success": True, "agent_id": "sma_agent"})
    store.append("equity_curve", {"tick": 1, "total_value": "300000", "krw_available": "300000", "position_value": "0", "reserve_capital": "0", "session_principal_total": "0", "sessions": {}, "_ts": 1})
    store.append("equity_curve", {"tick": 2, "total_value": "301000", "krw_available": "250000", "position_value": "51000", "reserve_capital": "0", "session_principal_total": "0", "sessions": {}, "_ts": 2})
    return root


def _write_sessions(root: Path) -> None:
    env = (root / ".env").read_text(encoding="utf-8")
    (root / ".env").write_text(env + "BOT_SESSION_ENABLED=true\n", encoding="utf-8")
    store = JsonlStore(root / "data")
    store.write_json("sessions", {
        "generation": 1,
        "sessions": {
            "session_trend_core": {
                "config": {
                    "session_id": "session_trend_core",
                    "provider_type": "multi_session",
                    "agent_ids": ["sma_agent", "alpha_agent", "steady_guard_agent", "momentum_agent", "breakout_agent"],
                    "initial_capital_krw": "100000",
                    "created_at": "2026-03-31T00:00:00Z",
                    "main_agent_id": "sma_agent",
                    "vote_weights": {
                        "sma_agent": 0.35,
                        "alpha_agent": 0.25,
                        "steady_guard_agent": 0.20,
                        "momentum_agent": 0.10,
                        "breakout_agent": 0.10,
                    },
                    "hyperparams": {"confidence_threshold": 0.27},
                },
                "is_active": True,
                "current_value_krw": "0",
                "peak_value_krw": "0",
                "total_pnl_krw": "0",
                "return_pct": "0",
                "total_trades": 3,
                "winning_trades": 2,
                "max_drawdown_pct": "0",
                "latest_action": "buy",
                "latest_confidence": 0.36,
                "latest_reason": "buy (buy_w=0.360, sell_w=0.000)",
                "latest_buy_vote": "0.36",
                "latest_sell_vote": "0",
                "latest_leader_agent_id": "sma_agent",
                "last_tick": 5,
                "vote_count": 5,
                "agreement_count": 4,
                "executed_count": 3,
            }
        },
    })
    StateStore(store).save_wallet("paper_session_trend_core", {
        "krw_available": "57500",
        "asset_available": "0.0005",
        "avg_buy_price": "85000000",
    })
    store.append("equity_curve", {
        "tick": 3,
        "total_value": "300500",
        "krw_available": "225000",
        "position_value": "75500",
        "reserve_capital": "0",
        "session_principal_total": "0",
        "sessions": {},
        "_ts": 3,
    })
    store.append("session_decisions", {
        "tick": 4,
        "market": "KRW-BTC",
        "session_id": "session_sma_main",
        "main_agent_id": "sma_agent",
        "action": "sell",
        "confidence": 0.28,
        "buy_vote": "0",
        "sell_vote": "0.280",
        "leader_agent_id": "sma_agent",
        "reason": "legacy session decision",
        "_ts": 3,
    })
    store.append("session_decisions", {
        "tick": 5,
        "market": "KRW-BTC",
        "session_id": "session_trend_core",
        "main_agent_id": "sma_agent",
        "action": "buy",
        "confidence": 0.36,
        "buy_vote": "0.360",
        "sell_vote": "0",
        "leader_agent_id": "sma_agent",
        "reason": "current session decision",
        "_ts": 4,
    })
    store.append("orders", {
        "side": "bid",
        "volume": "0.0005",
        "price": "85100000",
        "success": False,
        "agent_id": "meta_consensus",
        "leader_session_id": "session_trend_core",
        "session_id": "session_trend_core",
        "main_agent_id": "sma_agent",
        "message": "cooldown (16s remaining)",
        "_ts": 4,
    })
    store.append("orders", {
        "side": "ask",
        "volume": "0.0004",
        "price": "84900000",
        "success": False,
        "agent_id": "session_sma_main",
        "session_id": "session_sma_main",
        "main_agent_id": "sma_agent",
        "message": "position_limit (new=84000 > max=70000)",
        "_ts": 3,
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
        assert len(result) == 7

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
        _write_sessions(root)
        api = DashboardAPI(root)
        result = api.orders()
        assert len(result) == 1
        assert result[0]["side"] == "bid"
        assert result[0]["session_id"] == "session_trend_core"
        assert result[0]["result_code"] == "cooldown"

    def test_equity_history(self):
        root = _make_root()
        _write_sessions(root)
        api = DashboardAPI(root)
        result = api.equity_history()
        assert len(result["overall"]) >= 2
        assert result["overall"][-1]["total_value"] == 300500.0
        assert result["sessions"] == {}

    def test_sessions(self):
        root = _make_root()
        _write_sessions(root)
        api = DashboardAPI(root)
        result = api.sessions()
        assert len(result) == 1
        assert result[0]["session_id"] == "session_trend_core"
        assert result[0]["main_agent_id"] == "sma_agent"
        assert result[0]["latest_action"] == "buy"
        assert result[0]["latest_failure_code"] == "cooldown"
        assert result[0]["latest_failure_detail"] == "cooldown (16s remaining)"
        assert result[0]["vote_count"] == 5
        assert result[0]["agreement_count"] == 4

    def test_overview_splits_account_principal_and_session_principal(self):
        root = _make_root()
        _write_sessions(root)
        api = DashboardAPI(root)
        result = api.overview()
        assert result["initial_capital"] == 300000.0
        assert result["execution_mode"] == "multi"
        assert result["session_principal_total"] == 100000.0
        assert result["reserve_capital"] == 200000.0
        assert result["total_value"] == 300000.0
        assert result["order_failures"]["total_failures"] == 1
        assert result["order_failures"]["summary"][0]["code"] == "cooldown"
        assert result["order_failures"]["legacy_total_failures"] == 1
        assert result["order_failures"]["legacy_summary"][0]["code"] == "position_limit"
        assert result["meta_consensus"] == {}
        assert result["session_wallets"]["session_trend_core"]["total_value"] == 100000.0

    def test_session_timeline_filters_legacy_rows(self):
        root = _make_root()
        _write_sessions(root)
        api = DashboardAPI(root)
        result = api.session_timeline()
        assert result["ticks"] == [5]
        assert result["legacy_count"] == 1
        assert "session_trend_core" in result["sessions"]
        assert result["sessions"]["session_trend_core"][0]["action"] == "buy"

    def test_session_decisions_filters_legacy_rows(self):
        root = _make_root()
        _write_sessions(root)
        api = DashboardAPI(root)
        result = api.session_decisions()
        assert result["legacy_count"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["session_id"] == "session_trend_core"
        assert result["items"][0]["action"] == "buy"

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
        assert result["session_capital_krw"] == 0.0
        assert result["max_order_krw"] == 100000.0
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
                             "/api/orders", "/api/meta_decisions", "/api/session_timeline", "/api/session_decisions", "/api/equity", "/api/risk", "/api/settings", "/api/report"]:
                resp = urllib.request.urlopen(f"{base}{endpoint}")
                assert resp.status == 200
                data = json.loads(resp.read())
                assert data is not None
        finally:
            server.shutdown()
