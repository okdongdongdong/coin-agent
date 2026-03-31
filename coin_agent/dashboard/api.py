from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

from ..config.settings import Settings
from ..exchange.bithumb_client import BithumbAPIError, BithumbClient
from ..execution.broker import LiveBroker
from ..storage.jsonl_store import JsonlStore
from ..storage.state_store import StateStore

_TRACKED_AGENT_IDS = {
    "sma_agent",
    "momentum_agent",
    "mean_reversion_agent",
    "breakout_agent",
}
_SESSION_MAIN_ORDER = {
    "sma_agent": 0,
    "momentum_agent": 1,
    "mean_reversion_agent": 2,
    "breakout_agent": 3,
}


def _dec(v: Any) -> Any:
    """Convert Decimal to float for JSON serialization."""
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, dict):
        return {k: _dec(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_dec(i) for i in v]
    return v


class DashboardAPI:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._settings = Settings.load(root)
        self._store = JsonlStore(self._settings.data_dir)
        self._state = StateStore(self._store)

    def overview(self) -> Dict[str, Any]:
        bot = self._state.get_bot_state()
        kill = self._state.is_kill_switch_active()
        sessions = self.sessions() if self._settings.session_enabled else []
        active_sessions = [s for s in sessions if s.get("is_active", True)]

        total_krw = Decimal("0")
        total_asset = Decimal("0")
        agent_wallets: Dict[str, Any] = {}
        session_wallets: Dict[str, Any] = {}

        if self._settings.dry_run:
            if active_sessions:
                wallets_dir = self._settings.data_dir / "wallets"
                if wallets_dir.exists():
                    for session in active_sessions:
                        wid = f"paper_{session['session_id']}"
                        wp = wallets_dir / f"{wid}.json"
                        default_capital = Decimal(str(session.get("initial_capital_krw", 0)))
                        data = {
                            "krw_available": str(default_capital),
                            "asset_available": "0",
                            "avg_buy_price": "0",
                        }
                        if wp.exists():
                            data = json.loads(wp.read_text(encoding="utf-8"))
                        krw = Decimal(str(data.get("krw_available", "0")))
                        asset = Decimal(str(data.get("asset_available", "0")))
                        avg = Decimal(str(data.get("avg_buy_price", "0")))
                        total_krw += krw
                        total_asset += asset
                        session_wallets[session["session_id"]] = {
                            "krw_available": float(krw),
                            "asset_available": float(asset),
                            "avg_buy_price": float(avg),
                        }
            else:
                wallets_dir = self._settings.data_dir / "wallets"
                if wallets_dir.exists():
                    for wp in wallets_dir.glob("*.json"):
                        wid = wp.stem
                        data = json.loads(wp.read_text(encoding="utf-8"))
                        krw = Decimal(str(data.get("krw_available", "0")))
                        asset = Decimal(str(data.get("asset_available", "0")))
                        avg = Decimal(str(data.get("avg_buy_price", "0")))
                        total_krw += krw
                        total_asset += asset
                        agent_wallets[wid] = {
                            "krw_available": float(krw),
                            "asset_available": float(asset),
                            "avg_buy_price": float(avg),
                        }
        else:
            try:
                broker = LiveBroker(BithumbClient(self._settings))
                wallet = broker.get_wallet("global", self._settings.market_asset(self._settings.primary_market))
                total_krw = wallet.krw_available
                total_asset = wallet.asset_available
            except (BithumbAPIError, ValueError):
                pass

        # Latest report info
        report_data = self._store.read_json("latest_report")
        report_text = report_data.get("text", "") if report_data else ""
        report_tick = report_data.get("tick", 0) if report_data else 0

        # Parse current price from report
        current_price = 0.0
        for line in report_text.splitlines():
            if "Current Price:" in line:
                try:
                    price_str = line.split("Current Price:")[-1].strip().replace(",", "").replace(" KRW", "")
                    current_price = float(price_str)
                except (ValueError, IndexError):
                    pass
                break

        position_value = float(total_asset) * current_price
        total_value = float(total_krw) + position_value
        initial = float(self._settings.paper_krw_balance)
        pnl = total_value - initial if total_value > 0 else 0
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0

        return {
            "bot_status": bot.get("status", "idle"),
            "tick_count": bot.get("tick_count", 0),
            "last_tick": bot.get("last_tick", ""),
            "kill_switch": kill,
            "mode": "Paper" if self._settings.dry_run else "LIVE",
            "markets": self._settings.markets,
            "initial_capital": initial,
            "total_value": round(total_value, 0),
            "krw_available": float(total_krw),
            "position_value": round(position_value, 0),
            "total_asset": float(total_asset),
            "current_price": current_price,
            "pnl": round(pnl, 0),
            "pnl_pct": round(pnl_pct, 2),
            "agent_wallets": agent_wallets,
            "session_wallets": session_wallets,
            "active_sessions": len(active_sessions),
            "report_tick": report_tick,
        }

    def leaderboard(self) -> List[Dict[str, Any]]:
        scores = self._store.read_lines("agent_scores", last_n=1)
        if not scores:
            # Build from agent states
            agents_dir = self._settings.data_dir / "agent_states"
            result = []
            if agents_dir.exists():
                for ap in agents_dir.glob("*.json"):
                    data = json.loads(ap.read_text(encoding="utf-8"))
                    result.append({
                        "agent_id": ap.stem,
                        "composite": 50.0,
                        "allocation_pct": 25.0,
                        "total_trades": data.get("trades", 0),
                        "win_rate": 0,
                        "pnl": 0,
                        "max_drawdown": 0,
                        "phase": data.get("phase", "warmup"),
                    })
            if not result:
                for aid in ["sma_agent", "momentum_agent", "mean_reversion_agent", "breakout_agent"]:
                    result.append({
                        "agent_id": aid,
                        "composite": 50.0,
                        "allocation_pct": 25.0,
                        "total_trades": 0,
                        "win_rate": 0,
                        "pnl": 0,
                        "max_drawdown": 0,
                        "phase": "warmup",
                    })
            return result

        latest = scores[-1]
        ranking = latest.get("ranking", [])
        result = []
        for entry in ranking:
            result.append(_dec(entry))
        return result

    def signals(self, last_n: int = 30) -> List[Dict[str, Any]]:
        signal_lines = self._store.read_lines("signals", last_n=last_n)
        if signal_lines:
            return [_dec(l) for l in signal_lines if l.get("agent_id") in _TRACKED_AGENT_IDS]

        decisions = self._store.read_lines("decisions", last_n=last_n)
        flattened: List[Dict[str, Any]] = []
        for item in decisions:
            if "agent_id" in item and "action" in item:
                if item.get("agent_id") not in _TRACKED_AGENT_IDS:
                    continue
                flattened.append(item)
                continue
            signals = item.get("signals", {})
            for agent_id, signal in signals.items():
                if agent_id not in _TRACKED_AGENT_IDS:
                    continue
                flattened.append({
                    "tick": item.get("tick"),
                    "market": item.get("market"),
                    "price": item.get("price"),
                    "agent_id": agent_id,
                    "action": signal.get("action", "hold"),
                    "confidence": signal.get("confidence", 0.0),
                    "reason": signal.get("reason", ""),
                    "_ts": item.get("_ts", time.time()),
                })
        return [_dec(l) for l in flattened]

    def sessions(self) -> List[Dict[str, Any]]:
        if not self._settings.session_enabled:
            return []
        data = self._store.read_json("sessions")
        raw_sessions = data.get("sessions", {}) if data else {}
        result: List[Dict[str, Any]] = []
        for session_id, session in raw_sessions.items():
            config = session.get("config", {})
            result.append(_dec({
                "session_id": session_id,
                "provider_type": config.get("provider_type", "technical_consensus"),
                "main_agent_id": config.get("main_agent_id", ""),
                "agent_ids": config.get("agent_ids", []),
                "vote_weights": config.get("vote_weights", {}),
                "initial_capital_krw": config.get("initial_capital_krw", "0"),
                "is_active": session.get("is_active", True),
                "current_value_krw": session.get("current_value_krw", "0"),
                "peak_value_krw": session.get("peak_value_krw", "0"),
                "total_pnl_krw": session.get("total_pnl_krw", "0"),
                "return_pct": session.get("return_pct", "0"),
                "total_trades": session.get("total_trades", 0),
                "winning_trades": session.get("winning_trades", 0),
                "max_drawdown_pct": session.get("max_drawdown_pct", "0"),
                "latest_action": session.get("latest_action", "hold"),
                "latest_confidence": session.get("latest_confidence", 0.0),
                "latest_reason": session.get("latest_reason", ""),
                "latest_buy_vote": session.get("latest_buy_vote", "0"),
                "latest_sell_vote": session.get("latest_sell_vote", "0"),
                "latest_leader_agent_id": session.get("latest_leader_agent_id", ""),
                "last_tick": session.get("last_tick", 0),
            }))

        result.sort(
            key=lambda item: (
                _SESSION_MAIN_ORDER.get(str(item.get("main_agent_id", "")), 99),
                str(item.get("session_id", "")),
            )
        )
        return result

    def orders(self, last_n: int = 30) -> List[Dict[str, Any]]:
        lines = self._store.read_lines("orders", last_n=last_n)
        return [_dec(l) for l in lines]

    def allocations(self) -> List[Dict[str, Any]]:
        lines = self._store.read_lines("allocations", last_n=10)
        return [_dec(l) for l in lines]

    def risk_status(self) -> Dict[str, Any]:
        cb_path = self._settings.data_dir / "circuit_breaker.json"
        cb_data: Dict[str, Any] = {}
        if cb_path.exists():
            cb_data = json.loads(cb_path.read_text(encoding="utf-8"))

        return {
            "kill_switch": self._state.is_kill_switch_active(),
            "circuit_breaker": _dec(cb_data),
            "max_daily_loss": float(self._settings.max_daily_loss_krw),
            "max_total_loss": float(self._settings.max_total_loss_krw),
            "max_position_pct": float(self._settings.max_position_pct),
            "order_cooldown_sec": self._settings.order_cooldown_sec,
        }

    def settings_info(self) -> Dict[str, Any]:
        s = self._settings
        return {
            "markets": s.markets,
            "dry_run": s.dry_run,
            "paper_krw_balance": float(s.paper_krw_balance),
            "fee_rate": float(s.fee_rate),
            "loop_interval_sec": s.loop_interval_sec,
            "candle_unit": s.candle_unit,
            "candle_count": s.candle_count,
            "order_cooldown_sec": s.order_cooldown_sec,
            "max_daily_loss_krw": float(s.max_daily_loss_krw),
            "max_total_loss_krw": float(s.max_total_loss_krw),
            "max_position_pct": float(s.max_position_pct),
            "bench_threshold": s.bench_threshold,
            "min_active_agents": s.min_active_agents,
            "rebalance_interval": s.rebalance_interval,
            "softmax_temperature": s.softmax_temperature,
            "min_alloc_pct": float(s.min_alloc_pct),
            "max_alloc_pct": float(s.max_alloc_pct),
            "session_enabled": s.session_enabled,
            "session_min_count": s.session_min_count,
            "session_max_count": s.session_max_count,
        }

    def report_text(self) -> str:
        data = self._store.read_json("latest_report")
        if not data:
            return ""
        return data.get("text", "")
