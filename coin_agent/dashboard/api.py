from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

from ..config.settings import Settings
from ..exchange.bithumb_client import BithumbAPIError, BithumbClient
from ..execution.broker import LiveBroker, PaperBroker
from ..session.manager import AGENT_IDS, SESSION_TEMPLATE_ORDER
from ..storage.jsonl_store import JsonlStore
from ..storage.state_store import StateStore

_TRACKED_AGENT_IDS = set(AGENT_IDS)
_ORDER_RESULT_LABELS = {
    "submitted": "Submitted",
    "filled_on_check": "Filled on check",
    "partial_fill_on_check": "Partial fill on check",
    "cancelled_partial_next_tick": "Partial fill, remainder canceled",
    "cancelled_unfilled_next_tick": "Unfilled by next tick",
    "cancelled_without_fill": "Canceled without fill",
    "cancel_error": "Cancel error",
    "cooldown": "Cooldown",
    "position_limit": "Position limit",
    "order_too_small": "Order too small",
    "insufficient_krw": "Insufficient KRW",
    "max_exposure": "Portfolio exposure limit",
    "daily_loss_limit": "Daily loss limit",
    "total_loss_limit": "Total loss limit",
    "kill_switch": "Kill switch",
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


def _order_result_meta(message: Any, success: bool) -> Dict[str, Any]:
    raw = str(message or "").strip()
    code = "unknown"
    category = "other"
    is_failure = not success

    if raw == "submitted":
        code = "submitted"
        category = "submitted"
        is_failure = False
    elif raw == "filled_on_check":
        code = "filled_on_check"
        category = "filled"
        is_failure = False
    elif raw == "partial_fill_on_check":
        code = "partial_fill_on_check"
        category = "partial_fill"
        is_failure = False
    elif raw == "cancelled_partial_next_tick":
        code = "cancelled_partial_next_tick"
        category = "partial_fill"
        is_failure = False
    elif raw == "cancelled_unfilled_next_tick":
        code = "cancelled_unfilled_next_tick"
        category = "unfilled"
        is_failure = True
    elif raw == "cancelled_without_fill":
        code = "cancelled_without_fill"
        category = "cancelled"
        is_failure = True
    elif raw.startswith("cancel_error:"):
        code = "cancel_error"
        category = "error"
        is_failure = True
    elif raw.startswith("cooldown"):
        code = "cooldown"
        category = "blocked"
        is_failure = True
    elif raw.startswith("position_limit"):
        code = "position_limit"
        category = "blocked"
        is_failure = True
    elif raw.startswith("order_too_small"):
        code = "order_too_small"
        category = "blocked"
        is_failure = True
    elif raw.startswith("insufficient_krw"):
        code = "insufficient_krw"
        category = "blocked"
        is_failure = True
    elif raw.startswith("max_exposure"):
        code = "max_exposure"
        category = "blocked"
        is_failure = True
    elif raw.startswith("daily_loss_limit"):
        code = "daily_loss_limit"
        category = "blocked"
        is_failure = True
    elif raw.startswith("total_loss_limit"):
        code = "total_loss_limit"
        category = "blocked"
        is_failure = True
    elif raw.startswith("kill_switch"):
        code = "kill_switch"
        category = "blocked"
        is_failure = True
    elif raw:
        code = raw.split("(", 1)[0].split(":", 1)[0].strip().replace(" ", "_")
        category = "error" if not success else "other"

    return {
        "result_code": code,
        "result_label": _ORDER_RESULT_LABELS.get(code, code.replace("_", " ").title()),
        "result_detail": raw or "-",
        "result_category": category,
        "is_failure": is_failure,
    }


class DashboardAPI:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._settings = Settings.load(root)
        self._store = JsonlStore(self._settings.data_dir)
        self._state = StateStore(self._store)

    def _enrich_order(self, item: Dict[str, Any]) -> Dict[str, Any]:
        enriched = _dec(item)
        enriched.update(
            _order_result_meta(
                message=enriched.get("message", ""),
                success=bool(enriched.get("success", False)),
            )
        )
        return enriched

    def _order_failure_context(self, last_n: int = 200) -> Dict[str, Any]:
        current_session_ids = self._current_session_ids()
        orders = [
            self._enrich_order(line)
            for line in self._store.read_lines("orders", last_n=last_n)
        ]
        summary_map: Dict[str, Dict[str, Any]] = {}
        legacy_summary_map: Dict[str, Dict[str, Any]] = {}
        latest_failure: Dict[str, Any] | None = None
        latest_legacy_failure: Dict[str, Any] | None = None
        latest_failure_by_session: Dict[str, Dict[str, Any]] = {}

        for order in orders:
            if not order.get("is_failure"):
                continue

            is_current = self._is_current_order(order, current_session_ids)
            code = str(order.get("result_code", "unknown"))
            target_map = summary_map if is_current else legacy_summary_map
            bucket = target_map.setdefault(code, {
                "code": code,
                "label": order.get("result_label", code),
                "count": 0,
            })
            bucket["count"] += 1

            ts = float(order.get("_ts", order.get("timestamp", 0)) or 0)
            session_id = str(
                order.get("session_id")
                or order.get("leader_session_id")
                or order.get("agent_id")
                or "global"
            )
            failure_entry = {
                "session_id": session_id,
                "agent_id": order.get("agent_id", ""),
                "main_agent_id": order.get("main_agent_id", ""),
                "side": order.get("side", ""),
                "timestamp": ts,
                "result_code": code,
                "result_label": order.get("result_label", code),
                "result_detail": order.get("result_detail", ""),
            }

            previous = latest_failure_by_session.get(session_id)
            if is_current and (previous is None or ts >= float(previous.get("timestamp", 0) or 0)):
                latest_failure_by_session[session_id] = failure_entry

            if is_current and (latest_failure is None or ts >= float(latest_failure.get("timestamp", 0) or 0)):
                latest_failure = failure_entry
            if (not is_current) and (latest_legacy_failure is None or ts >= float(latest_legacy_failure.get("timestamp", 0) or 0)):
                latest_legacy_failure = failure_entry

        summary = sorted(summary_map.values(), key=lambda item: (-int(item["count"]), str(item["code"])))
        legacy_summary = sorted(legacy_summary_map.values(), key=lambda item: (-int(item["count"]), str(item["code"])))
        return {
            "total_failures": sum(int(item["count"]) for item in summary),
            "summary": summary,
            "latest": latest_failure or {},
            "by_session": latest_failure_by_session,
            "legacy_total_failures": sum(int(item["count"]) for item in legacy_summary),
            "legacy_summary": legacy_summary,
            "legacy_latest": latest_legacy_failure or {},
        }

    def _current_session_ids(self) -> set[str]:
        data = self._store.read_json("sessions")
        raw_sessions = data.get("sessions", {}) if data else {}
        return set(raw_sessions.keys())

    def _current_system_started_at(self) -> float:
        bot = self._state.get_bot_state()
        started_at = float(bot.get("system_started_at", 0) or 0)
        if started_at > 0:
            return started_at
        path = self._settings.data_dir / "meta_decisions.jsonl"
        if not path.exists():
            return 0.0
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue
            return float(line.get("_ts", 0) or 0)
        return 0.0

    def _is_current_order(self, order: Dict[str, Any], current_session_ids: set[str]) -> bool:
        ts = float(order.get("_ts", order.get("timestamp", 0)) or 0)
        started_at = self._current_system_started_at()
        if started_at > 0 and ts < started_at:
            return False
        session_id = str(order.get("session_id") or "")
        leader_session_id = str(order.get("leader_session_id") or "")
        agent_id = str(order.get("agent_id") or "")
        if session_id in current_session_ids or leader_session_id in current_session_ids:
            return True
        if agent_id == "meta_consensus":
            return True
        if started_at > 0 and ts >= started_at and str(order.get("meta_action") or "") in {"buy", "sell", "hold"}:
            return True
        return False

    def _latest_report_info(self) -> Dict[str, Any]:
        report_data = self._store.read_json("latest_report")
        report_text = report_data.get("text", "") if report_data else ""
        report_tick = report_data.get("tick", 0) if report_data else 0

        current_price = 0.0
        for line in report_text.splitlines():
            if "Current Price:" not in line:
                continue
            try:
                price_str = line.split("Current Price:")[-1].strip().replace(",", "").replace(" KRW", "")
                current_price = float(price_str)
            except (ValueError, IndexError):
                current_price = 0.0
            break

        return {
            "report_text": report_text,
            "report_tick": report_tick,
            "current_price": current_price,
        }

    def _global_wallet_snapshot(self, current_price: float) -> Dict[str, Any]:
        if self._settings.dry_run:
            broker = PaperBroker(self._settings, self._state)
            wallet = broker.get_wallet("global", self._settings.market_asset(self._settings.primary_market))
        else:
            broker = LiveBroker(BithumbClient(self._settings))
            wallet = broker.get_wallet("global", self._settings.market_asset(self._settings.primary_market))
        position_value = float(wallet.asset_available) * current_price
        return {
            "krw_available": float(wallet.krw_available),
            "asset_available": float(wallet.asset_available),
            "avg_buy_price": float(wallet.avg_buy_price),
            "position_value": position_value,
            "total_value": float(wallet.krw_available) + position_value,
        }

    def _current_execution_mode(self) -> str:
        if not self._settings.session_enabled:
            return "global"
        return self._settings.session_execution_mode

    def _session_wallet_snapshot(
        self,
        session_id: str,
        initial_capital_krw: Decimal,
        current_price: float,
    ) -> Dict[str, Any]:
        wallet_key = f"{'paper' if self._settings.dry_run else 'live'}_{session_id}"
        default_capital = str(initial_capital_krw)
        wallet = self._state.get_wallet(
            wallet_key,
            {
                "krw_available": default_capital,
                "asset_available": "0",
                "avg_buy_price": "0",
            },
        )
        krw_available = float(wallet.get("krw_available", default_capital) or 0)
        asset_available = float(wallet.get("asset_available", 0) or 0)
        avg_buy_price = float(wallet.get("avg_buy_price", 0) or 0)
        position_value = asset_available * current_price
        return {
            "krw_available": krw_available,
            "asset_available": asset_available,
            "avg_buy_price": avg_buy_price,
            "position_value": position_value,
            "total_value": krw_available + position_value,
        }

    def _latest_meta_decision(self) -> Dict[str, Any]:
        lines = self._store.read_lines("meta_decisions", last_n=1)
        if not lines:
            return {}
        return _dec(lines[-1])

    def overview(self) -> Dict[str, Any]:
        bot = self._state.get_bot_state()
        kill = self._state.is_kill_switch_active()
        sessions = self.sessions() if self._settings.session_enabled else []
        execution_mode = self._current_execution_mode()
        initial = float(self._settings.paper_krw_balance)
        report_info = self._latest_report_info()
        current_price = report_info["current_price"]

        report_tick = report_info["report_tick"]
        if self._settings.session_enabled and execution_mode == "multi":
            session_principal_total = sum(
                float(session.get("initial_capital_krw", 0) or 0)
                for session in sessions
            )
            reserve_capital = max(0.0, initial - session_principal_total)
            session_wallets = {
                str(session["session_id"]): {
                    "krw_available": float(session.get("krw_available", 0) or 0),
                    "asset_available": float(session.get("asset_available", 0) or 0),
                    "avg_buy_price": float(session.get("avg_buy_price", 0) or 0),
                    "position_value": float(session.get("position_value", 0) or 0),
                    "total_value": float(session.get("wallet_total_value", 0) or 0),
                }
                for session in sessions
            }
            total_krw = reserve_capital + sum(
                float(wallet.get("krw_available", 0) or 0)
                for wallet in session_wallets.values()
            )
            total_asset = sum(
                float(wallet.get("asset_available", 0) or 0)
                for wallet in session_wallets.values()
            )
            position_value = sum(
                float(wallet.get("position_value", 0) or 0)
                for wallet in session_wallets.values()
            )
            total_value = total_krw + position_value
            wallet_snapshot = {
                "krw_available": total_krw,
                "asset_available": total_asset,
                "avg_buy_price": 0.0,
                "position_value": position_value,
                "total_value": total_value,
            }
            meta_consensus = {}
        else:
            try:
                wallet_snapshot = self._global_wallet_snapshot(current_price)
            except (BithumbAPIError, ValueError):
                wallet_snapshot = {
                    "krw_available": initial,
                    "asset_available": 0.0,
                    "avg_buy_price": 0.0,
                    "position_value": 0.0,
                    "total_value": initial,
                }
            total_value = float(wallet_snapshot["total_value"])
            session_principal_total = 0.0
            reserve_capital = 0.0
            session_wallets = {}
            meta_consensus = self._latest_meta_decision()

        pnl = total_value - initial if total_value > 0 else 0
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0
        order_failures = self._order_failure_context()

        return {
            "bot_status": bot.get("status", "idle"),
            "tick_count": bot.get("tick_count", 0),
            "last_tick": bot.get("last_tick", ""),
            "kill_switch": kill,
            "mode": "Paper" if self._settings.dry_run else "LIVE",
            "execution_mode": execution_mode,
            "markets": self._settings.markets,
            "initial_capital": initial,
            "total_value": round(total_value, 0),
            "krw_available": float(wallet_snapshot["krw_available"]),
            "position_value": round(float(wallet_snapshot["position_value"]), 0),
            "total_asset": float(wallet_snapshot["asset_available"]),
            "current_price": current_price,
            "pnl": round(pnl, 0),
            "pnl_pct": round(pnl_pct, 2),
            "session_principal_total": round(session_principal_total, 0),
            "reserve_capital": round(reserve_capital, 0),
            "agent_wallets": {},
            "session_wallets": session_wallets,
            "active_sessions": len([s for s in sessions if s.get("is_active", True)]),
            "report_tick": report_tick,
            "order_failures": order_failures,
            "meta_consensus": meta_consensus,
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
                for aid in AGENT_IDS:
                    result.append({
                        "agent_id": aid,
                        "composite": 50.0,
                        "allocation_pct": round(100 / max(len(AGENT_IDS), 1), 2),
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
        failure_by_session = self._order_failure_context().get("by_session", {})
        current_price = self._latest_report_info()["current_price"]
        result: List[Dict[str, Any]] = []
        for session_id, session in raw_sessions.items():
            config = session.get("config", {})
            failure = failure_by_session.get(session_id, {})
            initial_capital_krw = Decimal(str(config.get("initial_capital_krw", "0")))
            wallet_snapshot = self._session_wallet_snapshot(
                session_id=session_id,
                initial_capital_krw=initial_capital_krw,
                current_price=current_price,
            )
            vote_count = int(session.get("vote_count", 0) or 0)
            agreement_count = int(session.get("agreement_count", 0) or 0)
            executed_count = int(session.get("executed_count", 0) or 0)
            agreement_rate = (agreement_count / vote_count * 100) if vote_count > 0 else 0.0
            win_rate = (
                int(session.get("winning_trades", 0) or 0) / executed_count * 100
                if executed_count > 0 else 0.0
            )
            result.append(_dec({
                "session_id": session_id,
                "provider_type": config.get("provider_type", "meta_consensus"),
                "main_agent_id": config.get("main_agent_id", ""),
                "agent_ids": config.get("agent_ids", []),
                "vote_weights": config.get("vote_weights", {}),
                "initial_capital_krw": config.get("initial_capital_krw", "0"),
                "krw_available": wallet_snapshot["krw_available"],
                "asset_available": wallet_snapshot["asset_available"],
                "avg_buy_price": wallet_snapshot["avg_buy_price"],
                "position_value": wallet_snapshot["position_value"],
                "wallet_total_value": wallet_snapshot["total_value"],
                "is_active": session.get("is_active", True),
                "current_value_krw": wallet_snapshot["total_value"],
                "peak_value_krw": session.get("peak_value_krw", "0"),
                "total_pnl_krw": wallet_snapshot["total_value"] - float(initial_capital_krw),
                "return_pct": (
                    ((wallet_snapshot["total_value"] - float(initial_capital_krw)) / float(initial_capital_krw) * 100)
                    if initial_capital_krw > 0 else 0
                ),
                "total_trades": session.get("total_trades", 0),
                "winning_trades": session.get("winning_trades", 0),
                "max_drawdown_pct": session.get("max_drawdown_pct", "0"),
                "latest_action": session.get("latest_action", "hold"),
                "latest_confidence": session.get("latest_confidence", 0.0),
                "latest_reason": session.get("latest_reason", ""),
                "latest_buy_vote": session.get("latest_buy_vote", "0"),
                "latest_sell_vote": session.get("latest_sell_vote", "0"),
                "latest_leader_agent_id": session.get("latest_leader_agent_id", ""),
                "latest_failure_code": failure.get("result_code", ""),
                "latest_failure_label": failure.get("result_label", ""),
                "latest_failure_detail": failure.get("result_detail", ""),
                "latest_failure_at": failure.get("timestamp", 0),
                "last_tick": session.get("last_tick", 0),
                "vote_count": vote_count,
                "agreement_count": agreement_count,
                "agreement_rate": agreement_rate,
                "executed_count": executed_count,
                "win_rate": win_rate,
            }))

        result.sort(
            key=lambda item: (
                SESSION_TEMPLATE_ORDER.get(str(item.get("session_id", "")), 99),
                str(item.get("session_id", "")),
            )
        )
        return result

    def orders(self, last_n: int = 30) -> List[Dict[str, Any]]:
        lines = self._store.read_lines("orders", last_n=last_n)
        current_session_ids = self._current_session_ids()
        return [
            enriched
            for enriched in (self._enrich_order(line) for line in lines)
            if self._is_current_order(enriched, current_session_ids)
        ]

    def equity_history(self, last_n: int = 240) -> Dict[str, Any]:
        started_at = self._current_system_started_at()
        lines = [
            line
            for line in self._store.read_lines("equity_curve", last_n=last_n * 4)
            if started_at <= 0 or float(line.get("_ts", 0) or 0) >= started_at
        ]
        if len(lines) > last_n:
            lines = lines[-last_n:]
        overall: List[Dict[str, Any]] = []
        sessions: Dict[str, List[Dict[str, Any]]] = {}

        for line in lines:
            point = {
                "tick": line.get("tick", 0),
                "timestamp": line.get("_ts", time.time()),
                "total_value": float(line.get("total_value", 0) or 0),
                "krw_available": float(line.get("krw_available", 0) or 0),
                "position_value": float(line.get("position_value", 0) or 0),
                "reserve_capital": float(line.get("reserve_capital", 0) or 0),
                "session_principal_total": float(line.get("session_principal_total", 0) or 0),
            }
            overall.append(point)
            for session_id, session_point in (line.get("sessions", {}) or {}).items():
                sessions.setdefault(str(session_id), []).append({
                    "tick": line.get("tick", 0),
                    "timestamp": line.get("_ts", time.time()),
                    "total_value": float(session_point.get("total_value", 0) or 0),
                    "krw_available": float(session_point.get("krw_available", 0) or 0),
                    "asset_available": float(session_point.get("asset_available", 0) or 0),
                    "avg_buy_price": float(session_point.get("avg_buy_price", 0) or 0),
                    "position_value": float(session_point.get("position_value", 0) or 0),
                })

        return {"overall": overall, "sessions": sessions}

    def meta_decisions(self, last_n: int = 30) -> List[Dict[str, Any]]:
        started_at = self._current_system_started_at()
        return [
            _dec(line)
            for line in self._store.read_lines("meta_decisions", last_n=last_n)
            if started_at <= 0 or float(line.get("_ts", 0) or 0) >= started_at
        ]

    def session_timeline(self, last_n_ticks: int = 60) -> Dict[str, Any]:
        current_session_ids = self._current_session_ids()
        if not current_session_ids:
            return {"ticks": [], "sessions": {}, "legacy_count": 0}

        lines = self._store.read_lines("session_decisions", last_n=max(last_n_ticks * max(len(current_session_ids), 1) * 4, 500))
        started_at = self._current_system_started_at()
        filtered: List[Dict[str, Any]] = []
        legacy_count = 0
        for line in lines:
            if started_at > 0 and float(line.get("_ts", 0) or 0) < started_at:
                legacy_count += 1
                continue
            session_id = str(line.get("session_id", ""))
            if session_id in current_session_ids:
                filtered.append(line)
            elif session_id:
                legacy_count += 1

        ticks = sorted({int(line.get("tick", 0) or 0) for line in filtered})
        if len(ticks) > last_n_ticks:
            ticks = ticks[-last_n_ticks:]
        allowed_ticks = set(ticks)

        sessions: Dict[str, List[Dict[str, Any]]] = {session_id: [] for session_id in current_session_ids}
        for line in filtered:
            tick = int(line.get("tick", 0) or 0)
            if tick not in allowed_ticks:
                continue
            session_id = str(line.get("session_id", ""))
            action = str(line.get("action", "hold"))
            confidence = float(line.get("confidence", 0.0) or 0.0)
            score = 0.0
            if action == "buy":
                score = confidence
            elif action == "sell":
                score = -confidence
            sessions.setdefault(session_id, []).append({
                "tick": tick,
                "timestamp": float(line.get("_ts", 0) or 0),
                "action": action,
                "confidence": confidence,
                "score": score,
                "buy_vote": float(line.get("buy_vote", 0) or 0),
                "sell_vote": float(line.get("sell_vote", 0) or 0),
                "reason": str(line.get("reason", "")),
            })

        for series in sessions.values():
            series.sort(key=lambda item: int(item.get("tick", 0)))

        return {
            "ticks": ticks,
            "sessions": sessions,
            "legacy_count": legacy_count,
        }

    def session_decisions(self, last_n: int = 90) -> Dict[str, Any]:
        current_session_ids = self._current_session_ids()
        if not current_session_ids:
            return {"items": [], "legacy_count": 0}

        lines = self._store.read_lines(
            "session_decisions",
            last_n=max(last_n * 3, 300),
        )
        started_at = self._current_system_started_at()
        items: List[Dict[str, Any]] = []
        legacy_count = 0

        for line in reversed(lines):
            if started_at > 0 and float(line.get("_ts", 0) or 0) < started_at:
                legacy_count += 1
                continue
            session_id = str(line.get("session_id", ""))
            if not session_id:
                continue
            if session_id not in current_session_ids:
                legacy_count += 1
                continue
            items.append({
                "tick": int(line.get("tick", 0) or 0),
                "timestamp": float(line.get("_ts", 0) or 0),
                "market": str(line.get("market", "")),
                "session_id": session_id,
                "main_agent_id": str(line.get("main_agent_id", "")),
                "leader_agent_id": str(line.get("leader_agent_id", "")),
                "action": str(line.get("action", "hold")),
                "confidence": float(line.get("confidence", 0.0) or 0.0),
                "buy_vote": float(line.get("buy_vote", 0) or 0),
                "sell_vote": float(line.get("sell_vote", 0) or 0),
                "reason": str(line.get("reason", "")),
            })
            if len(items) >= last_n:
                break

        return {
            "items": items,
            "legacy_count": legacy_count,
        }

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
            "max_order_krw": float(s.max_order_krw),
            "bench_threshold": s.bench_threshold,
            "min_active_agents": s.min_active_agents,
            "rebalance_interval": s.rebalance_interval,
            "softmax_temperature": s.softmax_temperature,
            "min_alloc_pct": float(s.min_alloc_pct),
            "max_alloc_pct": float(s.max_alloc_pct),
            "session_enabled": s.session_enabled,
            "session_execution_mode": s.session_execution_mode,
            "session_min_count": s.session_min_count,
            "session_max_count": s.session_max_count,
            "session_capital_krw": float(s.session_capital_krw),
        }

    def report_text(self) -> str:
        data = self._store.read_json("latest_report")
        if not data:
            return ""
        return data.get("text", "")
