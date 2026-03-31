from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .jsonl_store import JsonlStore


class StateStore:
    def __init__(self, store: JsonlStore) -> None:
        self._store = store

    def _substore(self, name: str) -> JsonlStore:
        path = self._store.data_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return JsonlStore(path)

    def get_bot_state(self) -> Dict[str, Any]:
        return self._store.read_json("bot_state", {"status": "idle", "tick_count": 0})

    def save_bot_state(self, state: Dict[str, Any]) -> None:
        self._store.write_json("bot_state", state)

    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        sub = self._substore("agent_states")
        return sub.read_json(agent_id, {"agent_id": agent_id, "active": True, "trades": 0})

    def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        sub = self._substore("agent_states")
        sub.write_json(agent_id, state)

    def get_wallet(self, wallet_id: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        sub = self._substore("wallets")
        return sub.read_json(wallet_id, default or {})

    def save_wallet(self, wallet_id: str, data: Dict[str, Any]) -> None:
        sub = self._substore("wallets")
        sub.write_json(wallet_id, data)

    def get_pending_orders(self) -> Dict[str, Any]:
        sub = self._substore("pending")
        return sub.read_json("orders", {"orders": {}}).get("orders", {})

    def save_pending_orders(self, orders: Dict[str, Any]) -> None:
        sub = self._substore("pending")
        sub.write_json("orders", {"orders": orders})

    def is_kill_switch_active(self) -> bool:
        return (self._store.data_dir / "KILL").exists()
