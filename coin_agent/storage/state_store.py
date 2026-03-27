from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .jsonl_store import JsonlStore


class StateStore:
    def __init__(self, store: JsonlStore) -> None:
        self._store = store

    def get_bot_state(self) -> Dict[str, Any]:
        return self._store.read_json("bot_state", {"status": "idle", "tick_count": 0})

    def save_bot_state(self, state: Dict[str, Any]) -> None:
        self._store.write_json("bot_state", state)

    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        path = self._store.data_dir / "agent_states"
        path.mkdir(parents=True, exist_ok=True)
        sub = JsonlStore(path)
        return sub.read_json(agent_id, {"agent_id": agent_id, "active": True, "trades": 0})

    def save_agent_state(self, agent_id: str, state: Dict[str, Any]) -> None:
        path = self._store.data_dir / "agent_states"
        path.mkdir(parents=True, exist_ok=True)
        sub = JsonlStore(path)
        sub.write_json(agent_id, state)

    def get_wallet(self, wallet_id: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        path = self._store.data_dir / "wallets"
        path.mkdir(parents=True, exist_ok=True)
        sub = JsonlStore(path)
        return sub.read_json(wallet_id, default or {})

    def save_wallet(self, wallet_id: str, data: Dict[str, Any]) -> None:
        path = self._store.data_dir / "wallets"
        path.mkdir(parents=True, exist_ok=True)
        sub = JsonlStore(path)
        sub.write_json(wallet_id, data)

    def is_kill_switch_active(self) -> bool:
        return (self._store.data_dir / "KILL").exists()
