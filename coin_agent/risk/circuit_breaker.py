from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal

from ..config.settings import Settings
from ..models.performance import PerformanceMetrics
from ..storage.jsonl_store import JsonlStore
from ..storage.state_store import StateStore

LOGGER = logging.getLogger(__name__)

MAX_CONSECUTIVE_LOSSES = 5


@dataclass(frozen=True)
class CircuitBreakerResult:
    tripped: bool
    reason: str


class CircuitBreaker:
    def __init__(self, settings: Settings, store: JsonlStore, state: StateStore) -> None:
        self.settings = settings
        self.store = store
        self.state = state

    def check(
        self,
        current_value: Decimal,
        daily_pnl: Decimal,
        max_consecutive_losses: int = 0,
    ) -> CircuitBreakerResult:
        # 1. Kill switch file
        if self.state.is_kill_switch_active():
            return CircuitBreakerResult(True, "kill_switch_active")

        # 2. Daily loss limit
        if daily_pnl < -self.settings.max_daily_loss_krw:
            self._record_trip(f"daily_loss: {daily_pnl:+,.0f}")
            return CircuitBreakerResult(
                True, f"daily_loss_limit ({daily_pnl:+,.0f} KRW)"
            )

        # 3. Total loss limit
        total_pnl = current_value - self.settings.paper_krw_balance
        if total_pnl < -self.settings.max_total_loss_krw:
            self._record_trip(f"total_loss: {total_pnl:+,.0f}")
            return CircuitBreakerResult(
                True, f"total_loss_limit ({total_pnl:+,.0f} KRW)"
            )

        # 4. Consecutive losses
        if max_consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self._record_trip(f"consecutive_losses: {max_consecutive_losses}")
            return CircuitBreakerResult(
                True, f"consecutive_losses ({max_consecutive_losses} >= {MAX_CONSECUTIVE_LOSSES})"
            )

        return CircuitBreakerResult(False, "ok")

    def _record_trip(self, reason: str) -> None:
        LOGGER.warning("CIRCUIT BREAKER TRIPPED: %s", reason)
        self.store.write_json("circuit_breaker", {
            "tripped": True,
            "reason": reason,
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Update bot state
        bot = self.state.get_bot_state()
        bot["status"] = "circuit_breaker_tripped"
        bot["circuit_breaker_reason"] = reason
        self.state.save_bot_state(bot)

    def is_tripped(self) -> bool:
        data = self.store.read_json("circuit_breaker")
        return data.get("tripped", False)

    def reset(self) -> None:
        self.store.write_json("circuit_breaker", {"tripped": False})
        LOGGER.info("Circuit breaker reset.")
