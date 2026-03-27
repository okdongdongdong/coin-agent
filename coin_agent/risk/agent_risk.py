from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ..config.settings import Settings
from ..models.trading import OrderIntent

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRiskResult:
    allowed: bool
    reason: str
    adjusted_volume: Optional[Decimal] = None


class AgentRiskManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_order_time: dict[str, float] = {}

    def check(
        self,
        intent: OrderIntent,
        agent_capital: Decimal,
        agent_krw: Decimal,
        agent_asset_value: Decimal,
    ) -> AgentRiskResult:
        # 1. Cooldown check
        agent_id = intent.agent_id
        now = time.time()
        last = self._last_order_time.get(agent_id, 0)
        if now - last < self.settings.order_cooldown_sec:
            remaining = int(self.settings.order_cooldown_sec - (now - last))
            return AgentRiskResult(False, f"cooldown ({remaining}s remaining)")

        # 2. Position size limit
        order_value = intent.volume * intent.price
        if intent.side == "bid":
            new_position_value = agent_asset_value + order_value
            max_position = agent_capital * self.settings.max_position_pct / Decimal("100")
            if new_position_value > max_position:
                return AgentRiskResult(False, f"position_limit (new={new_position_value:.0f} > max={max_position:.0f})")

        # 3. Minimum order size (Bithumb minimum: 5,000 KRW)
        if order_value < Decimal("5000"):
            return AgentRiskResult(False, f"order_too_small ({order_value:.0f} < 5000 KRW)")

        # 4. Insufficient funds check
        if intent.side == "bid":
            total_cost = order_value * (Decimal("1") + self.settings.fee_rate)
            if total_cost > agent_krw:
                return AgentRiskResult(False, f"insufficient_krw ({total_cost:.0f} > {agent_krw:.0f})")

        # Mark order time
        self._last_order_time[agent_id] = now

        return AgentRiskResult(True, "passed")

    def reset_cooldown(self, agent_id: str) -> None:
        self._last_order_time.pop(agent_id, None)
