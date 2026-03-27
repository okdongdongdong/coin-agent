from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict

from ..exchange.market_data import MarketSnapshot
from ..models.agent import Signal


class SubAgent(ABC):
    def __init__(self, agent_id: str, config: Dict[str, Any] | None = None) -> None:
        self.agent_id = agent_id
        self.config = config or {}

    @abstractmethod
    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        ...

    @abstractmethod
    def strategy_name(self) -> str:
        ...

    def hold_signal(self, reason: str = "no_signal") -> Signal:
        return Signal(
            agent_id=self.agent_id,
            action="hold",
            confidence=0.0,
            reason=reason,
        )
