from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Signal:
    agent_id: str
    action: str  # "buy", "sell", "hold"
    confidence: float  # 0.0 to 1.0
    reason: str
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    agent_id: str
    strategy_name: str
    is_active: bool = True
    phase: str = "warmup"  # "warmup", "active", "benched", "retired"
    allocated_capital_krw: Decimal = Decimal("0")
    total_trades: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "strategy_name": self.strategy_name,
            "is_active": self.is_active,
            "phase": self.phase,
            "allocated_capital_krw": str(self.allocated_capital_krw),
            "total_trades": self.total_trades,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        return cls(
            agent_id=data["agent_id"],
            strategy_name=data.get("strategy_name", "unknown"),
            is_active=data.get("is_active", True),
            phase=data.get("phase", "warmup"),
            allocated_capital_krw=Decimal(data.get("allocated_capital_krw", "0")),
            total_trades=data.get("total_trades", 0),
            config=data.get("config", {}),
        )
