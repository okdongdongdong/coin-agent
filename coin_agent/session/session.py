from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional


@dataclass
class SessionConfig:
    """Configuration for a trading session."""

    session_id: str                  # unique ID like "session_001"
    provider_type: str               # "claude", "codex", "hybrid", "technical"
    agent_ids: List[str]             # which agents are in this session
    initial_capital_krw: Decimal     # starting capital
    created_at: str                  # ISO timestamp
    hyperparams: Dict[str, Any] = field(default_factory=dict)  # softmax temp, thresholds, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "provider_type": self.provider_type,
            "agent_ids": self.agent_ids,
            "initial_capital_krw": str(self.initial_capital_krw),
            "created_at": self.created_at,
            "hyperparams": self.hyperparams,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionConfig":
        return cls(
            session_id=d["session_id"],
            provider_type=d["provider_type"],
            agent_ids=d.get("agent_ids", []),
            initial_capital_krw=Decimal(d.get("initial_capital_krw", "0")),
            created_at=d.get("created_at", ""),
            hyperparams=d.get("hyperparams", {}),
        )


@dataclass
class SessionState:
    """Runtime state of a session."""

    config: SessionConfig
    is_active: bool = True
    current_value_krw: Decimal = Decimal("0")
    peak_value_krw: Decimal = Decimal("0")
    total_pnl_krw: Decimal = Decimal("0")
    return_pct: Decimal = Decimal("0")       # (current - initial) / initial * 100
    total_trades: int = 0
    winning_trades: int = 0
    max_drawdown_pct: Decimal = Decimal("0")
    generation: int = 1                      # increments when session is replaced
    eliminated_at: Optional[str] = None      # ISO timestamp if eliminated
    elimination_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "is_active": self.is_active,
            "current_value_krw": str(self.current_value_krw),
            "peak_value_krw": str(self.peak_value_krw),
            "total_pnl_krw": str(self.total_pnl_krw),
            "return_pct": str(self.return_pct),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "generation": self.generation,
            "eliminated_at": self.eliminated_at,
            "elimination_reason": self.elimination_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionState":
        config = SessionConfig.from_dict(d["config"])
        return cls(
            config=config,
            is_active=d.get("is_active", True),
            current_value_krw=Decimal(d.get("current_value_krw", "0")),
            peak_value_krw=Decimal(d.get("peak_value_krw", "0")),
            total_pnl_krw=Decimal(d.get("total_pnl_krw", "0")),
            return_pct=Decimal(d.get("return_pct", "0")),
            total_trades=d.get("total_trades", 0),
            winning_trades=d.get("winning_trades", 0),
            max_drawdown_pct=Decimal(d.get("max_drawdown_pct", "0")),
            generation=d.get("generation", 1),
            eliminated_at=d.get("eliminated_at"),
            elimination_reason=d.get("elimination_reason"),
        )
