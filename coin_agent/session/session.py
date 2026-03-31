from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional


@dataclass
class SessionConfig:
    """Configuration for a trading session."""

    session_id: str                  # unique ID like "session_sma_main"
    provider_type: str               # retained for backward compatibility
    agent_ids: List[str]             # all technical agents participating
    initial_capital_krw: Decimal     # starting capital
    created_at: str                  # ISO timestamp
    main_agent_id: str = ""          # the 40% voting agent in this session
    vote_weights: Dict[str, Any] = field(default_factory=dict)
    hyperparams: Dict[str, Any] = field(default_factory=dict)  # thresholds, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "provider_type": self.provider_type,
            "agent_ids": self.agent_ids,
            "initial_capital_krw": str(self.initial_capital_krw),
            "created_at": self.created_at,
            "main_agent_id": self.main_agent_id,
            "vote_weights": self.vote_weights,
            "hyperparams": self.hyperparams,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionConfig":
        return cls(
            session_id=d["session_id"],
            provider_type=d.get("provider_type", "technical_consensus"),
            agent_ids=d.get("agent_ids", []),
            initial_capital_krw=Decimal(d.get("initial_capital_krw", "0")),
            created_at=d.get("created_at", ""),
            main_agent_id=d.get("main_agent_id", ""),
            vote_weights=d.get("vote_weights", {}),
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
    latest_action: str = "hold"
    latest_confidence: float = 0.0
    latest_reason: str = ""
    latest_buy_vote: Decimal = Decimal("0")
    latest_sell_vote: Decimal = Decimal("0")
    latest_leader_agent_id: str = ""
    last_tick: int = 0
    vote_count: int = 0
    agreement_count: int = 0
    executed_count: int = 0

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
            "latest_action": self.latest_action,
            "latest_confidence": self.latest_confidence,
            "latest_reason": self.latest_reason,
            "latest_buy_vote": str(self.latest_buy_vote),
            "latest_sell_vote": str(self.latest_sell_vote),
            "latest_leader_agent_id": self.latest_leader_agent_id,
            "last_tick": self.last_tick,
            "vote_count": self.vote_count,
            "agreement_count": self.agreement_count,
            "executed_count": self.executed_count,
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
            latest_action=d.get("latest_action", "hold"),
            latest_confidence=float(d.get("latest_confidence", 0.0)),
            latest_reason=d.get("latest_reason", ""),
            latest_buy_vote=Decimal(d.get("latest_buy_vote", "0")),
            latest_sell_vote=Decimal(d.get("latest_sell_vote", "0")),
            latest_leader_agent_id=d.get("latest_leader_agent_id", ""),
            last_tick=d.get("last_tick", 0),
            vote_count=d.get("vote_count", 0),
            agreement_count=d.get("agreement_count", 0),
            executed_count=d.get("executed_count", 0),
        )
