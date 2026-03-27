from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List
import time


@dataclass(frozen=True)
class TradeRecord:
    agent_id: str
    market: str
    side: str  # "bid" or "ask"
    volume: Decimal
    price: Decimal
    pnl_krw: Decimal  # realized P&L for this trade (0 for buys, calculated for sells)
    timestamp: float
    order_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "market": self.market,
            "side": self.side,
            "volume": str(self.volume),
            "price": str(self.price),
            "pnl_krw": str(self.pnl_krw),
            "timestamp": self.timestamp,
            "order_id": self.order_id,
        }


@dataclass
class PerformanceMetrics:
    agent_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_krw: Decimal = Decimal("0")
    max_drawdown_pct: float = 0.0
    peak_value_krw: Decimal = Decimal("0")
    current_value_krw: Decimal = Decimal("0")
    trade_pnls: List[Decimal] = field(default_factory=list)
    consecutive_losses: int = 0
    last_updated: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(p for p in self.trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in self.trade_pnls if p < 0))
        if gross_loss == 0:
            return float(gross_profit) if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trade_pnls) < 2:
            return 0.0
        returns = [float(p) for p in self.trade_pnls]
        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = variance ** 0.5
        if std_r == 0:
            return 0.0
        return mean_r / std_r

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl_krw": str(self.total_pnl_krw),
            "max_drawdown_pct": self.max_drawdown_pct,
            "peak_value_krw": str(self.peak_value_krw),
            "current_value_krw": str(self.current_value_krw),
            "trade_pnls": [str(p) for p in self.trade_pnls],
            "consecutive_losses": self.consecutive_losses,
            "last_updated": self.last_updated,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        return cls(
            agent_id=data["agent_id"],
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            total_pnl_krw=Decimal(data.get("total_pnl_krw", "0")),
            max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
            peak_value_krw=Decimal(data.get("peak_value_krw", "0")),
            current_value_krw=Decimal(data.get("current_value_krw", "0")),
            trade_pnls=[Decimal(p) for p in data.get("trade_pnls", [])],
            consecutive_losses=data.get("consecutive_losses", 0),
            last_updated=data.get("last_updated", 0.0),
        )
