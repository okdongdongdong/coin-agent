from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class OrderIntent:
    market: str
    side: str  # "bid" (buy) or "ask" (sell)
    volume: Decimal
    price: Decimal
    ord_type: str = "limit"
    agent_id: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    mode: str  # "paper" or "live"
    success: bool
    message: str
    order_id: Optional[str]
    order_payload: Dict[str, Any] = field(default_factory=dict)
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WalletSnapshot:
    krw_available: Decimal
    asset_available: Decimal
    avg_buy_price: Decimal

    @property
    def total_value_krw(self) -> Decimal:
        return self.krw_available + (self.asset_available * self.avg_buy_price)


@dataclass(frozen=True)
class PositionSnapshot:
    market: str
    asset_balance: Decimal
    position_value_krw: Decimal
    average_price: Decimal
    unrealized_pnl_krw: Decimal
