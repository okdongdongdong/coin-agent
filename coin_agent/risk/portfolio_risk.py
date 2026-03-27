from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List

from ..config.settings import Settings
from ..models.trading import OrderIntent
from ..storage.jsonl_store import JsonlStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortfolioRiskResult:
    allowed: bool
    reason: str


class PortfolioRiskManager:
    def __init__(self, settings: Settings, store: JsonlStore) -> None:
        self.settings = settings
        self.store = store

    def check(
        self,
        intent: OrderIntent,
        total_krw: Decimal,
        total_asset_value: Decimal,
    ) -> PortfolioRiskResult:
        total_value = total_krw + total_asset_value

        # 1. Check daily P&L
        daily_pnl = self._calc_daily_pnl()
        if daily_pnl < -self.settings.max_daily_loss_krw:
            return PortfolioRiskResult(
                False,
                f"daily_loss_limit (P&L={daily_pnl:+,.0f} < -{self.settings.max_daily_loss_krw:,.0f})"
            )

        # 2. Check total P&L
        total_pnl = total_value - self.settings.paper_krw_balance
        if total_pnl < -self.settings.max_total_loss_krw:
            return PortfolioRiskResult(
                False,
                f"total_loss_limit (P&L={total_pnl:+,.0f} < -{self.settings.max_total_loss_krw:,.0f})"
            )

        # 3. Max portfolio exposure (prevent going all-in)
        if intent.side == "bid":
            order_value = intent.volume * intent.price
            new_exposure = total_asset_value + order_value
            max_exposure = total_value * Decimal("0.8")  # Max 80% in positions
            if new_exposure > max_exposure:
                return PortfolioRiskResult(
                    False,
                    f"max_exposure ({new_exposure:,.0f} > {max_exposure:,.0f})"
                )

        return PortfolioRiskResult(True, "passed")

    def get_daily_pnl(self) -> Decimal:
        return self._calc_daily_pnl()

    def get_total_pnl(self, current_value: Decimal) -> Decimal:
        return current_value - self.settings.paper_krw_balance

    def _calc_daily_pnl(self) -> Decimal:
        trades = self.store.read_lines("trades")
        if not trades:
            return Decimal("0")

        today_start = _today_start_ts()
        daily_pnl = Decimal("0")
        for t in trades:
            ts = t.get("timestamp", t.get("_ts", 0))
            if ts >= today_start:
                pnl = Decimal(t.get("pnl_krw", "0"))
                daily_pnl += pnl

        return daily_pnl


def _today_start_ts() -> float:
    t = time.localtime()
    return time.mktime(time.strptime(time.strftime("%Y-%m-%d", t), "%Y-%m-%d"))
