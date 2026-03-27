from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict

from ..models.trading import PositionSnapshot, WalletSnapshot
from ..storage.state_store import StateStore

LOGGER = logging.getLogger(__name__)


class PositionTracker:
    def __init__(self, state: StateStore) -> None:
        self.state = state

    def get_position(self, agent_id: str, market: str, current_price: Decimal) -> PositionSnapshot:
        wallet_data = self.state.get_wallet(f"paper_{agent_id}", {
            "krw_available": "0",
            "asset_available": "0",
            "avg_buy_price": "0",
        })
        asset_balance = Decimal(wallet_data.get("asset_available", "0"))
        avg_price = Decimal(wallet_data.get("avg_buy_price", "0"))
        position_value = asset_balance * current_price
        unrealized_pnl = (current_price - avg_price) * asset_balance if asset_balance > 0 else Decimal("0")

        return PositionSnapshot(
            market=market,
            asset_balance=asset_balance,
            position_value_krw=position_value,
            average_price=avg_price,
            unrealized_pnl_krw=unrealized_pnl,
        )

    def get_agent_wallet(self, agent_id: str, default_krw: Decimal = Decimal("0")) -> WalletSnapshot:
        data = self.state.get_wallet(f"paper_{agent_id}", {
            "krw_available": str(default_krw),
            "asset_available": "0",
            "avg_buy_price": "0",
        })
        return WalletSnapshot(
            krw_available=Decimal(data["krw_available"]),
            asset_available=Decimal(data["asset_available"]),
            avg_buy_price=Decimal(data["avg_buy_price"]),
        )

    def get_total_value(self, agent_id: str, current_price: Decimal, default_krw: Decimal = Decimal("0")) -> Decimal:
        wallet = self.get_agent_wallet(agent_id, default_krw)
        return wallet.krw_available + wallet.asset_available * current_price
