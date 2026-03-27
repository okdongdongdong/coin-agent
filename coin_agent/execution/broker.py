from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Dict

from ..config.settings import Settings
from ..exchange.bithumb_client import BithumbClient
from ..models.trading import ExecutionResult, OrderIntent, WalletSnapshot
from ..storage.state_store import StateStore

LOGGER = logging.getLogger(__name__)


class PaperBroker:
    def __init__(self, settings: Settings, state: StateStore) -> None:
        self.settings = settings
        self.state = state

    def get_wallet(self, agent_id: str = "global") -> WalletSnapshot:
        default = {
            "krw_available": str(self.settings.paper_krw_balance),
            "asset_available": "0",
            "avg_buy_price": "0",
        }
        data = self.state.get_wallet(f"paper_{agent_id}", default)
        return WalletSnapshot(
            krw_available=Decimal(data["krw_available"]),
            asset_available=Decimal(data["asset_available"]),
            avg_buy_price=Decimal(data["avg_buy_price"]),
        )

    def execute(self, intent: OrderIntent, agent_id: str = "global") -> ExecutionResult:
        wallet = self.get_wallet(agent_id)
        fee_rate = self.settings.fee_rate
        notional = intent.price * intent.volume

        if intent.side == "bid":
            fee_mult = Decimal("1") + fee_rate
            total_spend = notional * fee_mult
            if total_spend > wallet.krw_available:
                return ExecutionResult("paper", False, "insufficient_krw", None)
            new_asset = wallet.asset_available + intent.volume
            avg_price = wallet.avg_buy_price
            if new_asset > 0:
                avg_price = (wallet.asset_available * wallet.avg_buy_price + notional) / new_asset
            state = {
                "krw_available": str(wallet.krw_available - total_spend),
                "asset_available": str(new_asset),
                "avg_buy_price": str(avg_price),
            }
        elif intent.side == "ask":
            if intent.volume > wallet.asset_available:
                return ExecutionResult("paper", False, "insufficient_asset", None)
            fee_mult = Decimal("1") - fee_rate
            total_receive = notional * fee_mult
            remaining = wallet.asset_available - intent.volume
            state = {
                "krw_available": str(wallet.krw_available + total_receive),
                "asset_available": str(remaining),
                "avg_buy_price": str(Decimal("0") if remaining <= 0 else wallet.avg_buy_price),
            }
        else:
            return ExecutionResult("paper", False, f"invalid_side:{intent.side}", None)

        self.state.save_wallet(f"paper_{agent_id}", state)
        order_id = f"paper-{int(time.time() * 1000)}"
        payload = {
            "market": intent.market,
            "side": intent.side,
            "volume": str(intent.volume),
            "price": str(intent.price),
            "agent_id": agent_id,
            "reason": intent.reason,
        }
        LOGGER.info("Paper %s: %s %s @ %s (agent=%s)", intent.side, intent.volume, intent.market, intent.price, agent_id)
        return ExecutionResult("paper", True, "filled", order_id, payload, state)


class LiveBroker:
    def __init__(self, client: BithumbClient) -> None:
        self.client = client

    def get_wallet(self, asset_currency: str) -> WalletSnapshot:
        accounts = self.client.get_accounts()
        acct_map: Dict[str, dict] = {item["currency"]: item for item in accounts}
        krw = acct_map.get("KRW", {})
        asset = acct_map.get(asset_currency, {})
        return WalletSnapshot(
            krw_available=Decimal(krw.get("balance", "0")),
            asset_available=Decimal(asset.get("balance", "0")),
            avg_buy_price=Decimal(asset.get("avg_buy_price", "0")),
        )

    def execute(self, intent: OrderIntent) -> ExecutionResult:
        resp = self.client.place_limit_order(
            market=intent.market,
            side=intent.side,
            volume=intent.volume,
            price=intent.price,
        )
        order_id = resp.get("uuid")
        payload = {
            "market": intent.market,
            "side": intent.side,
            "volume": str(intent.volume),
            "price": str(intent.price),
            "agent_id": intent.agent_id,
            "reason": intent.reason,
        }
        return ExecutionResult(
            mode="live",
            success=bool(order_id),
            message="submitted" if order_id else "unknown_response",
            order_id=order_id,
            order_payload=payload,
            raw_response=resp,
        )
