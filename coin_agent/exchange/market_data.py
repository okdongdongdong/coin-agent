from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List

from ..config.settings import Settings
from .bithumb_client import BithumbClient

LOGGER = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    market: str
    current_price: Decimal
    candles: List[Dict[str, Any]]
    ticker: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    @property
    def closes(self) -> List[Decimal]:
        return [Decimal(str(c["trade_price"])) for c in self.candles]

    @property
    def volumes(self) -> List[Decimal]:
        return [Decimal(str(c["candle_acc_trade_volume"])) for c in self.candles]

    @property
    def highs(self) -> List[Decimal]:
        return [Decimal(str(c["high_price"])) for c in self.candles]

    @property
    def lows(self) -> List[Decimal]:
        return [Decimal(str(c["low_price"])) for c in self.candles]

    @property
    def opens(self) -> List[Decimal]:
        return [Decimal(str(c["opening_price"])) for c in self.candles]


class MarketDataCollector:
    def __init__(self, client: BithumbClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    def snapshot(self, market: str) -> MarketSnapshot:
        candles = self.client.get_minute_candles(
            market=market,
            unit=self.settings.candle_unit,
            count=self.settings.candle_count,
        )
        tickers = self.client.get_ticker([market])
        ticker = tickers[0] if tickers else {}
        current_price = Decimal(str(ticker.get("trade_price", 0)))
        LOGGER.info("Snapshot %s: price=%s, candles=%d", market, current_price, len(candles))
        return MarketSnapshot(
            market=market,
            current_price=current_price,
            candles=candles,
            ticker=ticker,
        )
