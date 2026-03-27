from __future__ import annotations

from decimal import Decimal

from ..base import SubAgent
from ...models.agent import Signal
from ...exchange.market_data import MarketSnapshot
from ...utils.indicators import sma


class SMAAgent(SubAgent):
    def __init__(self, agent_id: str = "sma_agent", config: dict | None = None) -> None:
        super().__init__(agent_id, config)
        self.short_window = self.config.get("short_window", 5)
        self.long_window = self.config.get("long_window", 20)
        self.band_bps = Decimal(str(self.config.get("band_bps", 15)))

    def strategy_name(self) -> str:
        return "sma_crossover"

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        closes = snapshot.closes
        if len(closes) < self.long_window:
            return self.hold_signal("insufficient_data")

        short_ma = sma(closes, self.short_window)
        long_ma = sma(closes, self.long_window)
        band = long_ma * self.band_bps / Decimal("10000")
        upper = long_ma + band
        lower = long_ma - band

        if short_ma > upper:
            diff = float((short_ma - upper) / upper * 100)
            confidence = min(0.5 + diff * 0.1, 0.95)
            return Signal(
                agent_id=self.agent_id,
                action="buy",
                confidence=confidence,
                reason=f"SMA({self.short_window})={short_ma:.0f} > upper_band={upper:.0f}",
                metadata={"short_ma": str(short_ma), "long_ma": str(long_ma)},
            )
        elif short_ma < lower:
            diff = float((lower - short_ma) / lower * 100)
            confidence = min(0.5 + diff * 0.1, 0.95)
            return Signal(
                agent_id=self.agent_id,
                action="sell",
                confidence=confidence,
                reason=f"SMA({self.short_window})={short_ma:.0f} < lower_band={lower:.0f}",
                metadata={"short_ma": str(short_ma), "long_ma": str(long_ma)},
            )
        else:
            return self.hold_signal(f"inside_band (SMA5={short_ma:.0f}, SMA20={long_ma:.0f})")
