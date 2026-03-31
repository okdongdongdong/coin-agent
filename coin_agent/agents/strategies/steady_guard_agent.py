from __future__ import annotations

from decimal import Decimal

from ..base import SubAgent
from ...exchange.market_data import MarketSnapshot
from ...models.agent import Signal
from ...utils.indicators import atr, bollinger_bands, ema, rsi


class SteadyGuardAgent(SubAgent):
    def __init__(self, agent_id: str = "steady_guard_agent", config: dict | None = None) -> None:
        super().__init__(agent_id, config)
        self.fast_ema = int(self.config.get("fast_ema", 21))
        self.slow_ema = int(self.config.get("slow_ema", 55))
        self.max_atr_ratio = Decimal(str(self.config.get("max_atr_ratio", 1.05)))

    def strategy_name(self) -> str:
        return "trend_filtered_pullback"

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        closes = snapshot.closes
        highs = snapshot.highs
        lows = snapshot.lows

        need = max(self.slow_ema + 1, 22)
        if len(closes) < need or len(highs) < 22 or len(lows) < 22:
            return self.hold_signal("insufficient_data")

        price = snapshot.current_price
        fast = ema(closes, self.fast_ema)
        slow = ema(closes, self.slow_ema)
        band_upper, band_mid, band_lower = bollinger_bands(closes, 20)
        band_width = band_upper - band_lower
        if band_width <= 0:
            return self.hold_signal("zero_bandwidth")

        band_position = (price - band_lower) / band_width
        atr_ratio = atr(highs, lows, closes, 7) / atr(highs, lows, closes, 21)
        current_rsi = rsi(closes, 14)

        if atr_ratio > self.max_atr_ratio:
            return self.hold_signal(f"high_volatility (atr_ratio={atr_ratio:.2f})")

        if fast > slow and band_position <= Decimal("0.35") and Decimal("40") <= current_rsi <= Decimal("58"):
            pullback_score = min((Decimal("0.35") - band_position) * Decimal("1.6"), Decimal("0.18"))
            confidence = min(0.45 + float(max(Decimal("0"), pullback_score)) + 0.08, 0.78)
            return Signal(
                agent_id=self.agent_id,
                action="buy",
                confidence=confidence,
                reason=(
                    f"trend_pullback_up ema21={fast:.0f} > ema55={slow:.0f} "
                    f"(band={band_position:.2f}, atr={atr_ratio:.2f})"
                ),
                metadata={
                    "ema_fast": str(fast),
                    "ema_slow": str(slow),
                    "band_position": str(band_position),
                    "atr_ratio": str(atr_ratio),
                    "rsi": str(current_rsi),
                },
            )

        if fast < slow and band_position >= Decimal("0.65") and Decimal("42") <= current_rsi <= Decimal("60"):
            rebound_score = min((band_position - Decimal("0.65")) * Decimal("1.6"), Decimal("0.18"))
            confidence = min(0.45 + float(max(Decimal("0"), rebound_score)) + 0.08, 0.78)
            return Signal(
                agent_id=self.agent_id,
                action="sell",
                confidence=confidence,
                reason=(
                    f"trend_pullback_down ema21={fast:.0f} < ema55={slow:.0f} "
                    f"(band={band_position:.2f}, atr={atr_ratio:.2f})"
                ),
                metadata={
                    "ema_fast": str(fast),
                    "ema_slow": str(slow),
                    "band_position": str(band_position),
                    "atr_ratio": str(atr_ratio),
                    "rsi": str(current_rsi),
                },
            )

        return self.hold_signal(
            f"filtered_out (band={band_position:.2f}, atr={atr_ratio:.2f}, rsi={current_rsi:.1f})"
        )
