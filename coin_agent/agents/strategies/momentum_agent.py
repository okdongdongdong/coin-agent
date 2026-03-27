from __future__ import annotations

from decimal import Decimal

from ..base import SubAgent
from ...models.agent import Signal
from ...exchange.market_data import MarketSnapshot
from ...utils.indicators import rsi, sma


class MomentumAgent(SubAgent):
    def __init__(self, agent_id: str = "momentum_agent", config: dict | None = None) -> None:
        super().__init__(agent_id, config)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.oversold = Decimal(str(self.config.get("oversold", 30)))
        self.overbought = Decimal(str(self.config.get("overbought", 70)))

    def strategy_name(self) -> str:
        return "rsi_momentum"

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        closes = snapshot.closes
        if len(closes) < self.rsi_period + 1:
            return self.hold_signal("insufficient_data")

        current_rsi = rsi(closes, self.rsi_period)

        # Also check price momentum via SMA
        if len(closes) >= 10:
            sma5 = sma(closes, 5)
            sma10 = sma(closes, 10)
            trend_up = sma5 > sma10
        else:
            trend_up = None

        if current_rsi < self.oversold:
            # Oversold - potential buy
            strength = float((self.oversold - current_rsi) / self.oversold)
            confidence = min(0.5 + strength * 0.5, 0.9)
            return Signal(
                agent_id=self.agent_id,
                action="buy",
                confidence=confidence,
                reason=f"RSI={current_rsi:.1f} < {self.oversold} (oversold)",
                metadata={"rsi": str(current_rsi), "trend_up": trend_up},
            )
        elif current_rsi > self.overbought:
            # Overbought - potential sell
            strength = float((current_rsi - self.overbought) / (Decimal("100") - self.overbought))
            confidence = min(0.5 + strength * 0.5, 0.9)
            return Signal(
                agent_id=self.agent_id,
                action="sell",
                confidence=confidence,
                reason=f"RSI={current_rsi:.1f} > {self.overbought} (overbought)",
                metadata={"rsi": str(current_rsi), "trend_up": trend_up},
            )
        elif current_rsi > Decimal("40") and current_rsi < Decimal("60") and trend_up:
            # Mid RSI with uptrend - mild buy signal
            return Signal(
                agent_id=self.agent_id,
                action="buy",
                confidence=0.4,
                reason=f"RSI={current_rsi:.1f} mid-range + uptrend",
                metadata={"rsi": str(current_rsi), "trend_up": trend_up},
            )
        elif current_rsi > Decimal("40") and current_rsi < Decimal("60") and trend_up is False:
            return Signal(
                agent_id=self.agent_id,
                action="sell",
                confidence=0.35,
                reason=f"RSI={current_rsi:.1f} mid-range + downtrend",
                metadata={"rsi": str(current_rsi), "trend_up": trend_up},
            )
        else:
            return self.hold_signal(f"RSI={current_rsi:.1f} neutral")
