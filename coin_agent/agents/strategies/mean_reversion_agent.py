from __future__ import annotations

from decimal import Decimal

from ..base import SubAgent
from ...models.agent import Signal
from ...exchange.market_data import MarketSnapshot
from ...utils.indicators import bollinger_bands


class MeanReversionAgent(SubAgent):
    def __init__(self, agent_id: str = "mean_reversion_agent", config: dict | None = None) -> None:
        super().__init__(agent_id, config)
        self.bb_period = self.config.get("bb_period", 20)
        self.bb_std = Decimal(str(self.config.get("bb_std", 2)))

    def strategy_name(self) -> str:
        return "bollinger_mean_reversion"

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        closes = snapshot.closes
        if len(closes) < self.bb_period:
            return self.hold_signal("insufficient_data")

        upper, mid, lower = bollinger_bands(closes, self.bb_period, self.bb_std)
        price = snapshot.current_price
        band_width = upper - lower

        if band_width == 0:
            return self.hold_signal("zero_bandwidth")

        # Position within bands: 0 = lower, 1 = upper
        position = float((price - lower) / band_width)

        if price <= lower:
            # At or below lower band - buy signal (mean reversion up)
            overshoot = float((lower - price) / lower * 100) if lower > 0 else 0
            confidence = min(0.55 + overshoot * 0.15, 0.9)
            return Signal(
                agent_id=self.agent_id,
                action="buy",
                confidence=confidence,
                reason=f"price={price:.0f} <= BB_lower={lower:.0f}",
                target_price=mid,
                metadata={"bb_upper": str(upper), "bb_mid": str(mid), "bb_lower": str(lower), "position": position},
            )
        elif price >= upper:
            # At or above upper band - sell signal (mean reversion down)
            overshoot = float((price - upper) / upper * 100) if upper > 0 else 0
            confidence = min(0.55 + overshoot * 0.15, 0.9)
            return Signal(
                agent_id=self.agent_id,
                action="sell",
                confidence=confidence,
                reason=f"price={price:.0f} >= BB_upper={upper:.0f}",
                target_price=mid,
                metadata={"bb_upper": str(upper), "bb_mid": str(mid), "bb_lower": str(lower), "position": position},
            )
        else:
            return self.hold_signal(
                f"price={price:.0f} within bands (position={position:.2f})"
            )
