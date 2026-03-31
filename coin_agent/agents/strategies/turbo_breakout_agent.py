from __future__ import annotations

from decimal import Decimal

from ..base import SubAgent
from ...exchange.market_data import MarketSnapshot
from ...models.agent import Signal
from ...utils.indicators import atr, bollinger_bands, rsi, sma


class TurboBreakoutAgent(SubAgent):
    def __init__(self, agent_id: str = "turbo_breakout_agent", config: dict | None = None) -> None:
        super().__init__(agent_id, config)
        self.lookback = int(self.config.get("lookback", 20))
        self.volume_period = int(self.config.get("volume_period", 20))
        self.min_volume_ratio = Decimal(str(self.config.get("min_volume_ratio", 1.35)))
        self.min_atr_ratio = Decimal(str(self.config.get("min_atr_ratio", 1.10)))
        self.max_range_ratio = Decimal(str(self.config.get("max_range_ratio", 0.020)))

    def strategy_name(self) -> str:
        return "volatility_squeeze_breakout"

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        closes = snapshot.closes
        highs = snapshot.highs
        lows = snapshot.lows
        volumes = snapshot.volumes

        need = max(self.lookback + 1, self.volume_period, 22)
        if (
            len(closes) < need
            or len(highs) < need
            or len(lows) < need
            or len(volumes) < self.volume_period
        ):
            return self.hold_signal("insufficient_data")

        price = snapshot.current_price
        prior_high = max(highs[1:self.lookback + 1])
        prior_low = min(lows[1:self.lookback + 1])
        band_upper, band_mid, band_lower = bollinger_bands(closes, 20)
        band_width = band_upper - band_lower
        bandwidth_ratio = band_width / band_mid if band_mid > 0 else Decimal("0")
        avg_volume = sma(volumes, self.volume_period)
        vol_ratio = volumes[0] / avg_volume if avg_volume > 0 else Decimal("1")
        short_atr = atr(highs, lows, closes, 7)
        long_atr = atr(highs, lows, closes, 21)
        atr_ratio = short_atr / long_atr if long_atr > 0 else Decimal("1")
        current_rsi = rsi(closes, 14)

        squeezed = bandwidth_ratio <= self.max_range_ratio and atr_ratio >= self.min_atr_ratio
        if not squeezed:
            return self.hold_signal(
                f"no_squeeze (bw={bandwidth_ratio:.4f}, atr_ratio={atr_ratio:.2f})"
            )
        if vol_ratio < self.min_volume_ratio:
            return self.hold_signal(f"weak_volume (vol_ratio={vol_ratio:.2f})")

        if price > prior_high and current_rsi >= Decimal("55"):
            breakout_strength = (price - prior_high) / prior_high if prior_high > 0 else Decimal("0")
            confidence = min(
                0.58
                + float(min(breakout_strength * Decimal("200"), Decimal("0.16")))
                + float(min((vol_ratio - self.min_volume_ratio) * Decimal("0.12"), Decimal("0.08")))
                + float(min((atr_ratio - self.min_atr_ratio) * Decimal("0.18"), Decimal("0.08"))),
                0.90,
            )
            return Signal(
                agent_id=self.agent_id,
                action="buy",
                confidence=confidence,
                reason=(
                    f"breakout_up price={price:.0f} > {prior_high:.0f} "
                    f"(vol={vol_ratio:.2f}x, atr={atr_ratio:.2f}x)"
                ),
                metadata={
                    "prior_high": str(prior_high),
                    "vol_ratio": str(vol_ratio),
                    "atr_ratio": str(atr_ratio),
                    "rsi": str(current_rsi),
                },
            )

        if price < prior_low and current_rsi <= Decimal("45"):
            breakout_strength = (prior_low - price) / prior_low if prior_low > 0 else Decimal("0")
            confidence = min(
                0.58
                + float(min(breakout_strength * Decimal("200"), Decimal("0.16")))
                + float(min((vol_ratio - self.min_volume_ratio) * Decimal("0.12"), Decimal("0.08")))
                + float(min((atr_ratio - self.min_atr_ratio) * Decimal("0.18"), Decimal("0.08"))),
                0.90,
            )
            return Signal(
                agent_id=self.agent_id,
                action="sell",
                confidence=confidence,
                reason=(
                    f"breakout_down price={price:.0f} < {prior_low:.0f} "
                    f"(vol={vol_ratio:.2f}x, atr={atr_ratio:.2f}x)"
                ),
                metadata={
                    "prior_low": str(prior_low),
                    "vol_ratio": str(vol_ratio),
                    "atr_ratio": str(atr_ratio),
                    "rsi": str(current_rsi),
                },
            )

        return self.hold_signal(
            f"armed_no_breakout (vol={vol_ratio:.2f}x, atr={atr_ratio:.2f}x, rsi={current_rsi:.1f})"
        )
