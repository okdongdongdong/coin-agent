from __future__ import annotations

from decimal import Decimal

from ..base import SubAgent
from ...models.agent import Signal
from ...exchange.market_data import MarketSnapshot
from ...utils.indicators import sma, atr


class BreakoutAgent(SubAgent):
    def __init__(self, agent_id: str = "breakout_agent", config: dict | None = None) -> None:
        super().__init__(agent_id, config)
        self.lookback = self.config.get("lookback", 20)
        self.volume_mult = Decimal(str(self.config.get("volume_multiplier", "1.5")))
        self.atr_period = self.config.get("atr_period", 14)

    def strategy_name(self) -> str:
        return "volume_breakout"

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        closes = snapshot.closes
        volumes = snapshot.volumes
        highs = snapshot.highs
        lows = snapshot.lows

        needed = max(self.lookback, self.atr_period + 1)
        if len(closes) < needed or len(volumes) < needed:
            return self.hold_signal("insufficient_data")

        price = snapshot.current_price

        # Volume analysis
        avg_volume = sma(volumes, self.lookback)
        current_volume = volumes[0]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else Decimal("0")
        volume_spike = volume_ratio >= self.volume_mult

        # Price breakout analysis
        recent_high = max(highs[1:self.lookback + 1])  # Exclude current candle
        recent_low = min(lows[1:self.lookback + 1])

        # ATR for confidence calibration
        current_atr = atr(highs, lows, closes, self.atr_period)
        atr_pct = float(current_atr / price * 100) if price > 0 else 0

        if volume_spike and price > recent_high:
            # Bullish breakout
            breakout_strength = float((price - recent_high) / current_atr) if current_atr > 0 else 0
            confidence = min(0.5 + breakout_strength * 0.1 + float(volume_ratio - 1) * 0.1, 0.9)
            return Signal(
                agent_id=self.agent_id,
                action="buy",
                confidence=confidence,
                reason=f"breakout above {recent_high:.0f}, vol_ratio={volume_ratio:.1f}x",
                stop_loss=price - current_atr,
                metadata={
                    "recent_high": str(recent_high),
                    "volume_ratio": str(volume_ratio),
                    "atr": str(current_atr),
                },
            )
        elif volume_spike and price < recent_low:
            # Bearish breakdown
            breakdown_strength = float((recent_low - price) / current_atr) if current_atr > 0 else 0
            confidence = min(0.5 + breakdown_strength * 0.1 + float(volume_ratio - 1) * 0.1, 0.9)
            return Signal(
                agent_id=self.agent_id,
                action="sell",
                confidence=confidence,
                reason=f"breakdown below {recent_low:.0f}, vol_ratio={volume_ratio:.1f}x",
                metadata={
                    "recent_low": str(recent_low),
                    "volume_ratio": str(volume_ratio),
                    "atr": str(current_atr),
                },
            )
        elif volume_spike:
            return self.hold_signal(f"volume_spike ({volume_ratio:.1f}x) but no price breakout")
        else:
            return self.hold_signal(f"no breakout (vol_ratio={volume_ratio:.1f}x)")
