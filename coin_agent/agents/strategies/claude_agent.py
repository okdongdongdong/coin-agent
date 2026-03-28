from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..base import SubAgent
from ...exchange.market_data import MarketSnapshot
from ...models.agent import Signal
from ...utils.indicators import atr, bollinger_bands, rsi, sma

LOGGER = logging.getLogger(__name__)

_ACTION_MAP = {"buy": "buy", "sell": "sell", "hold": "hold"}


def _compute_indicators(snapshot: MarketSnapshot) -> Dict[str, Any]:
    closes = snapshot.closes
    highs = snapshot.highs
    lows = snapshot.lows
    indicators: Dict[str, Any] = {}

    try:
        indicators["sma_5"] = str(sma(closes, 5))
    except Exception:
        indicators["sma_5"] = "?"

    try:
        indicators["sma_20"] = str(sma(closes, 20))
    except Exception:
        indicators["sma_20"] = "?"

    try:
        indicators["rsi_14"] = str(rsi(closes, 14))
    except Exception:
        indicators["rsi_14"] = "?"

    try:
        upper, mid, lower = bollinger_bands(closes, 20)
        indicators["bb_upper"] = str(upper)
        indicators["bb_mid"] = str(mid)
        indicators["bb_lower"] = str(lower)
    except Exception:
        indicators["bb_upper"] = "?"
        indicators["bb_mid"] = "?"
        indicators["bb_lower"] = "?"

    try:
        indicators["atr_14"] = str(atr(highs, lows, closes, 14))
    except Exception:
        indicators["atr_14"] = "?"

    return indicators


def _candles_for_prompt(snapshot: MarketSnapshot) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for c in snapshot.candles:
        result.append({
            "open": str(c.get("opening_price", "?")),
            "high": str(c.get("high_price", "?")),
            "low": str(c.get("low_price", "?")),
            "close": str(c.get("trade_price", "?")),
            "volume": str(c.get("candle_acc_trade_volume", "?")),
        })
    return result


def _parse_signal(agent_id: str, result: Dict[str, Any], metadata: Dict[str, Any]) -> Signal:
    action = _ACTION_MAP.get(str(result.get("action", "hold")).lower(), "hold")
    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(result.get("reasoning", ""))
    raw_target = result.get("target_price")
    raw_stop = result.get("stop_loss")
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    try:
        if raw_target and raw_target != "null":
            target_price = Decimal(str(raw_target))
    except Exception:
        pass
    try:
        if raw_stop and raw_stop != "null":
            stop_loss = Decimal(str(raw_stop))
    except Exception:
        pass
    return Signal(
        agent_id=agent_id,
        action=action,
        confidence=confidence,
        reason=reasoning,
        target_price=target_price,
        stop_loss=stop_loss,
        metadata=metadata,
    )


class ClaudeAgent(SubAgent):
    """AI agent powered by Claude (Anthropic API)."""

    def __init__(self, agent_id: str = "claude_agent", model: str = "claude-haiku-4-5-20250610",
                 config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(agent_id, config)
        self._model = model
        self._provider: Optional[Any] = None  # lazy init

    def strategy_name(self) -> str:
        return "claude_ai"

    def _get_provider(self) -> Any:
        if self._provider is None:
            from ...ai.provider import ClaudeProvider
            self._provider = ClaudeProvider(model=self._model)
        return self._provider

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        from ...ai.prompts import SIGNAL_SYSTEM_PROMPT, build_market_prompt

        try:
            indicators = _compute_indicators(snapshot)
            candles = _candles_for_prompt(snapshot)
            prompt = build_market_prompt(
                market=snapshot.market,
                current_price=str(snapshot.current_price),
                candles=candles,
                indicators=indicators,
            )
            provider = self._get_provider()
            result = provider.generate_signal(prompt, SIGNAL_SYSTEM_PROMPT)
            if result is None:
                return self.hold_signal("claude_no_response")
            metadata: Dict[str, Any] = {
                "provider": "claude",
                "model": self._model,
                **{k: indicators[k] for k in indicators},
            }
            return _parse_signal(self.agent_id, result, metadata)
        except Exception as exc:
            LOGGER.error("ClaudeAgent.analyze error: %s", exc)
            return self.hold_signal(f"error: {exc}")
