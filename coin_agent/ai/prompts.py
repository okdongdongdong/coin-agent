from __future__ import annotations

from typing import Dict, List, Any


SIGNAL_SYSTEM_PROMPT = (
    "You are a crypto market signal generator. Analyze the provided market data and "
    "return a JSON object with your trading signal. "
    "Respond with ONLY valid JSON in this exact format: "
    '{"action": "buy"|"sell"|"hold", "confidence": 0.0-1.0, "reasoning": "brief explanation", '
    '"target_price": "optional price string or null", "stop_loss": "optional price string or null"}. '
    "Keep reasoning under 100 characters. Do not include any text outside the JSON object."
)


def build_market_prompt(
    market: str,
    current_price: str,
    candles: List[Dict[str, Any]],
    indicators: Dict[str, Any],
) -> str:
    """Formats market data into a compact prompt for signal generation."""
    recent = candles[-20:] if len(candles) > 20 else candles

    candle_lines = []
    for c in recent:
        candle_lines.append(
            f"o={c.get('open','?')} h={c.get('high','?')} "
            f"l={c.get('low','?')} c={c.get('close','?')} v={c.get('volume','?')}"
        )

    ind = indicators
    indicator_str = (
        f"sma5={ind.get('sma_5','?')} sma20={ind.get('sma_20','?')} "
        f"rsi={ind.get('rsi_14','?')} "
        f"bb_upper={ind.get('bb_upper','?')} bb_mid={ind.get('bb_mid','?')} bb_lower={ind.get('bb_lower','?')} "
        f"atr={ind.get('atr_14','?')}"
    )

    candles_str = "\n".join(candle_lines)

    return (
        f"Market: {market}\n"
        f"Price: {current_price}\n"
        f"Indicators: {indicator_str}\n"
        f"Last {len(recent)} candles (OHLCV):\n{candles_str}\n"
        "Generate trading signal JSON."
    )
