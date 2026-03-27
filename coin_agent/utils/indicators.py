from __future__ import annotations

from decimal import Decimal
from typing import List, Tuple


def sma(values: List[Decimal], period: int) -> Decimal:
    if len(values) < period:
        raise ValueError(f"Need at least {period} values, got {len(values)}")
    window = values[:period]
    return sum(window, Decimal("0")) / Decimal(period)


def ema(values: List[Decimal], period: int) -> Decimal:
    if len(values) < period:
        raise ValueError(f"Need at least {period} values, got {len(values)}")
    multiplier = Decimal("2") / (Decimal(period) + Decimal("1"))
    result = sma(list(reversed(values[-period:])), period)
    for val in reversed(values[:-period]):
        result = (val - result) * multiplier + result
    # Process from oldest to newest
    ordered = list(reversed(values))
    result = sma(ordered[:period], period)
    for i in range(period, len(ordered)):
        result = (ordered[i] - result) * multiplier + result
    return result


def rsi(closes: List[Decimal], period: int = 14) -> Decimal:
    if len(closes) < period + 1:
        raise ValueError(f"Need at least {period + 1} closes, got {len(closes)}")
    # closes[0] is most recent
    ordered = list(reversed(closes[:period + 1]))  # oldest first
    gains: List[Decimal] = []
    losses: List[Decimal] = []
    for i in range(1, len(ordered)):
        change = ordered[i] - ordered[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(abs(change))
    avg_gain = sum(gains, Decimal("0")) / Decimal(period)
    avg_loss = sum(losses, Decimal("0")) / Decimal(period)
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


def bollinger_bands(
    closes: List[Decimal], period: int = 20, num_std: Decimal = Decimal("2")
) -> Tuple[Decimal, Decimal, Decimal]:
    if len(closes) < period:
        raise ValueError(f"Need at least {period} closes, got {len(closes)}")
    mid = sma(closes, period)
    window = closes[:period]
    mean = mid
    variance = sum((x - mean) ** 2 for x in window) / Decimal(period)
    # Use Newton's method for sqrt since Decimal doesn't have sqrt
    std = _decimal_sqrt(variance)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def atr(
    highs: List[Decimal], lows: List[Decimal], closes: List[Decimal], period: int = 14
) -> Decimal:
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        raise ValueError(f"Need at least {period + 1} values for ATR")
    true_ranges: List[Decimal] = []
    for i in range(period):
        h = highs[i]
        l = lows[i]
        prev_c = closes[i + 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        true_ranges.append(tr)
    return sum(true_ranges, Decimal("0")) / Decimal(period)


def _decimal_sqrt(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal("0")
    x = value
    for _ in range(50):
        x = (x + value / x) / Decimal("2")
    return x
