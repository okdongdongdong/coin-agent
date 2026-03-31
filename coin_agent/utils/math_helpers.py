from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


def round_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def krw_tick_size(value: Decimal) -> Decimal:
    if value < Decimal("1"):
        return Decimal("0.0001")
    if value < Decimal("10"):
        return Decimal("0.001")
    if value < Decimal("100"):
        return Decimal("0.01")
    if value < Decimal("5000"):
        return Decimal("1")
    if value < Decimal("10000"):
        return Decimal("5")
    if value < Decimal("50000"):
        return Decimal("10")
    if value < Decimal("100000"):
        return Decimal("50")
    if value < Decimal("500000"):
        return Decimal("100")
    if value < Decimal("1000000"):
        return Decimal("500")
    return Decimal("1000")


def round_price(value: Decimal, tick_size: Decimal | None = None) -> Decimal:
    if tick_size is None:
        tick_size = krw_tick_size(value)
    return (value / tick_size).to_integral_value(rounding=ROUND_HALF_UP) * tick_size


def pct_change(old: Decimal, new: Decimal) -> Decimal:
    if old == 0:
        return Decimal("0")
    return ((new - old) / old) * Decimal("100")


def decimal_str(value: Decimal) -> str:
    return format(value.normalize(), "f")
