from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


def round_down(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def round_price(value: Decimal, tick_size: Decimal = Decimal("1")) -> Decimal:
    return (value / tick_size).to_integral_value(rounding=ROUND_HALF_UP) * tick_size


def pct_change(old: Decimal, new: Decimal) -> Decimal:
    if old == 0:
        return Decimal("0")
    return ((new - old) / old) * Decimal("100")


def decimal_str(value: Decimal) -> str:
    return format(value.normalize(), "f")
