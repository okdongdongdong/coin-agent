from decimal import Decimal

from coin_agent.utils.math_helpers import krw_tick_size, round_price


def test_krw_tick_size_for_high_price_band():
    assert krw_tick_size(Decimal("101150")) == Decimal("100")
    assert round_price(Decimal("101149.5")) == Decimal("101100")


def test_krw_tick_size_for_million_band():
    assert krw_tick_size(Decimal("1011000")) == Decimal("1000")
    assert round_price(Decimal("1011499.5")) == Decimal("1011000")
