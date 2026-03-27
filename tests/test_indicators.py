from decimal import Decimal

import pytest

from coin_agent.utils.indicators import sma, rsi, bollinger_bands, atr, _decimal_sqrt


class TestSMA:
    def test_basic(self):
        values = [Decimal(x) for x in [10, 20, 30, 40, 50]]
        assert sma(values, 3) == Decimal(20)  # (10+20+30)/3

    def test_period_equals_length(self):
        values = [Decimal(x) for x in [10, 20, 30]]
        assert sma(values, 3) == Decimal(20)

    def test_insufficient_data(self):
        with pytest.raises(ValueError):
            sma([Decimal(1)], 5)


class TestRSI:
    def test_all_gains(self):
        # Monotonically increasing -> RSI should be 100
        closes = list(reversed([Decimal(i) for i in range(1, 20)]))
        result = rsi(closes, 14)
        assert result == Decimal("100")

    def test_all_losses(self):
        # Monotonically decreasing -> RSI should be 0
        closes = [Decimal(i) for i in range(1, 20)]
        result = rsi(closes, 14)
        assert result == Decimal("0")

    def test_mixed(self):
        closes = [Decimal(x) for x in [50, 48, 52, 47, 53, 46, 54, 45, 55, 44, 56, 43, 57, 42, 58, 41]]
        result = rsi(closes, 14)
        assert Decimal("0") <= result <= Decimal("100")

    def test_insufficient_data(self):
        with pytest.raises(ValueError):
            rsi([Decimal(1)] * 5, 14)


class TestBollingerBands:
    def test_constant_values(self):
        values = [Decimal("100")] * 20
        upper, mid, lower = bollinger_bands(values, 20)
        assert mid == Decimal("100")
        assert upper == Decimal("100")
        assert lower == Decimal("100")

    def test_upper_gt_lower(self):
        values = [Decimal(x) for x in range(1, 25)]
        upper, mid, lower = bollinger_bands(values, 20)
        assert upper > mid > lower

    def test_insufficient_data(self):
        with pytest.raises(ValueError):
            bollinger_bands([Decimal(1)] * 5, 20)


class TestATR:
    def test_basic(self):
        highs = [Decimal(x) for x in [12, 13, 14, 11, 15, 12, 13, 14, 11, 15, 12, 13, 14, 11, 15]]
        lows = [Decimal(x) for x in [8, 9, 10, 7, 11, 8, 9, 10, 7, 11, 8, 9, 10, 7, 11]]
        closes = [Decimal(x) for x in [10, 11, 12, 9, 13, 10, 11, 12, 9, 13, 10, 11, 12, 9, 13]]
        result = atr(highs, lows, closes, 14)
        assert result > 0

    def test_insufficient_data(self):
        with pytest.raises(ValueError):
            atr([Decimal(1)] * 5, [Decimal(1)] * 5, [Decimal(1)] * 5, 14)


class TestDecimalSqrt:
    def test_zero(self):
        assert _decimal_sqrt(Decimal("0")) == Decimal("0")

    def test_positive(self):
        result = _decimal_sqrt(Decimal("4"))
        assert abs(result - Decimal("2")) < Decimal("0.0001")

    def test_negative(self):
        assert _decimal_sqrt(Decimal("-1")) == Decimal("0")
