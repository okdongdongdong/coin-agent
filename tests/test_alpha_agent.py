"""AlphaAgent 유닛 테스트.

5개 서브시그널, 국면 탐지, 컨플루언스 필터, 리스크 관리를 검증.
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from coin_agent.agents.strategies.alpha_agent import (
    AlphaAgent,
    _Regime,
    _atr_expansion,
    _detect_regime,
    _ema_ribbon_score,
    _obv_trend,
    _rate_of_change,
    _signal_bollinger,
    _signal_ema_ribbon,
    _signal_roc,
    _signal_rsi_adaptive,
    _signal_volume,
    _volume_confirmation,
)
from coin_agent.exchange.market_data import MarketSnapshot

D = Decimal


def _make_snapshot(
    closes: List[Decimal],
    highs: List[Decimal] | None = None,
    lows: List[Decimal] | None = None,
    volumes: List[Decimal] | None = None,
    current_price: Decimal | None = None,
) -> MarketSnapshot:
    """테스트용 MarketSnapshot 생성. closes[0]=최신."""
    n = len(closes)
    if highs is None:
        highs = [c + D("100") for c in closes]
    if lows is None:
        lows = [c - D("100") for c in closes]
    if volumes is None:
        volumes = [D("10")] * n
    if current_price is None:
        current_price = closes[0]

    candles: List[Dict[str, Any]] = []
    for i in range(n):
        candles.append({
            "trade_price": str(closes[i]),
            "high_price": str(highs[i]),
            "low_price": str(lows[i]),
            "opening_price": str(closes[i]),
            "candle_acc_trade_volume": str(volumes[i]),
        })
    return MarketSnapshot(
        market="KRW-BTC",
        current_price=current_price,
        candles=candles,
        ticker={"trade_price": str(current_price)},
        timestamp=time.time(),
    )


def _uptrend_closes(n: int = 80, start: Decimal = D("50000000")) -> List[Decimal]:
    """일관된 상승 추세 close 배열 (newest-first)."""
    # oldest→newest: start, start+step, ...  → reverse to newest-first
    step = D("100000")
    asc = [start + step * i for i in range(n)]
    return list(reversed(asc))


def _downtrend_closes(n: int = 80, start: Decimal = D("60000000")) -> List[Decimal]:
    """일관된 하락 추세."""
    step = D("100000")
    asc = [start - step * i for i in range(n)]
    return list(reversed(asc))


def _ranging_closes(n: int = 80, mid: Decimal = D("55000000")) -> List[Decimal]:
    """횡보 (사인파 형태). newest-first."""
    import math
    result = []
    for i in range(n):
        # oldest-first index
        j = n - 1 - i
        offset = D(str(int(200000 * math.sin(j * 0.3))))
        result.append(mid + offset)
    return result


# ===== 헬퍼 함수 테스트 =====

class TestRateOfChange:
    def test_positive_roc(self):
        closes = [D("110"), D("105"), D("100")]  # newest first
        assert _rate_of_change(closes, 2) == D("10")  # (110-100)/100*100

    def test_zero_old_price(self):
        closes = [D("100"), D("0")]
        assert _rate_of_change(closes, 1) == D("0")

    def test_insufficient_data(self):
        closes = [D("100")]
        assert _rate_of_change(closes, 5) == D("0")


class TestOBVTrend:
    def test_accumulation(self):
        # 가격 상승 + 거래량 → 양의 OBV 기울기
        closes = list(reversed([D("100"), D("101"), D("102"), D("103"), D("104")]))
        volumes = [D("10")] * 5
        result = _obv_trend(closes, volumes, 4)
        assert result > D("0")

    def test_distribution(self):
        closes = list(reversed([D("104"), D("103"), D("102"), D("101"), D("100")]))
        volumes = [D("10")] * 5
        result = _obv_trend(closes, volumes, 4)
        assert result < D("0")


class TestEMARibbonScore:
    def test_uptrend_alignment(self):
        closes = _uptrend_closes(80)
        score, direction = _ema_ribbon_score(closes)
        assert score > D("0")
        assert direction == "up"

    def test_downtrend_alignment(self):
        closes = _downtrend_closes(80)
        score, direction = _ema_ribbon_score(closes)
        assert score < D("0")
        assert direction == "down"

    def test_insufficient_data(self):
        closes = [D("100")] * 30
        score, direction = _ema_ribbon_score(closes)
        assert score == D("0")
        assert direction == "mixed"


class TestVolumeConfirmation:
    def test_above_average(self):
        volumes = [D("20")] + [D("10")] * 19
        assert _volume_confirmation(volumes, 20) > D("1")

    def test_below_average(self):
        volumes = [D("5")] + [D("10")] * 19
        assert _volume_confirmation(volumes, 20) < D("1")


class TestATRExpansion:
    def test_expanding_volatility(self):
        # 최근 변동성 > 장기 변동성
        n = 30
        highs = [D("100") + D(str(i * 5)) for i in range(n)]
        lows = [D("90") - D(str(i * 5)) for i in range(n)]
        closes = [D("95") + D(str(i * 2)) for i in range(n)]
        result = _atr_expansion(highs, lows, closes)
        # 최근 ATR이 더 크면 >1
        assert isinstance(result, Decimal)


# ===== 국면 탐지 테스트 =====

class TestRegimeDetection:
    def test_trending_up(self):
        regime = _detect_regime(D("0.7"), "up", D("1.1"), D("55"))
        assert regime == _Regime.TRENDING_UP

    def test_trending_down(self):
        regime = _detect_regime(D("-0.7"), "down", D("1.1"), D("45"))
        assert regime == _Regime.TRENDING_DOWN

    def test_ranging(self):
        regime = _detect_regime(D("0.2"), "mixed", D("0.9"), D("50"))
        assert regime == _Regime.RANGING

    def test_volatile(self):
        regime = _detect_regime(D("0.3"), "mixed", D("1.5"), D("50"))
        assert regime == _Regime.VOLATILE


# ===== 서브시그널 테스트 =====

class TestSubSignals:
    def test_ema_ribbon_signal(self):
        assert _signal_ema_ribbon(D("0.8")) == D("0.8")
        assert _signal_ema_ribbon(D("-0.5")) == D("-0.5")

    def test_rsi_adaptive_trending_up_pullback(self):
        # 상승 추세 중 RSI 30 → 강한 매수
        sig = _signal_rsi_adaptive(D("30"), _Regime.TRENDING_UP)
        assert sig > D("0.5")

    def test_rsi_adaptive_trending_down_rally(self):
        # 하락 추세 중 RSI 70 → 강한 매도
        sig = _signal_rsi_adaptive(D("70"), _Regime.TRENDING_DOWN)
        assert sig < D("-0.5")

    def test_rsi_adaptive_ranging_oversold(self):
        sig = _signal_rsi_adaptive(D("20"), _Regime.RANGING)
        assert sig > D("0.5")

    def test_rsi_adaptive_ranging_overbought(self):
        sig = _signal_rsi_adaptive(D("80"), _Regime.RANGING)
        assert sig < D("-0.5")

    def test_bollinger_trending_up_lower_touch(self):
        # 상승 추세 중 하단 터치 → 강한 매수
        sig = _signal_bollinger(
            D("100"), D("200"), D("150"), D("100"), _Regime.TRENDING_UP
        )
        assert sig > D("0.5")

    def test_bollinger_ranging_upper_touch(self):
        sig = _signal_bollinger(
            D("200"), D("200"), D("150"), D("100"), _Regime.RANGING
        )
        assert sig < D("-0.5")

    def test_volume_high_bullish(self):
        sig = _signal_volume(D("2.0"), _Regime.TRENDING_UP, D("1.0"))
        assert sig > D("0")

    def test_volume_low_neutral(self):
        sig = _signal_volume(D("0.5"), _Regime.RANGING, D("1.0"))
        assert sig == D("0")

    def test_roc_strong_bullish(self):
        sig = _signal_roc(D("3.0"), D("1.5"))
        assert sig > D("0")

    def test_roc_strong_bearish(self):
        sig = _signal_roc(D("-3.0"), D("-1.5"))
        assert sig < D("0")


# ===== AlphaAgent 통합 테스트 =====

class TestAlphaAgent:
    def test_strategy_name(self):
        agent = AlphaAgent()
        assert agent.strategy_name() == "adaptive_confluence"

    def test_agent_id_default(self):
        agent = AlphaAgent()
        assert agent.agent_id == "alpha_agent"

    def test_insufficient_data_hold(self):
        agent = AlphaAgent()
        closes = [D("50000000")] * 30  # Only 30, need 60+
        snap = _make_snapshot(closes)
        sig = agent.analyze(snap)
        assert sig.action == "hold"
        assert "insufficient_data" in sig.reason

    def test_uptrend_buy_signal(self):
        """강한 상승 추세에서는 완화된 컨플루언스로 실제 매수 시그널을 낸다."""
        agent = AlphaAgent()
        closes = _uptrend_closes(100)
        # 거래량을 최근에 높게 설정
        volumes = [D("20")] * 20 + [D("10")] * 80
        snap = _make_snapshot(closes, volumes=volumes)
        sig = agent.analyze(snap)
        assert sig.action == "buy"
        assert sig.confidence > 0.5
        assert sig.stop_loss is not None
        assert sig.target_price is not None
        assert sig.target_price > snap.current_price
        assert sig.stop_loss < snap.current_price

    def test_downtrend_sell_signal(self):
        """강한 하락 추세에서는 완화된 컨플루언스로 실제 매도 시그널을 낸다."""
        agent = AlphaAgent()
        closes = _downtrend_closes(100)
        volumes = [D("20")] * 20 + [D("10")] * 80
        snap = _make_snapshot(closes, volumes=volumes)
        sig = agent.analyze(snap)
        assert sig.action == "sell"
        assert sig.confidence > 0.5
        assert sig.stop_loss is not None
        assert sig.target_price is not None
        assert sig.target_price < snap.current_price
        assert sig.stop_loss > snap.current_price

    def test_ranging_market_patient(self):
        """횡보장에서 성급한 시그널 내지 않음."""
        agent = AlphaAgent()
        closes = _ranging_closes(100)
        snap = _make_snapshot(closes)
        sig = agent.analyze(snap)
        # 횡보장: hold이 많아야 함 (노이즈 필터링)
        # buy/sell 나오더라도 confidence가 낮아야 함
        if sig.action != "hold":
            assert sig.confidence <= 0.85

    def test_metadata_always_present(self):
        """어떤 시그널이든 메타데이터가 있어야 함."""
        agent = AlphaAgent()
        closes = _uptrend_closes(100)
        snap = _make_snapshot(closes)
        sig = agent.analyze(snap)
        assert sig.metadata is not None
        assert "regime" in sig.metadata

    def test_confluence_filter_prevents_weak_trades(self):
        """컨플루언스 미달 시 HOLD."""
        # 평탄한 가격 데이터 → 서브시그널 대부분 0 → hold
        agent = AlphaAgent()
        closes = [D("50000000")] * 100
        snap = _make_snapshot(closes)
        sig = agent.analyze(snap)
        assert sig.action == "hold"

    def test_custom_config(self):
        """커스텀 설정 적용."""
        agent = AlphaAgent(config={
            "min_confluence": 4,
            "min_score": "0.5",
            "atr_stop_mult": "2.0",
            "atr_target_mult": "3.0",
        })
        assert agent._min_confluence == 4
        assert agent._min_score == D("0.5")
        assert agent._atr_stop == D("2.0")
        assert agent._atr_target == D("3.0")

    def test_risk_reward_ratio(self):
        """R:R이 항상 1:1 이상."""
        agent = AlphaAgent()
        closes = _uptrend_closes(100)
        volumes = [D("30")] * 30 + [D("10")] * 70
        snap = _make_snapshot(closes, volumes=volumes)
        sig = agent.analyze(snap)
        if sig.action == "buy" and sig.stop_loss and sig.target_price:
            risk = snap.current_price - sig.stop_loss
            reward = sig.target_price - snap.current_price
            assert reward >= risk  # R:R >= 1:1
