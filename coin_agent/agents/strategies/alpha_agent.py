"""
Alpha Agent — Adaptive Confluence Trend Follower

다른 4개 에이전트를 압도하기 위해 설계된 고급 전략 에이전트.

핵심 차별점:
  1. 시장 국면 탐지 (추세/횡보/고변동) → 전략 적응
  2. 5개 독립 서브시그널 컨플루언스 → 고확신 진입만 허용
  3. ATR 기반 동적 손절/목표 → 리스크 관리 내재화
  4. 추세장 풀백 매수 / 횡보장 평균회귀 / 고변동장 돌파 추종

다른 에이전트와의 차이:
  - SMA: 단일 크로스오버 → Alpha는 EMA 리본 + 4개 보조 확인
  - Momentum: RSI 단독 → Alpha는 RSI를 국면별 동적 구간으로 활용
  - MeanReversion: BB 극단만 → Alpha는 추세 중 BB 풀백도 포착
  - Breakout: 거의 항상 HOLD → Alpha는 거래량을 보조 확인으로 유연하게 활용
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from ..base import SubAgent
from ...exchange.market_data import MarketSnapshot
from ...models.agent import Signal
from ...utils.indicators import atr, bollinger_bands, ema, rsi, sma

# ---------------------------------------------------------------------------
# Internal helpers (no external deps, all Decimal)
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")


def _rate_of_change(closes: List[Decimal], period: int) -> Decimal:
    """가격 변화율 (%). closes[0]=최신."""
    if len(closes) <= period:
        return _ZERO
    old = closes[period]
    if old == _ZERO:
        return _ZERO
    return (closes[0] - old) / old * _HUNDRED


def _obv_trend(closes: List[Decimal], volumes: List[Decimal], period: int) -> Decimal:
    """OBV(On Balance Volume) 기울기. 양수=매집, 음수=분산."""
    n = min(len(closes), len(volumes), period + 1)
    if n < 3:
        return _ZERO
    # oldest-first 순서로 변환
    c = list(reversed(closes[:n]))
    v = list(reversed(volumes[:n]))
    obv: List[Decimal] = [_ZERO]
    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            obv.append(obv[-1] + v[i])
        elif c[i] < c[i - 1]:
            obv.append(obv[-1] - v[i])
        else:
            obv.append(obv[-1])
    if len(obv) < 2:
        return _ZERO
    # 단순 기울기: (최신 OBV - period/2 전 OBV) / period
    mid_idx = len(obv) // 2
    diff = obv[-1] - obv[mid_idx]
    # 정규화: 평균 거래량 대비
    avg_v = sum(v, _ZERO) / Decimal(len(v)) if v else _ONE
    if avg_v == _ZERO:
        return _ZERO
    return diff / avg_v


def _ema_ribbon_score(closes: List[Decimal]) -> Tuple[Decimal, str]:
    """EMA(8, 21, 55) 리본 정렬도.
    Returns (score, direction).
      score:  -1.0 ~ +1.0  (양=상승정렬, 음=하락정렬, 0=혼재)
      direction: "up", "down", "mixed"
    """
    n = len(closes)
    if n < 55:
        return _ZERO, "mixed"

    e8 = ema(closes, 8)
    e21 = ema(closes, 21)
    e55 = ema(closes, 55)

    # 완전 상승 정렬: 8 > 21 > 55
    if e8 > e21 > e55:
        # 강도: (e8 - e55) / e55 를 % 로 (최대 1.0 클램프)
        strength = min((e8 - e55) / e55 * _HUNDRED, _ONE)
        return strength, "up"
    # 완전 하락 정렬: 8 < 21 < 55
    elif e8 < e21 < e55:
        strength = min((e55 - e8) / e55 * _HUNDRED, _ONE)
        return -strength, "down"
    else:
        # 부분 정렬
        if e8 > e21:
            return Decimal("0.3"), "mixed"
        elif e8 < e21:
            return Decimal("-0.3"), "mixed"
        return _ZERO, "mixed"


def _volume_confirmation(volumes: List[Decimal], lookback: int = 20) -> Decimal:
    """현재 거래량 vs 평균. >1이면 평균 이상."""
    if len(volumes) < lookback:
        return _ONE
    avg = sma(volumes, lookback)
    if avg <= _ZERO:
        return _ONE
    return volumes[0] / avg


def _atr_expansion(
    highs: List[Decimal], lows: List[Decimal], closes: List[Decimal],
) -> Decimal:
    """ATR(7) vs ATR(21) 비율. >1이면 변동성 확대."""
    if len(highs) < 22 or len(lows) < 22 or len(closes) < 22:
        return _ONE
    short_atr = atr(highs, lows, closes, 7)
    long_atr = atr(highs, lows, closes, 21)
    if long_atr <= _ZERO:
        return _ONE
    return short_atr / long_atr


# ---------------------------------------------------------------------------
# Market Regime
# ---------------------------------------------------------------------------

class _Regime:
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


def _detect_regime(
    ribbon_score: Decimal,
    ribbon_dir: str,
    atr_ratio: Decimal,
    current_rsi: Decimal,
) -> str:
    """3축(추세/변동성/모멘텀)으로 시장 국면 판정."""
    high_vol = atr_ratio > Decimal("1.3")
    strong_trend = abs(ribbon_score) > Decimal("0.5")

    if high_vol and not strong_trend:
        return _Regime.VOLATILE
    if ribbon_dir == "up" and strong_trend:
        return _Regime.TRENDING_UP
    if ribbon_dir == "down" and strong_trend:
        return _Regime.TRENDING_DOWN
    return _Regime.RANGING


# ---------------------------------------------------------------------------
# Sub-signal generators  (each returns  -1.0 ~ +1.0)
# ---------------------------------------------------------------------------

def _signal_ema_ribbon(ribbon_score: Decimal) -> Decimal:
    """EMA 리본 정렬 신호."""
    return ribbon_score  # already -1..+1


def _signal_rsi_adaptive(current_rsi: Decimal, regime: str) -> Decimal:
    """국면별 동적 RSI 신호.
    추세 상승: RSI 45-80 bullish zone  (풀백 40-45 = 매수 기회)
    추세 하락: RSI 20-55 bearish zone  (반등 55-60 = 매도 기회)
    횡보: 전통적 30/70
    """
    if regime == _Regime.TRENDING_UP:
        if current_rsi < Decimal("35"):
            return Decimal("0.8")   # 깊은 풀백 = 강한 매수
        if current_rsi < Decimal("45"):
            return Decimal("0.5")   # 적정 풀백
        if current_rsi > Decimal("80"):
            return Decimal("-0.3")  # 과열 경고 (추세 중이므로 약한 매도)
        return Decimal("0.2")      # 추세 중 기본 매수 편향
    elif regime == _Regime.TRENDING_DOWN:
        if current_rsi > Decimal("65"):
            return Decimal("-0.8")  # 과매수 반등 = 강한 매도
        if current_rsi > Decimal("55"):
            return Decimal("-0.5")  # 적정 반등
        if current_rsi < Decimal("20"):
            return Decimal("0.3")   # 극단 과매도 경고
        return Decimal("-0.2")     # 추세 중 기본 매도 편향
    else:  # RANGING or VOLATILE
        if current_rsi < Decimal("25"):
            return Decimal("0.7")
        if current_rsi < Decimal("35"):
            return Decimal("0.4")
        if current_rsi > Decimal("75"):
            return Decimal("-0.7")
        if current_rsi > Decimal("65"):
            return Decimal("-0.4")
        return _ZERO


def _signal_bollinger(price: Decimal, upper: Decimal, mid: Decimal, lower: Decimal,
                      regime: str) -> Decimal:
    """BB 포지션 신호.
    추세 상승: 하단 터치 = 강한 매수, 상단 터치 = 약한 hold
    추세 하락: 상단 터치 = 강한 매도, 하단 터치 = 약한 hold
    횡보: 전통적 평균회귀
    """
    band_width = upper - lower
    if band_width <= _ZERO:
        return _ZERO
    position = (price - lower) / band_width  # 0=하단, 1=상단

    if regime == _Regime.TRENDING_UP:
        if position < Decimal("0.2"):
            return Decimal("0.8")   # 추세 중 하단 풀백 = 최고의 매수
        if position < Decimal("0.4"):
            return Decimal("0.4")   # 중간 아래 = 양호 매수
        if position > Decimal("0.95"):
            return Decimal("-0.2")  # 과도한 상승 = 약한 주의
        return Decimal("0.1")
    elif regime == _Regime.TRENDING_DOWN:
        if position > Decimal("0.8"):
            return Decimal("-0.8")  # 추세 중 상단 반등 = 최고의 매도
        if position > Decimal("0.6"):
            return Decimal("-0.4")
        if position < Decimal("0.05"):
            return Decimal("0.2")   # 극단 = 약한 주의
        return Decimal("-0.1")
    else:  # RANGING or VOLATILE
        if position < Decimal("0.1"):
            return Decimal("0.7")
        if position < Decimal("0.25"):
            return Decimal("0.3")
        if position > Decimal("0.9"):
            return Decimal("-0.7")
        if position > Decimal("0.75"):
            return Decimal("-0.3")
        return _ZERO


def _signal_volume(vol_ratio: Decimal, regime: str, price_direction: Decimal) -> Decimal:
    """거래량 확인 신호.
    높은 거래량 + 가격 방향 = 강한 확인.
    높은 거래량 + 가격 역방향 = 경고 신호.
    """
    if vol_ratio < Decimal("0.7"):
        # 낮은 거래량 = 현재 움직임 불신
        return _ZERO
    if vol_ratio > Decimal("1.5"):
        # 높은 거래량 = 방향 확인 (가격 방향과 같은 방향)
        if price_direction > _ZERO:
            return Decimal("0.5")
        elif price_direction < _ZERO:
            return Decimal("-0.5")
    if vol_ratio > Decimal("1.1"):
        # 평균 이상 거래량
        if price_direction > _ZERO:
            return Decimal("0.2")
        elif price_direction < _ZERO:
            return Decimal("-0.2")
    return _ZERO


def _signal_roc(roc_5: Decimal, roc_10: Decimal) -> Decimal:
    """가격 변화율(ROC) 신호. 단기/중기 모멘텀 결합."""
    # 두 ROC의 부호가 같으면 강한 신호
    if roc_5 > _ONE and roc_10 > _ZERO:
        return min(roc_5 / Decimal("5"), _ONE)  # 최대 +1.0
    if roc_5 < -_ONE and roc_10 < _ZERO:
        return max(roc_5 / Decimal("5"), -_ONE)  # 최소 -1.0
    # 부호 혼재 = 약한 신호
    combined = (roc_5 + roc_10) / Decimal("10")
    return max(min(combined, _ONE), -_ONE)


# ---------------------------------------------------------------------------
# Regime-dependent weights
# ---------------------------------------------------------------------------

_WEIGHTS: Dict[str, Dict[str, Decimal]] = {
    _Regime.TRENDING_UP: {
        "ema_ribbon": Decimal("0.30"),  # 추세 판단 최우선
        "rsi":        Decimal("0.25"),  # 풀백 매수 타이밍
        "bollinger":  Decimal("0.20"),  # 밴드 내 위치
        "volume":     Decimal("0.15"),  # 거래량 확인
        "roc":        Decimal("0.10"),  # 모멘텀 확인
    },
    _Regime.TRENDING_DOWN: {
        "ema_ribbon": Decimal("0.30"),
        "rsi":        Decimal("0.25"),
        "bollinger":  Decimal("0.20"),
        "volume":     Decimal("0.15"),
        "roc":        Decimal("0.10"),
    },
    _Regime.RANGING: {
        "ema_ribbon": Decimal("0.10"),  # 추세 약하므로 낮은 비중
        "rsi":        Decimal("0.25"),  # 과매수/과매도 핵심
        "bollinger":  Decimal("0.30"),  # 평균회귀 최우선
        "volume":     Decimal("0.15"),
        "roc":        Decimal("0.20"),  # 방향 전환 감지
    },
    _Regime.VOLATILE: {
        "ema_ribbon": Decimal("0.15"),
        "rsi":        Decimal("0.15"),
        "bollinger":  Decimal("0.15"),
        "volume":     Decimal("0.30"),  # 거래량 돌파 최우선
        "roc":        Decimal("0.25"),  # 모멘텀 돌파 확인
    },
}


# ---------------------------------------------------------------------------
# Alpha Agent
# ---------------------------------------------------------------------------

class AlphaAgent(SubAgent):
    """적응형 컨플루언스 트렌드 팔로워.

    5개 독립 서브시그널의 국면별 가중 합산 → 컨플루언스 검증 →
    고확신 진입만 허용. ATR 기반 stop_loss/target_price 자동 설정.
    """

    # 기본 파라미터
    _MIN_CONFLUENCE = 3          # 기본: 최소 3/5 서브시그널 동의
    _TREND_MIN_CONFLUENCE = 2    # 추세/돌파 국면은 2개 핵심 신호만 맞아도 허용
    _MIN_SCORE = Decimal("0.25") # 추세장 2-신호 진입을 허용할 수준으로 완화
    _ATR_STOP_MULT = Decimal("1.5")
    _ATR_TARGET_MULT = Decimal("2.5")  # R:R ≈ 1:1.67

    def __init__(
        self,
        agent_id: str = "alpha_agent",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(agent_id, config or {})
        self._min_confluence = self.config.get("min_confluence", self._MIN_CONFLUENCE)
        self._trend_min_confluence = int(
            self.config.get(
                "trend_min_confluence",
                min(self._min_confluence, self._TREND_MIN_CONFLUENCE),
            )
        )
        self._min_score = Decimal(str(self.config.get("min_score", self._MIN_SCORE)))
        self._atr_stop = Decimal(str(self.config.get("atr_stop_mult", self._ATR_STOP_MULT)))
        self._atr_target = Decimal(str(self.config.get("atr_target_mult", self._ATR_TARGET_MULT)))

    def strategy_name(self) -> str:
        return "adaptive_confluence"

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        closes = snapshot.closes
        highs = snapshot.highs
        lows = snapshot.lows
        volumes = snapshot.volumes
        price = snapshot.current_price

        # 최소 데이터 요구량: EMA(55) + 여유
        if len(closes) < 60:
            return self.hold_signal("insufficient_data (need 60+ candles)")

        # ====== Layer 1: 시장 국면 탐지 ======
        ribbon_score, ribbon_dir = _ema_ribbon_score(closes)
        current_rsi = rsi(closes, 14)
        atr_ratio = _atr_expansion(highs, lows, closes)
        regime = _detect_regime(ribbon_score, ribbon_dir, atr_ratio, current_rsi)

        # ====== Layer 2: 5개 서브시그널 ======
        upper, mid, lower = bollinger_bands(closes, 20, _TWO)
        vol_ratio = _volume_confirmation(volumes, 20)
        roc_5 = _rate_of_change(closes, 5)
        roc_10 = _rate_of_change(closes, 10)

        # 가격 방향 (단기 ROC 기반)
        price_direction = roc_5

        sub_signals: Dict[str, Decimal] = {
            "ema_ribbon": _signal_ema_ribbon(ribbon_score),
            "rsi":        _signal_rsi_adaptive(current_rsi, regime),
            "bollinger":  _signal_bollinger(price, upper, mid, lower, regime),
            "volume":     _signal_volume(vol_ratio, regime, price_direction),
            "roc":        _signal_roc(roc_5, roc_10),
        }

        # ====== Layer 3: 컨플루언스 점수 계산 ======
        weights = _WEIGHTS[regime]
        weighted_score = _ZERO
        buy_count = 0
        sell_count = 0

        for key, signal_val in sub_signals.items():
            w = weights[key]
            weighted_score += signal_val * w
            if signal_val > Decimal("0.1"):
                buy_count += 1
            elif signal_val < Decimal("-0.1"):
                sell_count += 1

        # 방향 결정
        if weighted_score > _ZERO:
            direction = "buy"
            confluence_count = buy_count
        elif weighted_score < _ZERO:
            direction = "sell"
            confluence_count = sell_count
        else:
            return self._hold_with_metadata(
                "neutral_score", regime, sub_signals, weighted_score, 0
            )

        abs_score = abs(weighted_score)
        required_confluence = self._required_confluence(regime)

        # ====== 컨플루언스 필터: 핵심 차별점 ======
        if confluence_count < required_confluence:
            return self._hold_with_metadata(
                f"low_confluence ({confluence_count}/{required_confluence})",
                regime, sub_signals, weighted_score, confluence_count,
            )

        if abs_score < self._min_score:
            return self._hold_with_metadata(
                f"weak_score ({abs_score:.3f} < {self._min_score})",
                regime, sub_signals, weighted_score, confluence_count,
            )

        # ====== Layer 4: 신뢰도 + 리스크 관리 ======
        # 컨플루언스 + 점수 강도 → confidence
        # 2~3/5 합의 + score 0.25+ → 0.48~ 기본
        # 5/5 합의 + score 1.0 → 0.95 최대
        base_conf = float(Decimal("0.4") + abs_score * Decimal("0.3"))
        confluence_bonus = max(confluence_count - required_confluence, 0) * 0.08
        regime_bonus = 0.05 if regime in (_Regime.TRENDING_UP, _Regime.TRENDING_DOWN) else 0.0
        confidence = min(base_conf + confluence_bonus + regime_bonus, 0.95)

        # ATR 기반 stop / target
        current_atr = atr(highs, lows, closes, 14)
        if direction == "buy":
            stop = price - self._atr_stop * current_atr
            target = price + self._atr_target * current_atr
        else:
            stop = price + self._atr_stop * current_atr
            target = price - self._atr_target * current_atr

        # 메타데이터 구성
        metadata = self._build_metadata(
            regime, sub_signals, weights, weighted_score,
            confluence_count, current_rsi, vol_ratio, atr_ratio,
            current_atr, upper, mid, lower, roc_5, roc_10,
        )

        reason_parts = []
        for key, val in sub_signals.items():
            if (direction == "buy" and val > Decimal("0.1")) or \
               (direction == "sell" and val < Decimal("-0.1")):
                reason_parts.append(f"{key}={val:+.2f}")
        reason = (
            f"{regime} | score={weighted_score:+.3f} | "
            f"confluence={confluence_count}/5 | "
            f"{', '.join(reason_parts)}"
        )

        return Signal(
            agent_id=self.agent_id,
            action=direction,
            confidence=confidence,
            reason=reason,
            target_price=target,
            stop_loss=stop,
            metadata=metadata,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _required_confluence(self, regime: str) -> int:
        if regime in (_Regime.TRENDING_UP, _Regime.TRENDING_DOWN, _Regime.VOLATILE):
            return self._trend_min_confluence
        return self._min_confluence

    def _hold_with_metadata(
        self,
        reason: str,
        regime: str,
        sub_signals: Dict[str, Decimal],
        score: Decimal,
        confluence: int,
    ) -> Signal:
        """HOLD 시에도 메타데이터 제공 (디버깅/리포트용)."""
        return Signal(
            agent_id=self.agent_id,
            action="hold",
            confidence=0.0,
            reason=reason,
            metadata={
                "regime": regime,
                "weighted_score": str(score),
                "confluence": confluence,
                **{f"sig_{k}": str(v) for k, v in sub_signals.items()},
            },
        )

    @staticmethod
    def _build_metadata(
        regime: str,
        sub_signals: Dict[str, Decimal],
        weights: Dict[str, Decimal],
        score: Decimal,
        confluence: int,
        current_rsi: Decimal,
        vol_ratio: Decimal,
        atr_ratio: Decimal,
        current_atr: Decimal,
        bb_upper: Decimal,
        bb_mid: Decimal,
        bb_lower: Decimal,
        roc_5: Decimal,
        roc_10: Decimal,
    ) -> Dict[str, Any]:
        return {
            "regime": regime,
            "weighted_score": str(score),
            "confluence": confluence,
            "rsi": str(current_rsi),
            "vol_ratio": str(vol_ratio),
            "atr_ratio": str(atr_ratio),
            "atr": str(current_atr),
            "bb_upper": str(bb_upper),
            "bb_mid": str(bb_mid),
            "bb_lower": str(bb_lower),
            "roc_5": str(roc_5),
            "roc_10": str(roc_10),
            **{f"sig_{k}": str(v) for k, v in sub_signals.items()},
            **{f"wgt_{k}": str(v) for k, v in weights.items()},
        }
