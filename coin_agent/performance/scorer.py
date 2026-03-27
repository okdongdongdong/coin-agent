from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..models.performance import PerformanceMetrics


@dataclass(frozen=True)
class ScoreBreakdown:
    agent_id: str
    sharpe_score: float
    win_rate_score: float
    profit_factor_score: float
    drawdown_score: float
    consistency_score: float
    composite: float


class PerformanceScorer:
    # Weights (must sum to 1.0)
    W_SHARPE = 0.30
    W_WIN_RATE = 0.20
    W_PROFIT_FACTOR = 0.20
    W_DRAWDOWN = 0.20
    W_CONSISTENCY = 0.10

    def score(self, metrics: PerformanceMetrics) -> ScoreBreakdown:
        sharpe_score = self._normalize_sharpe(metrics.sharpe_ratio)
        win_rate_score = metrics.win_rate * 100  # 0-100
        pf_score = self._normalize_profit_factor(metrics.profit_factor)
        dd_score = self._normalize_drawdown(metrics.max_drawdown_pct)
        consistency_score = self._calc_consistency(metrics)

        composite = (
            sharpe_score * self.W_SHARPE
            + win_rate_score * self.W_WIN_RATE
            + pf_score * self.W_PROFIT_FACTOR
            + dd_score * self.W_DRAWDOWN
            + consistency_score * self.W_CONSISTENCY
        )
        composite = max(0.0, min(100.0, composite))

        return ScoreBreakdown(
            agent_id=metrics.agent_id,
            sharpe_score=sharpe_score,
            win_rate_score=win_rate_score,
            profit_factor_score=pf_score,
            drawdown_score=dd_score,
            consistency_score=consistency_score,
            composite=composite,
        )

    def score_all(self, all_metrics: Dict[str, PerformanceMetrics]) -> Dict[str, ScoreBreakdown]:
        return {aid: self.score(m) for aid, m in all_metrics.items()}

    @staticmethod
    def _normalize_sharpe(sharpe: float) -> float:
        # Sharpe: -2 to +3 range mapped to 0-100
        clamped = max(-2.0, min(3.0, sharpe))
        return (clamped + 2.0) / 5.0 * 100

    @staticmethod
    def _normalize_profit_factor(pf: float) -> float:
        # PF: 0 to 3+ mapped to 0-100
        clamped = max(0.0, min(3.0, pf))
        return clamped / 3.0 * 100

    @staticmethod
    def _normalize_drawdown(mdd: float) -> float:
        # Lower drawdown = higher score. MDD 0% = 100, MDD 50%+ = 0
        clamped = max(0.0, min(50.0, mdd))
        return (1.0 - clamped / 50.0) * 100

    @staticmethod
    def _calc_consistency(metrics: PerformanceMetrics) -> float:
        if len(metrics.trade_pnls) < 3:
            return 50.0  # Neutral score when not enough data
        pnls = [float(p) for p in metrics.trade_pnls]
        positive_count = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        # Consistency = smoothed win rate (favoring consistent winners)
        return (positive_count / total) * 100 if total > 0 else 50.0
