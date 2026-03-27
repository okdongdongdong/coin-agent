from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from ..models.performance import PerformanceMetrics
from ..performance.scorer import PerformanceScorer, ScoreBreakdown
from ..storage.jsonl_store import JsonlStore


class Leaderboard:
    def __init__(self, scorer: PerformanceScorer, store: JsonlStore) -> None:
        self.scorer = scorer
        self.store = store

    def rank(self, all_metrics: Dict[str, PerformanceMetrics]) -> List[Tuple[str, ScoreBreakdown]]:
        scored = self.scorer.score_all(all_metrics)
        ranking = sorted(scored.items(), key=lambda x: x[1].composite, reverse=True)
        return ranking

    def save_snapshot(self, all_metrics: Dict[str, PerformanceMetrics]) -> None:
        ranking = self.rank(all_metrics)
        entry = {
            "timestamp": time.time(),
            "ranking": [
                {
                    "rank": i + 1,
                    "agent_id": aid,
                    "composite": sb.composite,
                    "sharpe": sb.sharpe_score,
                    "win_rate": sb.win_rate_score,
                    "profit_factor": sb.profit_factor_score,
                    "drawdown": sb.drawdown_score,
                    "consistency": sb.consistency_score,
                }
                for i, (aid, sb) in enumerate(ranking)
            ],
        }
        self.store.append("agent_scores", entry)

    def get_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        return self.store.read_lines("agent_scores", last_n)
