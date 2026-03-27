from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Dict, Optional

from ..models.performance import PerformanceMetrics, TradeRecord
from ..storage.jsonl_store import JsonlStore

LOGGER = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self, store: JsonlStore) -> None:
        self.store = store
        self._metrics: Dict[str, PerformanceMetrics] = {}

    def get_metrics(self, agent_id: str) -> PerformanceMetrics:
        if agent_id not in self._metrics:
            saved = self.store.read_json(f"perf_{agent_id}")
            if saved:
                self._metrics[agent_id] = PerformanceMetrics.from_dict(saved)
            else:
                self._metrics[agent_id] = PerformanceMetrics(agent_id=agent_id)
        return self._metrics[agent_id]

    def record_trade(self, trade: TradeRecord) -> None:
        metrics = self.get_metrics(trade.agent_id)
        metrics.total_trades += 1

        if trade.side == "ask" and trade.pnl_krw != 0:
            metrics.trade_pnls.append(trade.pnl_krw)
            metrics.total_pnl_krw += trade.pnl_krw

            if trade.pnl_krw > 0:
                metrics.winning_trades += 1
                metrics.consecutive_losses = 0
            else:
                metrics.losing_trades += 1
                metrics.consecutive_losses += 1

        metrics.last_updated = time.time()
        self._save(metrics)

        # Also log trade
        self.store.append("trades", trade.to_dict())

    def update_value(self, agent_id: str, current_value: Decimal) -> None:
        metrics = self.get_metrics(agent_id)
        metrics.current_value_krw = current_value

        if current_value > metrics.peak_value_krw:
            metrics.peak_value_krw = current_value

        if metrics.peak_value_krw > 0:
            drawdown = float(
                (metrics.peak_value_krw - current_value) / metrics.peak_value_krw * 100
            )
            if drawdown > metrics.max_drawdown_pct:
                metrics.max_drawdown_pct = drawdown

        metrics.last_updated = time.time()
        self._save(metrics)

    def all_metrics(self) -> Dict[str, PerformanceMetrics]:
        return dict(self._metrics)

    def _save(self, metrics: PerformanceMetrics) -> None:
        self._metrics[metrics.agent_id] = metrics
        self.store.write_json(f"perf_{metrics.agent_id}", metrics.to_dict())
