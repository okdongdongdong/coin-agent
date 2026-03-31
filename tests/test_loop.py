from __future__ import annotations

import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from coin_agent.config.settings import Settings
from coin_agent.models.agent import Signal
from coin_agent.models.performance import PerformanceMetrics
from coin_agent.models.trading import ExecutionResult, OrderIntent, WalletSnapshot
from coin_agent.storage.jsonl_store import JsonlStore
from coin_agent.storage.state_store import StateStore


def _make_settings(root: Path) -> Settings:
    return Settings(
        access_key="",
        secret_key="",
        markets=["KRW-BTC"],
        dry_run=True,
        log_level="WARNING",
        data_dir=root / "data",
        loop_interval_sec=37,
        request_timeout_sec=10,
        candle_unit=1,
        candle_count=200,
        paper_krw_balance=Decimal("300000"),
        fee_rate=Decimal("0.0025"),
        order_cooldown_sec=60,
        max_daily_loss_krw=Decimal("30000"),
        max_total_loss_krw=Decimal("90000"),
        max_position_pct=Decimal("40"),
        min_agent_trades=10,
        bench_threshold=30.0,
        min_active_agents=2,
        rebalance_interval=50,
        softmax_temperature=2.0,
        min_alloc_pct=Decimal("5"),
        max_alloc_pct=Decimal("40"),
        anthropic_api_key="",
        openai_api_key="",
        claude_backend="anthropic",
        claude_model="claude-haiku-4-5-20250610",
        openai_model="gpt-5.4",
        openai_backend="codex_cli",
        session_enabled=False,
        session_min_count=3,
        session_max_count=5,
        session_eval_interval=5,
        session_min_ticks_before_eval=10,
    )


@dataclass
class _Snapshot:
    market: str
    current_price: Decimal
    timestamp: float


class _Collector:
    def snapshot(self, market: str) -> _Snapshot:
        return _Snapshot(market=market, current_price=Decimal("100000000"), timestamp=1.0)


class _Broker:
    def get_wallet(self, agent_id: str = "global", asset_currency: str = "") -> WalletSnapshot:
        return WalletSnapshot(
            krw_available=Decimal("300000"),
            asset_available=Decimal("0"),
            avg_buy_price=Decimal("0"),
        )


class _Registry:
    def active_agents(self):
        return []

    def all_agents(self):
        return []


class _Orchestrator:
    def __init__(self) -> None:
        self.decide_called = False
        self.execute_called = False
        self.reconcile_called = False
        self.rebalanced = False

    def rebalance(self):
        self.rebalanced = True
        return {}

    def run_tick(self, snapshot):
        return {
            "codex_agent": Signal(
                agent_id="codex_agent",
                action="buy",
                confidence=0.8,
                reason="test",
            )
        }

    def reconcile_pending_live_orders(self, current_tick: int):
        self.reconcile_called = True
        return []

    def has_pending_live_order(self, wallet_id: str) -> bool:
        return False

    def decide(self, snapshot, tech_signals, claude_signals=None, meta_action=None):
        self.decide_called = True
        return OrderIntent(
            market=snapshot.market,
            side="bid",
            volume=Decimal("0.0008"),
            price=Decimal("99950000"),
            agent_id="orchestrator",
            reason="consensus_buy (w=0.8)",
        )

    def execute_order(self, intent):
        self.execute_called = True
        return ExecutionResult(
            mode="paper",
            success=True,
            message="filled",
            order_id="paper-1",
        )

    def generate_report(self, snapshot, signals):
        return "report"


class _Leaderboard:
    def save_snapshot(self, metrics):
        return None


def test_run_loop_executes_decision_path(monkeypatch):
    from coin_agent.engine import loop as loop_module

    root = Path(tempfile.mkdtemp())
    settings = _make_settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    orchestrator = _Orchestrator()
    perf_tracker = type(
        "PerfTracker",
        (),
        {
            "get_metrics": lambda self, agent_id: PerformanceMetrics(agent_id=agent_id),
            "all_metrics": lambda self: {},
        },
    )()

    def fake_build_system(root_path):
        return (
            settings,
            object(),
            _Collector(),
            _Registry(),
            orchestrator,
            _Broker(),
            perf_tracker,
            object(),
            _Leaderboard(),
            store,
            state,
            None,
            None,
        )

    monkeypatch.setattr(loop_module, "build_system", fake_build_system)

    loop_module.run_loop(root, interval=0, max_ticks=1)

    assert orchestrator.rebalanced is True
    assert orchestrator.reconcile_called is True
    assert orchestrator.decide_called is True
    assert orchestrator.execute_called is True
