from __future__ import annotations

import tempfile
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from coin_agent.agents.orchestrator import Orchestrator
from coin_agent.agents.registry import AgentRegistry
from coin_agent.agents.strategies.breakout_agent import BreakoutAgent
from coin_agent.agents.strategies.mean_reversion_agent import MeanReversionAgent
from coin_agent.agents.strategies.momentum_agent import MomentumAgent
from coin_agent.agents.strategies.sma_agent import SMAAgent
from coin_agent.config.settings import Settings
from coin_agent.models.agent import Signal
from coin_agent.models.trading import ExecutionResult, OrderIntent, WalletSnapshot
from coin_agent.performance.scorer import PerformanceScorer
from coin_agent.performance.tracker import PerformanceTracker
from coin_agent.session.manager import SessionManager
from coin_agent.storage.jsonl_store import JsonlStore
from coin_agent.storage.state_store import StateStore


class _Broker:
    def get_wallet(self, agent_id: str = "global", asset_currency: str = "") -> WalletSnapshot:
        return WalletSnapshot(
            krw_available=Decimal("300000"),
            asset_available=Decimal("0.01"),
            avg_buy_price=Decimal("100000000"),
        )

    def execute(self, intent, agent_id: str = "global"):
        raise AssertionError("execute should not be called in this test")


class _PositionTracker:
    def get_total_value(self, agent_id: str, current_price: Decimal, default_krw: Decimal = Decimal("0")) -> Decimal:
        return default_krw


class _LiveBroker:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.order_status: dict = {
            "uuid": "live-1",
            "price": "100001000",
            "executed_volume": "0",
            "remaining_volume": "0.0002",
            "state": "wait",
        }

    def get_wallet(self, agent_id: str = "global", asset_currency: str = "") -> WalletSnapshot:
        return WalletSnapshot(
            krw_available=Decimal("300000"),
            asset_available=Decimal("0.01"),
            avg_buy_price=Decimal("100000000"),
        )

    def execute(self, intent, agent_id: str = "global"):
        return ExecutionResult(
            mode="live",
            success=True,
            message="submitted",
            order_id="live-1",
        )

    def get_order(self, order_id: str) -> dict:
        return dict(self.order_status)

    def cancel_order(self, order_id: str) -> dict:
        self.cancelled.append(order_id)
        return {"uuid": order_id, "state": "cancel"}


class _HighBalanceBroker:
    def get_wallet(self, agent_id: str = "global", asset_currency: str = "") -> WalletSnapshot:
        return WalletSnapshot(
            krw_available=Decimal("600000"),
            asset_available=Decimal("0.03"),
            avg_buy_price=Decimal("100000000"),
        )

    def execute(self, intent, agent_id: str = "global"):
        raise AssertionError("execute should not be called in this test")


def _settings(root: Path) -> Settings:
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
        max_order_krw=Decimal("100000"),
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
        session_execution_mode="multi",
        session_min_count=3,
        session_max_count=5,
        session_eval_interval=5,
        session_min_ticks_before_eval=10,
    )


def _live_settings(root: Path) -> Settings:
    return replace(_settings(root), dry_run=False)


class _Snapshot:
    market = "KRW-BTC"
    current_price = Decimal("100000000")


def test_decide_builds_consensus_buy_intent():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    for agent_id, alloc in {
        "sma_agent": Decimal("75000"),
        "momentum_agent": Decimal("150000"),
        "mean_reversion_agent": Decimal("37500"),
        "breakout_agent": Decimal("37500"),
    }.items():
        agent_state = registry.get_state(agent_id)
        agent_state.allocated_capital_krw = alloc
        registry.set_state(agent_id, agent_state)

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_Broker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )

    intent = orchestrator.decide(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "sell", 0.9, "sell"),
            "momentum_agent": Signal("momentum_agent", "buy", 0.8, "buy"),
            "mean_reversion_agent": Signal("mean_reversion_agent", "hold", 0.0, "hold"),
            "breakout_agent": Signal("breakout_agent", "sell", 0.55, "sell"),
        },
    )

    assert intent is not None
    assert intent.agent_id == "momentum_agent"
    assert intent.side == "bid"
    assert "consensus_buy" in intent.reason


def test_build_session_decisions_bias_main_agent_vote():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_Broker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=9, max_sessions=9)
    sessions = session_mgr.initialize_sessions()

    decisions = orchestrator.build_session_decisions(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "buy", 0.9, "sma"),
            "momentum_agent": Signal("momentum_agent", "hold", 0.0, "hold"),
            "mean_reversion_agent": Signal("mean_reversion_agent", "hold", 0.0, "hold"),
            "breakout_agent": Signal("breakout_agent", "hold", 0.0, "hold"),
        },
        sessions=sessions,
    )

    decisions_by_id = {item["session_id"]: item for item in decisions}
    assert decisions_by_id["session_trend_core"]["action"] == "buy"
    assert decisions_by_id["session_trend_core"]["intent"] is not None
    assert decisions_by_id["session_breakout_core"]["action"] == "hold"
    assert decisions_by_id["session_breakout_core"]["intent"] is None


def test_build_session_decisions_uses_current_consensus_threshold():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_Broker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=9, max_sessions=9)
    sessions = session_mgr.initialize_sessions()

    decisions = orchestrator.build_session_decisions(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "hold", 0.0, "hold"),
            "momentum_agent": Signal("momentum_agent", "buy", 0.5, "momentum"),
            "alpha_agent": Signal("alpha_agent", "buy", 0.6, "alpha"),
            "turbo_breakout_agent": Signal("turbo_breakout_agent", "buy", 0.8, "turbo"),
            "breakout_agent": Signal("breakout_agent", "hold", 0.0, "hold"),
        },
        sessions=sessions,
    )

    decisions_by_id = {item["session_id"]: item for item in decisions}
    assert decisions_by_id["session_momentum_core"]["action"] == "buy"
    assert decisions_by_id["session_momentum_core"]["intent"] is not None
    assert "buy_w=0.450" in decisions_by_id["session_momentum_core"]["reason"]


def test_build_session_decisions_allows_main_override():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_Broker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=9, max_sessions=9)
    sessions = session_mgr.initialize_sessions()

    decisions = orchestrator.build_session_decisions(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "hold", 0.0, "hold"),
            "momentum_agent": Signal("momentum_agent", "hold", 0.0, "hold"),
            "breakout_agent": Signal("breakout_agent", "sell", 0.7, "volume breakout"),
        },
        sessions=sessions,
    )

    decisions_by_id = {item["session_id"]: item for item in decisions}
    breakout_session = decisions_by_id["session_breakout_core"]
    assert breakout_session["action"] == "sell"
    assert breakout_session["intent"] is None
    assert "main_override_sell" in breakout_session["reason"]


def test_build_meta_decision_sizes_order_from_average_confidence():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_Broker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )
    meta = orchestrator.build_meta_decision(
        snapshot=_Snapshot(),
        session_decisions=[
            {"session_id": "s1", "action": "buy", "confidence": 0.60},
            {"session_id": "s2", "action": "buy", "confidence": 0.65},
            {"session_id": "s3", "action": "buy", "confidence": 0.70},
            {"session_id": "s4", "action": "buy", "confidence": 0.66},
            {"session_id": "s5", "action": "buy", "confidence": 0.64},
            {"session_id": "s6", "action": "buy", "confidence": 0.63},
            {"session_id": "s7", "action": "hold", "confidence": 0.0},
            {"session_id": "s8", "action": "sell", "confidence": 0.52},
            {"session_id": "s9", "action": "hold", "confidence": 0.0},
        ],
    )

    assert meta["action"] == "buy"
    assert meta["intent"] is not None
    assert meta["intent"].volume == Decimal("0.00054")


def test_decide_uses_one_tick_offset_prices():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_Broker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )

    buy_intent = orchestrator.decide(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "hold", 0.0, "hold"),
            "momentum_agent": Signal("momentum_agent", "buy", 0.8, "buy"),
            "mean_reversion_agent": Signal("mean_reversion_agent", "buy", 0.6, "buy"),
            "breakout_agent": Signal("breakout_agent", "hold", 0.0, "hold"),
        },
    )
    sell_intent = orchestrator.decide(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "sell", 0.9, "sell"),
            "momentum_agent": Signal("momentum_agent", "sell", 0.8, "sell"),
            "mean_reversion_agent": Signal("mean_reversion_agent", "hold", 0.0, "hold"),
            "breakout_agent": Signal("breakout_agent", "hold", 0.0, "hold"),
        },
    )

    assert buy_intent is not None
    assert buy_intent.price == Decimal("100001000")
    assert sell_intent is not None
    assert sell_intent.price == Decimal("99999000")


def test_build_meta_decision_uses_one_tick_buy_price():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_Broker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )
    meta = orchestrator.build_meta_decision(
        snapshot=_Snapshot(),
        session_decisions=[
            {"session_id": "s1", "action": "buy", "confidence": 0.80},
            {"session_id": "s2", "action": "buy", "confidence": 0.75},
            {"session_id": "s3", "action": "buy", "confidence": 0.78},
            {"session_id": "s4", "action": "buy", "confidence": 0.82},
            {"session_id": "s5", "action": "buy", "confidence": 0.79},
            {"session_id": "s6", "action": "buy", "confidence": 0.81},
            {"session_id": "s7", "action": "hold", "confidence": 0.0},
            {"session_id": "s8", "action": "sell", "confidence": 0.40},
            {"session_id": "s9", "action": "hold", "confidence": 0.0},
        ],
    )

    assert meta["intent"] is not None
    assert meta["intent"].price == Decimal("100001000")


def test_build_meta_decision_caps_single_order_to_max_order_krw():
    root = Path(tempfile.mkdtemp())
    settings = _settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=_HighBalanceBroker(),
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )
    meta = orchestrator.build_meta_decision(
        snapshot=_Snapshot(),
        session_decisions=[
            {"session_id": "s1", "action": "buy", "confidence": 0.92},
            {"session_id": "s2", "action": "buy", "confidence": 0.88},
            {"session_id": "s3", "action": "buy", "confidence": 0.91},
            {"session_id": "s4", "action": "buy", "confidence": 0.89},
            {"session_id": "s5", "action": "buy", "confidence": 0.90},
            {"session_id": "s6", "action": "buy", "confidence": 0.87},
            {"session_id": "s7", "action": "hold", "confidence": 0.0},
            {"session_id": "s8", "action": "hold", "confidence": 0.0},
            {"session_id": "s9", "action": "hold", "confidence": 0.0},
        ],
    )

    assert meta["intent"] is not None
    assert meta["intent"].volume == Decimal("0.00100000")


def test_execute_order_live_session_stores_pending_until_checked():
    root = Path(tempfile.mkdtemp())
    settings = _live_settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    broker = _LiveBroker()

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=broker,
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )

    result = orchestrator.execute_order(
        OrderIntent(
            market="KRW-BTC",
            side="bid",
            volume=Decimal("0.0002"),
            price=Decimal("100001000"),
            agent_id="session_sma_main",
            reason="session_consensus_buy",
        ),
        wallet_id="session_sma_main",
        initial_capital_krw=Decimal("100000"),
        extra_log={"session_id": "session_sma_main", "main_agent_id": "sma_agent"},
    )

    pending_orders = state.get_pending_orders()
    assert result.success is True
    assert result.message == "submitted"
    assert "live-1" in pending_orders
    assert pending_orders["live-1"]["wallet_id"] == "session_sma_main"


def test_reconcile_pending_live_order_applies_fill_and_clears_pending():
    root = Path(tempfile.mkdtemp())
    settings = _live_settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    broker = _LiveBroker()
    broker.order_status = {
        "uuid": "live-1",
        "price": "100001000",
        "executed_volume": "0.0002",
        "remaining_volume": "0",
        "state": "done",
    }

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=broker,
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )

    orchestrator.execute_order(
        OrderIntent(
            market="KRW-BTC",
            side="bid",
            volume=Decimal("0.0002"),
            price=Decimal("100001000"),
            agent_id="session_sma_main",
            reason="session_consensus_buy",
        ),
        wallet_id="session_sma_main",
        initial_capital_krw=Decimal("100000"),
        extra_log={"session_id": "session_sma_main", "main_agent_id": "sma_agent"},
    )

    outcomes = orchestrator.reconcile_pending_live_orders(current_tick=1)
    wallet = orchestrator.trading_wallet(
        "session_sma_main",
        "KRW-BTC",
        Decimal("100000"),
    )

    assert len(outcomes) == 1
    assert outcomes[0]["filled_volume"] == Decimal("0.0002")
    assert state.get_pending_orders() == {}
    assert wallet.asset_available == Decimal("0.0002")
    assert wallet.krw_available < Decimal("100000")


def test_reconcile_pending_live_order_cancels_unfilled_next_tick():
    root = Path(tempfile.mkdtemp())
    settings = _live_settings(root)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    registry = AgentRegistry()
    broker = _LiveBroker()

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=broker,
        perf_tracker=PerformanceTracker(store),
        scorer=PerformanceScorer(),
        pos_tracker=_PositionTracker(),
        store=store,
        state=state,
    )

    orchestrator.execute_order(
        OrderIntent(
            market="KRW-BTC",
            side="bid",
            volume=Decimal("0.0002"),
            price=Decimal("100001000"),
            agent_id="session_sma_main",
            reason="session_consensus_buy",
        ),
        wallet_id="session_sma_main",
        initial_capital_krw=Decimal("100000"),
        extra_log={"session_id": "session_sma_main", "main_agent_id": "sma_agent"},
    )

    outcomes = orchestrator.reconcile_pending_live_orders(current_tick=1)
    order_logs = store.read_lines("orders")

    assert outcomes == []
    assert broker.cancelled == ["live-1"]
    assert state.get_pending_orders() == {}
    assert order_logs[-1]["message"] == "cancelled_unfilled_next_tick"
