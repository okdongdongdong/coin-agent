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
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=4, max_sessions=4)
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
    assert decisions_by_id["session_sma_main"]["action"] == "buy"
    assert decisions_by_id["session_sma_main"]["intent"] is not None
    assert decisions_by_id["session_breakout_main"]["action"] == "hold"
    assert decisions_by_id["session_breakout_main"]["intent"] is None


def test_build_session_decisions_uses_lowered_consensus_threshold():
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
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=4, max_sessions=4)
    sessions = session_mgr.initialize_sessions()

    decisions = orchestrator.build_session_decisions(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "hold", 0.0, "hold"),
            "momentum_agent": Signal("momentum_agent", "buy", 0.4, "momentum"),
            "mean_reversion_agent": Signal("mean_reversion_agent", "buy", 0.55, "mean"),
            "breakout_agent": Signal("breakout_agent", "hold", 0.0, "hold"),
        },
        sessions=sessions,
    )

    decisions_by_id = {item["session_id"]: item for item in decisions}
    assert decisions_by_id["session_momentum_main"]["action"] == "buy"
    assert decisions_by_id["session_momentum_main"]["intent"] is not None
    assert "buy_w=0.270" in decisions_by_id["session_momentum_main"]["reason"]


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
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=4, max_sessions=4)
    sessions = session_mgr.initialize_sessions()

    decisions = orchestrator.build_session_decisions(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "hold", 0.0, "hold"),
            "momentum_agent": Signal("momentum_agent", "hold", 0.0, "hold"),
            "mean_reversion_agent": Signal("mean_reversion_agent", "buy", 0.55, "lower band"),
            "breakout_agent": Signal("breakout_agent", "hold", 0.0, "hold"),
        },
        sessions=sessions,
    )

    decisions_by_id = {item["session_id"]: item for item in decisions}
    mean_session = decisions_by_id["session_mean_reversion_main"]
    assert mean_session["action"] == "buy"
    assert mean_session["intent"] is not None
    assert "main_override_buy" in mean_session["reason"]


def test_build_session_decisions_caps_order_size_to_twenty_percent():
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
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=4, max_sessions=4)
    sessions = session_mgr.initialize_sessions()

    decisions = orchestrator.build_session_decisions(
        snapshot=_Snapshot(),
        tech_signals={
            "sma_agent": Signal("sma_agent", "hold", 0.0, "hold"),
            "momentum_agent": Signal("momentum_agent", "hold", 0.0, "hold"),
            "mean_reversion_agent": Signal("mean_reversion_agent", "buy", 0.80, "lower band"),
            "breakout_agent": Signal("breakout_agent", "hold", 0.0, "hold"),
        },
        sessions=sessions,
    )

    mean_session = {item["session_id"]: item for item in decisions}["session_mean_reversion_main"]
    assert mean_session["intent"] is not None
    assert mean_session["intent"].volume == Decimal("0.00018750")


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


def test_build_session_decisions_use_one_tick_buy_price():
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
    session_mgr = SessionManager(settings=settings, store=store, min_sessions=4, max_sessions=4)
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

    sma_session = {item["session_id"]: item for item in decisions}["session_sma_main"]
    assert sma_session["intent"] is not None
    assert sma_session["intent"].price == Decimal("100001000")


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
