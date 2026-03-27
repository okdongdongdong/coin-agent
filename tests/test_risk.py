from decimal import Decimal
from pathlib import Path
import tempfile

from coin_agent.config.settings import Settings
from coin_agent.models.trading import OrderIntent
from coin_agent.risk.agent_risk import AgentRiskManager
from coin_agent.risk.portfolio_risk import PortfolioRiskManager
from coin_agent.risk.circuit_breaker import CircuitBreaker
from coin_agent.storage.jsonl_store import JsonlStore
from coin_agent.storage.state_store import StateStore


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        access_key="", secret_key="", markets=["KRW-BTC"], dry_run=True,
        log_level="WARNING", data_dir=Path(tempfile.mkdtemp()), loop_interval_sec=60,
        request_timeout_sec=10, candle_unit=1, candle_count=200,
        paper_krw_balance=Decimal("300000"), fee_rate=Decimal("0.0025"),
        order_cooldown_sec=0, max_daily_loss_krw=Decimal("30000"),
        max_total_loss_krw=Decimal("90000"), max_position_pct=Decimal("40"),
        min_agent_trades=10, bench_threshold=30.0, min_active_agents=2,
        rebalance_interval=50, softmax_temperature=2.0,
        min_alloc_pct=Decimal("5"), max_alloc_pct=Decimal("40"),
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestAgentRisk:
    def test_pass(self):
        s = _make_settings()
        arm = AgentRiskManager(s)
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.001"), Decimal("10000000"), agent_id="test")
        r = arm.check(intent, Decimal("100000"), Decimal("100000"), Decimal("0"))
        assert r.allowed

    def test_position_limit(self):
        s = _make_settings()
        arm = AgentRiskManager(s)
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.01"), Decimal("85000000"), agent_id="test")
        r = arm.check(intent, Decimal("100000"), Decimal("100000"), Decimal("0"))
        assert not r.allowed
        assert "position_limit" in r.reason

    def test_order_too_small(self):
        s = _make_settings()
        arm = AgentRiskManager(s)
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.00000001"), Decimal("1000"), agent_id="test")
        r = arm.check(intent, Decimal("100000"), Decimal("100000"), Decimal("0"))
        assert not r.allowed
        assert "order_too_small" in r.reason

    def test_cooldown(self):
        s = _make_settings(order_cooldown_sec=9999)
        arm = AgentRiskManager(s)
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.001"), Decimal("10000000"), agent_id="test")
        # First call sets the cooldown
        arm.check(intent, Decimal("100000"), Decimal("100000"), Decimal("0"))
        # Second call should be blocked
        r = arm.check(intent, Decimal("100000"), Decimal("100000"), Decimal("0"))
        assert not r.allowed
        assert "cooldown" in r.reason


class TestPortfolioRisk:
    def test_pass(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        prm = PortfolioRiskManager(s, store)
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.001"), Decimal("10000000"))
        r = prm.check(intent, Decimal("200000"), Decimal("100000"))
        assert r.allowed

    def test_max_exposure(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        prm = PortfolioRiskManager(s, store)
        intent = OrderIntent("KRW-BTC", "bid", Decimal("1"), Decimal("300000"))
        r = prm.check(intent, Decimal("100000"), Decimal("250000"))
        assert not r.allowed
        assert "max_exposure" in r.reason


class TestCircuitBreaker:
    def test_normal(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        state = StateStore(store)
        cb = CircuitBreaker(s, store, state)
        r = cb.check(Decimal("300000"), Decimal("0"))
        assert not r.tripped

    def test_daily_loss_trip(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        state = StateStore(store)
        cb = CircuitBreaker(s, store, state)
        r = cb.check(Decimal("270000"), Decimal("-35000"))
        assert r.tripped
        assert "daily_loss" in r.reason

    def test_total_loss_trip(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        state = StateStore(store)
        cb = CircuitBreaker(s, store, state)
        r = cb.check(Decimal("200000"), Decimal("0"))
        assert r.tripped
        assert "total_loss" in r.reason

    def test_consecutive_losses_trip(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        state = StateStore(store)
        cb = CircuitBreaker(s, store, state)
        r = cb.check(Decimal("300000"), Decimal("0"), max_consecutive_losses=5)
        assert r.tripped
        assert "consecutive_losses" in r.reason

    def test_kill_switch(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        state = StateStore(store)
        kill_path = s.data_dir / "KILL"
        kill_path.touch()
        cb = CircuitBreaker(s, store, state)
        r = cb.check(Decimal("300000"), Decimal("0"))
        assert r.tripped
        assert "kill_switch" in r.reason
        kill_path.unlink()

    def test_reset(self):
        s = _make_settings()
        store = JsonlStore(s.data_dir)
        state = StateStore(store)
        cb = CircuitBreaker(s, store, state)
        cb._record_trip("test")
        assert cb.is_tripped()
        cb.reset()
        assert not cb.is_tripped()
