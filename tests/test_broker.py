from decimal import Decimal
from pathlib import Path
import tempfile

from coin_agent.config.settings import Settings
from coin_agent.execution.broker import PaperBroker
from coin_agent.models.trading import OrderIntent
from coin_agent.storage.jsonl_store import JsonlStore
from coin_agent.storage.state_store import StateStore


def _make_env() -> tuple:
    data_dir = Path(tempfile.mkdtemp())
    s = Settings(
        access_key="", secret_key="", markets=["KRW-BTC"], dry_run=True,
        log_level="WARNING", data_dir=data_dir, loop_interval_sec=60,
        request_timeout_sec=10, candle_unit=1, candle_count=200,
        paper_krw_balance=Decimal("300000"), fee_rate=Decimal("0.0025"),
        order_cooldown_sec=60, max_daily_loss_krw=Decimal("30000"),
        max_total_loss_krw=Decimal("90000"), max_position_pct=Decimal("40"),
        min_agent_trades=10, bench_threshold=30.0, min_active_agents=2,
        rebalance_interval=50, softmax_temperature=2.0,
        min_alloc_pct=Decimal("5"), max_alloc_pct=Decimal("40"),
        anthropic_api_key="", openai_api_key="",
        claude_model="claude-haiku-4-5-20250610", openai_model="gpt-4o-mini",
        session_enabled=False, session_min_count=3, session_max_count=5,
        session_eval_interval=5, session_min_ticks_before_eval=10,
    )
    store = JsonlStore(data_dir)
    state = StateStore(store)
    broker = PaperBroker(s, state)
    return s, broker


class TestPaperBroker:
    def test_initial_wallet(self):
        s, broker = _make_env()
        w = broker.get_wallet("test_agent")
        assert w.krw_available == Decimal("300000")
        assert w.asset_available == Decimal("0")

    def test_buy(self):
        s, broker = _make_env()
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.001"), Decimal("85000000"), agent_id="test")
        result = broker.execute(intent, "test_agent")
        assert result.success
        assert result.mode == "paper"

        w = broker.get_wallet("test_agent")
        assert w.asset_available == Decimal("0.001")
        assert w.krw_available < Decimal("300000")

    def test_sell(self):
        s, broker = _make_env()
        # Buy first
        buy = OrderIntent("KRW-BTC", "bid", Decimal("0.001"), Decimal("85000000"))
        broker.execute(buy, "test_agent")

        # Then sell
        sell = OrderIntent("KRW-BTC", "ask", Decimal("0.001"), Decimal("86000000"))
        result = broker.execute(sell, "test_agent")
        assert result.success

        w = broker.get_wallet("test_agent")
        assert w.asset_available == Decimal("0")
        assert w.krw_available > Decimal("0")

    def test_insufficient_krw(self):
        s, broker = _make_env()
        intent = OrderIntent("KRW-BTC", "bid", Decimal("1"), Decimal("85000000"))
        result = broker.execute(intent, "test_agent")
        assert not result.success
        assert "insufficient_krw" in result.message

    def test_insufficient_asset(self):
        s, broker = _make_env()
        intent = OrderIntent("KRW-BTC", "ask", Decimal("1"), Decimal("85000000"))
        result = broker.execute(intent, "test_agent")
        assert not result.success
        assert "insufficient_asset" in result.message

    def test_per_agent_wallets(self):
        s, broker = _make_env()
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.001"), Decimal("85000000"))
        broker.execute(intent, "agent_a")

        # agent_b should still have full balance
        w_b = broker.get_wallet("agent_b")
        assert w_b.krw_available == Decimal("300000")

        # agent_a should have less
        w_a = broker.get_wallet("agent_a")
        assert w_a.krw_available < Decimal("300000")

    def test_fee_applied(self):
        s, broker = _make_env()
        intent = OrderIntent("KRW-BTC", "bid", Decimal("0.001"), Decimal("10000000"))
        broker.execute(intent, "test")
        w = broker.get_wallet("test")
        # Cost = 10000 * 1.0025 = 10025
        expected_remaining = Decimal("300000") - Decimal("10000") * Decimal("1.0025")
        assert w.krw_available == expected_remaining
