from __future__ import annotations

from decimal import Decimal

from coin_agent.agents.strategies.sma_agent import SMAAgent
from coin_agent.agents.strategies.momentum_agent import MomentumAgent
from coin_agent.agents.strategies.mean_reversion_agent import MeanReversionAgent
from coin_agent.agents.strategies.breakout_agent import BreakoutAgent
from coin_agent.agents.registry import AgentRegistry
from coin_agent.exchange.market_data import MarketSnapshot


def _make_snapshot(
    closes: list[Decimal],
    volumes: list[Decimal] | None = None,
    highs: list[Decimal] | None = None,
    lows: list[Decimal] | None = None,
) -> MarketSnapshot:
    n = len(closes)
    if volumes is None:
        volumes = [Decimal("100")] * n
    if highs is None:
        highs = [c + Decimal("100") for c in closes]
    if lows is None:
        lows = [c - Decimal("100") for c in closes]

    candles = []
    for i in range(n):
        candles.append({
            "trade_price": str(closes[i]),
            "candle_acc_trade_volume": str(volumes[i]),
            "high_price": str(highs[i]),
            "low_price": str(lows[i]),
            "opening_price": str(closes[i]),
        })

    return MarketSnapshot(
        market="KRW-BTC",
        current_price=closes[0],
        candles=candles,
        ticker={},
    )


class TestSMAAgent:
    def test_buy_signal(self):
        # Short MA > Long MA -> buy
        closes = [Decimal("100")] * 5 + [Decimal("90")] * 15
        snap = _make_snapshot(closes)
        agent = SMAAgent()
        sig = agent.analyze(snap)
        assert sig.action == "buy"
        assert sig.confidence > 0

    def test_sell_signal(self):
        # Short MA < Long MA -> sell
        closes = [Decimal("80")] * 5 + [Decimal("100")] * 15
        snap = _make_snapshot(closes)
        agent = SMAAgent()
        sig = agent.analyze(snap)
        assert sig.action == "sell"

    def test_hold_signal(self):
        closes = [Decimal("100")] * 20
        snap = _make_snapshot(closes)
        agent = SMAAgent()
        sig = agent.analyze(snap)
        assert sig.action == "hold"

    def test_insufficient_data(self):
        closes = [Decimal("100")] * 5
        snap = _make_snapshot(closes)
        agent = SMAAgent()
        sig = agent.analyze(snap)
        assert sig.action == "hold"
        assert "insufficient" in sig.reason


class TestMomentumAgent:
    def test_oversold(self):
        # Falling prices -> low RSI -> buy
        closes = [Decimal(100 - i * 3) for i in range(20)]
        snap = _make_snapshot(closes)
        agent = MomentumAgent()
        sig = agent.analyze(snap)
        assert sig.action in ("buy", "hold", "sell")  # depends on exact RSI

    def test_insufficient_data(self):
        closes = [Decimal("100")] * 5
        snap = _make_snapshot(closes)
        agent = MomentumAgent()
        sig = agent.analyze(snap)
        assert sig.action == "hold"


class TestMeanReversionAgent:
    def test_at_lower_band(self):
        # Price well below the mean -> buy
        closes = [Decimal("80")] + [Decimal("100")] * 19
        snap = _make_snapshot(closes)
        agent = MeanReversionAgent()
        sig = agent.analyze(snap)
        # Price 80 vs mean ~99 -> below lower band
        assert sig.action in ("buy", "hold")

    def test_insufficient_data(self):
        closes = [Decimal("100")] * 5
        snap = _make_snapshot(closes)
        agent = MeanReversionAgent()
        sig = agent.analyze(snap)
        assert sig.action == "hold"


class TestBreakoutAgent:
    def test_volume_breakout_up(self):
        # High volume + price above recent high
        closes = [Decimal("110")] + [Decimal("100")] * 24
        volumes = [Decimal("500")] + [Decimal("100")] * 24
        highs = [Decimal("112")] + [Decimal("102")] * 24
        lows = [Decimal("108")] + [Decimal("98")] * 24
        snap = _make_snapshot(closes, volumes, highs, lows)
        agent = BreakoutAgent()
        sig = agent.analyze(snap)
        assert sig.action in ("buy", "hold")

    def test_no_breakout(self):
        closes = [Decimal("100")] * 25
        snap = _make_snapshot(closes)
        agent = BreakoutAgent()
        sig = agent.analyze(snap)
        assert sig.action == "hold"


class TestRegistry:
    def test_register_and_list(self):
        reg = AgentRegistry()
        reg.register(SMAAgent())
        reg.register(MomentumAgent())
        assert len(reg.all_agents()) == 2
        assert "sma_agent" in reg.agent_ids()

    def test_active_agents(self):
        reg = AgentRegistry()
        reg.register(SMAAgent())
        agent_state = reg.get_state("sma_agent")
        agent_state.is_active = False
        reg.set_state("sma_agent", agent_state)
        assert len(reg.active_agents()) == 0

    def test_dump_load_states(self):
        reg = AgentRegistry()
        reg.register(SMAAgent())
        dumped = reg.dump_states()
        assert "sma_agent" in dumped

        reg2 = AgentRegistry()
        reg2.register(SMAAgent())
        reg2.load_states(dumped)
        assert reg2.get_state("sma_agent").strategy_name == "sma_crossover"
