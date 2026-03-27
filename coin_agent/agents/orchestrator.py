from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config.settings import Settings
from ..exchange.market_data import MarketSnapshot
from ..models.agent import Signal, AgentState
from ..models.trading import OrderIntent
from ..models.performance import TradeRecord
from ..agents.registry import AgentRegistry
from ..agents.allocator import softmax_allocate
from ..execution.broker import PaperBroker
from ..execution.position_tracker import PositionTracker
from ..performance.tracker import PerformanceTracker
from ..performance.scorer import PerformanceScorer
from ..storage.jsonl_store import JsonlStore
from ..storage.state_store import StateStore
from ..utils.math_helpers import round_down, round_price

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        registry: AgentRegistry,
        broker: PaperBroker,
        perf_tracker: PerformanceTracker,
        scorer: PerformanceScorer,
        pos_tracker: PositionTracker,
        store: JsonlStore,
        state: StateStore,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.broker = broker
        self.perf_tracker = perf_tracker
        self.scorer = scorer
        self.pos_tracker = pos_tracker
        self.store = store
        self.state = state
        self._tick_count = 0

    def run_tick(self, snapshot: MarketSnapshot) -> Dict[str, Signal]:
        """Run one tick: collect all technical agent signals."""
        self._tick_count += 1
        signals: Dict[str, Signal] = {}

        for agent in self.registry.active_agents():
            try:
                signal = agent.analyze(snapshot)
                signals[agent.agent_id] = signal
                LOGGER.info(
                    "Signal %s: %s (conf=%.2f) %s",
                    agent.agent_id, signal.action, signal.confidence, signal.reason,
                )
            except Exception as e:
                LOGGER.error("Agent %s error: %s", agent.agent_id, e)
                signals[agent.agent_id] = Signal(
                    agent_id=agent.agent_id,
                    action="hold",
                    confidence=0.0,
                    reason=f"error: {e}",
                )

        # Update position values
        for agent in self.registry.all_agents():
            total = self.pos_tracker.get_total_value(
                agent.agent_id, snapshot.current_price, self.settings.paper_krw_balance / Decimal(str(len(self.registry.all_agents())))
            )
            self.perf_tracker.update_value(agent.agent_id, total)

        # Log signals
        self.store.append("decisions", {
            "tick": self._tick_count,
            "market": snapshot.market,
            "price": str(snapshot.current_price),
            "signals": {k: {"action": v.action, "confidence": v.confidence, "reason": v.reason} for k, v in signals.items()},
        })

        return signals

    def decide(
        self,
        snapshot: MarketSnapshot,
        tech_signals: Dict[str, Signal],
        claude_signals: Optional[Dict[str, Dict[str, Any]]] = None,
        meta_action: Optional[str] = None,
    ) -> Optional[OrderIntent]:
        """Aggregate signals and produce an order intent."""
        all_signals = dict(tech_signals)

        # Merge Claude Code signals (sentiment, pattern)
        if claude_signals:
            for name, sig_data in claude_signals.items():
                all_signals[name] = Signal(
                    agent_id=name,
                    action=sig_data.get("action", "hold"),
                    confidence=float(sig_data.get("confidence", 0.0)),
                    reason=sig_data.get("reason", "claude_analysis"),
                )

        # Calculate weighted vote
        allocations = self._get_current_allocations()
        total_alloc = sum(allocations.values())

        buy_weight = Decimal("0")
        sell_weight = Decimal("0")

        for agent_id, signal in all_signals.items():
            alloc = allocations.get(agent_id, Decimal("0"))
            weight = alloc / total_alloc if total_alloc > 0 else Decimal("1") / Decimal(str(len(all_signals)))
            confidence = Decimal(str(signal.confidence))

            if signal.action == "buy":
                buy_weight += weight * confidence
            elif signal.action == "sell":
                sell_weight += weight * confidence

        # Meta override from Claude Code
        if meta_action and meta_action in ("buy", "sell"):
            action = meta_action
        elif buy_weight > sell_weight and buy_weight > Decimal("0.3"):
            action = "buy"
        elif sell_weight > buy_weight and sell_weight > Decimal("0.3"):
            action = "sell"
        else:
            LOGGER.info("No consensus: buy_w=%.3f sell_w=%.3f -> hold", buy_weight, sell_weight)
            return None

        # Calculate order size
        wallet = self.broker.get_wallet("global")
        price = snapshot.current_price

        if action == "buy":
            max_spend = wallet.krw_available * Decimal("0.3")  # Use 30% of available
            if max_spend < Decimal("5000"):
                LOGGER.info("Insufficient KRW for buy: %s", wallet.krw_available)
                return None
            volume = round_down(max_spend / price, Decimal("0.00000001"))
            if volume <= 0:
                return None
            return OrderIntent(
                market=snapshot.market,
                side="bid",
                volume=volume,
                price=round_price(price * Decimal("0.9995")),  # -0.05% offset
                agent_id="orchestrator",
                reason=f"consensus_buy (w={buy_weight:.3f})",
            )
        else:  # sell
            if wallet.asset_available <= 0:
                LOGGER.info("No asset to sell")
                return None
            volume = wallet.asset_available  # Sell all
            return OrderIntent(
                market=snapshot.market,
                side="ask",
                volume=volume,
                price=round_price(price * Decimal("1.0005")),  # +0.05% offset
                agent_id="orchestrator",
                reason=f"consensus_sell (w={sell_weight:.3f})",
            )

    def execute_order(self, intent: OrderIntent) -> "ExecutionResult":
        """Execute an order through the broker."""
        from ..models.trading import ExecutionResult
        result = self.broker.execute(intent)

        if result.success:
            # Record trade for performance tracking
            wallet = self.broker.get_wallet("global")
            pnl = Decimal("0")
            if intent.side == "ask":
                pnl = (intent.price - wallet.avg_buy_price) * intent.volume

            trade = TradeRecord(
                agent_id=intent.agent_id,
                market=intent.market,
                side=intent.side,
                volume=intent.volume,
                price=intent.price,
                pnl_krw=pnl,
                timestamp=time.time(),
                order_id=result.order_id or "",
            )
            self.perf_tracker.record_trade(trade)
            self.store.append("orders", {
                "order_id": result.order_id,
                "side": intent.side,
                "volume": str(intent.volume),
                "price": str(intent.price),
                "agent_id": intent.agent_id,
                "reason": intent.reason,
                "success": result.success,
            })

        return result

    def rebalance(self, custom_allocations: Optional[Dict[str, float]] = None) -> Dict[str, Decimal]:
        """Rebalance capital across agents."""
        if custom_allocations:
            total = self.settings.paper_krw_balance
            result = {k: Decimal(str(v)) * total for k, v in custom_allocations.items()}
        else:
            all_metrics = self.perf_tracker.all_metrics()
            scores = {}
            for aid, m in all_metrics.items():
                breakdown = self.scorer.score(m)
                scores[aid] = breakdown.composite

            # Include agents without metrics at neutral score
            for aid in self.registry.agent_ids():
                if aid not in scores:
                    scores[aid] = 50.0

            result = softmax_allocate(
                scores=scores,
                total_capital=self.settings.paper_krw_balance,
                temperature=self.settings.softmax_temperature,
                min_alloc_pct=float(self.settings.min_alloc_pct),
                max_alloc_pct=float(self.settings.max_alloc_pct),
                bench_threshold=self.settings.bench_threshold,
                min_active=self.settings.min_active_agents,
            )

        # Update agent states
        for aid, alloc in result.items():
            st = self.registry.get_state(aid)
            if st:
                st.allocated_capital_krw = alloc
                if alloc > 0 and st.phase == "benched":
                    st.phase = "active"
                elif alloc == 0 and st.phase == "active":
                    st.phase = "benched"
                self.registry.set_state(aid, st)

        self.store.append("allocations", {k: str(v) for k, v in result.items()})
        return result

    def _get_current_allocations(self) -> Dict[str, Decimal]:
        states = self.registry.all_states()
        allocs = {aid: st.allocated_capital_krw for aid, st in states.items()}
        if all(v == 0 for v in allocs.values()):
            n = len(allocs) or 1
            equal = self.settings.paper_krw_balance / Decimal(str(n))
            return {aid: equal for aid in allocs}
        return allocs

    def generate_report(self, snapshot: MarketSnapshot, signals: Dict[str, Signal]) -> str:
        """Generate the report text that Claude Code will read."""
        lines = []
        lines.append("=== COIN-AGENT TICK REPORT ===")
        lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S KST')}")
        lines.append(f"Market: {snapshot.market}")
        lines.append(f"Tick: #{self._tick_count}")
        lines.append("")

        # Market Data
        lines.append("--- Market Data ---")
        lines.append(f"Current Price: {snapshot.current_price:,.0f} KRW")
        ticker = snapshot.ticker
        if ticker:
            change_rate = ticker.get("signed_change_rate", 0)
            if change_rate:
                lines.append(f"24h Change: {float(change_rate)*100:+.2f}%")
            acc_vol = ticker.get("acc_trade_volume_24h", "?")
            lines.append(f"Volume (24h): {acc_vol}")
        lines.append("")

        # Technical Indicators
        from ..utils.indicators import sma as calc_sma, rsi as calc_rsi, bollinger_bands as calc_bb, atr as calc_atr
        closes = snapshot.closes
        lines.append("--- Technical Indicators ---")
        if len(closes) >= 20:
            lines.append(f"SMA(5): {calc_sma(closes, 5):,.0f}  SMA(20): {calc_sma(closes, 20):,.0f}")
        if len(closes) >= 15:
            lines.append(f"RSI(14): {calc_rsi(closes, 14):.1f}")
        if len(closes) >= 20:
            u, m, l = calc_bb(closes, 20)
            lines.append(f"BB Upper: {u:,.0f}  Mid: {m:,.0f}  Lower: {l:,.0f}")
        if len(snapshot.highs) >= 15:
            a = calc_atr(snapshot.highs, snapshot.lows, closes, 14)
            lines.append(f"ATR(14): {a:,.0f}")
        lines.append("")

        # Candles summary (last 10)
        lines.append("--- Recent Candles (last 10) ---")
        for c in snapshot.candles[:10]:
            ts = c.get("candle_date_time_kst", "?")
            o = c.get("opening_price", "?")
            h = c.get("high_price", "?")
            lo = c.get("low_price", "?")
            cl = c.get("trade_price", "?")
            v = c.get("candle_acc_trade_volume", "?")
            lines.append(f"  {ts} O={o} H={h} L={lo} C={cl} V={v}")
        lines.append("")

        # Agent Signals
        lines.append("--- Agent Signals ---")
        for aid, sig in signals.items():
            lines.append(f"  {aid:25s} {sig.action.upper():5s} (confidence: {sig.confidence:.2f}, reason: {sig.reason})")
        lines.append("")

        lines.append("--- Awaiting Claude Signals ---")
        lines.append("  sentiment_agent:  PENDING (Claude Code to analyze)")
        lines.append("  pattern_agent:    PENDING (Claude Code to analyze)")
        lines.append("")

        # Portfolio State
        lines.append("--- Portfolio State ---")
        wallet = self.broker.get_wallet("global")
        total = wallet.krw_available + wallet.asset_available * snapshot.current_price
        lines.append(f"Total Value: {total:,.0f} KRW")
        lines.append(f"KRW Available: {wallet.krw_available:,.0f} KRW")
        if wallet.asset_available > 0:
            asset_val = wallet.asset_available * snapshot.current_price
            lines.append(f"Position: {wallet.asset_available} ({asset_val:,.0f} KRW)")
        lines.append(f"Initial Capital: {self.settings.paper_krw_balance:,.0f} KRW")
        pnl = total - self.settings.paper_krw_balance
        pnl_pct = float(pnl / self.settings.paper_krw_balance * 100) if self.settings.paper_krw_balance > 0 else 0
        lines.append(f"P&L: {pnl:+,.0f} KRW ({pnl_pct:+.2f}%)")
        lines.append("")

        # Leaderboard
        lines.append("--- Leaderboard ---")
        all_metrics = self.perf_tracker.all_metrics()
        scored = []
        for aid in self.registry.agent_ids():
            m = all_metrics.get(aid)
            if m:
                sb = self.scorer.score(m)
                scored.append((aid, sb.composite, m.total_pnl_krw))
            else:
                scored.append((aid, 50.0, Decimal("0")))
        scored.sort(key=lambda x: x[1], reverse=True)
        allocs = self._get_current_allocations()
        total_alloc = sum(allocs.values())
        for rank, (aid, score, pnl) in enumerate(scored, 1):
            alloc = allocs.get(aid, Decimal("0"))
            pct = float(alloc / total_alloc * 100) if total_alloc > 0 else 0
            lines.append(f"  #{rank} {aid:25s} Score: {score:5.1f}  Alloc: {pct:4.0f}%  P&L: {pnl:+,.0f}")

        return "\n".join(lines)
