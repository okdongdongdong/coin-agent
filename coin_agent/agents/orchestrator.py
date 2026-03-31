from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config.settings import Settings
from ..exchange.market_data import MarketSnapshot
from ..models.agent import Signal
from ..models.trading import OrderIntent
from ..models.performance import TradeRecord
from ..agents.registry import AgentRegistry
from ..agents.allocator import softmax_allocate
from ..execution.broker import LiveBroker, PaperBroker
from ..execution.position_tracker import PositionTracker
from ..performance.tracker import PerformanceTracker
from ..performance.scorer import PerformanceScorer
from ..storage.jsonl_store import JsonlStore
from ..storage.state_store import StateStore
from ..session.session import SessionState
from ..utils.math_helpers import krw_tick_size, round_down, round_price

LOGGER = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        registry: AgentRegistry,
        broker: PaperBroker | LiveBroker,
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

    def _global_wallet(self, market: str) -> "WalletSnapshot":
        asset_currency = self.settings.market_asset(market)
        return self.broker.get_wallet("global", asset_currency)

    def trading_wallet(
        self,
        wallet_id: str,
        market: str,
        initial_capital_krw: Optional[Decimal] = None,
    ) -> "WalletSnapshot":
        if wallet_id == "global":
            return self._global_wallet(market)

        asset_currency = self.settings.market_asset(market)
        if self.settings.dry_run:
            default_capital = initial_capital_krw or self.settings.paper_krw_balance
            data = self.state.get_wallet(
                f"paper_{wallet_id}",
                {
                    "krw_available": str(default_capital),
                    "asset_available": "0",
                    "avg_buy_price": "0",
                },
            )
            if not data:
                data = {
                    "krw_available": str(default_capital),
                    "asset_available": "0",
                    "avg_buy_price": "0",
                }
                self.state.save_wallet(f"paper_{wallet_id}", data)
            from ..models.trading import WalletSnapshot
            return WalletSnapshot(
                krw_available=Decimal(str(data.get("krw_available", "0"))),
                asset_available=Decimal(str(data.get("asset_available", "0"))),
                avg_buy_price=Decimal(str(data.get("avg_buy_price", "0"))),
            )

        default_capital = initial_capital_krw or self.settings.paper_krw_balance
        data = self.state.get_wallet(
            f"live_{wallet_id}",
            {
                "krw_available": str(default_capital),
                "asset_available": "0",
                "avg_buy_price": "0",
            },
        )
        from ..models.trading import WalletSnapshot
        return WalletSnapshot(
            krw_available=Decimal(str(data.get("krw_available", "0"))),
            asset_available=Decimal(str(data.get("asset_available", "0"))),
            avg_buy_price=Decimal(str(data.get("avg_buy_price", "0"))),
        )

    def aggregate_session_wallets(
        self,
        market: str,
        sessions: List[SessionState],
    ) -> "WalletSnapshot":
        from ..models.trading import WalletSnapshot

        total_krw = Decimal("0")
        total_asset = Decimal("0")
        weighted_avg_numerator = Decimal("0")

        for session in sessions:
            wallet = self.trading_wallet(
                session.config.session_id,
                market,
                session.config.initial_capital_krw,
            )
            total_krw += wallet.krw_available
            total_asset += wallet.asset_available
            weighted_avg_numerator += wallet.asset_available * wallet.avg_buy_price

        avg_buy_price = Decimal("0")
        if total_asset > 0:
            avg_buy_price = weighted_avg_numerator / total_asset

        return WalletSnapshot(
            krw_available=total_krw,
            asset_available=total_asset,
            avg_buy_price=avg_buy_price,
        )

    def current_allocations(self) -> Dict[str, Decimal]:
        return self._get_current_allocations()

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
        for agent_id, signal in signals.items():
            self.store.append("signals", {
                "tick": self._tick_count,
                "market": snapshot.market,
                "price": str(snapshot.current_price),
                "agent_id": agent_id,
                "action": signal.action,
                "confidence": signal.confidence,
                "reason": signal.reason,
            })

        return signals

    def decide(
        self,
        snapshot: MarketSnapshot,
        tech_signals: Dict[str, Signal],
        claude_signals: Optional[Dict[str, Dict[str, Any]]] = None,
        meta_action: Optional[str] = None,
    ) -> Optional[OrderIntent]:
        """Build a single weighted-consensus order intent."""
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

        allocations = self._get_current_allocations()
        normalized_weights = self._normalize_weights(allocations)
        summary = self._summarize_consensus(
            signals=all_signals,
            weights=normalized_weights,
            threshold=Decimal("0.25"),
            meta_action=meta_action,
        )

        if summary["action"] == "hold":
            LOGGER.info(
                "No consensus -> hold (buy_w=%.3f sell_w=%.3f)",
                float(summary["buy_weight"]),
                float(summary["sell_weight"]),
            )
            return None

        action = str(summary["action"])
        leader_agent_id = str(summary["leader_agent_id"])

        wallet = self._global_wallet(snapshot.market)
        price = snapshot.current_price
        allocation_budget = allocations.get(leader_agent_id, Decimal("0"))
        if allocation_budget <= 0:
            allocation_budget = wallet.krw_available / Decimal(str(max(len(all_signals), 1)))

        if action == "buy":
            max_spend = min(wallet.krw_available, allocation_budget)
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
                price=self._price_with_one_tick_edge(price, "bid"),
                agent_id=leader_agent_id,
                reason=(
                    f"consensus_buy (buy_w={summary['buy_weight']:.3f}, "
                    f"sell_w={summary['sell_weight']:.3f}, leader={leader_agent_id})"
                ),
            )
        else:  # sell
            if wallet.asset_available <= 0:
                LOGGER.info("No asset to sell")
                return None
            max_notional = min(wallet.asset_available * price, allocation_budget)
            if max_notional < Decimal("5000"):
                LOGGER.info("Sellable notional too small: %s", max_notional)
                return None
            volume = round_down(max_notional / price, Decimal("0.00000001"))
            if volume <= 0:
                return None
            return OrderIntent(
                market=snapshot.market,
                side="ask",
                volume=volume,
                price=self._price_with_one_tick_edge(price, "ask"),
                agent_id=leader_agent_id,
                reason=(
                    f"consensus_sell (buy_w={summary['buy_weight']:.3f}, "
                    f"sell_w={summary['sell_weight']:.3f}, leader={leader_agent_id})"
                ),
            )

    def build_session_decisions(
        self,
        snapshot: MarketSnapshot,
        tech_signals: Dict[str, Signal],
        sessions: List[SessionState],
    ) -> List[Dict[str, Any]]:
        decisions: List[Dict[str, Any]] = []
        for session in sessions:
            session_signals = {
                agent_id: tech_signals[agent_id]
                for agent_id in session.config.agent_ids
                if agent_id in tech_signals
            }
            weights = {
                agent_id: Decimal(str(session.config.vote_weights.get(agent_id, 0.0)))
                for agent_id in session.config.agent_ids
            }
            threshold = Decimal(
                str(session.config.hyperparams.get("confidence_threshold", 0.3))
            )
            summary = self._summarize_consensus(
                signals=session_signals,
                weights=self._normalize_weights(weights),
                threshold=threshold,
                main_agent_id=session.config.main_agent_id,
                override_threshold=Decimal(
                    str(session.config.hyperparams.get("main_agent_override_threshold", 0.55))
                ),
            )
            wallet = self.trading_wallet(
                session.config.session_id,
                snapshot.market,
                session.config.initial_capital_krw,
            )
            intent = self._build_session_intent(
                snapshot=snapshot,
                session=session,
                wallet=wallet,
                summary=summary,
            )
            decisions.append({
                "session_id": session.config.session_id,
                "main_agent_id": session.config.main_agent_id,
                "action": summary["action"],
                "confidence": float(summary["confidence"]),
                "buy_vote": summary["buy_weight"],
                "sell_vote": summary["sell_weight"],
                "leader_agent_id": summary["leader_agent_id"],
                "reason": summary["reason"],
                "intent": intent,
            })
        return decisions

    def execute_order(
        self,
        intent: OrderIntent,
        wallet_id: Optional[str] = None,
        initial_capital_krw: Optional[Decimal] = None,
        extra_log: Optional[Dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """Execute an order through the broker."""
        from ..models.trading import ExecutionResult

        execution_wallet_id = wallet_id or intent.agent_id or "global"
        broker_wallet_id = execution_wallet_id if self.settings.dry_run else "global"
        pre_wallet = self.trading_wallet(
            execution_wallet_id,
            intent.market,
            initial_capital_krw,
        )

        result = self.broker.execute(intent, broker_wallet_id)

        order_record = {
            "order_id": result.order_id,
            "side": intent.side,
            "volume": str(intent.volume),
            "price": str(intent.price),
            "agent_id": intent.agent_id,
            "wallet_id": execution_wallet_id,
            "reason": intent.reason,
            "success": result.success,
            "message": result.message,
            "mode": result.mode,
        }
        if extra_log:
            order_record.update(extra_log)
        self.store.append("orders", order_record)

        if result.success and result.mode == "live" and result.order_id:
            self._save_pending_live_order(
                order_id=result.order_id,
                intent=intent,
                wallet_id=execution_wallet_id,
                initial_capital_krw=initial_capital_krw,
                extra_log=extra_log,
            )

        if result.success and result.mode == "paper" and result.message == "filled":
            self._record_trade_fill(
                wallet_id=execution_wallet_id,
                intent=intent,
                order_id=result.order_id or "",
                pre_wallet=pre_wallet,
            )

        return result

    def has_pending_live_order(self, wallet_id: str) -> bool:
        if self.settings.dry_run:
            return False
        return any(
            str(entry.get("wallet_id", "")) == wallet_id
            for entry in self.state.get_pending_orders().values()
        )

    def reconcile_pending_live_orders(self, current_tick: int) -> List[Dict[str, Any]]:
        if (
            self.settings.dry_run
            or not hasattr(self.broker, "get_order")
            or not hasattr(self.broker, "cancel_order")
        ):
            return []

        pending_orders = dict(self.state.get_pending_orders())
        if not pending_orders:
            return []

        outcomes: List[Dict[str, Any]] = []
        updated_orders = dict(pending_orders)
        dirty = False

        for order_id, pending in pending_orders.items():
            created_tick = int(pending.get("tick", 0))
            if current_tick <= created_tick:
                continue
            try:
                order_data = self.broker.get_order(order_id)
            except Exception as exc:
                LOGGER.warning("Pending order check failed for %s: %s", order_id, exc)
                continue

            outcome_batch, updated_pending = self._handle_pending_live_order(
                order_id=order_id,
                pending=pending,
                order_data=order_data,
                current_tick=current_tick,
            )
            if outcome_batch:
                outcomes.extend(outcome_batch)
            if updated_pending is None:
                updated_orders.pop(order_id, None)
                dirty = True
            elif updated_pending != pending:
                updated_orders[order_id] = updated_pending
                dirty = True

        if dirty:
            self.state.save_pending_orders(updated_orders)
        return outcomes

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
        allocs = {aid: Decimal(str(st.allocated_capital_krw)) for aid, st in states.items()}
        if all(v == 0 for v in allocs.values()):
            n = len(allocs) or 1
            equal = self.settings.paper_krw_balance / Decimal(str(n))
            return {aid: equal for aid in allocs}
        return allocs

    def generate_report(
        self,
        snapshot: MarketSnapshot,
        signals: Dict[str, Signal],
        session_decisions: Optional[List[Dict[str, Any]]] = None,
        sessions: Optional[List[SessionState]] = None,
    ) -> str:
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

        if session_decisions:
            lines.append("--- Session Consensus ---")
            for decision in session_decisions:
                lines.append(
                    "  "
                    f"{decision['session_id']:25s} "
                    f"{str(decision['action']).upper():5s} "
                    f"(main={decision['main_agent_id']}, "
                    f"buy={Decimal(str(decision['buy_vote'])):.3f}, "
                    f"sell={Decimal(str(decision['sell_vote'])):.3f}, "
                    f"leader={decision['leader_agent_id'] or '-'})"
                )
            lines.append("")

        # Portfolio State
        lines.append("--- Portfolio State ---")
        if sessions:
            wallet = self.aggregate_session_wallets(snapshot.market, sessions)
        else:
            wallet = self._global_wallet(snapshot.market)
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

    def _normalize_weights(self, weights: Dict[str, Decimal]) -> Dict[str, Decimal]:
        positive_weights = {k: max(Decimal("0"), v) for k, v in weights.items()}
        total = sum(positive_weights.values())
        if total <= 0:
            n = len(positive_weights) or 1
            equal = Decimal("1") / Decimal(str(n))
            return {k: equal for k in positive_weights}
        return {k: v / total for k, v in positive_weights.items()}

    def _summarize_consensus(
        self,
        signals: Dict[str, Signal],
        weights: Dict[str, Decimal],
        threshold: Decimal,
        meta_action: Optional[str] = None,
        main_agent_id: Optional[str] = None,
        override_threshold: Optional[Decimal] = None,
    ) -> Dict[str, Any]:
        buy_weight = Decimal("0")
        sell_weight = Decimal("0")
        buy_leader = {"agent_id": "", "score": Decimal("0"), "reason": ""}
        sell_leader = {"agent_id": "", "score": Decimal("0"), "reason": ""}

        for agent_id, signal in signals.items():
            action = str(signal.action).lower()
            if action not in {"buy", "sell"}:
                continue
            weight = Decimal(str(weights.get(agent_id, Decimal("0"))))
            confidence = Decimal(str(signal.confidence))
            weighted_vote = weight * confidence
            if action == "buy":
                buy_weight += weighted_vote
                if weighted_vote >= buy_leader["score"]:
                    buy_leader = {
                        "agent_id": agent_id,
                        "score": weighted_vote,
                        "reason": signal.reason,
                    }
            else:
                sell_weight += weighted_vote
                if weighted_vote >= sell_leader["score"]:
                    sell_leader = {
                        "agent_id": agent_id,
                        "score": weighted_vote,
                        "reason": signal.reason,
                    }

        action = "hold"
        leader = {"agent_id": "", "reason": ""}
        confidence = max(buy_weight, sell_weight)
        override_applied = False

        if main_agent_id and override_threshold is not None:
            main_signal = signals.get(main_agent_id)
            if main_signal is not None:
                main_action = str(main_signal.action).lower()
                main_confidence = Decimal(str(main_signal.confidence))
                if (
                    main_action in {"buy", "sell"}
                    and main_confidence >= override_threshold
                    and (meta_action is None or meta_action == main_action)
                ):
                    action = main_action
                    leader = {"agent_id": main_agent_id, "reason": main_signal.reason}
                    confidence = main_confidence
                    override_applied = True

        if not override_applied:
            if meta_action == "buy":
                if buy_weight >= threshold:
                    action = "buy"
                    leader = buy_leader
                    confidence = buy_weight
            elif meta_action == "sell":
                if sell_weight >= threshold:
                    action = "sell"
                    leader = sell_leader
                    confidence = sell_weight
            else:
                if buy_weight >= threshold and buy_weight > sell_weight:
                    action = "buy"
                    leader = buy_leader
                    confidence = buy_weight
                elif sell_weight >= threshold and sell_weight > buy_weight:
                    action = "sell"
                    leader = sell_leader
                    confidence = sell_weight

        if action == "hold":
            reason = f"hold (buy_w={buy_weight:.3f}, sell_w={sell_weight:.3f})"
        elif override_applied:
            reason = (
                f"main_override_{action} (main={main_agent_id}, "
                f"conf={confidence:.3f}, buy_w={buy_weight:.3f}, sell_w={sell_weight:.3f}, "
                f"reason={leader['reason'] or '-'})"
            )
        else:
            reason = (
                f"{action} (buy_w={buy_weight:.3f}, sell_w={sell_weight:.3f}, "
                f"leader={leader['agent_id'] or '-'}: {leader['reason'] or '-'})"
            )

        return {
            "action": action,
            "confidence": confidence,
            "buy_weight": buy_weight,
            "sell_weight": sell_weight,
            "leader_agent_id": leader.get("agent_id", ""),
            "reason": reason,
            "override_applied": override_applied,
        }

    def _build_session_intent(
        self,
        snapshot: MarketSnapshot,
        session: SessionState,
        wallet: "WalletSnapshot",
        summary: Dict[str, Any],
    ) -> Optional[OrderIntent]:
        action = str(summary["action"])
        if action not in {"buy", "sell"}:
            return None

        price = snapshot.current_price
        strength = Decimal(str(summary["confidence"]))
        leader_agent_id = str(summary["leader_agent_id"])
        order_fraction_target = Decimal(
            str(session.config.hyperparams.get("order_fraction", 0.2))
        )
        order_fraction_max = Decimal(
            str(session.config.hyperparams.get("order_fraction_max", 0.25))
        )
        trade_fraction = min(
            max(Decimal("0"), strength),
            max(order_fraction_target, order_fraction_max),
        )

        if action == "buy":
            max_spend = wallet.krw_available * trade_fraction
            if max_spend < Decimal("5000"):
                return None
            volume = round_down(max_spend / price, Decimal("0.00000001"))
            if volume <= 0:
                return None
            return OrderIntent(
                market=snapshot.market,
                side="bid",
                volume=volume,
                price=self._price_with_one_tick_edge(price, "bid"),
                agent_id=session.config.session_id,
                reason=(
                    f"session_consensus_buy ({session.config.session_id}, "
                    f"buy_w={summary['buy_weight']:.3f}, sell_w={summary['sell_weight']:.3f}, "
                    f"leader={leader_agent_id})"
                ),
            )

        if wallet.asset_available <= 0:
            return None
        volume = round_down(wallet.asset_available * trade_fraction, Decimal("0.00000001"))
        if volume <= 0 or volume * price < Decimal("5000"):
            return None
        return OrderIntent(
            market=snapshot.market,
            side="ask",
            volume=volume,
            price=self._price_with_one_tick_edge(price, "ask"),
            agent_id=session.config.session_id,
            reason=(
                f"session_consensus_sell ({session.config.session_id}, "
                f"buy_w={summary['buy_weight']:.3f}, sell_w={summary['sell_weight']:.3f}, "
                f"leader={leader_agent_id})"
            ),
        )

    def _handle_pending_live_order(
        self,
        order_id: str,
        pending: Dict[str, Any],
        order_data: Dict[str, Any],
        current_tick: int,
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        from ..models.trading import WalletSnapshot

        outcomes: List[Dict[str, Any]] = []
        wallet_id = str(pending.get("wallet_id", "global"))
        market = str(pending.get("market", ""))
        side = str(pending.get("side", ""))
        agent_id = str(pending.get("agent_id", ""))
        reason = str(pending.get("reason", ""))
        price = Decimal(str(order_data.get("price", pending.get("price", "0"))))
        requested_volume = Decimal(str(pending.get("volume", "0")))
        executed_volume = Decimal(str(order_data.get("executed_volume", "0")))
        applied_volume = Decimal(str(pending.get("applied_volume", "0")))
        remaining_volume = Decimal(
            str(order_data.get("remaining_volume", requested_volume - executed_volume))
        )
        if remaining_volume < 0:
            remaining_volume = Decimal("0")
        fill_delta = executed_volume - applied_volume
        if fill_delta < 0:
            fill_delta = Decimal("0")
        state = str(order_data.get("state", ""))

        if fill_delta > 0:
            initial_capital_krw = self._optional_decimal(pending.get("initial_capital_krw"))
            fill_intent = OrderIntent(
                market=market,
                side=side,
                volume=fill_delta,
                price=price,
                agent_id=agent_id,
                reason=reason,
            )
            pre_wallet: Optional[WalletSnapshot] = None
            profitable = None
            if wallet_id != "global":
                pre_wallet = self.trading_wallet(wallet_id, market, initial_capital_krw)
                if side == "ask" and pre_wallet.avg_buy_price > 0:
                    profitable = fill_intent.price > pre_wallet.avg_buy_price
                self._apply_virtual_live_fill(
                    wallet_id=wallet_id,
                    intent=fill_intent,
                    current_wallet=pre_wallet,
                )
            self._record_trade_fill(
                wallet_id=wallet_id,
                intent=fill_intent,
                order_id=order_id,
                pre_wallet=pre_wallet,
            )
            self.store.append(
                "orders",
                self._pending_order_log(
                    pending,
                    order_id=order_id,
                    message="partial_fill_on_check" if remaining_volume > 0 else "filled_on_check",
                    success=True,
                    extra={
                        "filled_volume": str(fill_delta),
                        "executed_volume": str(executed_volume),
                        "remaining_volume": str(remaining_volume),
                        "state": state,
                        "checked_tick": current_tick,
                    },
                ),
            )
            outcomes.append(
                {
                    "wallet_id": wallet_id,
                    "session_id": pending.get("session_id", wallet_id),
                    "main_agent_id": pending.get("main_agent_id", ""),
                    "order_id": order_id,
                    "filled_volume": fill_delta,
                    "profitable": profitable,
                    "state": state,
                }
            )

        if state in {"done", "cancel"} or remaining_volume <= 0:
            if executed_volume <= 0 and state == "cancel":
                self.store.append(
                    "orders",
                    self._pending_order_log(
                        pending,
                        order_id=order_id,
                        message="cancelled_without_fill",
                        success=False,
                        extra={
                            "filled_volume": "0",
                            "executed_volume": str(executed_volume),
                            "remaining_volume": str(remaining_volume),
                            "state": state,
                            "checked_tick": current_tick,
                        },
                    ),
                )
            return outcomes, None

        try:
            self.broker.cancel_order(order_id)
            self.store.append(
                "orders",
                self._pending_order_log(
                    pending,
                    order_id=order_id,
                    message="cancelled_partial_next_tick" if executed_volume > 0 else "cancelled_unfilled_next_tick",
                    success=executed_volume > 0,
                    extra={
                        "filled_volume": str(executed_volume),
                        "executed_volume": str(executed_volume),
                        "remaining_volume": str(remaining_volume),
                        "state": "cancel",
                        "checked_tick": current_tick,
                    },
                ),
            )
            return outcomes, None
        except Exception as exc:
            LOGGER.warning("Pending order cancel failed for %s: %s", order_id, exc)
            pending_copy = dict(pending)
            pending_copy["applied_volume"] = str(executed_volume)
            pending_copy["last_checked_tick"] = current_tick
            self.store.append(
                "orders",
                self._pending_order_log(
                    pending_copy,
                    order_id=order_id,
                    message=f"cancel_error:{exc}",
                    success=False,
                    extra={
                        "filled_volume": str(executed_volume),
                        "executed_volume": str(executed_volume),
                        "remaining_volume": str(remaining_volume),
                        "state": state or "wait",
                        "checked_tick": current_tick,
                    },
                ),
            )
            return outcomes, pending_copy

    def _price_with_one_tick_edge(self, price: Decimal, side: str) -> Decimal:
        tick = krw_tick_size(price)
        if side == "bid":
            return round_price(price + tick, tick)
        if side == "ask":
            return max(tick, round_price(price - tick, tick))
        return round_price(price, tick)

    def _save_pending_live_order(
        self,
        order_id: str,
        intent: OrderIntent,
        wallet_id: str,
        initial_capital_krw: Optional[Decimal],
        extra_log: Optional[Dict[str, Any]],
    ) -> None:
        pending_orders = dict(self.state.get_pending_orders())
        pending_entry: Dict[str, Any] = {
            "order_id": order_id,
            "wallet_id": wallet_id,
            "market": intent.market,
            "side": intent.side,
            "volume": str(intent.volume),
            "price": str(intent.price),
            "agent_id": intent.agent_id,
            "reason": intent.reason,
            "tick": self._tick_count,
            "submitted_at": time.time(),
            "applied_volume": "0",
        }
        if initial_capital_krw is not None:
            pending_entry["initial_capital_krw"] = str(initial_capital_krw)
        if extra_log:
            pending_entry.update(extra_log)
        pending_orders[order_id] = pending_entry
        self.state.save_pending_orders(pending_orders)

    def _pending_order_log(
        self,
        pending: Dict[str, Any],
        order_id: str,
        message: str,
        success: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "order_id": order_id,
            "side": pending.get("side"),
            "volume": pending.get("volume"),
            "price": pending.get("price"),
            "agent_id": pending.get("agent_id"),
            "wallet_id": pending.get("wallet_id"),
            "reason": pending.get("reason"),
            "success": success,
            "message": message,
            "mode": "live",
        }
        for key in ("session_id", "main_agent_id", "leader_agent_id"):
            if key in pending:
                record[key] = pending[key]
        if extra:
            record.update(extra)
        return record

    def _optional_decimal(self, value: Any) -> Optional[Decimal]:
        if value in (None, ""):
            return None
        return Decimal(str(value))

    def _apply_virtual_live_fill(
        self,
        wallet_id: str,
        intent: OrderIntent,
        current_wallet: "WalletSnapshot",
    ) -> None:
        notional = intent.price * intent.volume
        fee_rate = self.settings.fee_rate

        if intent.side == "bid":
            total_spend = notional * (Decimal("1") + fee_rate)
            new_asset = current_wallet.asset_available + intent.volume
            avg_price = current_wallet.avg_buy_price
            if new_asset > 0:
                avg_price = (
                    current_wallet.asset_available * current_wallet.avg_buy_price + notional
                ) / new_asset
            state = {
                "krw_available": str(current_wallet.krw_available - total_spend),
                "asset_available": str(new_asset),
                "avg_buy_price": str(avg_price),
            }
        else:
            remaining = current_wallet.asset_available - intent.volume
            total_receive = notional * (Decimal("1") - fee_rate)
            state = {
                "krw_available": str(current_wallet.krw_available + total_receive),
                "asset_available": str(max(Decimal("0"), remaining)),
                "avg_buy_price": str(
                    Decimal("0") if remaining <= 0 else current_wallet.avg_buy_price
                ),
            }

        self.state.save_wallet(f"live_{wallet_id}", state)

    def _record_trade_fill(
        self,
        wallet_id: str,
        intent: OrderIntent,
        order_id: str,
        pre_wallet: Optional["WalletSnapshot"],
    ) -> None:
        pnl = Decimal("0")
        if intent.side == "ask" and pre_wallet is not None:
            pnl = (intent.price - pre_wallet.avg_buy_price) * intent.volume

        trade = TradeRecord(
            agent_id=wallet_id,
            market=intent.market,
            side=intent.side,
            volume=intent.volume,
            price=intent.price,
            pnl_krw=pnl,
            timestamp=time.time(),
            order_id=order_id,
        )
        self.perf_tracker.record_trade(trade)
