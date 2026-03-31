from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from ..config.settings import Settings
from ..exchange.bithumb_client import BithumbClient, BithumbAPIError
from ..exchange.market_data import MarketDataCollector
from ..agents.registry import AgentRegistry
from ..agents.orchestrator import Orchestrator
from ..agents.strategies.sma_agent import SMAAgent
from ..agents.strategies.momentum_agent import MomentumAgent
from ..agents.strategies.mean_reversion_agent import MeanReversionAgent
from ..agents.strategies.breakout_agent import BreakoutAgent
from ..execution.broker import LiveBroker, PaperBroker
from ..execution.position_tracker import PositionTracker
from ..performance.tracker import PerformanceTracker
from ..performance.scorer import PerformanceScorer
from ..performance.leaderboard import Leaderboard
from ..risk.agent_risk import AgentRiskManager
from ..risk.circuit_breaker import CircuitBreaker
from ..risk.portfolio_risk import PortfolioRiskManager
from ..storage.jsonl_store import JsonlStore
from ..storage.state_store import StateStore
from ..session.manager import SessionManager
from ..utils.logging_setup import setup_logging

LOGGER = logging.getLogger(__name__)


def build_system(root: Path) -> tuple:
    """Build the complete system from settings."""
    settings = Settings.load(root)
    setup_logging(settings.log_level)

    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    client = BithumbClient(settings)
    collector = MarketDataCollector(client, settings)

    registry = AgentRegistry()
    registry.register(SMAAgent())
    registry.register(MomentumAgent())
    registry.register(MeanReversionAgent())
    registry.register(BreakoutAgent())

    broker = PaperBroker(settings, state) if settings.dry_run else LiveBroker(client)
    pos_tracker = PositionTracker(state)
    perf_tracker = PerformanceTracker(store)
    scorer = PerformanceScorer()
    leaderboard = Leaderboard(scorer, store)

    orchestrator = Orchestrator(
        settings=settings,
        registry=registry,
        broker=broker,
        perf_tracker=perf_tracker,
        scorer=scorer,
        pos_tracker=pos_tracker,
        store=store,
        state=state,
    )

    # Session management (optional)
    session_mgr: Optional[SessionManager] = None
    if settings.session_enabled:
        session_mgr = SessionManager(
            settings=settings,
            store=store,
            min_sessions=settings.session_min_count,
            max_sessions=settings.session_max_count,
        )
        LOGGER.info("Session consensus mode enabled (4 fixed sessions)")

    return settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state, session_mgr, None


def run_loop(root: Path, interval: int = 0, max_ticks: int = 0) -> None:
    """Run the main bot loop."""
    settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state, session_mgr, evolution = build_system(root)
    agent_risk = AgentRiskManager(settings)
    portfolio_risk = PortfolioRiskManager(settings, store)
    circuit_breaker = CircuitBreaker(settings, store, state)

    if interval <= 0:
        interval = settings.loop_interval_sec

    market = settings.primary_market
    tick_count = 0

    LOGGER.info("Starting bot loop: market=%s, interval=%ds, mode=%s", market, interval, "paper" if settings.dry_run else "LIVE")

    # Initial equal allocation
    orchestrator.rebalance()

    # Initialize sessions if enabled
    if session_mgr is not None:
        try:
            session_mgr.load_state()
            sessions = session_mgr.ensure_consensus_layout()
            session_mgr.save_state()
            LOGGER.info("Initialized %d consensus sessions", len(sessions))
        except Exception as e:
            LOGGER.error("Session init error: %s", e, exc_info=True)

    try:
        while True:
            # Check kill switch
            if state.is_kill_switch_active():
                LOGGER.warning("Kill switch active. Stopping.")
                print("Kill switch active. Bot stopped.")
                break

            tick_count += 1
            LOGGER.info("=== Tick %d ===", tick_count)

            try:
                snapshot = collector.snapshot(market)
                pending_outcomes = orchestrator.reconcile_pending_live_orders(tick_count)
                for outcome in pending_outcomes:
                    session_id = str(outcome.get("session_id", ""))
                    if session_mgr is not None and session_id:
                        session_mgr.record_trade(
                            session_id,
                            outcome.get("profitable"),
                        )
                    LOGGER.info(
                        "Pending order reconciled: wallet=%s order=%s filled=%s state=%s",
                        outcome.get("wallet_id", ""),
                        outcome.get("order_id", ""),
                        outcome.get("filled_volume", "0"),
                        outcome.get("state", ""),
                    )
                signals = orchestrator.run_tick(snapshot)

                if session_mgr is not None:
                    portfolio_wallet = orchestrator.aggregate_session_wallets(
                        market=snapshot.market,
                        sessions=session_mgr.active_sessions(),
                    ) if settings.dry_run else broker.get_wallet("global", settings.market_asset(market))
                else:
                    portfolio_wallet = broker.get_wallet("global", settings.market_asset(market))

                wallet = portfolio_wallet
                current_value = wallet.krw_available + wallet.asset_available * snapshot.current_price
                daily_pnl = portfolio_risk.get_daily_pnl()
                orchestrator_metrics = perf_tracker.get_metrics("orchestrator")
                breaker = circuit_breaker.check(
                    current_value=current_value,
                    daily_pnl=daily_pnl,
                    max_consecutive_losses=orchestrator_metrics.consecutive_losses,
                )
                if breaker.tripped:
                    LOGGER.warning("Circuit breaker active: %s", breaker.reason)
                    break

                session_decisions = None
                if session_mgr is not None:
                    session_decisions = orchestrator.build_session_decisions(
                        snapshot=snapshot,
                        tech_signals=signals,
                        sessions=session_mgr.active_sessions(),
                    )
                    for decision in session_decisions:
                        session_mgr.update_session_decision(
                            session_id=decision["session_id"],
                            tick=tick_count,
                            action=str(decision["action"]),
                            confidence=float(decision["confidence"]),
                            buy_vote=decision["buy_vote"],
                            sell_vote=decision["sell_vote"],
                            leader_agent_id=str(decision["leader_agent_id"]),
                            reason=str(decision["reason"]),
                        )
                        store.append("session_decisions", {
                            "tick": tick_count,
                            "market": snapshot.market,
                            "session_id": decision["session_id"],
                            "main_agent_id": decision["main_agent_id"],
                            "action": decision["action"],
                            "confidence": decision["confidence"],
                            "buy_vote": str(decision["buy_vote"]),
                            "sell_vote": str(decision["sell_vote"]),
                            "leader_agent_id": decision["leader_agent_id"],
                            "reason": decision["reason"],
                        })

                    for decision in session_decisions:
                        intent = decision["intent"]
                        if intent is None:
                            continue

                        session = session_mgr.get_session(decision["session_id"])
                        if session is None:
                            continue
                        if orchestrator.has_pending_live_order(session.config.session_id):
                            LOGGER.info(
                                "Skipping %s: pending live order still open",
                                session.config.session_id,
                            )
                            continue
                        session_wallet = orchestrator.trading_wallet(
                            session.config.session_id,
                            snapshot.market,
                            session.config.initial_capital_krw,
                        )
                        session_value = (
                            session_wallet.krw_available
                            + session_wallet.asset_available * snapshot.current_price
                        )
                        total_asset_value = wallet.asset_available * snapshot.current_price
                        portfolio_check = portfolio_risk.check(
                            intent=intent,
                            total_krw=wallet.krw_available,
                            total_asset_value=total_asset_value,
                        )
                        if not portfolio_check.allowed:
                            LOGGER.warning(
                                "Portfolio risk blocked order for %s: %s",
                                session.config.session_id,
                                portfolio_check.reason,
                            )
                            store.append("orders", {
                                "order_id": None,
                                "side": intent.side,
                                "volume": str(intent.volume),
                                "price": str(intent.price),
                                "agent_id": intent.agent_id,
                                "session_id": session.config.session_id,
                                "main_agent_id": session.config.main_agent_id,
                                "reason": intent.reason,
                                "success": False,
                                "message": portfolio_check.reason,
                                "mode": "paper" if settings.dry_run else "live",
                            })
                            continue

                        agent_check = agent_risk.check(
                            intent=intent,
                            agent_capital=session_value,
                            agent_krw=session_wallet.krw_available,
                            agent_asset_value=session_wallet.asset_available * snapshot.current_price,
                        )
                        if not agent_check.allowed:
                            LOGGER.warning(
                                "Session risk blocked order for %s: %s",
                                session.config.session_id,
                                agent_check.reason,
                            )
                            store.append("orders", {
                                "order_id": None,
                                "side": intent.side,
                                "volume": str(intent.volume),
                                "price": str(intent.price),
                                "agent_id": intent.agent_id,
                                "session_id": session.config.session_id,
                                "main_agent_id": session.config.main_agent_id,
                                "reason": intent.reason,
                                "success": False,
                                "message": agent_check.reason,
                                "mode": "paper" if settings.dry_run else "live",
                            })
                            continue

                        profitable = None
                        if intent.side == "ask" and session_wallet.avg_buy_price > 0:
                            profitable = intent.price > session_wallet.avg_buy_price

                        result = orchestrator.execute_order(
                            intent,
                            wallet_id=session.config.session_id,
                            initial_capital_krw=session.config.initial_capital_krw,
                            extra_log={
                                "session_id": session.config.session_id,
                                "main_agent_id": session.config.main_agent_id,
                                "leader_agent_id": decision["leader_agent_id"],
                            },
                        )
                        if result.success and (settings.dry_run or result.message == "filled"):
                            session_mgr.record_trade(session.config.session_id, profitable)
                            LOGGER.info(
                                "Session order %s: %s %s @ %s (%s)",
                                session.config.session_id,
                                intent.side,
                                intent.volume,
                                intent.price,
                                result.order_id or "no-order-id",
                            )
                        elif result.success:
                            LOGGER.info(
                                "Session order submitted %s: %s %s @ %s (%s)",
                                session.config.session_id,
                                intent.side,
                                intent.volume,
                                intent.price,
                                result.order_id or "no-order-id",
                            )
                        else:
                            LOGGER.warning("Order failed for %s: %s", session.config.session_id, result.message)

                        if session_mgr is not None:
                            wallet = orchestrator.aggregate_session_wallets(
                                market=snapshot.market,
                                sessions=session_mgr.active_sessions(),
                            ) if settings.dry_run else broker.get_wallet("global", settings.market_asset(market))
                else:
                    intent = orchestrator.decide(snapshot, signals)
                    if intent is not None and orchestrator.has_pending_live_order("global"):
                        LOGGER.info("Skipping global order: pending live order still open")
                    elif intent is not None:
                        total_asset_value = wallet.asset_available * snapshot.current_price
                        portfolio_check = portfolio_risk.check(
                            intent=intent,
                            total_krw=wallet.krw_available,
                            total_asset_value=total_asset_value,
                        )
                        if not portfolio_check.allowed:
                            LOGGER.warning("Portfolio risk blocked order: %s", portfolio_check.reason)
                            store.append("orders", {
                                "order_id": None,
                                "side": intent.side,
                                "volume": str(intent.volume),
                                "price": str(intent.price),
                                "agent_id": intent.agent_id,
                                "reason": intent.reason,
                                "success": False,
                                "message": portfolio_check.reason,
                                "mode": "paper" if settings.dry_run else "live",
                            })
                        else:
                            agent_check = agent_risk.check(
                                intent=intent,
                                agent_capital=current_value,
                                agent_krw=wallet.krw_available,
                                agent_asset_value=total_asset_value,
                            )
                            if not agent_check.allowed:
                                LOGGER.warning("Agent risk blocked order: %s", agent_check.reason)
                                store.append("orders", {
                                    "order_id": None,
                                    "side": intent.side,
                                    "volume": str(intent.volume),
                                    "price": str(intent.price),
                                    "agent_id": intent.agent_id,
                                    "reason": intent.reason,
                                    "success": False,
                                    "message": agent_check.reason,
                                    "mode": "paper" if settings.dry_run else "live",
                                })
                            else:
                                result = orchestrator.execute_order(intent)
                                if result.success:
                                    LOGGER.info(
                                        "Order %s: %s %s @ %s (%s)",
                                        result.message,
                                        intent.side,
                                        intent.volume,
                                        intent.price,
                                        result.order_id or "no-order-id",
                                    )
                                else:
                                    LOGGER.warning("Order failed: %s", result.message)

                if session_mgr is not None:
                    for session in session_mgr.active_sessions():
                        session_wallet = orchestrator.trading_wallet(
                            session.config.session_id,
                            snapshot.market,
                            session.config.initial_capital_krw,
                        )
                        session_value = (
                            session_wallet.krw_available
                            + session_wallet.asset_available * snapshot.current_price
                        )
                        session_mgr.update_session_value(
                            session.config.session_id,
                            session_value,
                        )
                    session_mgr.save_state()

                # Save latest report for `report` command
                report = orchestrator.generate_report(
                    snapshot,
                    signals,
                    session_decisions=session_decisions,
                    sessions=session_mgr.active_sessions() if session_mgr is not None else None,
                )
                store.write_json("latest_report", {"text": report, "tick": tick_count})

                # Periodic rebalance
                if tick_count % settings.rebalance_interval == 0:
                    LOGGER.info("Rebalancing...")
                    orchestrator.rebalance()
                    leaderboard.save_snapshot(perf_tracker.all_metrics())

                # Update bot state
                bot = state.get_bot_state()
                bot["tick_count"] = tick_count
                bot["last_tick"] = snapshot.timestamp
                bot["status"] = "running"
                if session_mgr:
                    bot["active_sessions"] = len(session_mgr.active_sessions())
                state.save_bot_state(bot)

            except BithumbAPIError as e:
                LOGGER.error("API error: %s", e)
            except Exception as e:
                LOGGER.error("Tick error: %s", e, exc_info=True)

            if max_ticks > 0 and tick_count >= max_ticks:
                LOGGER.info("Max ticks reached (%d). Stopping.", max_ticks)
                break

            LOGGER.info("Sleeping %ds...", interval)
            time.sleep(interval)

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        bot = state.get_bot_state()
        bot["status"] = "stopped"
        state.save_bot_state(bot)
        if session_mgr:
            session_mgr.save_state()
        LOGGER.info("Bot stopped after %d ticks.", tick_count)
