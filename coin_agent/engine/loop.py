from __future__ import annotations

import logging
import time
from decimal import Decimal
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
from ..agents.strategies.alpha_agent import AlphaAgent
from ..agents.strategies.turbo_breakout_agent import TurboBreakoutAgent
from ..agents.strategies.steady_guard_agent import SteadyGuardAgent
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
    registry.register(AlphaAgent())
    registry.register(TurboBreakoutAgent())
    registry.register(SteadyGuardAgent())

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
        LOGGER.info(
            "Session %s mode enabled (7 agents / 9 sessions)",
            settings.session_execution_mode,
        )

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

    system_mode = (
        f"session_{settings.session_execution_mode}"
        if session_mgr is not None else "global_consensus"
    )
    bot = state.get_bot_state()
    if bot.get("system_mode") != system_mode:
        bot["system_mode"] = system_mode
        bot["system_started_at"] = time.time()
    if session_mgr is not None:
        bot["active_sessions"] = len(session_mgr.active_sessions())
    state.save_bot_state(bot)

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
                    LOGGER.info(
                        "Pending order reconciled: wallet=%s order=%s filled=%s state=%s",
                        outcome.get("wallet_id", ""),
                        outcome.get("order_id", ""),
                        outcome.get("filled_volume", "0"),
                        outcome.get("state", ""),
                    )
                    if session_mgr is not None and settings.session_execution_mode == "multi":
                        session_id = str(outcome.get("session_id", ""))
                        if session_id:
                            session_mgr.record_trade(
                                session_id=session_id,
                                profitable=outcome.get("profitable"),
                            )
                signals = orchestrator.run_tick(snapshot)
                reserve_capital = Decimal("0")
                if session_mgr is not None and settings.session_execution_mode == "multi":
                    wallet = orchestrator.aggregate_session_wallets(
                        market=snapshot.market,
                        sessions=session_mgr.active_sessions(),
                    )
                    reserve_capital = session_mgr.reserve_capital()
                    current_value = (
                        wallet.krw_available
                        + wallet.asset_available * snapshot.current_price
                        + reserve_capital
                    )
                else:
                    wallet = broker.get_wallet("global", settings.market_asset(market))
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
                meta_decision = None
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
                            "execution_mode": settings.session_execution_mode,
                        })
                    if settings.session_execution_mode == "meta":
                        meta_decision = orchestrator.build_meta_decision(
                            snapshot=snapshot,
                            session_decisions=session_decisions,
                        )
                        store.append("meta_decisions", {
                            "tick": tick_count,
                            "market": snapshot.market,
                            "action": meta_decision["action"],
                            "confidence": meta_decision["confidence"],
                            "agree_count": meta_decision["agree_count"],
                            "buy_count": meta_decision["buy_count"],
                            "sell_count": meta_decision["sell_count"],
                            "agreeing_session_ids": meta_decision["agreeing_session_ids"],
                            "leader_session_id": meta_decision["leader_session_id"],
                            "reason": meta_decision["reason"],
                            "execution_mode": settings.session_execution_mode,
                        })

                        meta_executed = False
                        intent = meta_decision.get("intent")
                        if intent is None:
                            LOGGER.info("Meta consensus: %s", meta_decision["reason"])
                        elif orchestrator.has_pending_live_order("global"):
                            LOGGER.info("Skipping meta order: pending live order still open")
                        else:
                            total_asset_value = wallet.asset_available * snapshot.current_price
                            portfolio_check = portfolio_risk.check(
                                intent=intent,
                                total_krw=wallet.krw_available,
                                total_asset_value=total_asset_value,
                            )
                            if not portfolio_check.allowed:
                                LOGGER.warning("Portfolio risk blocked meta order: %s", portfolio_check.reason)
                                store.append("orders", {
                                    "order_id": None,
                                    "side": intent.side,
                                    "volume": str(intent.volume),
                                    "price": str(intent.price),
                                    "agent_id": intent.agent_id,
                                    "leader_session_id": meta_decision["leader_session_id"],
                                    "agree_count": meta_decision["agree_count"],
                                    "agreeing_session_ids": meta_decision["agreeing_session_ids"],
                                    "reason": intent.reason,
                                    "success": False,
                                    "message": portfolio_check.reason,
                                    "mode": "paper" if settings.dry_run else "live",
                                    "execution_mode": settings.session_execution_mode,
                                })
                            else:
                                agent_check = agent_risk.check(
                                    intent=intent,
                                    agent_capital=current_value,
                                    agent_krw=wallet.krw_available,
                                    agent_asset_value=total_asset_value,
                                )
                                if not agent_check.allowed:
                                    LOGGER.warning("Meta agent risk blocked order: %s", agent_check.reason)
                                    store.append("orders", {
                                        "order_id": None,
                                        "side": intent.side,
                                        "volume": str(intent.volume),
                                        "price": str(intent.price),
                                        "agent_id": intent.agent_id,
                                        "leader_session_id": meta_decision["leader_session_id"],
                                        "agree_count": meta_decision["agree_count"],
                                        "agreeing_session_ids": meta_decision["agreeing_session_ids"],
                                        "reason": intent.reason,
                                        "success": False,
                                        "message": agent_check.reason,
                                        "mode": "paper" if settings.dry_run else "live",
                                        "execution_mode": settings.session_execution_mode,
                                    })
                                else:
                                    result = orchestrator.execute_order(
                                        intent,
                                        wallet_id="global",
                                        initial_capital_krw=settings.paper_krw_balance,
                                        extra_log={
                                            "leader_session_id": meta_decision["leader_session_id"],
                                            "agree_count": meta_decision["agree_count"],
                                            "agreeing_session_ids": meta_decision["agreeing_session_ids"],
                                            "meta_action": meta_decision["action"],
                                            "execution_mode": settings.session_execution_mode,
                                        },
                                    )
                                    meta_executed = bool(result.success)
                                    if result.success:
                                        LOGGER.info(
                                            "Meta order %s: %s %s @ %s (%s)",
                                            result.message,
                                            intent.side,
                                            intent.volume,
                                            intent.price,
                                            result.order_id or "no-order-id",
                                        )
                                    else:
                                        LOGGER.warning("Meta order failed: %s", result.message)

                        session_mgr.record_meta_consensus(
                            final_action=str(meta_decision["action"]),
                            executed=meta_executed,
                            agreeing_session_ids=list(meta_decision["agreeing_session_ids"]),
                        )
                    else:
                        for decision in session_decisions:
                            session_id = str(decision["session_id"])
                            session = session_mgr.get_session(session_id)
                            if session is None:
                                continue
                            session_wallet = orchestrator.trading_wallet(
                                session_id,
                                snapshot.market,
                                session.config.initial_capital_krw,
                            )
                            session_value = (
                                session_wallet.krw_available
                                + session_wallet.asset_available * snapshot.current_price
                            )
                            session_mgr.update_session_value(session_id, session_value)

                            intent = decision.get("intent")
                            if intent is None:
                                continue
                            if orchestrator.has_pending_live_order(session_id):
                                LOGGER.info(
                                    "Skipping %s order: pending live order still open",
                                    session_id,
                                )
                                continue

                            aggregate_wallet = orchestrator.aggregate_session_wallets(
                                market=snapshot.market,
                                sessions=session_mgr.active_sessions(),
                            )
                            total_asset_value = aggregate_wallet.asset_available * snapshot.current_price
                            portfolio_check = portfolio_risk.check(
                                intent=intent,
                                total_krw=aggregate_wallet.krw_available + reserve_capital,
                                total_asset_value=total_asset_value,
                            )
                            if not portfolio_check.allowed:
                                LOGGER.warning(
                                    "Portfolio risk blocked %s order: %s",
                                    session_id,
                                    portfolio_check.reason,
                                )
                                store.append("orders", {
                                    "order_id": None,
                                    "side": intent.side,
                                    "volume": str(intent.volume),
                                    "price": str(intent.price),
                                    "agent_id": intent.agent_id,
                                    "session_id": session_id,
                                    "main_agent_id": session.config.main_agent_id,
                                    "leader_agent_id": decision["leader_agent_id"],
                                    "reason": intent.reason,
                                    "success": False,
                                    "message": portfolio_check.reason,
                                    "mode": "paper" if settings.dry_run else "live",
                                    "execution_mode": settings.session_execution_mode,
                                })
                                continue

                            session_asset_value = session_wallet.asset_available * snapshot.current_price
                            agent_check = agent_risk.check(
                                intent=intent,
                                agent_capital=session_value,
                                agent_krw=session_wallet.krw_available,
                                agent_asset_value=session_asset_value,
                            )
                            if not agent_check.allowed:
                                LOGGER.warning(
                                    "Session risk blocked %s order: %s",
                                    session_id,
                                    agent_check.reason,
                                )
                                store.append("orders", {
                                    "order_id": None,
                                    "side": intent.side,
                                    "volume": str(intent.volume),
                                    "price": str(intent.price),
                                    "agent_id": intent.agent_id,
                                    "session_id": session_id,
                                    "main_agent_id": session.config.main_agent_id,
                                    "leader_agent_id": decision["leader_agent_id"],
                                    "reason": intent.reason,
                                    "success": False,
                                    "message": agent_check.reason,
                                    "mode": "paper" if settings.dry_run else "live",
                                    "execution_mode": settings.session_execution_mode,
                                })
                                continue

                            result = orchestrator.execute_order(
                                intent,
                                wallet_id=session_id,
                                initial_capital_krw=session.config.initial_capital_krw,
                                extra_log={
                                    "session_id": session_id,
                                    "main_agent_id": session.config.main_agent_id,
                                    "leader_agent_id": decision["leader_agent_id"],
                                    "execution_mode": settings.session_execution_mode,
                                },
                            )
                            if result.success:
                                LOGGER.info(
                                    "Session order %s: %s %s @ %s (%s)",
                                    result.message,
                                    intent.side,
                                    intent.volume,
                                    intent.price,
                                    result.order_id or "no-order-id",
                                )
                                if settings.dry_run:
                                    profitable = None
                                    if intent.side == "ask" and session_wallet.avg_buy_price > 0:
                                        profitable = intent.price > session_wallet.avg_buy_price
                                    session_mgr.record_trade(session_id, profitable)
                            else:
                                LOGGER.warning("Session order failed: %s", result.message)
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

                session_values = {}
                session_principal_total = Decimal("0")
                display_wallet = wallet
                display_reserve_capital = reserve_capital
                display_total_value = current_value
                if session_mgr is not None:
                    if settings.session_execution_mode == "multi":
                        session_principal_total = session_mgr.session_principal_total()
                        display_reserve_capital = session_mgr.reserve_capital()
                        for session in session_mgr.active_sessions():
                            session_wallet = orchestrator.trading_wallet(
                                session.config.session_id,
                                snapshot.market,
                                session.config.initial_capital_krw,
                            )
                            session_total = (
                                session_wallet.krw_available
                                + session_wallet.asset_available * snapshot.current_price
                            )
                            session_mgr.update_session_value(
                                session.config.session_id,
                                session_total,
                            )
                            session_values[session.config.session_id] = {
                                "total_value": str(session_total),
                                "krw_available": str(session_wallet.krw_available),
                                "asset_available": str(session_wallet.asset_available),
                                "avg_buy_price": str(session_wallet.avg_buy_price),
                                "position_value": str(
                                    session_wallet.asset_available * snapshot.current_price
                                ),
                            }
                        display_wallet = orchestrator.aggregate_session_wallets(
                            market=snapshot.market,
                            sessions=session_mgr.active_sessions(),
                        )
                        display_total_value = (
                            display_wallet.krw_available
                            + display_wallet.asset_available * snapshot.current_price
                            + display_reserve_capital
                        )
                    else:
                        session_principal_total = session_mgr.session_principal_total()
                    session_mgr.save_state()

                store.append("equity_curve", {
                    "tick": tick_count,
                    "market": snapshot.market,
                    "price": str(snapshot.current_price),
                    "total_value": str(display_total_value),
                    "krw_available": str(display_wallet.krw_available + display_reserve_capital),
                    "position_value": str(display_wallet.asset_available * snapshot.current_price),
                    "session_principal_total": str(session_principal_total),
                    "reserve_capital": str(display_reserve_capital),
                    "sessions": session_values,
                    "execution_mode": settings.session_execution_mode if session_mgr is not None else "global",
                })

                # Save latest report for `report` command
                report = orchestrator.generate_report(
                    snapshot,
                    signals,
                    session_decisions=session_decisions,
                    meta_decision=meta_decision,
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
                bot["system_mode"] = system_mode
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
