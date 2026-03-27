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
from ..execution.broker import PaperBroker
from ..execution.position_tracker import PositionTracker
from ..performance.tracker import PerformanceTracker
from ..performance.scorer import PerformanceScorer
from ..performance.leaderboard import Leaderboard
from ..storage.jsonl_store import JsonlStore
from ..storage.state_store import StateStore
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

    broker = PaperBroker(settings, state)
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

    return settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state


def run_loop(root: Path, interval: int = 0, max_ticks: int = 0) -> None:
    """Run the main bot loop."""
    settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state = build_system(root)

    if interval <= 0:
        interval = settings.loop_interval_sec

    market = settings.primary_market
    tick_count = 0

    LOGGER.info("Starting bot loop: market=%s, interval=%ds, mode=%s", market, interval, "paper" if settings.dry_run else "LIVE")

    # Initial equal allocation
    orchestrator.rebalance()

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
                signals = orchestrator.run_tick(snapshot)

                # Save latest report for `report` command
                report = orchestrator.generate_report(snapshot, signals)
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
        LOGGER.info("Bot stopped after %d ticks.", tick_count)
