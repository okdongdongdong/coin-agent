from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Optional

from .config.settings import Settings
from .exchange.bithumb_client import BithumbClient, BithumbAPIError
from .exchange.market_data import MarketDataCollector
from .storage.jsonl_store import JsonlStore
from .storage.state_store import StateStore
from .utils.logging_setup import setup_logging

LOGGER = logging.getLogger(__name__)


def _build(root: Path):
    from .engine.loop import build_system
    result = build_system(root)
    # Return first 11 elements for backward compatibility (ignore session_mgr, evolution)
    return result[:11]


def _build_full(root: Path):
    from .engine.loop import build_system
    return build_system(root)


def cmd_doctor(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings = Settings.load(root)
    setup_logging(settings.log_level)
    client = BithumbClient(settings)

    print("=== Coin-Agent Doctor ===")
    print(f"Markets: {', '.join(settings.markets)}")
    print(f"Mode: {'Paper' if settings.dry_run else 'LIVE'}")
    print(f"Paper Capital: {settings.paper_krw_balance:,.0f} KRW")
    print()

    print("[1/3] Public API (markets)...", end=" ")
    try:
        markets = client.get_markets()
        print(f"OK ({len(markets)} markets)")
    except BithumbAPIError as e:
        print(f"FAIL: {e}")
        return

    print("[2/3] Ticker...", end=" ")
    try:
        tickers = client.get_ticker(settings.markets)
        for t in tickers:
            price = t.get("trade_price", "?")
            print(f"OK ({t.get('market', '?')}: {price:,} KRW)")
    except BithumbAPIError as e:
        print(f"FAIL: {e}")

    print("[3/3] Private API...", end=" ")
    if not settings.has_private_api_keys:
        print("SKIP (no API keys configured)")
    else:
        try:
            accounts = client.get_accounts()
            print(f"OK ({len(accounts)} accounts)")
        except BithumbAPIError as e:
            print(f"FAIL: {e}")

    print("\nDoctor complete.")


def cmd_tick(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state = _build(root)

    market = settings.primary_market
    snapshot = collector.snapshot(market)
    signals = orchestrator.run_tick(snapshot)

    # Save report
    report = orchestrator.generate_report(snapshot, signals)
    store.write_json("latest_report", {"text": report, "tick": 0})

    # Print signals summary
    print(f"=== Tick: {market} @ {snapshot.current_price:,.0f} KRW ===")
    for aid, sig in signals.items():
        print(f"  {aid:25s} {sig.action.upper():5s} conf={sig.confidence:.2f}  {sig.reason}")

    # Update bot state
    bot = state.get_bot_state()
    bot["tick_count"] = bot.get("tick_count", 0) + 1
    bot["last_tick"] = snapshot.timestamp
    state.save_bot_state(bot)

    print(f"\nReport saved. Run 'report' to see full analysis.")


def cmd_report(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings = Settings.load(root)
    store = JsonlStore(settings.data_dir)
    data = store.read_json("latest_report")
    if not data:
        print("No report available. Run 'tick' first.")
        return
    print(data.get("text", "Empty report"))


def cmd_decide(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state = _build(root)

    market = settings.primary_market
    snapshot = collector.snapshot(market)

    # First run tick to get fresh signals
    signals = orchestrator.run_tick(snapshot)

    # Parse Claude signals
    claude_signals = None
    if args.claude_signals:
        try:
            claude_signals = json.loads(args.claude_signals)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON for --claude-signals: {e}")
            return

    # Generate order intent
    intent = orchestrator.decide(
        snapshot=snapshot,
        tech_signals=signals,
        claude_signals=claude_signals,
        meta_action=args.meta_action,
    )

    if intent is None:
        print("Decision: HOLD (no trade)")
        return

    print(f"Decision: {intent.side.upper()} {intent.volume} @ {intent.price:,.0f} KRW")
    print(f"Reason: {intent.reason}")

    if args.execute:
        result = orchestrator.execute_order(intent)
        if result.success:
            print(f"Executed: {result.message} (order_id={result.order_id})")
        else:
            print(f"Failed: {result.message}")
    else:
        print("(dry run - add --execute to submit order)")


def cmd_rebalance(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state = _build(root)

    custom = None
    if args.allocations:
        try:
            custom = json.loads(args.allocations)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            return

    result = orchestrator.rebalance(custom)

    print("=== Capital Rebalance ===")
    total = sum(result.values())
    for aid, alloc in sorted(result.items(), key=lambda x: x[1], reverse=True):
        pct = float(alloc / total * 100) if total > 0 else 0
        print(f"  {aid:25s} {alloc:>10,.0f} KRW ({pct:5.1f}%)")
    print(f"  {'TOTAL':25s} {total:>10,.0f} KRW")


def cmd_status(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings = Settings.load(root)
    setup_logging(settings.log_level)
    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    from .execution.broker import PaperBroker
    broker = PaperBroker(settings, state)

    print("=== Coin-Agent Status ===")
    print(f"Mode: {'Paper' if settings.dry_run else 'LIVE'}")
    print(f"Markets: {', '.join(settings.markets)}")

    bot = state.get_bot_state()
    print(f"Status: {bot.get('status', 'idle')}")
    print(f"Ticks: {bot.get('tick_count', 0)}")
    print()

    if settings.dry_run:
        wallet = broker.get_wallet()
        print("--- Paper Wallet (Global) ---")
        print(f"  KRW: {wallet.krw_available:,.0f}")
        print(f"  Asset: {wallet.asset_available}")
        print(f"  Avg Buy Price: {wallet.avg_buy_price:,.0f}")
    print()

    if state.is_kill_switch_active():
        print("!! KILL SWITCH ACTIVE - trading halted !!")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings, client, collector, registry, orchestrator, broker, perf_tracker, scorer, leaderboard, store, state = _build(root)

    all_metrics = perf_tracker.all_metrics()
    # Ensure all agents have metrics
    for aid in registry.agent_ids():
        if aid not in all_metrics:
            perf_tracker.get_metrics(aid)
    all_metrics = perf_tracker.all_metrics()

    ranking = leaderboard.rank(all_metrics)

    print("=== Agent Leaderboard ===")
    print(f"{'#':>3} {'Agent':25s} {'Score':>7} {'Trades':>7} {'WinRate':>8} {'P&L':>12} {'MDD':>7}")
    print("-" * 75)
    for rank, (aid, sb) in enumerate(ranking, 1):
        m = all_metrics.get(aid)
        trades = m.total_trades if m else 0
        wr = f"{m.win_rate*100:.0f}%" if m else "N/A"
        pnl = m.total_pnl_krw if m else Decimal("0")
        mdd = f"{m.max_drawdown_pct:.1f}%" if m else "0.0%"
        print(f"{rank:>3} {aid:25s} {sb.composite:>7.1f} {trades:>7} {wr:>8} {pnl:>+12,.0f} {mdd:>7}")


def cmd_run(args: argparse.Namespace) -> None:
    from .engine.loop import run_loop
    root = Path(args.root)
    interval = args.interval
    max_ticks = args.max_ticks
    print(f"Starting bot loop (interval={interval}s, max_ticks={max_ticks or 'unlimited'})...")
    print("Press Ctrl+C to stop.\n")
    run_loop(root, interval=interval, max_ticks=max_ticks)


def cmd_stop(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings = Settings.load(root)
    kill_path = settings.data_dir / "KILL"
    kill_path.parent.mkdir(parents=True, exist_ok=True)
    kill_path.touch()
    print("Kill switch activated.")

    store = JsonlStore(settings.data_dir)
    state = StateStore(store)
    bot = state.get_bot_state()
    bot["status"] = "stopped"
    state.save_bot_state(bot)


def cmd_dashboard(args: argparse.Namespace) -> None:
    from .dashboard.server import run_dashboard
    root = Path(args.root)
    run_dashboard(root, host=args.host, port=args.port)


def cmd_session(args: argparse.Namespace) -> None:
    root = Path(args.root)
    result = _build_full(root)
    settings, store, state = result[0], result[9], result[10]
    session_mgr, evolution = result[11], result[12]

    if session_mgr is None:
        print("Session mode is disabled. Set BOT_SESSION_ENABLED=true in .env")
        return

    sub = args.session_action

    if sub == "init":
        session_mgr.initialize_sessions()
        session_mgr.save_state()
        print(f"Initialized {len(session_mgr.active_sessions())} sessions:")
        for s in session_mgr.active_sessions():
            print(f"  {s.config.session_id} ({s.config.provider_type}) "
                  f"capital={s.config.initial_capital_krw:,.0f} KRW")

    elif sub == "list":
        try:
            session_mgr.load_state()
        except Exception:
            pass
        sessions = session_mgr.active_sessions()
        if not sessions:
            print("No active sessions. Run 'session init' first.")
            return
        print(f"=== Active Sessions ({len(sessions)}) ===")
        print(f"{'ID':30s} {'Provider':10s} {'Return%':>8} {'PnL':>12} {'Trades':>7} {'MDD':>7}")
        print("-" * 80)
        for s in session_mgr.rank_sessions():
            print(f"{s.config.session_id:30s} {s.config.provider_type:10s} "
                  f"{s.return_pct:>+7.2f}% {s.total_pnl_krw:>+12,.0f} "
                  f"{s.total_trades:>7} {s.max_drawdown_pct:>6.1f}%")

    elif sub == "evolve":
        try:
            session_mgr.load_state()
        except Exception:
            pass
        event = evolution.evaluate()
        if event:
            print(f"Evolution event:")
            print(f"  Eliminated: {event['eliminated']} ({event['reason']})")
            print(f"  New session: {event['new_session']}")
            session_mgr.save_state()
        else:
            print("No evolution needed at this time.")

    elif sub == "history":
        history = evolution.get_evolution_history()
        if not history:
            print("No evolution history yet.")
            return
        print(f"=== Evolution History ({len(history)} events) ===")
        for i, evt in enumerate(history, 1):
            print(f"  [{i}] eliminated={evt.get('eliminated', '?')} "
                  f"reason={evt.get('reason', '?')} "
                  f"new={evt.get('new_session', '?')}")


def cmd_history(args: argparse.Namespace) -> None:
    root = Path(args.root)
    settings = Settings.load(root)
    store = JsonlStore(settings.data_dir)

    last_n = args.last
    trades = store.read_lines("orders", last_n)

    if not trades:
        print("No trade history.")
        return

    print(f"=== Trade History (last {last_n}) ===")
    for t in trades:
        ts = time.strftime("%m-%d %H:%M", time.localtime(t.get("_ts", 0)))
        side = t.get("side", "?")
        vol = t.get("volume", "?")
        price = t.get("price", "?")
        agent = t.get("agent_id", "?")
        reason = t.get("reason", "")
        success = "OK" if t.get("success") else "FAIL"
        print(f"  [{ts}] {side:4s} vol={vol} price={price} agent={agent} {success} {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="coin-agent", description="Multi-agent crypto trading system")
    parser.add_argument("--root", default=".", help="Project root directory")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Check API connectivity")
    sub.add_parser("tick", help="Run one tick (data + signals)")
    sub.add_parser("report", help="Show latest analysis report")

    p_decide = sub.add_parser("decide", help="Submit Claude signals and execute")
    p_decide.add_argument("--claude-signals", default=None, help='JSON: {"sentiment": {"action": "buy", "confidence": 0.7}, ...}')
    p_decide.add_argument("--meta-action", default=None, choices=["buy", "sell"], help="Override action")
    p_decide.add_argument("--execute", action="store_true", help="Actually execute the order")

    p_rebalance = sub.add_parser("rebalance", help="Rebalance capital allocation")
    p_rebalance.add_argument("--allocations", default=None, help='JSON: {"sma_agent": 0.25, ...}')

    sub.add_parser("status", help="Show portfolio status")
    sub.add_parser("leaderboard", help="Show agent rankings")

    p_run = sub.add_parser("run", help="Run auto-loop")
    p_run.add_argument("--interval", type=int, default=60, help="Tick interval seconds")
    p_run.add_argument("--max-ticks", type=int, default=0, help="Max ticks (0=unlimited)")

    sub.add_parser("stop", help="Activate kill switch")

    p_dash = sub.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--host", default="0.0.0.0", help="Bind host")
    p_dash.add_argument("--port", type=int, default=8080, help="Bind port")

    p_session = sub.add_parser("session", help="Manage competing sessions")
    p_session.add_argument("session_action", choices=["init", "list", "evolve", "history"],
                           help="init=create sessions, list=show active, evolve=force evaluation, history=evolution log")

    p_history = sub.add_parser("history", help="Show trade history")
    p_history.add_argument("--last", type=int, default=20, help="Number of recent trades")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "doctor": cmd_doctor,
        "tick": cmd_tick,
        "report": cmd_report,
        "decide": cmd_decide,
        "rebalance": cmd_rebalance,
        "status": cmd_status,
        "leaderboard": cmd_leaderboard,
        "run": cmd_run,
        "stop": cmd_stop,
        "dashboard": cmd_dashboard,
        "session": cmd_session,
        "history": cmd_history,
    }
    fn = commands.get(args.command)
    if fn:
        fn(args)
    else:
        parser.print_help()
        sys.exit(1)
