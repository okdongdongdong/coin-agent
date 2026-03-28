from __future__ import annotations

import datetime
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config.settings import Settings
from ..models.performance import PerformanceMetrics
from ..storage.jsonl_store import JsonlStore
from .session import SessionConfig, SessionState

LOGGER = logging.getLogger(__name__)

# The 4 technical agent IDs present in every session
_TECHNICAL_AGENT_IDS: List[str] = [
    "sma_agent",
    "momentum_agent",
    "mean_reversion_agent",
    "breakout_agent",
]

# Provider-specific AI agent IDs
_PROVIDER_AGENT_IDS: Dict[str, str] = {
    "claude": "claude_agent",
    "codex": "codex_agent",
    "hybrid": "hybrid_agent",
}


class SessionManager:
    """Manages multiple competing trading sessions."""

    def __init__(
        self,
        settings: Settings,
        store: JsonlStore,
        min_sessions: int = 3,
        max_sessions: int = 5,
    ) -> None:
        self._settings = settings
        self._store = store
        self._sessions: Dict[str, SessionState] = {}
        self._min_sessions = min_sessions
        self._max_sessions = max_sessions
        self._generation = 1

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        provider_type: str,
        capital_krw: Optional[Decimal] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """Create a new trading session.

        provider_type: "claude", "codex", "hybrid", "technical"
        capital_krw: defaults to settings.paper_krw_balance / max_sessions
        hyperparams: optional overrides for softmax_temperature, etc.

        Each session contains ALL 4 technical agents + 1 AI agent (based on
        provider_type).  "technical" sessions have only the 4 technical agents.
        """
        if capital_krw is None:
            capital_krw = self._settings.paper_krw_balance / Decimal(
                str(self._max_sessions)
            )

        # Build agent list
        agent_ids: List[str] = list(_TECHNICAL_AGENT_IDS)
        if provider_type in _PROVIDER_AGENT_IDS:
            agent_ids.append(_PROVIDER_AGENT_IDS[provider_type])

        session_id = f"session_{self._generation:03d}_{provider_type}"

        default_hyperparams: Dict[str, Any] = {
            "softmax_temperature": self._settings.softmax_temperature,
            "confidence_threshold": 0.3,
            "min_alloc_pct": float(self._settings.min_alloc_pct),
            "max_alloc_pct": float(self._settings.max_alloc_pct),
        }
        if hyperparams:
            default_hyperparams.update(hyperparams)

        config = SessionConfig(
            session_id=session_id,
            provider_type=provider_type,
            agent_ids=agent_ids,
            initial_capital_krw=capital_krw,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            hyperparams=default_hyperparams,
        )

        state = SessionState(
            config=config,
            is_active=True,
            current_value_krw=capital_krw,
            peak_value_krw=capital_krw,
            total_pnl_krw=Decimal("0"),
            return_pct=Decimal("0"),
            generation=self._generation,
        )

        self._sessions[session_id] = state
        self._generation += 1

        LOGGER.info(
            "Created session %s (provider=%s, capital=%s KRW, agents=%s)",
            session_id,
            provider_type,
            capital_krw,
            agent_ids,
        )
        return state

    def initialize_sessions(self) -> List[SessionState]:
        """Create the initial set of sessions with diverse providers.

        Default setup:
        - 1 Claude session
        - 1 Codex session
        - 1 Hybrid session (Claude -> Codex fallback)
        - 1 Technical-only session (baseline)

        Capital is split equally among sessions.
        """
        num_sessions = 4
        capital_each = self._settings.paper_krw_balance / Decimal(str(num_sessions))

        sessions: List[SessionState] = []
        for provider in ("claude", "codex", "hybrid", "technical"):
            s = self.create_session(provider_type=provider, capital_krw=capital_each)
            sessions.append(s)

        self.save_state()
        return sessions

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def active_sessions(self) -> List[SessionState]:
        """Return all active (non-eliminated) sessions."""
        return [s for s in self._sessions.values() if s.is_active]

    def all_sessions(self) -> List[SessionState]:
        return list(self._sessions.values())

    # ------------------------------------------------------------------
    # Metrics update
    # ------------------------------------------------------------------

    def update_session_metrics(
        self,
        session_id: str,
        metrics: Dict[str, PerformanceMetrics],
    ) -> None:
        """Update session state from agent performance metrics.

        Aggregates all agent metrics within the session:
        - total_pnl = sum of agent PnLs
        - current_value = initial_capital + total_pnl
        - return_pct = (current_value - initial) / initial * 100
        - Tracks peak value and max drawdown
        """
        session = self._sessions.get(session_id)
        if session is None:
            LOGGER.warning("update_session_metrics: unknown session %s", session_id)
            return

        # Only aggregate metrics for agents belonging to this session
        total_pnl = Decimal("0")
        total_trades = 0
        winning_trades = 0

        for agent_id, m in metrics.items():
            if agent_id in session.config.agent_ids:
                total_pnl += m.total_pnl_krw
                total_trades += m.total_trades
                winning_trades += m.winning_trades

        initial = session.config.initial_capital_krw
        current = initial + total_pnl
        if current < Decimal("0"):
            current = Decimal("0")

        # Peak tracking
        peak = session.peak_value_krw
        if current > peak:
            peak = current

        # Max drawdown
        max_dd = session.max_drawdown_pct
        if peak > 0:
            dd = (peak - current) / peak * Decimal("100")
            if dd > max_dd:
                max_dd = dd

        # Return %
        return_pct = Decimal("0")
        if initial > 0:
            return_pct = (current - initial) / initial * Decimal("100")

        session.total_pnl_krw = total_pnl
        session.current_value_krw = current
        session.peak_value_krw = peak
        session.max_drawdown_pct = max_dd
        session.return_pct = return_pct
        session.total_trades = total_trades
        session.winning_trades = winning_trades

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_sessions(self) -> List[SessionState]:
        """Rank active sessions by return_pct descending."""
        active = self.active_sessions()
        active.sort(key=lambda s: s.return_pct, reverse=True)
        return active

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist all session states to JSON."""
        data: Dict[str, Any] = {
            "generation": self._generation,
            "sessions": {sid: s.to_dict() for sid, s in self._sessions.items()},
        }
        self._store.write_json("sessions", data)
        LOGGER.debug("Session state saved (%d sessions)", len(self._sessions))

    def load_state(self) -> None:
        """Load session states from JSON."""
        data = self._store.read_json("sessions")
        if not data:
            LOGGER.info("No saved session state found")
            return
        self._generation = data.get("generation", 1)
        sessions_raw = data.get("sessions", {})
        self._sessions = {
            sid: SessionState.from_dict(s) for sid, s in sessions_raw.items()
        }
        LOGGER.info(
            "Loaded %d sessions (generation=%d)", len(self._sessions), self._generation
        )
