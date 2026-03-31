from __future__ import annotations

import datetime
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..config.settings import Settings
from ..storage.jsonl_store import JsonlStore
from .session import SessionConfig, SessionState

LOGGER = logging.getLogger(__name__)

TECHNICAL_AGENT_IDS: List[str] = [
    "sma_agent",
    "momentum_agent",
    "mean_reversion_agent",
    "breakout_agent",
]

MAIN_AGENT_WEIGHT = Decimal("0.4")
PEER_AGENT_WEIGHT = Decimal("0.2")
DEFAULT_CONSENSUS_THRESHOLD = 0.25
DEFAULT_MAIN_OVERRIDE_THRESHOLD = 0.55
DEFAULT_ORDER_FRACTION = 0.2
DEFAULT_ORDER_FRACTION_MAX = 0.25


def default_vote_weights(main_agent_id: str) -> Dict[str, float]:
    if main_agent_id not in TECHNICAL_AGENT_IDS:
        raise ValueError(f"Unknown main agent: {main_agent_id}")
    return {
        agent_id: float(MAIN_AGENT_WEIGHT if agent_id == main_agent_id else PEER_AGENT_WEIGHT)
        for agent_id in TECHNICAL_AGENT_IDS
    }


class SessionManager:
    """Manages four fixed consensus sessions over the same technical agents."""

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
        main_agent_id: str,
        capital_krw: Optional[Decimal] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """Create a fixed technical consensus session."""
        if capital_krw is None:
            capital_krw = self._settings.paper_krw_balance / Decimal("4")

        agent_ids: List[str] = list(TECHNICAL_AGENT_IDS)
        session_id = f"session_{main_agent_id.replace('_agent', '')}_main"

        default_hyperparams: Dict[str, Any] = self._default_hyperparams()
        if hyperparams:
            default_hyperparams.update(hyperparams)

        config = SessionConfig(
            session_id=session_id,
            provider_type="technical_consensus",
            agent_ids=agent_ids,
            initial_capital_krw=capital_krw,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            main_agent_id=main_agent_id,
            vote_weights=default_vote_weights(main_agent_id),
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
            "Created session %s (main=%s, capital=%s KRW, weights=%s)",
            session_id,
            main_agent_id,
            capital_krw,
            config.vote_weights,
        )
        return state

    def initialize_sessions(self) -> List[SessionState]:
        """Create four fixed technical consensus sessions."""
        self._sessions = {}
        num_sessions = 4
        capital_each = self._settings.paper_krw_balance / Decimal(str(num_sessions))

        sessions: List[SessionState] = []
        for main_agent_id in TECHNICAL_AGENT_IDS:
            s = self.create_session(main_agent_id=main_agent_id, capital_krw=capital_each)
            sessions.append(s)

        self.save_state()
        return sessions

    def ensure_consensus_layout(self) -> List[SessionState]:
        if self._has_expected_layout():
            self._sync_consensus_defaults()
            return self.active_sessions()
        LOGGER.info("Resetting session layout to four fixed consensus sessions")
        return self.initialize_sessions()

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

    def update_session_value(
        self,
        session_id: str,
        current_value_krw: Decimal,
    ) -> None:
        """Update session return and drawdown from its own wallet value."""
        session = self._sessions.get(session_id)
        if session is None:
            LOGGER.warning("update_session_value: unknown session %s", session_id)
            return

        initial = session.config.initial_capital_krw
        current = current_value_krw
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

        session.total_pnl_krw = current - initial
        session.current_value_krw = current
        session.peak_value_krw = peak
        session.max_drawdown_pct = max_dd
        session.return_pct = return_pct

    def update_session_decision(
        self,
        session_id: str,
        tick: int,
        action: str,
        confidence: float,
        buy_vote: Decimal,
        sell_vote: Decimal,
        leader_agent_id: str,
        reason: str,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            LOGGER.warning("update_session_decision: unknown session %s", session_id)
            return
        session.last_tick = tick
        session.latest_action = action
        session.latest_confidence = confidence
        session.latest_buy_vote = buy_vote
        session.latest_sell_vote = sell_vote
        session.latest_leader_agent_id = leader_agent_id
        session.latest_reason = reason

    def record_trade(self, session_id: str, profitable: Optional[bool]) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            LOGGER.warning("record_trade: unknown session %s", session_id)
            return
        session.total_trades += 1
        if profitable is True:
            session.winning_trades += 1

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_sessions(self) -> List[SessionState]:
        """Rank active sessions by return_pct descending."""
        active = self.active_sessions()
        active.sort(
            key=lambda s: (s.return_pct, s.current_value_krw, s.config.main_agent_id),
            reverse=True,
        )
        return active

    def _has_expected_layout(self) -> bool:
        active = self.active_sessions()
        if len(active) != 4:
            return False
        main_agents = {s.config.main_agent_id for s in active}
        if main_agents != set(TECHNICAL_AGENT_IDS):
            return False
        for session in active:
            if session.config.agent_ids != TECHNICAL_AGENT_IDS:
                return False
            weights = session.config.vote_weights or {}
            if set(weights.keys()) != set(TECHNICAL_AGENT_IDS):
                return False
        return True

    def _default_hyperparams(self) -> Dict[str, Any]:
        return {
            "confidence_threshold": DEFAULT_CONSENSUS_THRESHOLD,
            "main_agent_override_threshold": DEFAULT_MAIN_OVERRIDE_THRESHOLD,
            "order_fraction": DEFAULT_ORDER_FRACTION,
            "order_fraction_max": DEFAULT_ORDER_FRACTION_MAX,
        }

    def _sync_consensus_defaults(self) -> None:
        defaults = self._default_hyperparams()
        for session in self.active_sessions():
            session.config.vote_weights = default_vote_weights(session.config.main_agent_id)
            for key, value in defaults.items():
                session.config.hyperparams[key] = value

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
