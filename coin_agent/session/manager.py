from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional

from ..config.settings import Settings
from ..storage.jsonl_store import JsonlStore
from .session import SessionConfig, SessionState

LOGGER = logging.getLogger(__name__)

AGENT_IDS: List[str] = [
    "sma_agent",
    "momentum_agent",
    "mean_reversion_agent",
    "breakout_agent",
    "alpha_agent",
    "turbo_breakout_agent",
    "steady_guard_agent",
]

SESSION_DISPLAY_NAMES: Dict[str, str] = {
    "sma_agent": "The Tracker",
    "momentum_agent": "The Pulse",
    "mean_reversion_agent": "The Rubber Band",
    "breakout_agent": "The Cannon",
    "alpha_agent": "The Apex",
    "turbo_breakout_agent": "The Turbo",
    "steady_guard_agent": "The Anchor",
}

DEFAULT_CONSENSUS_THRESHOLD = 0.25
DEFAULT_MAIN_OVERRIDE_THRESHOLD = 0.55


@dataclass(frozen=True)
class SessionTemplate:
    session_id: str
    provider_type: str
    display_name: str
    main_agent_id: str
    vote_weights: Dict[str, float]
    hyperparams: Dict[str, Any]

    @property
    def agent_ids(self) -> List[str]:
        return list(self.vote_weights.keys())


def _hp(
    *,
    display_name: str,
    confidence_threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
    main_agent_override_threshold: float = DEFAULT_MAIN_OVERRIDE_THRESHOLD,
) -> Dict[str, Any]:
    return {
        "display_name": display_name,
        "confidence_threshold": confidence_threshold,
        "main_agent_override_threshold": main_agent_override_threshold,
    }


SESSION_TEMPLATES: List[SessionTemplate] = [
    SessionTemplate(
        session_id="session_trend_core",
        provider_type="meta_consensus",
        display_name="Trend Core",
        main_agent_id="sma_agent",
        vote_weights={
            "sma_agent": 0.40,
            "alpha_agent": 0.20,
            "steady_guard_agent": 0.20,
            "momentum_agent": 0.10,
            "breakout_agent": 0.10,
        },
        hyperparams=_hp(display_name="Trend Core", confidence_threshold=0.27, main_agent_override_threshold=0.57),
    ),
    SessionTemplate(
        session_id="session_momentum_core",
        provider_type="meta_consensus",
        display_name="Momentum Core",
        main_agent_id="momentum_agent",
        vote_weights={
            "momentum_agent": 0.40,
            "alpha_agent": 0.15,
            "turbo_breakout_agent": 0.20,
            "sma_agent": 0.15,
            "breakout_agent": 0.10,
        },
        hyperparams=_hp(display_name="Momentum Core", confidence_threshold=0.26, main_agent_override_threshold=0.56),
    ),
    SessionTemplate(
        session_id="session_reversion_core",
        provider_type="meta_consensus",
        display_name="Reversion Core",
        main_agent_id="mean_reversion_agent",
        vote_weights={
            "mean_reversion_agent": 0.45,
            "steady_guard_agent": 0.20,
            "alpha_agent": 0.15,
            "sma_agent": 0.10,
            "momentum_agent": 0.10,
        },
        hyperparams=_hp(display_name="Reversion Core", confidence_threshold=0.26, main_agent_override_threshold=0.56),
    ),
    SessionTemplate(
        session_id="session_breakout_core",
        provider_type="meta_consensus",
        display_name="Breakout Core",
        main_agent_id="breakout_agent",
        vote_weights={
            "breakout_agent": 0.40,
            "turbo_breakout_agent": 0.25,
            "alpha_agent": 0.15,
            "momentum_agent": 0.10,
            "sma_agent": 0.10,
        },
        hyperparams=_hp(display_name="Breakout Core", confidence_threshold=0.27, main_agent_override_threshold=0.57),
    ),
    SessionTemplate(
        session_id="session_alpha_core",
        provider_type="meta_consensus",
        display_name="Alpha Core",
        main_agent_id="alpha_agent",
        vote_weights={
            "alpha_agent": 0.40,
            "sma_agent": 0.15,
            "momentum_agent": 0.15,
            "mean_reversion_agent": 0.10,
            "breakout_agent": 0.10,
            "turbo_breakout_agent": 0.05,
            "steady_guard_agent": 0.05,
        },
        hyperparams=_hp(display_name="Alpha Core", confidence_threshold=0.28, main_agent_override_threshold=0.60),
    ),
    SessionTemplate(
        session_id="session_turbo_core",
        provider_type="meta_consensus",
        display_name="Turbo Core",
        main_agent_id="turbo_breakout_agent",
        vote_weights={
            "turbo_breakout_agent": 0.50,
            "breakout_agent": 0.20,
            "momentum_agent": 0.15,
            "alpha_agent": 0.10,
            "sma_agent": 0.05,
        },
        hyperparams=_hp(display_name="Turbo Core", confidence_threshold=0.26, main_agent_override_threshold=0.58),
    ),
    SessionTemplate(
        session_id="session_steady_core",
        provider_type="meta_consensus",
        display_name="Steady Core",
        main_agent_id="steady_guard_agent",
        vote_weights={
            "steady_guard_agent": 0.50,
            "sma_agent": 0.20,
            "mean_reversion_agent": 0.15,
            "alpha_agent": 0.10,
            "momentum_agent": 0.05,
        },
        hyperparams=_hp(display_name="Steady Core", confidence_threshold=0.32, main_agent_override_threshold=0.68),
    ),
    SessionTemplate(
        session_id="session_balanced_offense",
        provider_type="meta_consensus",
        display_name="Balanced Offense",
        main_agent_id="alpha_agent",
        vote_weights={
            "alpha_agent": 0.20,
            "turbo_breakout_agent": 0.20,
            "momentum_agent": 0.15,
            "breakout_agent": 0.15,
            "sma_agent": 0.10,
            "mean_reversion_agent": 0.10,
            "steady_guard_agent": 0.10,
        },
        hyperparams=_hp(display_name="Balanced Offense", confidence_threshold=0.27, main_agent_override_threshold=0.57),
    ),
    SessionTemplate(
        session_id="session_balanced_defense",
        provider_type="meta_consensus",
        display_name="Balanced Defense",
        main_agent_id="steady_guard_agent",
        vote_weights={
            "steady_guard_agent": 0.25,
            "sma_agent": 0.20,
            "alpha_agent": 0.15,
            "mean_reversion_agent": 0.15,
            "momentum_agent": 0.10,
            "breakout_agent": 0.10,
            "turbo_breakout_agent": 0.05,
        },
        hyperparams=_hp(display_name="Balanced Defense", confidence_threshold=0.30, main_agent_override_threshold=0.62),
    ),
]

SESSION_TEMPLATE_BY_ID: Dict[str, SessionTemplate] = {
    template.session_id: template for template in SESSION_TEMPLATES
}
SESSION_TEMPLATE_ORDER: Dict[str, int] = {
    template.session_id: idx for idx, template in enumerate(SESSION_TEMPLATES)
}


def default_session_capital(settings: Settings) -> Decimal:
    if settings.session_capital_krw > 0:
        return settings.session_capital_krw
    session_count = max(len(SESSION_TEMPLATES), 1)
    return (settings.paper_krw_balance / Decimal(str(session_count))).quantize(
        Decimal("1"),
        rounding=ROUND_DOWN,
    )


class SessionManager:
    """Manages the fixed 9-session layout for either multi-session or meta mode."""

    def __init__(
        self,
        settings: Settings,
        store: JsonlStore,
        min_sessions: int = 3,
        max_sessions: int = 9,
    ) -> None:
        self._settings = settings
        self._store = store
        self._sessions: Dict[str, SessionState] = {}
        self._min_sessions = min_sessions
        self._max_sessions = max_sessions
        self._generation = 1

    def _session_state_from_template(self, template: SessionTemplate) -> SessionState:
        config = SessionConfig(
            session_id=template.session_id,
            provider_type=self._provider_type(template),
            agent_ids=list(template.agent_ids),
            initial_capital_krw=default_session_capital(self._settings),
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            main_agent_id=template.main_agent_id,
            vote_weights=dict(template.vote_weights),
            hyperparams=dict(template.hyperparams),
        )
        return SessionState(
            config=config,
            is_active=True,
            current_value_krw=Decimal("0"),
            peak_value_krw=Decimal("0"),
            total_pnl_krw=Decimal("0"),
            return_pct=Decimal("0"),
            generation=self._generation,
        )

    def initialize_sessions(self) -> List[SessionState]:
        self._sessions = {}
        sessions: List[SessionState] = []
        for template in SESSION_TEMPLATES:
            state = self._session_state_from_template(template)
            self._sessions[template.session_id] = state
            sessions.append(state)
            self._generation += 1
        self.save_state()
        return sessions

    def ensure_consensus_layout(self) -> List[SessionState]:
        if self._has_expected_layout():
            self._sync_consensus_defaults()
            return self.active_sessions()
        LOGGER.info(
            "Resetting session layout to 9-session %s templates",
            self._settings.session_execution_mode,
        )
        return self.initialize_sessions()

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def active_sessions(self) -> List[SessionState]:
        return [s for s in self._sessions.values() if s.is_active]

    def all_sessions(self) -> List[SessionState]:
        return list(self._sessions.values())

    def update_session_value(self, session_id: str, current_value_krw: Decimal) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            LOGGER.warning("update_session_value: unknown session %s", session_id)
            return
        session.current_value_krw = current_value_krw

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
        session.executed_count += 1
        if profitable is True:
            session.winning_trades += 1

    def record_meta_consensus(
        self,
        final_action: str,
        executed: bool,
        agreeing_session_ids: List[str],
    ) -> None:
        if final_action not in {"buy", "sell"}:
            for session in self.active_sessions():
                if session.latest_action in {"buy", "sell"}:
                    session.vote_count += 1
            return

        agreeing = set(agreeing_session_ids)
        for session in self.active_sessions():
            if session.latest_action in {"buy", "sell"}:
                session.vote_count += 1
            if session.config.session_id in agreeing:
                session.agreement_count += 1
            if executed:
                session.total_trades += 1
                session.executed_count += 1
                if session.config.session_id in agreeing:
                    session.winning_trades += 1

    def session_principal_total(self) -> Decimal:
        return sum(
            (session.config.initial_capital_krw for session in self.active_sessions()),
            Decimal("0"),
        )

    def reserve_capital(self) -> Decimal:
        reserve = self._settings.paper_krw_balance - self.session_principal_total()
        return max(Decimal("0"), reserve)

    def rank_sessions(self) -> List[SessionState]:
        active = self.active_sessions()

        def _agreement_rate(session: SessionState) -> Decimal:
            if session.vote_count <= 0:
                return Decimal("0")
            return Decimal(session.agreement_count) / Decimal(session.vote_count)

        active.sort(
            key=lambda s: (
                _agreement_rate(s),
                Decimal(str(s.latest_confidence)),
                Decimal(s.winning_trades),
                Decimal(s.vote_count),
            ),
            reverse=True,
        )
        return active

    def _has_expected_layout(self) -> bool:
        active = self.active_sessions()
        if len(active) != len(SESSION_TEMPLATES):
            return False
        for template in SESSION_TEMPLATES:
            session = self._sessions.get(template.session_id)
            if session is None:
                return False
            if session.config.provider_type != self._provider_type(template):
                return False
            if session.config.main_agent_id != template.main_agent_id:
                return False
            if session.config.agent_ids != template.agent_ids:
                return False
            if session.config.vote_weights != template.vote_weights:
                return False
        return True

    def _sync_consensus_defaults(self) -> None:
        for template in SESSION_TEMPLATES:
            session = self._sessions.get(template.session_id)
            if session is None:
                continue
            session.config.provider_type = self._provider_type(template)
            session.config.main_agent_id = template.main_agent_id
            session.config.agent_ids = list(template.agent_ids)
            session.config.vote_weights = dict(template.vote_weights)
            session.config.hyperparams = dict(template.hyperparams)
            session.config.initial_capital_krw = default_session_capital(self._settings)

    def _provider_type(self, template: SessionTemplate) -> str:
        if self._settings.session_execution_mode == "multi":
            return "multi_session"
        return template.provider_type

    def save_state(self) -> None:
        data: Dict[str, Any] = {
            "generation": self._generation,
            "sessions": {sid: s.to_dict() for sid, s in self._sessions.items()},
        }
        self._store.write_json("sessions", data)
        LOGGER.debug("Session state saved (%d sessions)", len(self._sessions))

    def load_state(self) -> None:
        data = self._store.read_json("sessions")
        if not data:
            LOGGER.info("No saved session state found")
            return
        self._generation = data.get("generation", 1)
        sessions_raw = data.get("sessions", {})
        self._sessions = {sid: SessionState.from_dict(s) for sid, s in sessions_raw.items()}
        LOGGER.info(
            "Loaded %d sessions (generation=%d)", len(self._sessions), self._generation
        )
