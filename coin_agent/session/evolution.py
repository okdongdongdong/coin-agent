from __future__ import annotations

import datetime
import logging
import random
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .manager import SessionManager
from .session import SessionState

LOGGER = logging.getLogger(__name__)

# All known provider types for diversity tracking
_ALL_PROVIDERS = ("claude", "codex", "hybrid", "technical")


class SessionEvolution:
    """Handles evolutionary selection of sessions."""

    def __init__(
        self,
        manager: SessionManager,
        min_ticks_before_eval: int = 10,
        eval_interval_ticks: int = 5,
    ) -> None:
        self._manager = manager
        self._min_ticks = min_ticks_before_eval   # don't evaluate too early
        self._eval_interval = eval_interval_ticks
        self._tick_count = 0
        self._evolution_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main tick entry
    # ------------------------------------------------------------------

    def tick(self) -> Optional[Dict[str, Any]]:
        """Called each tick.  Returns evolution event dict if evolution happened.

        Returns None if no evolution this tick.
        Returns {"eliminated": session_id, "reason": str, "new_session": session_id}
        if evolution occurred.
        """
        self._tick_count += 1
        if self._tick_count < self._min_ticks:
            return None
        if self._tick_count % self._eval_interval != 0:
            return None
        return self.evaluate()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Optional[Dict[str, Any]]:
        """Evaluate all sessions and potentially eliminate the worst.

        Rules:
        1. Need at least min_sessions active.
        2. Only eliminate if worst session has negative returns.
        3. Only eliminate if worst session is significantly behind
           (>5% below average return).
        4. Don't eliminate sessions that are less than min_ticks old
           (proxied by generation gap – newly spawned sessions have the
           highest generation number).

        When eliminating:
        1. Mark session as eliminated with reason.
        2. Redistribute its capital to remaining sessions.
        3. Spawn a replacement session with a different provider/hyperparams.
        4. Log the evolution event.
        """
        active = self._manager.active_sessions()
        if len(active) < self._manager._min_sessions:
            LOGGER.debug(
                "evolution.evaluate: only %d active sessions, need >= %d",
                len(active),
                self._manager._min_sessions,
            )
            return None

        ranked = self._manager.rank_sessions()  # best -> worst
        worst = ranked[-1]

        # Rule 2: only eliminate if worst has negative return
        if worst.return_pct >= Decimal("0"):
            LOGGER.debug(
                "evolution.evaluate: worst session %s has non-negative return %.2f%%, skipping",
                worst.config.session_id,
                float(worst.return_pct),
            )
            return None

        # Rule 3: must be significantly behind average (> 5%)
        avg_return = sum(s.return_pct for s in active) / Decimal(str(len(active)))
        gap = avg_return - worst.return_pct
        if gap < Decimal("5"):
            LOGGER.debug(
                "evolution.evaluate: worst session %s gap=%.2f%% < 5%%, skipping",
                worst.config.session_id,
                float(gap),
            )
            return None

        # Rule 4: don't eliminate if it was just created (lowest generation gap).
        # A session is considered "too young" if its generation is the highest
        # (i.e. it was spawned most recently) AND there are older sessions with
        # negative returns too.
        max_gen = max(s.generation for s in active)
        if worst.generation == max_gen and len(active) > 1:
            # Find the second-worst that is old enough
            older_candidates = [s for s in ranked[:-1] if s.generation < max_gen]
            if older_candidates and older_candidates[-1].return_pct < Decimal("0"):
                second_worst = older_candidates[-1]
                gap2 = avg_return - second_worst.return_pct
                if gap2 >= Decimal("5"):
                    worst = second_worst
                    LOGGER.debug(
                        "evolution.evaluate: switching target to older session %s",
                        worst.config.session_id,
                    )

        # --- Perform elimination ---
        reason = (
            f"return_pct={float(worst.return_pct):.2f}% "
            f"(avg={float(avg_return):.2f}%, gap={float(gap):.2f}%)"
        )
        LOGGER.info(
            "Eliminating session %s: %s", worst.config.session_id, reason
        )

        worst.is_active = False
        worst.eliminated_at = datetime.datetime.utcnow().isoformat() + "Z"
        worst.elimination_reason = reason

        # Redistribute eliminated capital to survivors
        eliminated_capital = worst.current_value_krw
        survivors = [s for s in active if s.config.session_id != worst.config.session_id]
        if survivors:
            share = eliminated_capital / Decimal(str(len(survivors)))
            for s in survivors:
                s.config.initial_capital_krw += share
                s.current_value_krw += share
                s.peak_value_krw = max(s.peak_value_krw, s.current_value_krw)
            LOGGER.info(
                "Redistributed %s KRW (%s each) to %d survivors",
                eliminated_capital,
                share,
                len(survivors),
            )

        # Spawn replacement
        replacement_provider = self._choose_replacement_provider(
            worst.config.provider_type
        )
        base_hp = worst.config.hyperparams
        new_hp = self._mutate_hyperparams(base_hp)
        new_capital = eliminated_capital  # give it the same starting capital

        new_session = self._manager.create_session(
            provider_type=replacement_provider,
            capital_krw=new_capital,
            hyperparams=new_hp,
        )

        event: Dict[str, Any] = {
            "tick": self._tick_count,
            "eliminated": worst.config.session_id,
            "eliminated_provider": worst.config.provider_type,
            "reason": reason,
            "new_session": new_session.config.session_id,
            "new_provider": replacement_provider,
            "new_hyperparams": new_hp,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        self._evolution_history.append(event)
        self._manager.save_state()

        LOGGER.info(
            "Evolution complete: %s -> %s (%s)",
            worst.config.session_id,
            new_session.config.session_id,
            replacement_provider,
        )
        return event

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def _choose_replacement_provider(self, eliminated_provider: str) -> str:
        """Choose a provider type for the replacement session.

        Strategy: look at which provider types are performing best and
        create more of those, but always maintain diversity by preferring
        under-represented provider types.
        """
        active = self._manager.active_sessions()

        # Count current providers
        provider_counts: Dict[str, int] = {p: 0 for p in _ALL_PROVIDERS}
        for s in active:
            p = s.config.provider_type
            if p in provider_counts:
                provider_counts[p] += 1

        # Rank providers by average return among active sessions
        provider_returns: Dict[str, List[Decimal]] = {p: [] for p in _ALL_PROVIDERS}
        for s in active:
            p = s.config.provider_type
            if p in provider_returns:
                provider_returns[p].append(s.return_pct)

        provider_avg: Dict[str, Decimal] = {}
        for p, returns in provider_returns.items():
            if returns:
                provider_avg[p] = sum(returns) / Decimal(str(len(returns)))
            else:
                provider_avg[p] = Decimal("0")

        # Prefer provider types that are:
        # 1. Under-represented (count == 0 gets high priority)
        # 2. Performing well on average
        # 3. Different from eliminated provider (diversity)
        candidates = list(_ALL_PROVIDERS)

        def _score(p: str) -> float:
            diversity_bonus = 10.0 if p != eliminated_provider else 0.0
            underrep_bonus = 5.0 if provider_counts[p] == 0 else 0.0
            perf_score = float(provider_avg[p])
            return perf_score + diversity_bonus + underrep_bonus

        candidates.sort(key=_score, reverse=True)
        return candidates[0]

    # ------------------------------------------------------------------
    # Hyperparam mutation
    # ------------------------------------------------------------------

    def _mutate_hyperparams(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Slightly mutate hyperparameters for the new session.

        Mutates: softmax_temperature (+-0.1), confidence_threshold (+-0.05).
        Uses random within small range for exploration.
        """
        mutated = dict(base_params)

        # softmax_temperature: clamp to [0.5, 5.0]
        temp = float(mutated.get("softmax_temperature", 2.0))
        delta_temp = random.uniform(-0.1, 0.1)
        mutated["softmax_temperature"] = max(0.5, min(5.0, temp + delta_temp))

        # confidence_threshold: clamp to [0.1, 0.9]
        conf = float(mutated.get("confidence_threshold", 0.3))
        delta_conf = random.uniform(-0.05, 0.05)
        mutated["confidence_threshold"] = max(0.1, min(0.9, conf + delta_conf))

        return mutated

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Return list of all evolution events."""
        return list(self._evolution_history)
