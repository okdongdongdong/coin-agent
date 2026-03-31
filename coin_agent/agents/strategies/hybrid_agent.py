from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base import SubAgent
from ...exchange.market_data import MarketSnapshot
from ...models.agent import Signal
from .claude_agent import _candles_for_prompt, _compute_indicators, _parse_signal

LOGGER = logging.getLogger(__name__)


class HybridAgent(SubAgent):
    """AI agent with Claude primary, Codex fallback."""

    def __init__(self, agent_id: str = "hybrid_agent",
                 claude_model: str = "claude-haiku-4-5-20250610",
                 codex_model: str = "gpt-4o-mini",
                 claude_backend: str = "anthropic",
                 codex_backend: str = "api",
                 config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(agent_id, config)
        self._claude_model = claude_model
        self._codex_model = codex_model
        self._claude_backend = claude_backend
        self._codex_backend = codex_backend
        self._provider: Optional[Any] = None  # lazy init

    def strategy_name(self) -> str:
        return "hybrid_ai"

    def _get_provider(self) -> Any:
        if self._provider is None:
            from ...ai.provider import build_claude_provider, build_openai_provider
            from ...ai.fallback import FallbackProvider
            self._provider = FallbackProvider(
                primary=build_claude_provider(
                    model=self._claude_model,
                    backend=self._claude_backend,
                ),
                secondary=build_openai_provider(
                    model=self._codex_model,
                    backend=self._codex_backend,
                ),
            )
        return self._provider

    @property
    def active_provider_name(self) -> str:
        """Returns the currently active provider name."""
        if self._provider is None:
            return "none"
        return self._provider.active_provider

    def analyze(self, snapshot: MarketSnapshot) -> Signal:
        from ...ai.prompts import SIGNAL_SYSTEM_PROMPT, build_market_prompt

        try:
            indicators = _compute_indicators(snapshot)
            candles = _candles_for_prompt(snapshot)
            prompt = build_market_prompt(
                market=snapshot.market,
                current_price=str(snapshot.current_price),
                candles=candles,
                indicators=indicators,
            )
            provider = self._get_provider()
            result = provider.generate_signal(prompt, SIGNAL_SYSTEM_PROMPT)
            if result is None:
                return self.hold_signal("hybrid_no_response")
            active = provider.active_provider
            metadata: Dict[str, Any] = {
                "provider": active,
                "claude_model": self._claude_model,
                "codex_model": self._codex_model,
                "claude_backend": self._claude_backend,
                "codex_backend": self._codex_backend,
                **{k: indicators[k] for k in indicators},
            }
            return _parse_signal(self.agent_id, result, metadata)
        except Exception as exc:
            LOGGER.error("HybridAgent.analyze error: %s", exc)
            return self.hold_signal(f"error: {exc}")
