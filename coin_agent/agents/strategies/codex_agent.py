from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base import SubAgent
from ...exchange.market_data import MarketSnapshot
from ...models.agent import Signal
from .claude_agent import _candles_for_prompt, _compute_indicators, _parse_signal

LOGGER = logging.getLogger(__name__)


class CodexAgent(SubAgent):
    """AI agent powered by OpenAI/Codex."""

    def __init__(self, agent_id: str = "codex_agent", model: str = "gpt-4o-mini",
                 config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(agent_id, config)
        self._model = model
        self._provider: Optional[Any] = None  # lazy init

    def strategy_name(self) -> str:
        return "codex_ai"

    def _get_provider(self) -> Any:
        if self._provider is None:
            from ...ai.provider import OpenAIProvider
            self._provider = OpenAIProvider(model=self._model)
        return self._provider

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
                return self.hold_signal("codex_no_response")
            metadata: Dict[str, Any] = {
                "provider": "openai",
                "model": self._model,
                **{k: indicators[k] for k in indicators},
            }
            return _parse_signal(self.agent_id, result, metadata)
        except Exception as exc:
            LOGGER.error("CodexAgent.analyze error: %s", exc)
            return self.hold_signal(f"error: {exc}")
