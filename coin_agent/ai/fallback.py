from __future__ import annotations

import logging
import time
from typing import Optional

from coin_agent.ai.provider import AIProvider

logger = logging.getLogger(__name__)


class FallbackProvider(AIProvider):
    """Tries primary provider, falls back to secondary on rate limit."""

    def __init__(self, primary: AIProvider, secondary: AIProvider) -> None:
        self._primary = primary
        self._secondary = secondary
        self._primary_backoff_until: float = 0.0
        self._secondary_backoff_until: float = 0.0

    def name(self) -> str:
        return f"{self._primary.name()}\u2192{self._secondary.name()}"

    def is_available(self) -> bool:
        return self._primary.is_available() or self._secondary.is_available()

    @property
    def active_provider(self) -> str:
        """Returns which provider is currently active."""
        now = time.monotonic()
        if self._primary.is_available() and now >= self._primary_backoff_until:
            return self._primary.name()
        if self._secondary.is_available() and now >= self._secondary_backoff_until:
            return self._secondary.name()
        return "none"

    def generate_signal(self, prompt: str, system_prompt: str) -> Optional[dict]:
        now = time.monotonic()

        # Try primary unless in backoff
        if self._primary.is_available() and now >= self._primary_backoff_until:
            result = self._primary.generate_signal(prompt, system_prompt)
            if result is not None:
                logger.debug("FallbackProvider: signal from %s", self._primary.name())
                return result
            # Primary returned None - check if it just entered backoff
            # (provider sets its own internal backoff; mirror here for routing)
            logger.info(
                "FallbackProvider: primary %s returned None, trying secondary %s",
                self._primary.name(),
                self._secondary.name(),
            )
            self._primary_backoff_until = now + 60.0
        else:
            logger.debug(
                "FallbackProvider: primary %s in backoff, routing to secondary",
                self._primary.name(),
            )

        # Try secondary
        if self._secondary.is_available() and now >= self._secondary_backoff_until:
            result = self._secondary.generate_signal(prompt, system_prompt)
            if result is not None:
                logger.debug("FallbackProvider: signal from %s", self._secondary.name())
                return result
            logger.warning(
                "FallbackProvider: secondary %s returned None, both providers exhausted",
                self._secondary.name(),
            )
            self._secondary_backoff_until = now + 60.0
        else:
            logger.debug(
                "FallbackProvider: secondary %s in backoff",
                self._secondary.name(),
            )

        logger.warning("FallbackProvider: all providers failed, returning None (hold)")
        return None
