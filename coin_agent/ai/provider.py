from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30


class AIProvider(ABC):
    """Abstract AI signal provider."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signal(self, prompt: str, system_prompt: str) -> Optional[dict]:
        """Returns parsed signal dict or None on failure."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if API key is configured."""
        ...


class ClaudeProvider(AIProvider):
    """Uses Anthropic Messages API via urllib."""

    _API_URL = "https://api.anthropic.com/v1/messages"
    _API_VERSION = "2023-06-01"
    _ENV_KEY = "ANTHROPIC_API_KEY"

    def __init__(self, model: str = "claude-haiku-4-5-20250610") -> None:
        self._model = model
        self._api_key: str = os.environ.get(self._ENV_KEY, "")
        self._backoff_until: float = 0.0

    def name(self) -> str:
        return "claude"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _in_backoff(self) -> bool:
        return time.monotonic() < self._backoff_until

    def _set_backoff(self, retry_after: Optional[str], default_secs: float = 60.0) -> None:
        try:
            delay = float(retry_after) if retry_after else default_secs
        except (TypeError, ValueError):
            delay = default_secs
        self._backoff_until = time.monotonic() + delay
        logger.warning("ClaudeProvider: backoff for %.0fs", delay)

    def generate_signal(self, prompt: str, system_prompt: str) -> Optional[dict]:
        if not self.is_available():
            logger.debug("ClaudeProvider: no API key configured")
            return None
        if self._in_backoff():
            logger.debug("ClaudeProvider: in backoff, skipping")
            return None

        payload = json.dumps({
            "model": self._model,
            "max_tokens": 200,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")

        req = urllib.request.Request(
            self._API_URL,
            data=payload,
            method="POST",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": self._API_VERSION,
                "content-type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                text = body["content"][0]["text"]
                return json.loads(text)
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("retry-after") if exc.headers else None
            if exc.code in (429, 529):
                logger.warning("ClaudeProvider: HTTP %d, setting backoff", exc.code)
                self._set_backoff(retry_after)
            else:
                logger.error("ClaudeProvider: HTTP %d: %s", exc.code, exc.reason)
            return None
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error("ClaudeProvider: parse error: %s", exc)
            return None
        except Exception as exc:
            logger.error("ClaudeProvider: unexpected error: %s", exc)
            return None


class OpenAIProvider(AIProvider):
    """Uses OpenAI Chat Completions API via urllib."""

    _API_URL = "https://api.openai.com/v1/chat/completions"
    _ENV_KEY = "OPENAI_API_KEY"

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model
        self._api_key: str = os.environ.get(self._ENV_KEY, "")
        self._backoff_until: float = 0.0

    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _in_backoff(self) -> bool:
        return time.monotonic() < self._backoff_until

    def _set_backoff(self, retry_after: Optional[str], default_secs: float = 60.0) -> None:
        try:
            delay = float(retry_after) if retry_after else default_secs
        except (TypeError, ValueError):
            delay = default_secs
        self._backoff_until = time.monotonic() + delay
        logger.warning("OpenAIProvider: backoff for %.0fs", delay)

    def generate_signal(self, prompt: str, system_prompt: str) -> Optional[dict]:
        if not self.is_available():
            logger.debug("OpenAIProvider: no API key configured")
            return None
        if self._in_backoff():
            logger.debug("OpenAIProvider: in backoff, skipping")
            return None

        payload = json.dumps({
            "model": self._model,
            "max_completion_tokens": 200,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            self._API_URL,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                return json.loads(text)
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("retry-after") if exc.headers else None
            if exc.code == 429:
                logger.warning("OpenAIProvider: HTTP 429, setting backoff")
                self._set_backoff(retry_after)
            else:
                logger.error("OpenAIProvider: HTTP %d: %s", exc.code, exc.reason)
            return None
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error("OpenAIProvider: parse error: %s", exc)
            return None
        except Exception as exc:
            logger.error("OpenAIProvider: unexpected error: %s", exc)
            return None
