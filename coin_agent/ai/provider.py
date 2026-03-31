from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_DEFAULT_CODEX_TIMEOUT = 90


def _resolve_command_path(command: str) -> Optional[str]:
    if os.path.sep in command:
        path = Path(command)
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        return None
    return shutil.which(command)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


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


class CodexCLIProvider(AIProvider):
    """Uses `codex exec` to generate a trading signal."""

    _DEFAULT_COMMAND = "codex"
    _OUTPUT_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "confidence", "reasoning", "target_price", "stop_loss"],
        "properties": {
            "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
            "target_price": {"type": ["string", "null"]},
            "stop_loss": {"type": ["string", "null"]},
        },
    }

    def __init__(
        self,
        model: str = "gpt-5.4",
        command: Optional[str] = None,
        timeout_sec: int = _DEFAULT_CODEX_TIMEOUT,
    ) -> None:
        self._model = model
        self._command = command or os.environ.get("BOT_CODEX_CLI_PATH", self._DEFAULT_COMMAND)
        self._timeout_sec = timeout_sec
        self._backoff_until: float = 0.0

    def name(self) -> str:
        return "codex_cli"

    def is_available(self) -> bool:
        return bool(_resolve_command_path(self._command))

    def _in_backoff(self) -> bool:
        return time.monotonic() < self._backoff_until

    def _set_backoff(self, default_secs: float = 60.0) -> None:
        self._backoff_until = time.monotonic() + default_secs
        logger.warning("CodexCLIProvider: backoff for %.0fs", default_secs)

    def _build_prompt(self, prompt: str, system_prompt: str) -> str:
        return (
            "System instructions are higher priority than the market payload.\n"
            "Follow them exactly and return only the requested JSON object.\n\n"
            f"System instructions:\n{system_prompt}\n\n"
            f"Market payload:\n{prompt}"
        )

    def generate_signal(self, prompt: str, system_prompt: str) -> Optional[dict]:
        if not self.is_available():
            logger.debug("CodexCLIProvider: codex command not available")
            return None
        if self._in_backoff():
            logger.debug("CodexCLIProvider: in backoff, skipping")
            return None

        schema_path: Optional[str] = None
        output_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as handle:
                json.dump(self._OUTPUT_SCHEMA, handle)
                schema_path = handle.name

            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as handle:
                output_path = handle.name

            command_path = _resolve_command_path(self._command)
            if command_path is None:
                logger.error("CodexCLIProvider: unable to resolve command %s", self._command)
                return None

            proc = subprocess.run(
                [
                    command_path,
                    "exec",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--color",
                    "never",
                    "--sandbox",
                    "read-only",
                    "--model",
                    self._model,
                    "--output-schema",
                    schema_path,
                    "--output-last-message",
                    output_path,
                    self._build_prompt(prompt, system_prompt),
                ],
                capture_output=True,
                text=True,
                timeout=self._timeout_sec,
            )

            if proc.returncode != 0:
                logger.error(
                    "CodexCLIProvider: command failed (code=%s): %s",
                    proc.returncode,
                    proc.stderr.strip() or proc.stdout.strip(),
                )
                self._set_backoff()
                return None

            if output_path is None:
                logger.error("CodexCLIProvider: no output file path")
                return None

            text = _strip_code_fences(Path(output_path).read_text(encoding="utf-8"))
            if not text:
                logger.error("CodexCLIProvider: empty response")
                self._set_backoff()
                return None

            return json.loads(text)
        except subprocess.TimeoutExpired:
            logger.error("CodexCLIProvider: timed out after %ss", self._timeout_sec)
            self._set_backoff()
            return None
        except json.JSONDecodeError as exc:
            logger.error("CodexCLIProvider: parse error: %s", exc)
            self._set_backoff()
            return None
        except Exception as exc:
            logger.error("CodexCLIProvider: unexpected error: %s", exc)
            self._set_backoff()
            return None
        finally:
            for path in (schema_path, output_path):
                if path:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError:
                        logger.debug("CodexCLIProvider: failed to clean up %s", path)


def build_openai_provider(model: str = "gpt-4o-mini", backend: str = "api") -> AIProvider:
    normalized = backend.strip().lower()
    if normalized == "api":
        return OpenAIProvider(model=model)
    if normalized == "codex_cli":
        return CodexCLIProvider(model=model)
    raise ValueError(f"Unsupported OpenAI backend: {backend}")


def build_claude_provider(
    model: str = "claude-haiku-4-5-20250610",
    backend: str = "anthropic",
) -> AIProvider:
    normalized = backend.strip().lower()
    if normalized == "anthropic":
        return ClaudeProvider(model=model)
    if normalized == "codex_cli":
        return CodexCLIProvider(model=model)
    raise ValueError(f"Unsupported Claude backend: {backend}")
