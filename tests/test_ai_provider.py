from pathlib import Path
from types import SimpleNamespace

from coin_agent.ai.provider import (
    ClaudeProvider,
    CodexCLIProvider,
    OpenAIProvider,
    build_claude_provider,
    build_openai_provider,
)


def test_build_openai_provider_api():
    provider = build_openai_provider(model="gpt-4o-mini", backend="api")
    assert isinstance(provider, OpenAIProvider)


def test_build_claude_provider_api():
    provider = build_claude_provider(model="claude-haiku-4-5-20250610", backend="anthropic")
    assert isinstance(provider, ClaudeProvider)


def test_build_openai_provider_codex_cli(monkeypatch):
    monkeypatch.setattr("coin_agent.ai.provider._resolve_command_path", lambda command: "/tmp/codex")
    provider = build_openai_provider(model="gpt-5.4", backend="codex_cli")
    assert isinstance(provider, CodexCLIProvider)
    assert provider.is_available()


def test_build_claude_provider_codex_cli(monkeypatch):
    monkeypatch.setattr("coin_agent.ai.provider._resolve_command_path", lambda command: "/tmp/codex")
    provider = build_claude_provider(model="gpt-5.4-mini", backend="codex_cli")
    assert isinstance(provider, CodexCLIProvider)
    assert provider.is_available()


def test_codex_cli_provider_reads_json_output(monkeypatch):
    seen = {}
    monkeypatch.setattr("coin_agent.ai.provider._resolve_command_path", lambda command: "/tmp/codex")

    def fake_run(cmd, capture_output, text, timeout):
        seen["cmd"] = cmd
        output_path = cmd[cmd.index("--output-last-message") + 1]
        Path(output_path).write_text(
            "```json\n"
            '{"action":"hold","confidence":0.4,"reasoning":"flat","target_price":null,"stop_loss":null}\n'
            "```",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("coin_agent.ai.provider.subprocess.run", fake_run)

    provider = CodexCLIProvider(model="gpt-5.4")
    result = provider.generate_signal("prompt", "system prompt")

    assert result == {
        "action": "hold",
        "confidence": 0.4,
        "reasoning": "flat",
        "target_price": None,
        "stop_loss": None,
    }
    assert "--sandbox" in seen["cmd"]
    assert "read-only" in seen["cmd"]
    assert "--output-schema" in seen["cmd"]


def test_codex_cli_provider_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr("coin_agent.ai.provider._resolve_command_path", lambda command: "/tmp/codex")

    def fake_run(cmd, capture_output, text, timeout):
        return SimpleNamespace(returncode=1, stdout="", stderr="auth failed")

    monkeypatch.setattr("coin_agent.ai.provider.subprocess.run", fake_run)

    provider = CodexCLIProvider(model="gpt-5.4")
    assert provider.generate_signal("prompt", "system prompt") is None
