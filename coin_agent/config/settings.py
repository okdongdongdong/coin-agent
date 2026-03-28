from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List


DEFAULT_ENV_FILE = ".env"


def _load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _get(env: Dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key, env.get(key, default))


def _get_bool(env: Dict[str, str], key: str, default: bool) -> bool:
    value = _get(env, key, "true" if default else "false").lower()
    return value in {"1", "true", "yes", "on"}


def _get_int(env: Dict[str, str], key: str, default: int) -> int:
    return int(_get(env, key, str(default)))


def _get_decimal(env: Dict[str, str], key: str, default: str) -> Decimal:
    return Decimal(_get(env, key, default))


@dataclass(frozen=True)
class Settings:
    access_key: str
    secret_key: str
    markets: List[str]
    dry_run: bool
    log_level: str
    data_dir: Path
    loop_interval_sec: int
    request_timeout_sec: int
    candle_unit: int
    candle_count: int
    paper_krw_balance: Decimal
    fee_rate: Decimal
    order_cooldown_sec: int
    max_daily_loss_krw: Decimal
    max_total_loss_krw: Decimal
    max_position_pct: Decimal
    min_agent_trades: int
    bench_threshold: float
    min_active_agents: int
    rebalance_interval: int
    softmax_temperature: float
    min_alloc_pct: Decimal
    max_alloc_pct: Decimal

    # AI Provider Config
    anthropic_api_key: str
    openai_api_key: str
    claude_model: str
    openai_model: str

    # Session Config
    session_enabled: bool
    session_min_count: int
    session_max_count: int
    session_eval_interval: int
    session_min_ticks_before_eval: int

    @classmethod
    def load(cls, root: Path) -> "Settings":
        env = _load_env_file(root / DEFAULT_ENV_FILE)
        data_dir = root / _get(env, "BOT_DATA_DIR", "data")
        markets_raw = _get(env, "BOT_MARKETS", "KRW-BTC")
        markets = [m.strip().upper() for m in markets_raw.split(",") if m.strip()]
        return cls(
            access_key=_get(env, "BITHUMB_ACCESS_KEY", ""),
            secret_key=_get(env, "BITHUMB_SECRET_KEY", ""),
            markets=markets,
            dry_run=_get_bool(env, "BOT_DRY_RUN", True),
            log_level=_get(env, "BOT_LOG_LEVEL", "INFO").upper(),
            data_dir=data_dir,
            loop_interval_sec=_get_int(env, "BOT_LOOP_INTERVAL_SEC", 60),
            request_timeout_sec=_get_int(env, "BOT_REQUEST_TIMEOUT_SEC", 10),
            candle_unit=_get_int(env, "BOT_CANDLE_UNIT", 1),
            candle_count=_get_int(env, "BOT_CANDLE_COUNT", 200),
            paper_krw_balance=_get_decimal(env, "BOT_PAPER_KRW_BALANCE", "300000"),
            fee_rate=_get_decimal(env, "BOT_FEE_RATE", "0.0025"),
            order_cooldown_sec=_get_int(env, "BOT_ORDER_COOLDOWN_SEC", 60),
            max_daily_loss_krw=_get_decimal(env, "BOT_MAX_DAILY_LOSS_KRW", "30000"),
            max_total_loss_krw=_get_decimal(env, "BOT_MAX_TOTAL_LOSS_KRW", "90000"),
            max_position_pct=_get_decimal(env, "BOT_MAX_POSITION_PCT", "40"),
            min_agent_trades=_get_int(env, "BOT_MIN_AGENT_TRADES", 10),
            bench_threshold=float(_get(env, "BOT_BENCH_THRESHOLD", "30")),
            min_active_agents=_get_int(env, "BOT_MIN_ACTIVE_AGENTS", 2),
            rebalance_interval=_get_int(env, "BOT_REBALANCE_INTERVAL", 50),
            softmax_temperature=float(_get(env, "BOT_SOFTMAX_TEMPERATURE", "2.0")),
            min_alloc_pct=_get_decimal(env, "BOT_MIN_ALLOC_PCT", "5"),
            max_alloc_pct=_get_decimal(env, "BOT_MAX_ALLOC_PCT", "40"),
            # AI Provider Config
            anthropic_api_key=_get(env, "ANTHROPIC_API_KEY", ""),
            openai_api_key=_get(env, "OPENAI_API_KEY", ""),
            claude_model=_get(env, "BOT_CLAUDE_MODEL", "claude-haiku-4-5-20250610"),
            openai_model=_get(env, "BOT_OPENAI_MODEL", "gpt-4o-mini"),
            # Session Config
            session_enabled=_get_bool(env, "BOT_SESSION_ENABLED", False),
            session_min_count=_get_int(env, "BOT_SESSION_MIN_COUNT", 3),
            session_max_count=_get_int(env, "BOT_SESSION_MAX_COUNT", 5),
            session_eval_interval=_get_int(env, "BOT_SESSION_EVAL_INTERVAL", 5),
            session_min_ticks_before_eval=_get_int(env, "BOT_SESSION_MIN_TICKS_BEFORE_EVAL", 10),
        )

    @property
    def has_private_api_keys(self) -> bool:
        return bool(self.access_key and self.secret_key)

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def primary_market(self) -> str:
        return self.markets[0] if self.markets else "KRW-BTC"

    def market_base(self, market: str) -> str:
        return market.split("-", 1)[0]

    def market_asset(self, market: str) -> str:
        return market.split("-", 1)[1]
