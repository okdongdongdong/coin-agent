from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..config.settings import Settings
from ..utils.math_helpers import decimal_str

LOGGER = logging.getLogger(__name__)


class BithumbAPIError(RuntimeError):
    pass


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class BithumbClient:
    BASE_URL = "https://api.bithumb.com/v1"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ── Public API ──────────────────────────────────────────

    def get_markets(self, details: bool = False) -> List[dict]:
        return self._request("GET", "/market/all", params={"isDetails": str(details).lower()})

    def get_ticker(self, markets: Iterable[str]) -> List[dict]:
        return self._request("GET", "/ticker", params={"markets": ",".join(markets)})

    def get_minute_candles(self, market: str, unit: int = 1, count: int = 200) -> List[dict]:
        return self._request(
            "GET",
            f"/candles/minutes/{unit}",
            params={"market": market, "count": str(count)},
        )

    # ── Private API ─────────────────────────────────────────

    def get_accounts(self) -> List[dict]:
        return self._request("GET", "/accounts", auth=True)

    def get_order_chance(self, market: str) -> dict:
        return self._request("GET", "/orders/chance", params={"market": market}, auth=True)

    def get_order(self, uuid_val: str) -> dict:
        return self._request("GET", "/order", params={"uuid": uuid_val}, auth=True)

    def list_orders(self, market: str, state: str = "wait", limit: int = 100) -> List[dict]:
        return self._request(
            "GET",
            "/orders",
            params={"market": market, "state": state, "limit": str(limit), "page": "1", "order_by": "desc"},
            auth=True,
        )

    def place_limit_order(self, market: str, side: str, volume: Decimal, price: Decimal) -> dict:
        body = {
            "market": market,
            "side": side,
            "volume": decimal_str(volume),
            "price": decimal_str(price),
            "ord_type": "limit",
        }
        return self._request("POST", "/orders", body=body, auth=True)

    def cancel_order(self, uuid_val: str) -> dict:
        return self._request("DELETE", "/order", params={"uuid": uuid_val}, auth=True)

    # ── Internal ────────────────────────────────────────────

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Any:
        if auth and not self.settings.has_private_api_keys:
            raise BithumbAPIError("Private API keys not configured")

        query_string = self._build_qs(params or body or {})
        url = f"{self.BASE_URL}{path}"
        if params:
            url = f"{url}?{query_string}"

        headers: Dict[str, str] = {"Accept": "application/json"}
        payload_bytes: Optional[bytes] = None

        if auth:
            token = self._make_jwt(query_string if (params or body) else "")
            headers["Authorization"] = f"Bearer {token}"

        if body is not None:
            headers["Content-Type"] = "application/json"
            payload_bytes = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url=url, data=payload_bytes, method=method.upper(), headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.settings.request_timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            msg = raw
            try:
                parsed = json.loads(raw)
                msg = parsed.get("error", {}).get("message", raw)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise BithumbAPIError(f"{method} {path} ({exc.code}): {msg}") from exc
        except urllib.error.URLError as exc:
            raise BithumbAPIError(f"{method} {path}: {exc.reason}") from exc

    def _make_jwt(self, query_string: str) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        payload: Dict[str, Any] = {
            "access_key": self.settings.access_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
        }
        if query_string:
            payload["query_hash"] = hashlib.sha512(query_string.encode("utf-8")).hexdigest()
            payload["query_hash_alg"] = "SHA512"

        segments = [
            _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ]
        signing_input = ".".join(segments).encode("utf-8")
        signature = hmac.new(
            self.settings.secret_key.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        segments.append(_b64url(signature))
        return ".".join(segments)

    @staticmethod
    def _build_qs(payload: Dict[str, Any]) -> str:
        items: List[Tuple[str, str]] = []
        for key, value in payload.items():
            if value is None:
                continue
            if isinstance(value, list):
                lk = key if key.endswith("[]") else f"{key}[]"
                for item in value:
                    items.append((lk, str(item)))
            else:
                items.append((key, str(value)))
        return urllib.parse.urlencode(items)
