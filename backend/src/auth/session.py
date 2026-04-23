"""Lightweight HMAC-signed session cookies for demo/admin auth.

Stdlib-only. Deliberately small — this is a *demo-grade* access guard, not
a full identity platform. Two modes:

  * ``AUTH_MODE=off``  (default) — every /api/* endpoint is open; the
    optional ``X-API-Key`` header is still honored for back-compat.
  * ``AUTH_MODE=demo`` — callers must present either a valid signed
    session cookie (issued by ``POST /api/auth/login``) or the legacy
    ``X-API-Key`` header.

Credentials for the demo mode come from ``DEMO_USER`` and ``DEMO_PASSWORD``
env vars. The cookie is HMAC-SHA256 over ``{username, expires_at}`` using
``SESSION_SECRET`` (or an ephemeral in-process secret if unset).

Design note: a real system would hash passwords, rotate keys, store
sessions in a database, and set SameSite / Secure correctly for HTTPS.
This module does enough for a convincing demo guard and says so in the UI.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger("supplysync.auth")

COOKIE_NAME = "supplysync_session"
DEFAULT_TTL_SECONDS = 60 * 60 * 12  # 12 hours


@dataclass(frozen=True)
class AuthConfig:
    mode: str                  # "off" or "demo"
    demo_user: str
    demo_password: str
    session_secret: bytes
    api_key: Optional[str]     # legacy header auth
    ttl_seconds: int = DEFAULT_TTL_SECONDS

    @property
    def enabled(self) -> bool:
        return self.mode == "demo"


def _load_secret() -> bytes:
    raw = os.getenv("SESSION_SECRET")
    if raw:
        return raw.encode("utf-8")
    # Ephemeral secret — sessions won't survive a restart. Fine for local
    # demo; log once so the operator notices.
    logger.warning(
        "SESSION_SECRET unset — using a random per-process secret. Logins will "
        "be invalidated when the backend restarts."
    )
    return secrets.token_bytes(32)


def load_config() -> AuthConfig:
    mode_raw = os.getenv("AUTH_MODE", "off").strip().lower()
    if mode_raw not in {"off", "demo"}:
        logger.warning("AUTH_MODE=%r not recognized; treating as 'off'.", mode_raw)
        mode_raw = "off"

    return AuthConfig(
        mode=mode_raw,
        demo_user=os.getenv("DEMO_USER", "demo"),
        demo_password=os.getenv("DEMO_PASSWORD", ""),
        session_secret=_load_secret(),
        api_key=os.getenv("API_KEY"),
    )


# --- signing primitives -----------------------------------------------------


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = 4 - (len(s) % 4)
    if pad != 4:
        s = s + ("=" * pad)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def sign_session(username: str, cfg: AuthConfig, now: Optional[int] = None) -> str:
    """Return a compact ``<payload>.<signature>`` token."""
    now_ts = int(now) if now is not None else int(time.time())
    payload = json.dumps(
        {"sub": username, "exp": now_ts + cfg.ttl_seconds},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    sig = hmac.new(cfg.session_secret, payload, hashlib.sha256).digest()
    return f"{_b64url_encode(payload)}.{_b64url_encode(sig)}"


def verify_session(token: str, cfg: AuthConfig) -> Optional[dict]:
    """Return the payload dict if ``token`` is valid and unexpired, else ``None``."""
    if not token or token.count(".") != 1:
        return None
    payload_b64, sig_b64 = token.split(".", 1)
    try:
        payload = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)
    except Exception:  # noqa: BLE001 — invalid token shape
        return None
    expected = hmac.new(cfg.session_secret, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if int(data.get("exp", 0)) < int(time.time()):
        return None
    return data


# --- credential check --------------------------------------------------------


def check_credentials(username: str, password: str, cfg: AuthConfig) -> bool:
    """Constant-time credential comparison against the demo env vars."""
    if not cfg.demo_password:
        # If the operator forgot to set DEMO_PASSWORD, refuse all logins
        # instead of accepting an empty password.
        return False
    user_ok = hmac.compare_digest(username.encode("utf-8"), cfg.demo_user.encode("utf-8"))
    pass_ok = hmac.compare_digest(password.encode("utf-8"), cfg.demo_password.encode("utf-8"))
    return user_ok and pass_ok
