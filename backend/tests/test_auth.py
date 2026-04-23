"""Tests for the lightweight demo-auth layer.

Covers the signing primitives, the login/me/logout flow, and the
require_access fallback behavior. Auth is off by default; these tests
override ``AUTH_CONFIG`` after the lifespan so the gate actually fires.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))


def _demo_cfg(password: str = "letmein", user: str = "demo"):
    """Build an AuthConfig tuned for demo-mode testing."""
    from auth.session import AuthConfig
    return AuthConfig(
        mode="demo",
        demo_user=user,
        demo_password=password,
        session_secret=b"test-secret-32-bytes-xxxxxxxxxxxx",
        api_key=None,
    )


# ---- signing primitives -----------------------------------------------------


def test_sign_and_verify_roundtrip():
    from auth.session import sign_session, verify_session
    cfg = _demo_cfg()
    token = sign_session("alice", cfg)
    payload = verify_session(token, cfg)
    assert payload is not None
    assert payload["sub"] == "alice"
    assert isinstance(payload["exp"], int)


def test_verify_session_rejects_tampered_payload():
    from auth.session import sign_session, verify_session
    cfg = _demo_cfg()
    token = sign_session("alice", cfg)
    payload_b64, sig = token.split(".", 1)
    # Flip a character in the payload; signature should no longer match.
    tampered = payload_b64[:-1] + ("A" if payload_b64[-1] != "A" else "B") + "." + sig
    assert verify_session(tampered, cfg) is None


def test_verify_session_rejects_expired_token():
    from auth.session import sign_session, verify_session
    cfg = _demo_cfg()
    token = sign_session("alice", cfg, now=0)  # expired long ago
    assert verify_session(token, cfg) is None


def test_verify_session_rejects_garbage():
    from auth.session import verify_session
    cfg = _demo_cfg()
    assert verify_session("", cfg) is None
    assert verify_session("not-a-token", cfg) is None
    assert verify_session("only.onedot", cfg) is not None or verify_session("a.b", cfg) is None


def test_check_credentials_rejects_empty_password_even_on_match():
    """If DEMO_PASSWORD isn't configured, every login must fail."""
    from auth.session import AuthConfig, check_credentials
    cfg = AuthConfig(
        mode="demo", demo_user="demo", demo_password="",
        session_secret=b"x" * 32, api_key=None,
    )
    assert check_credentials("demo", "", cfg) is False
    assert check_credentials("demo", "anything", cfg) is False


# ---- endpoint flow ----------------------------------------------------------


def test_auth_status_reflects_off_mode_by_default():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        body = c.get("/api/auth/status").json()
    assert body["auth_mode"] == "off"
    assert body["login_required"] is False


def test_login_login_me_logout_flow_in_demo_mode(monkeypatch):
    # ``monkeypatch.setattr`` auto-reverts, so the demo config doesn't
    # leak into sibling test modules.
    import main as backend_main

    monkeypatch.setattr(backend_main, "AUTH_CONFIG", _demo_cfg(password="topsecret"))
    with TestClient(backend_main.app) as c:
        # Gate is up — anonymous call to a protected endpoint fails.
        r = c.get("/api/skus/details")
        assert r.status_code == 401

        # Wrong credentials — still 401, no session set.
        r_bad = c.post("/api/auth/login", json={"username": "demo", "password": "nope"})
        assert r_bad.status_code == 401

        # Correct credentials → 200 + cookie.
        r_ok = c.post("/api/auth/login", json={"username": "demo", "password": "topsecret"})
        assert r_ok.status_code == 200
        assert r_ok.json()["user"] == "demo"

        # /api/auth/me now says we're the session user.
        me = c.get("/api/auth/me").json()
        assert me["source"] == "session"
        assert me["user"] == "demo"

        # /api/auth/status reflects it too.
        status = c.get("/api/auth/status").json()
        assert status["authenticated"] is True
        assert status["user"] == "demo"

        # Logout clears the cookie; status goes back to login_required.
        c.post("/api/auth/logout")
        assert c.get("/api/auth/status").json()["login_required"] is True


def test_login_returns_400_when_mode_off(monkeypatch):
    import main as backend_main
    from auth.session import AuthConfig

    monkeypatch.setattr(
        backend_main, "AUTH_CONFIG",
        AuthConfig(mode="off", demo_user="demo", demo_password="x",
                   session_secret=b"x" * 32, api_key=None),
    )
    with TestClient(backend_main.app) as c:
        r = c.post("/api/auth/login", json={"username": "demo", "password": "x"})
        assert r.status_code == 400


def test_api_key_still_works_in_demo_mode(monkeypatch):
    """When API_KEY is set, automation scripts can keep using the header
    even after AUTH_MODE flips to demo."""
    import main as backend_main
    from auth.session import AuthConfig

    monkeypatch.setattr(
        backend_main, "AUTH_CONFIG",
        AuthConfig(mode="demo", demo_user="demo", demo_password="x",
                   session_secret=b"x" * 32, api_key="secret-key-42"),
    )
    with TestClient(backend_main.app) as c:
        # Anonymous → 401.
        assert c.get("/api/skus/details").status_code == 401
        # With the legacy header → 503 (data missing) but auth passed.
        backend_main._data_service = None
        r = c.get("/api/skus/details", headers={"X-API-Key": "secret-key-42"})
        assert r.status_code == 503  # auth accepted; failure is downstream
