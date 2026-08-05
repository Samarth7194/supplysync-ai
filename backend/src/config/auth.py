"""Authentication settings.

Auth primitives still live in auth/session.py. This domain module keeps
settings composition explicit without moving the session-signing logic.
"""

from __future__ import annotations

from auth.session import AuthConfig, load_config


def load_auth_settings() -> AuthConfig:
    return load_config()

