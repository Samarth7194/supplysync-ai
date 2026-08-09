"""Tests for the /health readiness fields.

Locks down the contract that /health reports on each backing artifact
(model, dataset, KPI cache) so a developer or Docker healthcheck can tell
what still needs generating.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))


def test_health_reports_readiness_fields():
    import main as backend_main

    with TestClient(backend_main.app) as c:
        res = c.get("/health")
        assert res.status_code == 200
        body = res.json()
        for key in ("status", "model_loaded", "data_available", "kpis_available", "database", "hint"):
            assert key in body, f"missing key {key!r} on /health"
        assert body["status"] == "online"
        # Types must be stable so a healthcheck can parse them
        assert isinstance(body["model_loaded"], bool)
        assert isinstance(body["data_available"], bool)
        assert isinstance(body["kpis_available"], bool)
        assert isinstance(body["database"], dict)
        assert isinstance(body["database"]["configured"], bool)
        assert isinstance(body["database"]["reachable"], bool)
        assert isinstance(body["database"]["dialect"], str)
        # hint is either None (all green) or a human-readable string
        assert body["hint"] is None or isinstance(body["hint"], str)
