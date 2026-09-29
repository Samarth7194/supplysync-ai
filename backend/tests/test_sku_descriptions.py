"""Tests for main._load_sku_descriptions: the committed
backend/data/sku_descriptions.json fallback for when the raw CSV (gitignored,
may be absent in a fresh clone or production deploy) isn't available.
"""

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

COMMITTED_PATH = BACKEND_DIR / "data" / "sku_descriptions.json"


def _committed_descriptions() -> dict:
    with COMMITTED_PATH.open() as fh:
        return json.load(fh)


def test_committed_file_used_when_raw_csv_is_unavailable(monkeypatch):
    import main as backend_main

    def _raise():
        raise FileNotFoundError("no raw CSV in this deployment")

    monkeypatch.setattr(backend_main, "get_sku_descriptions", _raise)
    result = backend_main._load_sku_descriptions()

    assert result == _committed_descriptions()
    assert len(result) > 0


def test_csv_descriptions_take_precedence_on_overlap(monkeypatch):
    import main as backend_main

    committed = _committed_descriptions()
    assert committed, "committed sku_descriptions.json must not be empty for this test to mean anything"
    overlapping_sku = next(iter(committed))

    monkeypatch.setattr(
        backend_main, "get_sku_descriptions",
        lambda: {overlapping_sku: "CSV-SOURCED DESCRIPTION", "NOT-IN-COMMITTED-FILE": "ONLY FROM CSV"},
    )
    result = backend_main._load_sku_descriptions()

    # CSV wins where both sources have the SKU...
    assert result[overlapping_sku] == "CSV-SOURCED DESCRIPTION"
    # ...but SKUs unique to either source both survive the merge.
    assert result["NOT-IN-COMMITTED-FILE"] == "ONLY FROM CSV"
    other_committed_skus = set(committed) - {overlapping_sku}
    assert other_committed_skus <= result.keys()
