"""Regression tests that keep committed artifacts free of machine-specific
residue.

A stale absolute path in ``lightgbm_demand_forecast_metadata.json`` was what
the original hiring-review checklist flagged, so these tests make sure the
metadata shape stays portable even if someone re-runs ``train_model.py``.
"""

import json
import re
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / "src"))

METADATA_PATH = (
    BACKEND_DIR / "saved_models" / "lightgbm_demand_forecast_metadata.json"
)


# ---- Committed metadata -----------------------------------------------------


def test_committed_metadata_has_no_absolute_paths():
    """The metadata JSON must describe the artifact logically, not with a
    filesystem path tied to one developer machine."""
    if not METADATA_PATH.exists():
        pytest.skip("metadata file not committed on this branch")
    meta = json.loads(METADATA_PATH.read_text())

    # Forbid the field that historically leaked the full local path.
    assert "model_path" not in meta, (
        "model_path is a machine-specific field; use artifact_file (filename only)."
    )

    # Scan every string value for obvious absolute-path shapes.
    abs_win = re.compile(r"^[A-Za-z]:[\\/]")
    for key, value in meta.items():
        if isinstance(value, str):
            assert not abs_win.match(value), (
                f"{key!r} looks like an absolute Windows path: {value!r}"
            )
            assert not value.startswith("/home/"), (
                f"{key!r} looks like an absolute Linux home path: {value!r}"
            )


def test_committed_metadata_has_expected_logical_fields():
    if not METADATA_PATH.exists():
        pytest.skip("metadata file not committed on this branch")
    meta = json.loads(METADATA_PATH.read_text())
    # What the code relies on
    assert meta.get("model_name") == "lightgbm_demand_forecast"
    assert isinstance(meta.get("features"), list) and meta["features"]
    assert meta.get("dataset")  # logical identifier, not a path
    # artifact_file is the portable replacement for the old model_path
    assert meta.get("artifact_file", "").endswith(".pkl")


# ---- Future-proofing: ModelService.save_model writes portable metadata ------


class _DummyModel:
    """Module-level class so ``pickle.dump`` can locate it by qualified name."""

    def predict(self, _):
        return [0.0]


def test_save_model_writes_portable_metadata(tmp_path):
    """If someone reruns train_model.py on another machine, the resulting
    metadata JSON must still be free of absolute paths."""
    from services.model_service import ModelService

    service = ModelService(model_dir=str(tmp_path))
    service.save_model(_DummyModel(), "dummy_model", metadata={"note": "test"})

    written = json.loads((tmp_path / "dummy_model_metadata.json").read_text())
    assert "model_path" not in written, (
        "ModelService regressed: metadata carries an absolute model_path again."
    )
    assert written["artifact_file"] == "dummy_model.pkl"
    # saved_at must parse as ISO 8601; timezone-aware preferred
    datetime.fromisoformat(written["saved_at"])
    # Sanity: the actual pkl is where artifact_file says it is.
    assert (tmp_path / written["artifact_file"]).exists()
    # And it unpickles.
    with (tmp_path / written["artifact_file"]).open("rb") as fh:
        assert pickle.load(fh) is not None
