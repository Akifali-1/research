"""Synthetic registry tests; no training subprocesses are started."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_registry import BudgetExceeded, Registry, experiment_signature  # noqa: E402
from orchestrator import compatible_model_configuration  # noqa: E402


def _identity(model: str = "stgt"):
    return experiment_signature(model, {"seed": 42}, "abc123", {"sha256": "dataset"})


def test_completed_signature_is_skipped(tmp_path):
    registry = Registry(tmp_path / "registry.json", max_runs=10)
    signature = _identity()
    reservation = registry.reserve(
        model="stgt",
        signature=signature,
        configuration={"seed": 42},
        git_commit="abc123",
        dataset_manifest={"sha256": "dataset"},
        artifact_dir=tmp_path / "artifacts",
    )
    assert reservation["action"] == "reserved"
    registry.update(reservation["record"]["experiment_id"], "running")
    registry.update(reservation["record"]["experiment_id"], "completed", report="metrics.json")
    repeat = registry.reserve(
        model="stgt",
        signature=signature,
        configuration={"seed": 42},
        git_commit="abc123",
        dataset_manifest={"sha256": "dataset"},
        artifact_dir=tmp_path / "artifacts",
    )
    assert repeat["action"] == "skip_completed"
    assert registry.status()["submitted_runs"] == 1


def test_budget_blocks_second_submission(tmp_path):
    registry = Registry(tmp_path / "registry.json", max_runs=1)
    first = registry.reserve(
        model="stgt",
        signature=_identity("stgt"),
        configuration={},
        git_commit="a",
        dataset_manifest={},
        artifact_dir=tmp_path,
    )
    registry.update(first["record"]["experiment_id"], "failed", failure={"reason": "synthetic"})
    with pytest.raises(BudgetExceeded):
        registry.reserve(
            model="gat_lstm",
            signature=_identity("gat_lstm"),
            configuration={},
            git_commit="a",
            dataset_manifest={},
            artifact_dir=tmp_path,
        )


def test_interrupted_run_can_be_explicitly_retried(tmp_path):
    registry = Registry(tmp_path / "registry.json", max_runs=3, max_retries=2)
    signature = _identity()
    first = registry.reserve(
        model="stgt",
        signature=signature,
        configuration={},
        git_commit="a",
        dataset_manifest={},
        artifact_dir=tmp_path,
    )
    registry.update(first["record"]["experiment_id"], "running")
    assert registry.mark_interrupted_active("test restart") == 1
    second = registry.reserve(
        model="stgt",
        signature=signature,
        configuration={},
        git_commit="a",
        dataset_manifest={},
        artifact_dir=tmp_path,
        force_retry=True,
    )
    assert second["record"]["attempt"] == 2
    assert registry.status()["submitted_runs"] == 2


def test_adding_a_comparison_model_does_not_invalidate_verified_existing_model():
    old = {
        "seed": 42,
        "deterministic": True,
        "data": {"processed_dir": "/drive/processed", "horizon": 7, "lookback": 14},
        "training": {"loss": "huber", "epochs": 100},
        "models": {"stgt": {"d_model": 64}, "gat_lstm": {"gat_hidden": 48}},
    }
    new = {
        **old,
        "models": {
            **old["models"],
            "adaptive_stgt": {"d_model": 64},
        },
    }
    assert compatible_model_configuration(old, new, "stgt")
    assert compatible_model_configuration(old, new, "gat_lstm")
    changed = {**new, "data": {**new["data"], "horizon": 14}}
    assert not compatible_model_configuration(old, changed, "stgt")
