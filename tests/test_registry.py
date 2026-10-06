"""Synthetic registry tests; no training subprocesses are started."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment_registry import BudgetExceeded, Registry, experiment_signature  # noqa: E402


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
