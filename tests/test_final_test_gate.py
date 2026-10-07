"""Mocked final-test recovery verifies gates and budget; no GPU/inference jobs."""
import json
import subprocess
from pathlib import Path

import pytest
import yaml

import orchestrator
from src.evaluation_recovery import EvaluationWorkspace
from src.experiment_registry import BudgetExceeded, RecoveryRequired, Registry


@pytest.fixture
def setup_gate(tmp_path, monkeypatch):
    config = {"seed": 42, "seeds": [42], "deterministic": True, "data": {"processed_dir": str(tmp_path / "processed")},
              "training": {"batch_size": 1}, "models": {"stgt": {}, "adaptive_stgt": {}, "gat_lstm": {}},
              "experiment": {"max_gpu_runs": 10, "max_job_seconds": 30},
              "outputs": {"drive_artifact_root": str(tmp_path / "experiments")}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    registry = Registry(tmp_path / "registry.json")
    dataset = {"raw": {"raw": "synthetic"}, "processed": {"processed": "synthetic"}}
    ids = []
    for name in ("stgt", "adaptive_stgt", "gat_lstm"):
        record = registry.reserve(model=name, signature=name, configuration=config, git_commit="code1",
                                  dataset_manifest=dataset, artifact_dir=tmp_path / "attempts")["record"]
        registry.update(record["experiment_id"], "completed")
        ids.append(record["experiment_id"])
    monkeypatch.setattr(orchestrator, "require_colab_drive", lambda *args: None)
    monkeypatch.setattr(orchestrator, "validate_pushed_repository", lambda *args: "code1")
    monkeypatch.setattr(orchestrator, "validate_archives", lambda *args: dataset["raw"])
    monkeypatch.setattr(orchestrator, "verify_processed", lambda *args: dataset["processed"])
    monkeypatch.setattr(orchestrator, "validate_completed_artifacts", lambda *args: None)
    monkeypatch.setattr(orchestrator, "scientific_source_fingerprint", lambda *args: {"model": "same"})
    monkeypatch.setattr(orchestrator, "runtime_info", lambda: {"packages": {"torch": "synthetic"}})
    return tmp_path, path, registry, ids


def fake_success(command, **kwargs):
    """Produce checksummed synthetic stages, without starting a subprocess."""
    config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    frozen = config["final_evaluation"]
    workspace = EvaluationWorkspace(frozen["directory"], frozen["experiment_ids"])
    for name in ("stgt_seed42", "adaptive_stgt_seed42", "gat_lstm_seed42", "report"):
        def producer(folder):
            (folder / "synthetic.json").write_text('{"synthetic": true}')
            if name == "report":
                (folder / "model_comparison.csv").write_text("Model,MAE\nsynthetic,0\n")
        names = ["synthetic.json"] + (["model_comparison.csv"] if name == "report" else [])
        workspace.stage(name, {}, producer, names)
    Path(kwargs["log_path"]).write_text("synthetic recovery success\n")
    return subprocess.CompletedProcess(command, 0)


def test_explicit_recovery_reuses_frozen_selection_and_never_spends_training_budget(setup_gate, monkeypatch):
    root, config, registry, ids = setup_gate
    def interrupted(command, **kwargs):
        Path(kwargs["log_path"]).write_text("synthetic interruption\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(orchestrator, "run_logged", interrupted)
    with pytest.raises(subprocess.TimeoutExpired):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids)
    destination = root / "experiments/final_test"
    frozen_bytes = (destination / "selection.json").read_bytes()
    with pytest.raises(RecoveryRequired):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids)
    monkeypatch.setattr(orchestrator, "run_logged", fake_success)
    orchestrator.finalize_test(root, config, root / "raw", registry.path, ids, resume_final_test=True)
    assert frozen_bytes == (destination / "selection.json").read_bytes()
    assert registry.status()["submitted_runs"] == 3
    assert len(list((destination / "evaluation_attempts").iterdir())) == 2
    monkeypatch.setattr(orchestrator, "run_logged", lambda *args, **kwargs: pytest.fail("Completed evaluation repeated"))
    orchestrator.finalize_test(root, config, root / "raw", registry.path, ids)
    assert registry.status()["submitted_runs"] == 3


def test_completed_report_corruption_is_not_overwritten(setup_gate, monkeypatch):
    root, config, registry, ids = setup_gate
    monkeypatch.setattr(orchestrator, "run_logged", fake_success)
    destination = orchestrator.finalize_test(root, config, root / "raw", registry.path, ids)
    report = destination / "stages/report/model_comparison.csv"
    report.write_text("corrupted completed report")
    monkeypatch.setattr(orchestrator, "run_logged", lambda *args, **kwargs: pytest.fail("Must not overwrite corrupt completed report"))
    with pytest.raises(ValueError):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids, resume_final_test=True)
    assert report.read_text() == "corrupted completed report"


def test_legacy_partial_output_remains_and_package_drift_blocks_recovery(setup_gate, monkeypatch):
    root, config, registry, ids = setup_gate
    def interrupted(command, **kwargs):
        Path(kwargs["log_path"]).write_text("synthetic interruption")
        raise OSError("synthetic failure")
    monkeypatch.setattr(orchestrator, "run_logged", interrupted)
    with pytest.raises(OSError):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids)
    destination = root / "experiments/final_test"
    (destination / "metrics").mkdir()
    legacy = destination / "metrics/model_comparison.csv"
    legacy.write_text("legacy partial output")
    monkeypatch.setattr(orchestrator, "runtime_info", lambda: {"packages": {"torch": "changed"}})
    with pytest.raises(ValueError, match="package/protocol"):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids, resume_final_test=True)
    assert legacy.read_text() == "legacy partial output"


def test_compatible_evaluator_upgrade_requires_explicit_authorization(setup_gate, monkeypatch):
    root, config, registry, ids = setup_gate
    monkeypatch.setattr(orchestrator, "validate_pushed_repository", lambda *args: "code2")
    with pytest.raises(RecoveryRequired):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids)
    monkeypatch.setattr(orchestrator, "run_logged", fake_success)
    destination = orchestrator.finalize_test(root, config, root / "raw", registry.path, ids, allow_evaluator_upgrade=True)
    assert json.loads((destination / "selection.json").read_text())["git_commit"] == "code2"
    provenance = next((destination / "evaluation_attempts").glob("*/provenance.json"))
    assert json.loads(provenance.read_text())["evaluator_git_commit"] == "code2"


def test_incompatible_evaluator_upgrade_is_rejected(setup_gate, monkeypatch):
    root, config, registry, ids = setup_gate
    monkeypatch.setattr(orchestrator, "validate_pushed_repository", lambda *args: "code2")
    monkeypatch.setattr(orchestrator, "scientific_source_fingerprint", lambda *args: {"model": args[1]})
    with pytest.raises(ValueError, match="changes .* definitions"):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids, allow_evaluator_upgrade=True)
    assert registry.status()["submitted_runs"] == 3


def test_expired_campaign_blocks_evaluation_without_resetting_deadline(setup_gate, monkeypatch):
    root, config, registry, ids = setup_gate
    registry.establish_deadline(1)
    # Only the synthetic registry is changed; no real history or budget is touched.
    state = json.loads(registry.path.read_text())
    state["deadline_at"] = "2000-01-01T00:00:00+00:00"
    registry.path.write_text(json.dumps(state))
    monkeypatch.setattr(orchestrator, "run_logged", lambda *args, **kwargs: pytest.fail("Expired evaluation launched"))
    with pytest.raises(BudgetExceeded):
        orchestrator.finalize_test(root, config, root / "raw", registry.path, ids)
    assert registry.status()["deadline_at"] == "2000-01-01T00:00:00+00:00"
    assert registry.status()["submitted_runs"] == 3
