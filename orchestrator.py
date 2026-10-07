"""Semi-automated controller: user-authorized Colab, Drive, then sequential runs.

Only --allow-execution can start compute. Run signatures, immutable attempt
directories, counted retries, validation-only selection, runtime limits, and
artifact verification are shared by initial and resumed Colab sessions.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.experiment_registry import BudgetExceeded, RecoveryRequired, Registry, experiment_signature
from src.raw_data import dataset_layout_error, source_paths
from src.process_runner import run_logged
from src.source_schema import validate_source_schema
from src.evaluation_recovery import evaluation_lock
from src.integrity import atomic_json, verify_manifest

EXPECTED_ARCHIVES = {
    f"{name}.csv.7z" for name in ("train", "items", "stores", "transactions", "oil",
                                  "holidays_events", "test", "sample_submission")
}


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path, payload):
    atomic_json(path, payload)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root):
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def validate_pushed_repository(root, branch=None):
    commit = git_commit(root)
    branch = branch or subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
    if not branch:
        raise RuntimeError("A checked-out experiment branch is required")
    remote = subprocess.check_output(
        ["git", "-C", str(root), "ls-remote", "origin", f"refs/heads/{branch}"], text=True).strip()
    if not remote or remote.split()[0] != commit:
        raise RuntimeError(f"Push gate failed: {branch} is absent or differs from {commit}")
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("Source tree is dirty; commit/push source changes before running")
    return commit


def require_colab_drive(*paths):
    """This is a Colab-only execution boundary, not an authentication API."""
    try:
        available = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        available = False
    mount = Path("/content/drive")
    mydrive = mount / "MyDrive"
    if not available or not os.path.ismount(mount) or not mydrive.is_dir():
        raise RuntimeError("Authorize Drive interactively in Google Colab before running compute")
    for path in paths:
        if not Path(path).resolve().is_relative_to(mount.resolve()):
            raise RuntimeError(f"Dataset/artifact path must be on mounted Drive: {path}")


def runtime_info():
    import torch
    packages = {}
    for name in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "PyYAML", "py7zr"):
        packages[name] = importlib.metadata.version(name)
    free_disk = shutil.disk_usage("/content" if Path("/content").exists() else ".").free
    memory = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value, *_ = line.split()
                memory[key.rstrip(":")] = int(value) * 1024
    result = {"python": sys.version, "packages": packages, "cuda_available": torch.cuda.is_available(),
              "free_runtime_disk_bytes": free_disk, "ram": memory}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(gpu=torch.cuda.get_device_name(0), free_gpu_bytes=free, total_gpu_bytes=total)
    return result


def quality_gate(root):
    """Run synthetic CPU checks; failures occur before a budget slot is reserved."""
    for package in ("torch", "torch_geometric", "numpy", "pandas", "sklearn", "yaml", "pytest"):
        if importlib.util.find_spec(package) is None:
            raise RuntimeError(f"Missing required quality-gate dependency: {package}")
    subprocess.run([sys.executable, "-m", "compileall", "-q", str(root / "src"), str(root / "tests")], check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=root, check=True, timeout=180)


def file_manifest(directory, names):
    entries = []
    for name in sorted(names):
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required nonempty file is absent: {path}")
        entries.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"files": entries}


def validate_archives(raw_dir):
    import py7zr
    sources, missing = source_paths(raw_dir)
    if missing:
        raise dataset_layout_error(raw_dir, missing)
    manifest = file_manifest(raw_dir, [path.name for path in sources.values()])
    manifest["dataset"] = "Corporación Favorita Grocery Sales Forecasting"
    for entry in manifest["files"]:
        if entry["name"].endswith(".7z"):
            with py7zr.SevenZipFile(Path(raw_dir) / entry["name"], "r") as archive:
                members = archive.list()
                if len(members) != 1 or members[0].filename != entry["name"][:-3]:
                    raise ValueError(f"Unexpected archive members: {entry['name']}")
                entry["uncompressed_bytes"] = members[0].uncompressed
        else:
            entry["uncompressed_bytes"] = entry["bytes"]
    # All eight schemas, including compressed train/test, use bounded header
    # decoding. Full CRC integrity is checked at complete CSV extraction.
    for stem, path in sources.items():
        validate_source_schema(path, stem)
    return manifest


def scientific_configuration(config):
    data = {key: value for key, value in config["data"].items() if key != "processed_dir"}
    return {"seed": config["seed"], "deterministic": config.get("deterministic", True),
            "data": data, "training": config["training"], "models": config["models"]}


def model_scientific_configuration(config, model):
    """Return the protocol and selected model block used for reuse checks."""
    if model not in config.get("models", {}):
        raise ValueError(f"Configuration has no model block for {model}")
    data = {key: value for key, value in config["data"].items() if key != "processed_dir"}
    return {
        "seed": config["seed"],
        "deterministic": config.get("deterministic", True),
        "data": data,
        "training": config["training"],
        "model": config["models"][model],
    }


def compatible_model_configuration(record_config, expected_config, model):
    """Compare only fields that affect a selected model's scientific run.

    Adding a new model block must not invalidate an intact older checkpoint for
    an unchanged model. The selected model's own block and all shared protocol
    fields must still match exactly.
    """
    return model_scientific_configuration(record_config, model) == model_scientific_configuration(expected_config, model)


def verify_processed(data_dir, raw_manifest):
    from src.preprocessing import validate_outputs
    data_dir = Path(data_dir)
    completed = json.loads((data_dir / "complete.json").read_text())
    if completed["raw_manifest"] != raw_manifest:
        raise ValueError("Processed data does not match the verified Drive archives")
    manifest = file_manifest(data_dir, ("sales.csv", "nodes.csv", "edges.csv", "data_quality.json"))
    if manifest != completed["processed_manifest"]:
        raise ValueError("Processed data changed since preprocessing completed")
    validate_outputs(data_dir)
    return manifest


def validate_completed_artifacts(record):
    artifact = Path(record["artifact_dir"])
    manifest = record.get("artifact_manifest")
    if not manifest or file_manifest(artifact, [item["name"] for item in manifest["files"]]) != manifest:
        raise RuntimeError(f"Completed run artifacts are missing/changed: {record['experiment_id']}; do not retrain silently")


def prepare_data(root, config_path, raw_dir, processed_base, branch=None):
    """Colab preprocessing stage; versioned output, durable log, completion marker."""
    require_colab_drive(raw_dir, processed_base)
    commit = validate_pushed_repository(root, branch)
    config = load_config(config_path)
    raw = validate_archives(raw_dir)
    identity = experiment_signature("preprocessing", config["preprocessing"], commit, raw)
    destination = Path(processed_base) / identity[:16]
    if (destination / "complete.json").exists():
        verify_processed(destination, raw)
        print(f"Verified existing processed version: {destination}")
        return destination
    if destination.exists():
        raise RecoveryRequired(f"Incomplete preprocessing exists at {destination}; inspect it, then use a new processed base for an explicit restart")
    info = runtime_info()
    destination.mkdir(parents=True)
    write_json(destination / "preprocessing_state.json", {"status": "running", "git_commit": commit,
                                                         "configuration": config["preprocessing"], "raw_manifest": raw,
                                                         "runtime": info})
    prep = config["preprocessing"]
    command = [sys.executable, str(root / "src/preprocessing.py"), "--raw-dir", str(raw_dir),
               "--processed-dir", str(destination), "--cache-dir", "/content/favorita-csv-cache",
               "--start-date", str(prep["start_date"]), "--end-date", str(prep["end_date"]),
               "--chunksize", str(prep["chunksize"]), "--missing-policy", prep["missing_policy"]]
    try:
        with (destination / "preprocessing.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=int(prep["max_seconds"]))
        manifest = file_manifest(destination, ("sales.csv", "nodes.csv", "edges.csv", "data_quality.json"))
        write_json(destination / "complete.json", {"raw_manifest": raw, "processed_manifest": manifest,
                                                    "git_commit": commit, "configuration": prep})
        verify_processed(destination, raw)
        write_json(destination / "preprocessing_state.json", {"status": "completed", "git_commit": commit})
        return destination
    except BaseException as exc:
        write_json(destination / "preprocessing_state.json", {"status": "failed", "failure": str(exc), "git_commit": commit})
        raise


def validate_budget(registry):
    status = registry.status()
    if status["active_experiments"]:
        raise RecoveryRequired(f"Active record(s): {status['active_experiments']}. Confirm no runtime still runs before explicit recovery.")
    return status


def scientific_source_fingerprint(root, revision, model_names=None):
    """Compare evaluation-relevant definitions without conflating logging changes.

    Model/dataset/metric modules are checked in full. Shared loader, scaler and
    prediction helpers are AST-compared. This is not permission to evaluate
    different models or target semantics under the same frozen selection.
    """
    values = {}
    selected_models = set(model_names or ("stgt", "adaptive_stgt", "gat_lstm"))
    train_names = {"ExperimentData", "_absolute_arrays", "evaluate_loader"}

    def module_nodes(source, names=None):
        tree = ast.parse(source)
        if names is not None:
            tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
        elif source:
            tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        return tree

    modules = ["dataset", "metrics", "utils", "train"]
    if "stgt" in selected_models or "adaptive_stgt" in selected_models:
        modules.append("stgt")
    if "gat_lstm" in selected_models:
        modules.append("gat_lstm")
    for name in modules:
        source = subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:src/{name}.py"], text=True)
        if name == "train":
            tree = module_nodes(source, train_names)
        elif name == "utils":
            tree = module_nodes(source)
        elif name == "stgt":
            names = {"TemporalAttention", "ReferenceSTGT"}
            if "adaptive_stgt" in selected_models:
                names |= {"AdaptiveSTGTFusion", "AdaptiveSTGT"}
            tree = module_nodes(source, names)
        elif name == "gat_lstm":
            tree = module_nodes(source)
        else:
            tree = ast.parse(source)
        values[name] = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    return values


def finalize_test(root, config_path, raw_dir, registry_path, experiment_ids, branch=None,
                  resume_final_test=False, allow_evaluator_upgrade=False):
    """Explicit recovery reuses verified stages of the same frozen test pair."""
    config = load_config(config_path)
    artifacts = Path(config["outputs"]["drive_artifact_root"])
    require_colab_drive(raw_dir, registry_path, artifacts, config["data"]["processed_dir"])
    commit = validate_pushed_repository(root, branch)
    raw = validate_archives(raw_dir)
    processed = verify_processed(config["data"]["processed_dir"], raw)
    registry = Registry(registry_path, max_runs=int(config["experiment"]["max_gpu_runs"]))
    validate_budget(registry)
    state = registry.snapshot()
    selected = [record for record in state["experiments"] if record["experiment_id"] in set(experiment_ids)]
    expected_models = {"stgt", "adaptive_stgt", "gat_lstm"}
    expected_seeds = {int(seed) for seed in config.get("seeds", [config["seed"]])}
    expected_pairs = {(model, seed) for model in expected_models for seed in expected_seeds}
    selected_pairs = {(record["model"], int(record["configuration"]["seed"])) for record in selected}
    if len(selected) != len(expected_pairs) or selected_pairs != expected_pairs:
        raise ValueError("Select exactly one completed run for each configured model and seed")
    code_fingerprint = scientific_source_fingerprint(root, commit)
    for record in selected:
        if record["status"] != "completed":
            raise ValueError("Final test requires completed training artifacts")
        if record["dataset_manifest"] != {"raw": raw, "processed": processed}:
            raise ValueError("Selected runs use different dataset versions")
        expected_config = copy.deepcopy(config)
        expected_config["seed"] = int(record["configuration"]["seed"])
        if not compatible_model_configuration(record["configuration"], expected_config, record["model"]):
            raise ValueError("Selected runs/configurations differ from the shared comparison protocol")
        if record["git_commit"] != commit:
            if not allow_evaluator_upgrade:
                raise RecoveryRequired("Selected runs use an older commit. Use --allow-evaluator-upgrade only for an explicitly approved compatible evaluation update.")
            if scientific_source_fingerprint(root, record["git_commit"], {record["model"]}) != scientific_source_fingerprint(root, commit, {record["model"]}):
                raise ValueError(f"Evaluator upgrade changes {record['model']} definitions; frozen checkpoints cannot be silently reinterpreted")
        validate_completed_artifacts(record)
    destination = artifacts / "final_test"
    ids = sorted(experiment_ids)
    frozen_path = destination / "selection.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if (frozen["experiment_ids"] != ids or frozen["git_commit"] != commit
                or frozen["dataset_manifest"] != {"raw": raw, "processed": processed}):
            raise RuntimeError("Test selection is already frozen; do not retune against the test set")
        if (destination / "complete.json").exists():
            manifest = json.loads((destination / "complete.json").read_text())
            verify_manifest(destination, manifest)
            report = destination / "stages/report/model_comparison.csv"
            if not report.exists():
                report = destination / "metrics/model_comparison.csv"  # preserve legacy completed results
            print(f"Verified existing final-test report: {report}")
            return destination
        if not resume_final_test:
            raise RecoveryRequired(f"Incomplete final evaluation at {destination}; inspect status/logs, then use --resume-final-test with the same IDs")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        write_json(frozen_path, {"experiment_ids": ids, "git_commit": commit,
                                 "training_commits": {
                                     record["experiment_id"]: record["git_commit"] for record in selected
                                 },
                                 "dataset_manifest": {"raw": raw, "processed": processed}})
    with evaluation_lock(destination):
        deadline = registry.status().get("deadline_at")
        remaining = (datetime.fromisoformat(deadline).timestamp() - time.time()) if deadline else float("inf")
        if remaining <= 0:
            raise BudgetExceeded("Campaign deadline expired; final-test artifacts are preserved and no evaluation is launched")
        evaluation_limit = min(int(config["experiment"]["max_job_seconds"]), remaining)
        info = runtime_info()
        policy = {"code_fingerprint": code_fingerprint, "packages": info["packages"],
                  "configuration": scientific_configuration(config)}
        policy_path = destination / "evaluation_protocol.json"
        if policy_path.exists():
            if json.loads(policy_path.read_text()) != policy:
                raise ValueError("Evaluation code/package/protocol changed during recovery; existing stages were preserved")
        else:
            write_json(policy_path, policy)
        attempts = destination / "evaluation_attempts"
        attempts.mkdir(exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts))
        frozen_config = copy.deepcopy(config)
        frozen_config["final_evaluation"] = {"selection_frozen": True, "experiment_ids": ids,
                                             "directory": str(destination), "seeds": sorted(expected_seeds), **policy}
        final_config = attempt / "config.yaml"
        final_config.write_text(yaml.safe_dump(frozen_config, sort_keys=False))
        write_json(attempt / "provenance.json", {"training_git_commits": {
                                                    record["experiment_id"]: record["git_commit"] for record in selected
                                                },
                                                "evaluator_git_commit": commit,
                                                "resume_authorized": resume_final_test, "runtime": info})
        checkpoint_specs = [
            f"{record['model']}_seed{int(record['configuration']['seed'])}="
            f"{Path(record['artifact_dir']) / 'checkpoints/best.pt'}"
            for record in sorted(selected, key=lambda item: (item["model"], int(item["configuration"]["seed"])))
        ]
        command = [sys.executable, "-u", str(root / "src/evaluate.py"), "--project-root", str(root), "--config", str(final_config),
                   "--checkpoints", *checkpoint_specs]
        try:
            write_json(attempt / "status.json", {"status": "running"})
            result = run_logged(command, cwd=root, log_path=attempt / "evaluation.log",
                                timeout=evaluation_limit)
            if result.returncode:
                raise RuntimeError(f"Final evaluation exited {result.returncode}; see {attempt / 'evaluation.log'}")
            names = ["selection.json", "evaluation_protocol.json"]
            names.extend(str(path.relative_to(destination)) for path in (destination / "stages").rglob("*") if path.is_file())
            write_json(destination / "complete.json", file_manifest(destination, names))
            write_json(attempt / "status.json", {"status": "completed"})
            write_json(destination / "status.json", {"status": "completed", "attempt": str(attempt)})
            print(f"Test report: {destination / 'stages/report/model_comparison.csv'}")
            return destination
        except BaseException as exc:
            failure = {"status": "failed", "failure": str(exc), "attempt": str(attempt)}
            write_json(attempt / "status.json", failure)
            write_json(destination / "status.json", failure)
            raise


def run_sequential(*, project_root, config_path, raw_dir, registry_path, models,
                   max_runs=10, allow_execution=False, force_retry=False, restart_without_checkpoint=False, branch=None):
    if not allow_execution:
        raise RuntimeError("Execution is disabled; authorize notebook gates first")
    config = load_config(config_path)
    artifacts = Path(config["outputs"]["drive_artifact_root"])
    processed = Path(config["data"]["processed_dir"])
    require_colab_drive(raw_dir, registry_path, artifacts, processed)
    commit = validate_pushed_repository(project_root, branch)
    raw = validate_archives(raw_dir)
    processed_manifest = verify_processed(processed, raw)
    dataset = {"raw": raw, "processed": processed_manifest}
    info = runtime_info()
    if not info["cuda_available"]:
        raise RuntimeError("Select a GPU runtime in Colab; no training run reserved")
    quality_gate(project_root)
    registry = Registry(registry_path, max_runs=min(max_runs, int(config["experiment"]["max_gpu_runs"])),
                        max_retries=int(config["experiment"]["max_retries_per_signature"]))
    validate_budget(registry)
    deadline = registry.establish_deadline(int(config["experiment"]["overall_max_seconds"]))
    deadline_seconds = datetime.fromisoformat(deadline).timestamp()
    reports = []
    seeds = [int(seed) for seed in config.get("seeds", [config["seed"]])]
    for seed in seeds:
        run_configuration = copy.deepcopy(config)
        run_configuration["seed"] = seed
        for model in models:
            signature = experiment_signature(model, scientific_configuration(run_configuration), commit, dataset)
            matches = registry.matching(signature)
            completed = [record for record in matches if record["status"] == "completed"]
            if completed:
                validate_completed_artifacts(completed[-1])
                reports.append({"model": model, "seed": seed, "action": "skip_completed", "record": completed[-1]})
                print(f"Skipped completed run: {completed[-1]['experiment_id']}", flush=True)
                continue
            if not matches:
                compatible = [
                    record for record in registry.snapshot()["experiments"]
                    if record.get("status") == "completed"
                    and record.get("model") == model
                    and record.get("dataset_manifest") == dataset
                    and int(record.get("configuration", {}).get("seed", -1)) == seed
                    and compatible_model_configuration(record["configuration"], run_configuration, model)
                ]
                if compatible:
                    if len(compatible) > 1:
                        raise RuntimeError(f"Multiple compatible completed artifacts found for {model}, seed {seed}; inspect the registry before reuse")
                    validate_completed_artifacts(compatible[0])
                    reports.append({"model": model, "seed": seed, "action": "skip_verified_compatible", "record": compatible[0]})
                    print(f"Reused verified compatible run: {compatible[0]['experiment_id']}", flush=True)
                    continue
            if (artifacts / "final_test/selection.json").exists():
                raise RuntimeError("Test selection is frozen; no further training/search is permitted in this campaign")
            # Rejected recovery is checked before consuming a submission slot.
            if matches:
                if not force_retry:
                    raise RecoveryRequired("Prior attempt requires explicit RETRY authorization")
                if matches[-1].get("runtime", {}).get("packages") not in (None, info["packages"]):
                    raise RecoveryRequired("Package versions differ from the interrupted run. Reinstall its recorded versions before resume.")
                latest = Path(matches[-1]["artifact_dir"]) / "checkpoints/latest.pt"
                if not latest.exists() and not restart_without_checkpoint:
                    raise RecoveryRequired("No committed epoch checkpoint. Inspect the log and use --force-retry --restart-without-checkpoint for an explicit counted restart.")
            remaining_seconds = int(deadline_seconds - time.time())
            if remaining_seconds <= 0:
                raise BudgetExceeded("Campaign deadline reached; checkpoints and logs remain on Drive")
            run_limit = min(int(config["experiment"]["max_job_seconds"]), remaining_seconds)
            reservation = registry.reserve(
                model=model,
                signature=signature,
                configuration=run_configuration,
                git_commit=commit,
                dataset_manifest=dataset,
                artifact_dir=artifacts,
                force_retry=force_retry,
            )
            record = reservation["record"]
            experiment_id = record["experiment_id"]
            artifact = Path(record["artifact_dir"])
            try:
                artifact.mkdir(parents=True, exist_ok=False)
                run_config = copy.deepcopy(run_configuration)
                run_config["experiment"]["max_job_seconds"] = max(1, run_limit - 30)
                run_config["outputs"].update({
                    key: str(artifact / sub)
                    for key, sub in (
                        ("checkpoint_dir", "checkpoints"),
                        ("metrics_dir", "metrics"),
                        ("predictions_dir", "predictions"),
                        ("logs_dir", "logs"),
                    )
                })
                run_config["run"] = {"experiment_id": experiment_id, "signature": signature, "git_commit": commit}
                attempt_config = artifact / "config.yaml"
                attempt_config.write_text(yaml.safe_dump(run_config, sort_keys=False))
                write_json(artifact / "runtime.json", info)
                write_json(artifact / "dataset_manifest.json", dataset)
                command = [
                    sys.executable,
                    "-u",
                    str(project_root / "src/train.py"),
                    "--model",
                    model,
                    "--config",
                    str(attempt_config),
                    "--project-root",
                    str(project_root),
                ]
                if matches:
                    latest = Path(matches[-1]["artifact_dir"]) / "checkpoints/latest.pt"
                    if latest.exists():
                        command.extend(["--resume", str(latest)])
                    else:
                        registry.update(experiment_id, "reserved", restart_reason="explicit restart authorized; no epoch checkpoint exists")
                        print(f"Explicit counted restart from epoch 1: {experiment_id}", flush=True)
                log_path = artifact / "training.log"
                registry.update(experiment_id, "running", command=command, runtime=info, config_path=str(attempt_config))
                print(f"Running {experiment_id}; budget={registry.status()['submitted_runs']}/{registry.status()['max_runs']}; log={log_path}", flush=True)
                result = run_logged(command, cwd=project_root, log_path=log_path, timeout=run_limit)
                if result.returncode == 124:
                    raise subprocess.TimeoutExpired(command, run_limit)
                if result.returncode:
                    raise RuntimeError(f"Training exited {result.returncode}; see {log_path}")
                relative_files = [
                    "config.yaml",
                    "runtime.json",
                    "dataset_manifest.json",
                    "training.log",
                    "checkpoints/best.pt",
                    "checkpoints/latest.pt",
                    "predictions/validation.npz",
                    "metrics/validation.json",
                    "logs/history.json",
                ]
                report = json.loads((artifact / "metrics/validation.json").read_text())
                if report["experiment_id"] != experiment_id or report["signature"] != signature:
                    raise ValueError("Returned metrics do not match the registered attempt")
                artifact_manifest = file_manifest(artifact, relative_files)
                registry.update(
                    experiment_id,
                    "completed",
                    exit_code=0,
                    report=str(artifact / "metrics/validation.json"),
                    artifact_manifest=artifact_manifest,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                reports.append({"model": model, "seed": seed, "action": "completed", "record": registry.matching(signature)[-1]})
            except BaseException as exc:
                status = "interrupted" if isinstance(exc, (KeyboardInterrupt, subprocess.TimeoutExpired)) else "failed"
                registry.update(
                    experiment_id,
                    status,
                    failure={"type": type(exc).__name__, "reason": str(exc)},
                    latest_checkpoint=str(artifact / "checkpoints/latest.pt"),
                )
                # Stop rather than silently submit retries or consume the next slot.
                raise
    return reports


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["stgt", "adaptive_stgt", "gat_lstm"],
                        default=["stgt", "adaptive_stgt", "gat_lstm"])
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--allow-execution", action="store_true")
    parser.add_argument("--force-retry", action="store_true")
    parser.add_argument("--restart-without-checkpoint", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true")
    parser.add_argument("--finalize-test", action="store_true")
    parser.add_argument("--resume-final-test", action="store_true")
    parser.add_argument("--allow-evaluator-upgrade", action="store_true")
    parser.add_argument("--experiment-ids", nargs="+")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    root = args.project_root.resolve()
    registry = Registry(args.registry, max_runs=args.max_runs)
    if args.status_only:
        print(json.dumps(registry.status(), indent=2))
    elif args.recover_interrupted:
        confirm = input("Confirm no notebook/process is still running. Type RECOVER: ")
        if confirm != "RECOVER":
            raise SystemExit("Recovery not confirmed")
        print(registry.mark_interrupted_active("user-confirmed inactive previous session"))
    elif args.finalize_test:
        if not args.allow_execution or not args.experiment_ids or args.raw_dir is None:
            raise SystemExit("Final-test gate requires --allow-execution, --experiment-ids, and --raw-dir")
        finalize_test(root, args.config.resolve(), args.raw_dir, args.registry, args.experiment_ids, args.branch,
                      args.resume_final_test, args.allow_evaluator_upgrade)
    else:
        if args.raw_dir is None:
            raise SystemExit("--raw-dir is required for execution")
        if args.restart_without_checkpoint and not args.force_retry:
            raise SystemExit("--restart-without-checkpoint requires --force-retry")
        print(json.dumps(run_sequential(project_root=root, config_path=args.config.resolve(), raw_dir=args.raw_dir,
                                       registry_path=args.registry, models=args.models, max_runs=args.max_runs,
                                       allow_execution=args.allow_execution, force_retry=args.force_retry,
                                       restart_without_checkpoint=args.restart_without_checkpoint,
                                       branch=args.branch), indent=2, default=str))
