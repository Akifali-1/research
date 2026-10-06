"""Semi-automated controller: user-authorized Colab, Drive, then sequential runs.

Only --allow-execution can start compute. Run signatures, immutable attempt
directories, counted retries, validation-only selection, runtime limits, and
artifact verification are shared by initial and resumed Colab sessions.
"""
from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.experiment_registry import BudgetExceeded, RecoveryRequired, Registry, experiment_signature
from src.raw_data import dataset_layout_error, source_paths
from src.process_runner import run_logged

EXPECTED_ARCHIVES = {
    f"{name}.csv.7z" for name in ("train", "items", "stores", "transactions", "oil",
                                  "holidays_events", "test", "sample_submission")
}


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


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
        if not Path(path).resolve().is_relative_to(mydrive.resolve()):
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
    # Validate small metadata schemas without expanding the train archive.
    import tempfile
    import csv
    with tempfile.TemporaryDirectory(prefix="favorita-schema-") as folder:
        for stem, required in (("items", {"item_nbr", "family", "class", "perishable"}),
                               ("stores", {"store_nbr", "city", "state", "type", "cluster"})):
            csv_path = sources[stem]
            if csv_path.name.endswith(".7z"):
                with py7zr.SevenZipFile(csv_path, "r") as archive:
                    archive.extractall(path=folder)
                csv_path = Path(folder) / f"{stem}.csv"
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                columns = next(csv.reader(handle))
            if not required.issubset(columns):
                raise ValueError(f"{stem}.csv is not the expected Favorita schema")
    # Direct CSV input allows train-header verification without extraction.
    if sources["train"].suffix == ".csv":
        with sources["train"].open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        if not {"id", "date", "store_nbr", "item_nbr", "unit_sales"}.issubset(columns):
            raise ValueError("train.csv is not the expected Favorita schema")
    return manifest


def scientific_configuration(config):
    data = {key: value for key, value in config["data"].items() if key != "processed_dir"}
    return {"seed": config["seed"], "deterministic": config.get("deterministic", True),
            "data": data, "training": config["training"], "models": config["models"]}


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


def validate_training_commit_compatibility(root, training_commit, evaluation_commit):
    """Allow only the specifically reviewed logging-only commit transition."""
    if training_commit == evaluation_commit:
        return {
            "mode": "exact_commit",
            "training_commit": training_commit,
            "evaluation_commit": evaluation_commit,
        }

    approved_training = "ac172c779ae0c5194d9e7190c7369bd6f7fbb42e"
    logging_commit = "db84e9ebdf0bdcdb44bbaaec3207b2d64e9da2e9"
    expected_paths = {
        "README.md",
        "orchestrator.py",
        "src/process_runner.py",
        "src/train.py",
        "tests/test_process_runner.py",
    }

    if training_commit != approved_training:
        raise ValueError(
            f"Unapproved training commit for compatibility exception: {training_commit}"
        )

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    def is_ancestor(ancestor, descendant):
        result = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor",
             ancestor, descendant],
            capture_output=True,
        )
        return result.returncode == 0

    if not is_ancestor(training_commit, logging_commit):
        raise ValueError("Approved logging commit is not based on the training commit")
    if not is_ancestor(logging_commit, evaluation_commit):
        raise ValueError("Evaluation commit does not descend from the approved logging commit")

    original_changes = set(
        git("diff", "--name-only", training_commit, logging_commit).splitlines()
    )
    if original_changes != expected_paths:
        raise ValueError(
            f"Logging commit changed unexpected files: {sorted(original_changes)}"
        )

    # Require exactly one additional commit: the reviewed compatibility patch.
    if git("rev-list", "--count", f"{logging_commit}..{evaluation_commit}") != "1":
        raise ValueError("Expected exactly one compatibility-patch commit after the logging commit")
    if git("rev-parse", f"{evaluation_commit}^") != logging_commit:
        raise ValueError("Compatibility patch must be a direct child of the logging commit")

    patch_paths = set(
        git("diff", "--name-only", logging_commit, evaluation_commit).splitlines()
    )
    if patch_paths != {"orchestrator.py"}:
        raise ValueError(
            f"Compatibility patch may change only orchestrator.py: {sorted(patch_paths)}"
        )

    return {
        "mode": "approved_logging_only",
        "training_commit": training_commit,
        "evaluation_commit": evaluation_commit,
        "approved_logging_commit": logging_commit,
        "reviewed_changed_paths": sorted(original_changes),
        "compatibility_patch_paths": sorted(patch_paths),
    }


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


def finalize_test(root, config_path, raw_dir, registry_path, experiment_ids, branch=None):
    """Freeze validation-selected IDs before accessing test data; never retrain."""
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
    if len(selected) != 2 or {record["model"] for record in selected} != {"stgt", "gat_lstm"}:
        raise ValueError("Select exactly one completed STGT ID and one completed GAT-LSTM ID")
    compatibility_by_id = {}
    for record in selected:
        if record["status"] != "completed":
            raise ValueError("Final test requires completed training runs")
        compatibility_by_id[record["experiment_id"]] = (
            validate_training_commit_compatibility(
                root, record["git_commit"], commit
            )
        )
        if record["dataset_manifest"] != {"raw": raw, "processed": processed}:
            raise ValueError("Selected runs use different dataset versions")
        if scientific_configuration(record["configuration"]) != scientific_configuration(config):
            raise ValueError("Selected runs/configurations differ from the shared comparison protocol")
        validate_completed_artifacts(record)
    destination = artifacts / "final_test"
    ids = sorted(experiment_ids)
    frozen_path = destination / "selection.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        expected_training_commits = {
            record["experiment_id"]: record["git_commit"] for record in selected
        }
        if (
            frozen["experiment_ids"] != ids
            or frozen["git_commit"] != commit
            or frozen.get("training_commits") != expected_training_commits
        ):
            raise RuntimeError("Test selection is already frozen; do not retune against the test set")
        if (destination / "complete.json").exists():
            manifest = json.loads((destination / "complete.json").read_text())
            if file_manifest(destination, [item["name"] for item in manifest["files"]]) != manifest:
                raise ValueError("Final-test artifacts changed or are missing")
            print(f"Verified existing final-test report: {destination / 'metrics/model_comparison.csv'}")
            return destination
        raise RecoveryRequired(f"Incomplete final evaluation at {destination}; inspect failure. No training is repeated.")
    destination.mkdir(parents=True, exist_ok=False)
    write_json(frozen_path, {
        "experiment_ids": ids,
        "git_commit": commit,
        "training_commits": {
            record["experiment_id"]: record["git_commit"] for record in selected
        },
        "compatibility": compatibility_by_id,
        "dataset_manifest": {"raw": raw, "processed": processed},
    })
    frozen_config = copy.deepcopy(config)
    frozen_config["outputs"]["metrics_dir"] = str(destination / "metrics")
    frozen_config["outputs"]["predictions_dir"] = str(destination / "predictions")
    frozen_config["final_evaluation"] = {"selection_frozen": True, "experiment_ids": ids}
    final_config = destination / "config.yaml"
    final_config.write_text(yaml.safe_dump(frozen_config, sort_keys=False))
    checkpoint_paths = {record["model"]: Path(record["artifact_dir"]) / "checkpoints/best.pt" for record in selected}
    command = [sys.executable, "-u", str(root / "src/evaluate.py"), "--project-root", str(root), "--config", str(final_config),
               "--stgt-checkpoint", str(checkpoint_paths["stgt"]), "--gat-lstm-checkpoint", str(checkpoint_paths["gat_lstm"])]
    try:
        with (destination / "evaluation.log").open("w") as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=int(config["experiment"]["max_job_seconds"]))
        manifest = file_manifest(destination, [str(path.relative_to(destination)) for path in destination.rglob("*")
                                               if path.is_file() and path.name not in {"complete.json", "status.json"}])
        write_json(destination / "complete.json", manifest)
        write_json(destination / "status.json", {"status": "completed"})
        print(f"Test report: {destination / 'metrics/model_comparison.csv'}")
        return destination
    except BaseException as exc:
        write_json(destination / "status.json", {"status": "failed", "failure": str(exc)})
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
    for model in models:
        signature = experiment_signature(model, scientific_configuration(config), commit, dataset)
        matches = registry.matching(signature)
        completed = [record for record in matches if record["status"] == "completed"]
        if completed:
            validate_completed_artifacts(completed[-1])
            reports.append({"model": model, "action": "skip_completed", "record": completed[-1]})
            print(f"Skipped completed run: {completed[-1]['experiment_id']}", flush=True)
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
        reservation = registry.reserve(model=model, signature=signature, configuration=config, git_commit=commit,
                                       dataset_manifest=dataset, artifact_dir=artifacts, force_retry=force_retry)
        record = reservation["record"]
        experiment_id = record["experiment_id"]
        artifact = Path(record["artifact_dir"])
        try:
            artifact.mkdir(parents=True, exist_ok=False)
            run_config = copy.deepcopy(config)
            run_config["experiment"]["max_job_seconds"] = max(1, run_limit - 30)
            run_config["outputs"].update({key: str(artifact / sub) for key, sub in
                                         (("checkpoint_dir", "checkpoints"), ("metrics_dir", "metrics"),
                                          ("predictions_dir", "predictions"), ("logs_dir", "logs"))})
            run_config["run"] = {"experiment_id": experiment_id, "signature": signature, "git_commit": commit}
            attempt_config = artifact / "config.yaml"
            attempt_config.write_text(yaml.safe_dump(run_config, sort_keys=False))
            write_json(artifact / "runtime.json", info)
            write_json(artifact / "dataset_manifest.json", dataset)
            command = [sys.executable, "-u", str(project_root / "src/train.py"), "--model", model,
                       "--config", str(attempt_config), "--project-root", str(project_root)]
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
            if result.returncode:
                raise RuntimeError(f"Training exited {result.returncode}; see {log_path}")
            relative_files = ["config.yaml", "runtime.json", "dataset_manifest.json", "training.log",
                              "checkpoints/best.pt", "checkpoints/latest.pt", "predictions/validation.npz", "metrics/validation.json", "logs/history.json"]
            report = json.loads((artifact / "metrics/validation.json").read_text())
            if report["experiment_id"] != experiment_id or report["signature"] != signature:
                raise ValueError("Returned metrics do not match the registered attempt")
            artifact_manifest = file_manifest(artifact, relative_files)
            registry.update(experiment_id, "completed", exit_code=0, report=str(artifact / "metrics/validation.json"),
                            artifact_manifest=artifact_manifest, completed_at=datetime.now(timezone.utc).isoformat())
            reports.append({"model": model, "action": "completed", "record": registry.matching(signature)[-1]})
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, subprocess.TimeoutExpired)) else "failed"
            registry.update(experiment_id, status, failure={"type": type(exc).__name__, "reason": str(exc)},
                            latest_checkpoint=str(artifact / "checkpoints/latest.pt"))
            # Stop rather than silently submit retries or consume the next slot.
            raise
    return reports


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["stgt", "gat_lstm"], default=["stgt", "gat_lstm"])
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--allow-execution", action="store_true")
    parser.add_argument("--force-retry", action="store_true")
    parser.add_argument("--restart-without-checkpoint", action="store_true")
    parser.add_argument("--branch")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true")
    parser.add_argument("--finalize-test", action="store_true")
    parser.add_argument("--experiment-ids", nargs=2)
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
        finalize_test(root, args.config.resolve(), args.raw_dir, args.registry, args.experiment_ids, args.branch)
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
