"""Independent 15-run, validation-only Day 2 ablation campaign controller.

This controller deliberately uses a separate registry implementation and
artifact root. It does not call the original ten-run campaign registry and it
has no final-test or checkpoint-evaluation path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from orchestrator import (
    file_manifest,
    load_config,
    model_scientific_configuration,
    quality_gate,
    require_colab_drive,
    run_logged,
    runtime_info,
    validate_archives,
    validate_pushed_repository,
    verify_processed,
)
from src.experiment_registry import experiment_signature


MAX_RUNS = 15
SEEDS = (42, 123, 2026)
DAY2_VARIANTS = ("temporal_only", "spatial_only", "fixed_fusion", "stgt", "adaptive_stgt")


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


class Day2Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 60
        descriptor = None
        while descriptor is None:
            try:
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Could not acquire Day 2 registry lock: {self.lock_path}")
                time.sleep(0.1)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            yield
        finally:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    def default(self):
        now = datetime.now(timezone.utc).isoformat()
        return {
            "schema_version": 1,
            "campaign": "day2_ablation",
            "created_at": now,
            "updated_at": now,
            "budget": {"max_runs": MAX_RUNS, "submitted_runs": 0, "max_retries_per_signature": 0},
            "variants": list(DAY2_VARIANTS),
            "seeds": list(SEEDS),
            "experiments": [],
        }

    def read(self):
        if not self.path.exists():
            return self.default()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("campaign") != "day2_ablation" or payload.get("schema_version") != 1:
            raise ValueError("Registry is not the Day 2 ablation registry")
        if payload["budget"]["max_runs"] != MAX_RUNS:
            raise ValueError("Day 2 registry budget must remain exactly 15 runs")
        if payload["budget"]["submitted_runs"] != len(payload["experiments"]):
            raise ValueError("Day 2 registry budget count is inconsistent")
        return payload

    def write(self, payload):
        payload = dict(payload)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, payload)

    def snapshot(self):
        with self.lock():
            return self.read()

    def status(self):
        payload = self.snapshot()
        return {
            "campaign": payload["campaign"],
            "submitted_runs": payload["budget"]["submitted_runs"],
            "max_runs": payload["budget"]["max_runs"],
            "remaining_runs": MAX_RUNS - payload["budget"]["submitted_runs"],
            "active_experiments": [item["experiment_id"] for item in payload["experiments"] if item["status"] in {"reserved", "running"}],
            "completed_experiments": [item["experiment_id"] for item in payload["experiments"] if item["status"] == "completed"],
        }

    def reserve(self, *, model, seed, signature, configuration, commit, dataset, artifact_root):
        with self.lock():
            payload = self.read()
            matches = [item for item in payload["experiments"] if item["signature"] == signature]
            if any(item["status"] == "completed" for item in matches):
                return {"action": "skip_completed", "record": matches[-1]}
            if any(item["status"] in {"reserved", "running"} for item in payload["experiments"]):
                raise RuntimeError("Another Day 2 run is active")
            if payload["budget"]["submitted_runs"] >= MAX_RUNS:
                raise RuntimeError("Day 2 15-run budget is exhausted")
            experiment_id = f"day2_{model}-{signature[:12]}-attempt1"
            record = {
                "experiment_id": experiment_id,
                "campaign": "day2_ablation",
                "model": model,
                "seed": seed,
                "attempt": 1,
                "signature": signature,
                "status": "reserved",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": commit,
                "configuration": copy.deepcopy(configuration),
                "dataset_manifest": copy.deepcopy(dataset),
                "artifact_dir": str(Path(artifact_root) / experiment_id),
            }
            payload["experiments"].append(record)
            payload["budget"]["submitted_runs"] += 1
            self.write(payload)
            return {"action": "reserved", "record": record}

    def update(self, experiment_id, status, **fields):
        with self.lock():
            payload = self.read()
            for record in payload["experiments"]:
                if record["experiment_id"] == experiment_id:
                    record.update(fields)
                    record["status"] = status
                    record["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self.write(payload)
                    return record
        raise KeyError(experiment_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/day2_ablation.yaml"))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--branch", default="research/initial-pipeline")
    parser.add_argument("--allow-execution", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_config(args.config.resolve())
    registry_path = args.registry.resolve()
    registry = Day2Registry(registry_path)
    if args.status_only:
        print(json.dumps(registry.status(), indent=2))
        return
    if not args.allow_execution:
        raise SystemExit("Day 2 execution is disabled; add --allow-execution only in the authorized Colab runtime")
    if args.raw_dir is None:
        raise SystemExit("--raw-dir is required")
    if registry_path.name != "experiment_registry_day2_ablation.json":
        raise SystemExit("Use the new experiment_registry_day2_ablation.json path")
    artifact_root = Path(config["outputs"]["drive_artifact_root"])
    if "day2_ablation" not in str(artifact_root) or artifact_root.name != "day2_ablation":
        raise SystemExit("Day 2 artifacts must use the day2_ablation namespace")
    if (artifact_root / "final_test").exists():
        raise SystemExit("Day 2 namespace must not contain final_test artifacts")
    if "final_evaluation" in config:
        raise SystemExit("Day 2 config must not define final evaluation")
    if tuple(config.get("seeds", [])) != SEEDS:
        raise SystemExit("Day 2 requires seeds [42, 123, 2026]")
    if int(config["experiment"]["max_gpu_runs"]) != MAX_RUNS:
        raise SystemExit("Day 2 requires a 15-run budget")
    if set(config["models"]) != set(DAY2_VARIANTS):
        raise SystemExit(f"Day 2 config variants must be exactly {DAY2_VARIANTS}")

    require_colab_drive(args.raw_dir, registry_path, artifact_root, config["data"]["processed_dir"])
    commit = validate_pushed_repository(root, args.branch)
    raw_manifest = validate_archives(args.raw_dir)
    processed_manifest = verify_processed(config["data"]["processed_dir"], raw_manifest)
    dataset = {"raw": raw_manifest, "processed": processed_manifest}
    info = runtime_info()
    if not info["cuda_available"]:
        raise RuntimeError("No CUDA GPU is available; no Day 2 run was reserved")
    quality_gate(root)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=int(config["experiment"]["overall_max_seconds"]))

    reports = []
    for model in DAY2_VARIANTS:
        for seed in SEEDS:
            run_config = copy.deepcopy(config)
            run_config["seed"] = seed
            signature = experiment_signature(model, model_scientific_configuration(run_config, model), commit, dataset)
            reservation = registry.reserve(model=model, seed=seed, signature=signature,
                                           configuration=run_config, commit=commit, dataset=dataset,
                                           artifact_root=artifact_root)
            if reservation["action"] == "skip_completed":
                reports.append(reservation["record"])
                print(f"Skipped completed Day 2 run: {reservation['record']['experiment_id']}", flush=True)
                continue
            record = reservation["record"]
            experiment_id = record["experiment_id"]
            artifact = Path(record["artifact_dir"])
            artifact.mkdir(parents=True, exist_ok=False)
            remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
            if remaining <= 0:
                raise RuntimeError("Day 2 campaign deadline reached")
            run_config["experiment"]["max_job_seconds"] = min(int(config["experiment"]["max_job_seconds"]), max(1, remaining - 30))
            run_config["outputs"].update({
                "checkpoint_dir": str(artifact / "checkpoints"),
                "metrics_dir": str(artifact / "metrics"),
                "predictions_dir": str(artifact / "predictions"),
                "logs_dir": str(artifact / "logs"),
            })
            run_config["run"] = {"experiment_id": experiment_id, "signature": signature, "git_commit": commit}
            attempt_config = artifact / "config.yaml"
            attempt_config.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            (artifact / "runtime.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            (artifact / "dataset_manifest.json").write_text(json.dumps(dataset, indent=2, default=str), encoding="utf-8")
            command = [sys.executable, "-u", str(root / "src/train.py"), "--model", model,
                       "--config", str(attempt_config), "--project-root", str(root)]
            log_path = artifact / "training.log"
            registry.update(experiment_id, "running", command=command, runtime=info, config_path=str(attempt_config))
            print(f"Running {experiment_id}; log={log_path}", flush=True)
            try:
                result = run_logged(command, cwd=root, log_path=log_path, timeout=run_config["experiment"]["max_job_seconds"])
                if result.returncode:
                    raise RuntimeError(f"Day 2 training exited {result.returncode}; see {log_path}")
                relative_files = ["config.yaml", "runtime.json", "dataset_manifest.json", "training.log",
                                  "checkpoints/best.pt", "checkpoints/latest.pt", "predictions/validation.npz",
                                  "metrics/validation.json", "logs/history.json"]
                report = json.loads((artifact / "metrics/validation.json").read_text(encoding="utf-8"))
                if report["experiment_id"] != experiment_id or report["signature"] != signature:
                    raise ValueError("Validation report does not match the Day 2 registry record")
                manifest = file_manifest(artifact, relative_files)
                record = registry.update(experiment_id, "completed", exit_code=0,
                                         report=str(artifact / "metrics/validation.json"),
                                         artifact_manifest=manifest,
                                         completed_at=datetime.now(timezone.utc).isoformat())
                reports.append(record)
            except BaseException as exc:
                registry.update(experiment_id, "failed", failure={"type": type(exc).__name__, "reason": str(exc)})
                raise
    print(json.dumps({"campaign": "day2_ablation", "completed": len(reports), "registry": str(registry_path)}, indent=2, default=str))


if __name__ == "__main__":
    main()
