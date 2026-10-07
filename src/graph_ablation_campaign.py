"""Independent validation-only current-vs-geographic graph campaign."""

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

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestrator import (
    file_manifest,
    load_config,
    quality_gate,
    require_colab_drive,
    run_logged,
    runtime_info,
    validate_archives,
    validate_pushed_repository,
    verify_processed,
)


CAMPAIGN = "graph_ablation"
MAX_RUNS = 6
SEEDS = (42, 123, 2026)
VARIANTS = ("current", "geographic_only")


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def signature(model, variant, config, commit, dataset):
    payload = {"model": model, "variant": variant, "config": config, "commit": commit, "dataset": dataset}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class Registry:
    def __init__(self, path):
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
                    raise TimeoutError(f"Could not acquire graph-ablation registry lock: {self.lock_path}")
                time.sleep(0.1)
        try:
            os.write(descriptor, str(os.getpid()).encode())
            yield
        finally:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    def read(self):
        if not self.path.exists():
            now = datetime.now(timezone.utc).isoformat()
            return {"schema_version": 1, "campaign": CAMPAIGN, "created_at": now, "updated_at": now,
                    "budget": {"max_runs": MAX_RUNS, "submitted_runs": 0, "max_retries_per_signature": 0},
                    "variants": list(VARIANTS), "seeds": list(SEEDS), "experiments": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("campaign") != CAMPAIGN or payload.get("schema_version") != 1:
            raise ValueError("Wrong graph-ablation registry")
        if payload["budget"]["max_runs"] != MAX_RUNS or payload["budget"]["submitted_runs"] != len(payload["experiments"]):
            raise ValueError("Graph-ablation registry budget is invalid")
        return payload

    def write(self, payload):
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, payload)

    def status(self):
        with self.lock():
            payload = self.read()
        return {"campaign": CAMPAIGN, "submitted_runs": payload["budget"]["submitted_runs"],
                "max_runs": MAX_RUNS, "remaining_runs": MAX_RUNS - payload["budget"]["submitted_runs"],
                "active_experiments": [r["experiment_id"] for r in payload["experiments"] if r["status"] in {"reserved", "running"}],
                "completed_experiments": [r["experiment_id"] for r in payload["experiments"] if r["status"] == "completed"]}

    def reserve(self, model, variant, seed, sig, config, commit, dataset, artifact_root):
        with self.lock():
            payload = self.read()
            matches = [r for r in payload["experiments"] if r["signature"] == sig]
            if any(r["status"] == "completed" for r in matches):
                return matches[-1]
            if any(r["status"] in {"reserved", "running"} for r in payload["experiments"]):
                raise RuntimeError("Another graph-ablation run is active")
            if payload["budget"]["submitted_runs"] >= MAX_RUNS:
                raise RuntimeError("Graph-ablation six-run budget is exhausted")
            experiment_id = f"graph_ablation_{variant}_stgt_seed{seed}-{sig[:12]}"
            record = {"experiment_id": experiment_id, "campaign": CAMPAIGN, "model": model, "variant": variant,
                      "seed": seed, "signature": sig, "status": "reserved", "attempt": 1,
                      "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(),
                      "git_commit": commit, "configuration": copy.deepcopy(config), "dataset_manifest": copy.deepcopy(dataset),
                      "artifact_dir": str(Path(artifact_root) / experiment_id)}
            payload["experiments"].append(record)
            payload["budget"]["submitted_runs"] += 1
            self.write(payload)
            return record

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
    parser.add_argument("--config", type=Path, default=Path("configs/graph_ablation.yaml"))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--branch", default="research/initial-pipeline")
    parser.add_argument("--allow-execution", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    registry = Registry(args.registry.resolve())
    if args.status_only:
        print(json.dumps(registry.status(), indent=2))
        return
    if not args.allow_execution:
        raise SystemExit("Preparation only: add --allow-execution in authorized Colab to train")
    config = load_config(args.config.resolve())
    root = args.project_root.resolve()
    registry_path = args.registry.resolve()
    artifact_root = Path(config["outputs"]["drive_artifact_root"])
    if registry_path.name != "experiment_registry_graph_ablation.json":
        raise SystemExit("Use the separate experiment_registry_graph_ablation.json")
    if artifact_root.name != "graph_ablation" or "adaptive_stgt_campaign" not in str(artifact_root):
        raise SystemExit("Graph-ablation artifacts must use the graph_ablation namespace")
    if (artifact_root / "final_test").exists():
        raise SystemExit("Graph-ablation namespace must not contain final_test")
    if tuple(config["seeds"]) != SEEDS or set(config["graph_variants"]) != set(VARIANTS):
        raise SystemExit("Graph-ablation configuration does not contain the expected variants/seeds")

    require_colab_drive(args.raw_dir, registry_path, artifact_root, config["data"]["processed_dir"])
    commit = validate_pushed_repository(root, args.branch)
    raw_manifest = validate_archives(args.raw_dir)
    processed_manifest = verify_processed(config["data"]["processed_dir"], raw_manifest)
    dataset = {"raw": raw_manifest, "processed": processed_manifest}
    info = runtime_info()
    if not info["cuda_available"]:
        raise RuntimeError("No CUDA GPU available; no graph-ablation run was reserved")
    quality_gate(root)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=int(config["experiment"]["overall_max_seconds"]))

    for variant in VARIANTS:
        for seed in SEEDS:
            run_config = copy.deepcopy(config)
            run_config["seed"] = seed
            run_config["data"]["graph_edge_types"] = list(config["graph_variants"][variant]["graph_edge_types"])
            run_config["graph_variant"] = variant
            sig = signature("stgt", variant, run_config, commit, dataset)
            existing = registry.reserve("stgt", variant, seed, sig, run_config, commit, dataset, artifact_root)
            if existing.get("status") == "completed":
                print(f"Skipped completed graph-ablation run: {existing['experiment_id']}", flush=True)
                continue
            experiment_id = existing["experiment_id"]
            artifact = Path(existing["artifact_dir"])
            artifact.mkdir(parents=True, exist_ok=False)
            remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
            if remaining <= 0:
                raise RuntimeError("Graph-ablation campaign deadline reached")
            run_config["experiment"]["max_job_seconds"] = min(int(config["experiment"]["max_job_seconds"]), max(1, remaining - 30))
            run_config["outputs"].update({"checkpoint_dir": str(artifact / "checkpoints"), "metrics_dir": str(artifact / "metrics"),
                                          "predictions_dir": str(artifact / "predictions"), "logs_dir": str(artifact / "logs")})
            run_config["run"] = {"experiment_id": experiment_id, "signature": sig, "git_commit": commit}
            config_path = artifact / "config.yaml"
            config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            (artifact / "runtime.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            (artifact / "dataset_manifest.json").write_text(json.dumps(dataset, indent=2, default=str), encoding="utf-8")
            command = [sys.executable, "-u", str(root / "src/train.py"), "--model", "stgt", "--config", str(config_path), "--project-root", str(root)]
            log_path = artifact / "training.log"
            registry.update(experiment_id, "running", command=command, runtime=info, config_path=str(config_path))
            print(f"Running {experiment_id}; graph={variant}; log={log_path}", flush=True)
            try:
                result = run_logged(command, cwd=root, log_path=log_path, timeout=run_config["experiment"]["max_job_seconds"])
                if result.returncode:
                    raise RuntimeError(f"Graph-ablation training exited {result.returncode}; see {log_path}")
                relative_files = ["config.yaml", "runtime.json", "dataset_manifest.json", "training.log", "checkpoints/best.pt", "checkpoints/latest.pt", "predictions/validation.npz", "metrics/validation.json", "logs/history.json"]
                report = json.loads((artifact / "metrics/validation.json").read_text(encoding="utf-8"))
                if report["experiment_id"] != experiment_id or report["signature"] != sig:
                    raise ValueError("Validation report does not match graph-ablation registry")
                registry.update(experiment_id, "completed", exit_code=0, report=str(artifact / "metrics/validation.json"),
                                artifact_manifest=file_manifest(artifact, relative_files), completed_at=datetime.now(timezone.utc).isoformat())
            except BaseException as exc:
                registry.update(experiment_id, "failed", failure={"type": type(exc).__name__, "reason": str(exc)})
                raise


if __name__ == "__main__":
    main()
