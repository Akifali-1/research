"""Independent four-run ISOMORPH validation-only campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

from orchestrator import file_manifest, load_config, quality_gate, run_logged, runtime_info, validate_pushed_repository


MODELS = ("temporal_only", "spatial_only", "stgt", "adaptive_stgt")
SEED = 42
MAX_RUNS = 4


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def make_signature(model, config, commit, processed_manifest):
    payload = {"dataset": "isomorph_sample_inventory", "model": model, "config": config, "commit": commit, "processed": processed_manifest}
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
                    raise TimeoutError("Could not acquire ISOMORPH registry lock")
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
            return {"schema_version": 1, "campaign": "isomorph_sample_seed42", "created_at": now, "updated_at": now,
                    "budget": {"max_runs": MAX_RUNS, "submitted_runs": 0, "max_retries_per_signature": 0},
                    "models": list(MODELS), "seed": SEED, "experiments": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("campaign") != "isomorph_sample_seed42" or payload["budget"]["max_runs"] != MAX_RUNS:
            raise ValueError("Invalid ISOMORPH campaign registry")
        if payload["budget"]["submitted_runs"] != len(payload["experiments"]):
            raise ValueError("ISOMORPH registry budget count is inconsistent")
        return payload

    def write(self, payload):
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, payload)

    def status(self):
        with self.lock():
            payload = self.read()
        return {"campaign": payload["campaign"], "submitted_runs": payload["budget"]["submitted_runs"],
                "max_runs": MAX_RUNS, "remaining_runs": MAX_RUNS - payload["budget"]["submitted_runs"],
                "active_experiments": [r["experiment_id"] for r in payload["experiments"] if r["status"] in {"reserved", "running"}],
                "completed_experiments": [r["experiment_id"] for r in payload["experiments"] if r["status"] == "completed"]}

    def reserve(self, model, signature, config, commit, processed_manifest, artifact_root):
        with self.lock():
            payload = self.read()
            matches = [r for r in payload["experiments"] if r["signature"] == signature]
            if any(r["status"] == "completed" for r in matches):
                return matches[-1]
            if any(r["status"] in {"reserved", "running"} for r in payload["experiments"]):
                raise RuntimeError("Another ISOMORPH run is active")
            if payload["budget"]["submitted_runs"] >= MAX_RUNS:
                raise RuntimeError("ISOMORPH four-run budget is exhausted")
            experiment_id = f"isomorph_sample_{model}_seed42-{signature[:12]}"
            record = {"experiment_id": experiment_id, "campaign": "isomorph_sample_seed42", "model": model, "seed": SEED,
                      "signature": signature, "status": "reserved", "attempt": 1,
                      "created_at": datetime.now(timezone.utc).isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(),
                      "git_commit": commit, "configuration": copy.deepcopy(config), "processed_manifest": processed_manifest,
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
    parser.add_argument("--config", type=Path, default=Path("configs/isomorph_seed42.yaml"))
    parser.add_argument("--processed-root", type=Path, required=True)
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
        raise SystemExit("Preparation only; add --allow-execution in Colab to train")
    config = load_config(args.config.resolve())
    root = args.project_root.resolve()
    processed = args.processed_root.resolve()
    artifact_root = Path(config["outputs"]["drive_artifact_root"])
    if args.registry.name != "experiment_registry_isomorph_sample_seed42.json" or artifact_root.name != "isomorph_sample_seed42":
        raise SystemExit("Use the separate ISOMORPH registry and namespace")
    if (artifact_root / "final_test").exists():
        raise SystemExit("ISOMORPH namespace must not contain final_test")
    if not (processed / "target_definition.json").is_file():
        raise SystemExit("Processed ISOMORPH target_definition.json is missing")
    manifest = file_manifest(processed, ["nodes.csv", "edges.csv", "sales.csv", "target_definition.json", "data_quality.json"])
    commit = validate_pushed_repository(root, args.branch)
    info = runtime_info()
    if not info["cuda_available"]:
        raise RuntimeError("No CUDA GPU available; no ISOMORPH run was reserved")
    quality_gate(root)
    for model in MODELS:
        run_config = copy.deepcopy(config)
        run_config["seed"] = SEED
        run_config["data"]["processed_dir"] = str(processed)
        signature = make_signature(model, run_config, commit, manifest)
        existing = registry.reserve(model, signature, run_config, commit, manifest, artifact_root)
        if existing.get("status") == "completed":
            print(f"Skipped completed ISOMORPH run: {existing['experiment_id']}", flush=True)
            continue
        experiment_id = existing["experiment_id"]
        artifact = Path(existing["artifact_dir"])
        artifact.mkdir(parents=True, exist_ok=False)
        run_config["outputs"].update({"checkpoint_dir": str(artifact / "checkpoints"), "metrics_dir": str(artifact / "metrics"),
                                      "predictions_dir": str(artifact / "predictions"), "logs_dir": str(artifact / "logs")})
        run_config["run"] = {"experiment_id": experiment_id, "signature": signature, "git_commit": commit}
        config_path = artifact / "config.yaml"
        config_path.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        (artifact / "runtime.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        (artifact / "processed_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        command = [sys.executable, "-u", str(root / "src/train.py"), "--model", model, "--config", str(config_path), "--project-root", str(root)]
        log_path = artifact / "training.log"
        registry.update(experiment_id, "running", command=command, runtime=info, config_path=str(config_path))
        print(f"Running {experiment_id}; log={log_path}", flush=True)
        try:
            result = run_logged(command, cwd=root, log_path=log_path, timeout=int(config["experiment"]["max_job_seconds"]))
            if result.returncode:
                raise RuntimeError(f"ISOMORPH training exited {result.returncode}; see {log_path}")
            files = ["config.yaml", "runtime.json", "processed_manifest.json", "training.log", "checkpoints/best.pt", "checkpoints/latest.pt", "predictions/validation.npz", "metrics/validation.json", "logs/history.json"]
            report = json.loads((artifact / "metrics/validation.json").read_text(encoding="utf-8"))
            if report["experiment_id"] != experiment_id or report["signature"] != signature:
                raise ValueError("ISOMORPH validation report does not match registry")
            registry.update(experiment_id, "completed", exit_code=0, report=str(artifact / "metrics/validation.json"),
                            artifact_manifest=file_manifest(artifact, files), completed_at=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            registry.update(experiment_id, "failed", failure={"type": type(exc).__name__, "reason": str(exc)})
            raise


if __name__ == "__main__":
    main()
