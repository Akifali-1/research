"""Sequential, budget-limited experiment controller.

This controller is designed to run in Colab after the interactive Drive and
repository gates pass. It never submits more than one training subprocess at a
time and never repeats a completed experiment signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

from src.experiment_registry import (
    ActiveExperiment,
    BudgetExceeded,
    Registry,
    experiment_signature,
)


EXPECTED_ARCHIVES = {
    "train.csv.7z",
    "items.csv.7z",
    "stores.csv.7z",
    "transactions.csv.7z",
    "oil.csv.7z",
    "holidays_events.csv.7z",
    "test.csv.7z",
    "sample_submission.csv.7z",
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def git_commit(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()


def current_branch(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()


def remote_head(project_root: Path, branch: str | None = None) -> str:
    branch = branch or current_branch(project_root)
    return subprocess.check_output(
        ["git", "-C", str(project_root), "ls-remote", "origin", f"refs/heads/{branch}"], text=True
    ).split()[0]


def validate_pushed_repository(project_root: Path, branch: str | None = None) -> str:
    local = git_commit(project_root)
    remote = remote_head(project_root, branch)
    if not local or local != remote:
        raise RuntimeError(f"Repository is not pushed at the current commit: local={local}, remote={remote}")
    return local


def file_manifest(directory: Path, names: Iterable[str]) -> Dict[str, Any]:
    entries = []
    for name in sorted(names):
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty required archive: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        entries.append({"name": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return {"dataset": "favorita-grocery-sales", "archives": entries}


def validate_archives(raw_dir: Path) -> Dict[str, Any]:
    manifest = file_manifest(raw_dir, EXPECTED_ARCHIVES)
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before validating .7z archives") from exc
    archive_members = {}
    for name in sorted(EXPECTED_ARCHIVES):
        with py7zr.SevenZipFile(raw_dir / name, mode="r") as archive:
            members = archive.getnames()
        expected_member = name[:-3]
        if members != [expected_member]:
            raise ValueError(f"Unexpected member list in {name}: {members}")
        archive_members[name] = members
    manifest["archive_members"] = archive_members
    return manifest


def validate_budget(registry: Registry) -> Dict[str, Any]:
    status = registry.status()
    if status["active_experiments"]:
        raise RuntimeError(f"Active experiments require explicit recovery: {status['active_experiments']}")
    if status["remaining_runs"] <= 0:
        raise BudgetExceeded("No GPU experiment budget remains")
    return status


def run_sequential(
    *,
    project_root: Path,
    config_path: Path,
    raw_dir: Path,
    registry_path: Path,
    models: Iterable[str],
    max_runs: int,
    allow_execution: bool,
    force_retry: bool = False,
    branch: str | None = None,
) -> list[Dict[str, Any]]:
    if not allow_execution:
        raise RuntimeError("Execution is disabled. Pass --allow-execution only after all Colab gates pass.")
    commit = validate_pushed_repository(project_root, branch)
    manifest = validate_archives(raw_dir)
    config = load_config(config_path)
    registry = Registry(registry_path, max_runs=max_runs, max_retries=2)
    validate_budget(registry)
    reports = []
    for model in models:
        signature = experiment_signature(model, config, commit, manifest)
        artifact_dir = Path(config["outputs"].get("drive_artifact_root", "results")) / model / signature[:12]
        reservation = registry.reserve(
            model=model,
            signature=signature,
            configuration=config,
            git_commit=commit,
            dataset_manifest=manifest,
            artifact_dir=artifact_dir,
            force_retry=force_retry,
        )
        if reservation["action"] == "skip_completed":
            reports.append({"model": model, "action": "skip_completed", "record": reservation["record"]})
            continue
        record = reservation["record"]
        experiment_id = record["experiment_id"]
        registry.update(experiment_id, "running", started_at=record["updated_at"])
        command = [
            sys.executable,
            str(project_root / "src" / "train.py"),
            "--model",
            model,
            "--config",
            str(config_path),
            "--project-root",
            str(project_root),
        ]
        log_path = Path(record["artifact_dir"]) / "training.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=project_root, stdout=log, stderr=subprocess.STDOUT, check=False)
            if completed.returncode != 0:
                registry.update(
                    experiment_id,
                    "failed",
                    exit_code=completed.returncode,
                    failure={"reason": "training subprocess failed", "log": str(log_path)},
                )
                reports.append({"model": model, "action": "failed", "record": experiment_id})
                continue
            report_path = project_root / config["outputs"]["metrics_dir"] / f"{model}.json"
            if not report_path.exists():
                registry.update(
                    experiment_id,
                    "failed",
                    exit_code=completed.returncode,
                    failure={"reason": "training completed without metrics artifact", "log": str(log_path)},
                )
                reports.append({"model": model, "action": "failed_missing_artifact", "record": experiment_id})
                continue
            registry.update(experiment_id, "completed", exit_code=0, report=str(report_path), log=str(log_path))
            reports.append({"model": model, "action": "completed", "record": experiment_id, "report": str(report_path)})
        except KeyboardInterrupt:
            registry.update(experiment_id, "interrupted", failure={"reason": "keyboard interrupt", "log": str(log_path)})
            raise
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["stgt", "gat_lstm"], default=["stgt", "gat_lstm"])
    parser.add_argument("--max-runs", type=int, default=10)
    parser.add_argument("--allow-execution", action="store_true")
    parser.add_argument("--force-retry", action="store_true")
    parser.add_argument("--branch", default=None)
    parser.add_argument("--status-only", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    root = arguments.project_root.resolve()
    registry_path = arguments.registry if arguments.registry.is_absolute() else root / arguments.registry
    registry = Registry(registry_path, max_runs=arguments.max_runs, max_retries=2)
    if arguments.status_only:
        print(json.dumps(registry.status(), indent=2))
    else:
        raw = arguments.raw_dir if arguments.raw_dir.is_absolute() else root / arguments.raw_dir
        config = arguments.config if arguments.config.is_absolute() else root / arguments.config
        result = run_sequential(
            project_root=root,
            config_path=config,
            raw_dir=raw,
            registry_path=registry_path,
            models=arguments.models,
            max_runs=arguments.max_runs,
            allow_execution=arguments.allow_execution,
            force_retry=arguments.force_retry,
            branch=arguments.branch,
        )
        print(json.dumps(result, indent=2, default=str))
