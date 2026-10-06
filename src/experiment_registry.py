"""Durable, JSON-backed experiment registry for Drive/Colab runs.

The registry intentionally uses only the Python standard library so the
preflight and budget checks work before the ML stack is imported. A submitted
run consumes one budget slot before its training subprocess starts. Completed
signatures are never submitted again unless the user explicitly creates a
new configuration or chooses a retry for a failed/interrupted attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping


ACTIVE_STATUSES = {"reserved", "running"}
FINAL_STATUSES = {"completed", "failed", "interrupted", "rejected", "skipped"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def experiment_signature(
    model: str,
    configuration: Mapping[str, Any],
    git_commit: str,
    dataset_manifest: Mapping[str, Any],
) -> str:
    payload = {
        "model": model,
        "configuration": configuration,
        "git_commit": git_commit,
        "dataset_manifest": dataset_manifest,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class BudgetExceeded(RuntimeError):
    """Raised when no GPU-run budget slots remain."""


class ActiveExperiment(RuntimeError):
    """Raised when another run is already active."""


class RecoveryRequired(RuntimeError):
    """A failed/interrupted attempt requires explicit retry authorization."""


class Registry:
    def __init__(self, path: Path | str, max_runs: int = 10, max_retries: int = 2):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.max_runs = int(max_runs)
        self.max_retries = int(max_retries)
        if self.max_runs < 1 or self.max_retries < 0:
            raise ValueError("max_runs must be positive and max_retries cannot be negative")
        if self.max_runs > 10 or self.max_retries > 2:
            raise ValueError("Hard limits are 10 submitted runs and 2 retries; approval is required to raise them")

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + 30
        descriptor = None
        while descriptor is None:
            try:
                descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError(f"Could not acquire registry lock: {self.lock_path}")
                time.sleep(0.05)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            yield
        finally:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    def _default(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "budget": {
                "max_runs": self.max_runs,
                "max_retries_per_signature": self.max_retries,
                "submitted_runs": 0,
            },
            "experiments": [],
        }

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._default()
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported registry schema: {payload.get('schema_version')}")
        payload.setdefault("experiments", [])
        payload.setdefault("budget", {})
        payload["budget"].setdefault("max_runs", self.max_runs)
        payload["budget"].setdefault("max_retries_per_signature", self.max_retries)
        payload["budget"].setdefault("submitted_runs", 0)
        budget = payload["budget"]
        if int(budget["max_runs"]) > 10 or int(budget["max_retries_per_signature"]) > 2:
            raise ValueError("Registry exceeds the approved hard limits")
        # Config changes cannot silently increase an existing campaign's budget.
        budget["max_runs"] = min(int(budget["max_runs"]), self.max_runs)
        budget["max_retries_per_signature"] = min(int(budget["max_retries_per_signature"]), self.max_retries)
        if int(budget["submitted_runs"]) != len(payload["experiments"]):
            raise ValueError("Registry submission count is inconsistent; inspect it before execution")
        return payload

    def _write_unlocked(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(payload)
        payload["updated_at"] = utc_now()
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(self.path)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock():
            return self._read_unlocked()

    def status(self) -> Dict[str, Any]:
        payload = self.snapshot()
        budget = payload["budget"]
        return {
            "submitted_runs": int(budget["submitted_runs"]),
            "max_runs": int(budget["max_runs"]),
            "remaining_runs": max(0, int(budget["max_runs"]) - int(budget["submitted_runs"])),
            "active_experiments": [
                item["experiment_id"] for item in payload["experiments"] if item.get("status") in ACTIVE_STATUSES
            ],
            "completed_experiments": [
                item["experiment_id"] for item in payload["experiments"] if item.get("status") == "completed"
            ],
            "deadline_at": payload.get("deadline_at"),
        }

    def establish_deadline(self, maximum_seconds: int) -> str:
        """Persist a campaign deadline once; a runtime restart cannot reset it."""
        from datetime import timedelta
        if maximum_seconds <= 0:
            raise ValueError("Campaign runtime must be positive")
        with self._lock():
            payload = self._read_unlocked()
            if "deadline_at" not in payload:
                payload["deadline_at"] = (datetime.now(timezone.utc) + timedelta(seconds=maximum_seconds)).isoformat()
                self._write_unlocked(payload)
            return payload["deadline_at"]

    def _matching_unlocked(self, payload: Mapping[str, Any], signature: str):
        return [item for item in payload["experiments"] if item.get("signature") == signature]

    def reserve(
        self,
        *,
        model: str,
        signature: str,
        configuration: Mapping[str, Any],
        git_commit: str,
        dataset_manifest: Mapping[str, Any],
        artifact_dir: Path | str,
        force_retry: bool = False,
    ) -> Dict[str, Any]:
        """Reserve one GPU run, or return a completed record for idempotent resume."""
        with self._lock():
            payload = self._read_unlocked()
            matches = self._matching_unlocked(payload, signature)
            completed = [item for item in matches if item.get("status") == "completed"]
            if completed:
                return {"action": "skip_completed", "record": completed[-1]}
            active = [item for item in payload["experiments"] if item.get("status") in ACTIVE_STATUSES]
            if active:
                raise ActiveExperiment(f"An experiment is already active: {active[-1]['experiment_id']}")
            attempts = len(matches)
            if matches and not force_retry:
                raise RecoveryRequired("Previous attempt is unfinished or failed; use explicit retry/recovery")
            budget = payload["budget"]
            if attempts >= 1 + int(budget["max_retries_per_signature"]):
                raise BudgetExceeded(f"Retry limit exceeded for signature {signature[:12]}")
            if int(budget["submitted_runs"]) >= int(budget["max_runs"]):
                raise BudgetExceeded(
                    f"GPU experiment budget exhausted: {budget['submitted_runs']}/{budget['max_runs']}"
                )
            experiment_id = f"{model}-{signature[:12]}-attempt{attempts + 1}"
            record = {
                "experiment_id": experiment_id,
                "signature": signature,
                "model": model,
                "attempt": attempts + 1,
                "status": "reserved",
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "git_commit": git_commit,
                "configuration": dict(configuration),
                "dataset_manifest": dict(dataset_manifest),
                "artifact_dir": str(Path(artifact_dir) / experiment_id),
                "failure": None,
                "exit_code": None,
            }
            payload["experiments"].append(record)
            budget["submitted_runs"] = int(budget["submitted_runs"]) + 1
            self._write_unlocked(payload)
            return {"action": "reserved", "record": record}

    def update(self, experiment_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        if status not in ACTIVE_STATUSES | FINAL_STATUSES:
            raise ValueError(f"Unsupported experiment status: {status}")
        with self._lock():
            payload = self._read_unlocked()
            for record in payload["experiments"]:
                if record.get("experiment_id") == experiment_id:
                    if record.get("status") == "completed" and status != "completed":
                        raise ValueError("Completed experiment records are immutable")
                    record.update(fields)
                    record["status"] = status
                    record["updated_at"] = utc_now()
                    self._write_unlocked(payload)
                    return record
        raise KeyError(f"Unknown experiment: {experiment_id}")

    def matching(self, signature: str) -> list[Dict[str, Any]]:
        payload = self.snapshot()
        return self._matching_unlocked(payload, signature)

    def mark_interrupted_active(self, reason: str = "session resumed") -> int:
        count = 0
        with self._lock():
            payload = self._read_unlocked()
            for record in payload["experiments"]:
                if record.get("status") in ACTIVE_STATUSES:
                    record["status"] = "interrupted"
                    record["failure"] = {"reason": reason, "at": utc_now()}
                    record["updated_at"] = utc_now()
                    count += 1
            if count:
                self._write_unlocked(payload)
        return count
