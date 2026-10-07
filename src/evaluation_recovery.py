"""Immutable final-test stages: verify completed work, retry incomplete work."""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .integrity import commit_stage, fingerprint, verify_stage


@contextmanager
def evaluation_lock(directory):
    """Fail closed for competing sessions; stale locks need explicit inspection."""
    path = Path(directory) / ".evaluation.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


class EvaluationWorkspace:
    def __init__(self, directory, experiment_ids):
        self.root = Path(directory)
        self.selection = json.loads((self.root / "selection.json").read_text(encoding="utf-8"))
        if sorted(experiment_ids) != sorted(self.selection["experiment_ids"]):
            raise ValueError("Cannot resume evaluation with different frozen model IDs")
        self.selection_sha256 = fingerprint(self.root / "selection.json")["sha256"]

    def stage(self, name, metadata, producer, names):
        """The producer is called only for a missing stage; completed files are immutable.

        Recovery reuses fully checksum-verified stages. Incomplete scratch output
        is never treated as a completed stage. Handled exceptions remove only
        this invocation's scratch directory; old files and user results remain.
        """
        if name != "report" and (
            not name
            or not name[0].isalpha()
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name)
        ):
            raise ValueError("Invalid evaluation stage name")
        metadata = {**metadata, "selection_sha256": self.selection_sha256, "stage": name}
        destination = self.root / "stages" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            verify_stage(destination, metadata)
            return destination
        scratch = self.root / ".work"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{name}-", dir=scratch) as folder:
            folder = Path(folder)
            producer(folder)
            commit_stage(folder, destination, names, metadata)
        return destination
