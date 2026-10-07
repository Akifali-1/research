"""Standard-library checksums, atomic metadata, and immutable stage artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zlib
from pathlib import Path


def fingerprint(path):
    digest, crc, count = hashlib.sha256(), 0, 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            crc = zlib.crc32(block, crc)
            count += len(block)
    return {"bytes": count, "sha256": digest.hexdigest(), "crc32": crc & 0xffffffff}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def manifest_for(directory, names):
    entries = []
    for name in sorted(names):
        path = Path(directory) / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Required nonempty file is absent: {path}")
        value = fingerprint(path)
        entries.append({"name": name, "bytes": value["bytes"], "sha256": value["sha256"]})
    return {"files": entries}


def verify_manifest(directory, manifest):
    if not manifest or not manifest.get("files"):
        raise ValueError("A nonempty integrity manifest is required")
    for entry in manifest["files"]:
        relative = Path(entry["name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest contains an unsafe relative path")
    if manifest_for(directory, [entry["name"] for entry in manifest["files"]]) != manifest:
        raise ValueError(f"Artifact integrity check failed in {directory}; existing files were preserved")


def commit_stage(staged_dir, destination, names, metadata):
    """Promote a verified complete stage, never replacing an existing stage."""
    staged_dir, destination = Path(staged_dir), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Completed/partial destination already exists: {destination}")
    manifest = manifest_for(staged_dir, names)
    atomic_json(staged_dir / "complete.json", {"metadata": metadata, "manifest": manifest})
    staged_dir.rename(destination)
    return manifest


def verify_stage(directory, metadata):
    directory = Path(directory)
    record = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if record["metadata"] != metadata:
        raise ValueError("Evaluation stage belongs to a different frozen selection/configuration")
    verify_manifest(directory, record["manifest"])
    return record
