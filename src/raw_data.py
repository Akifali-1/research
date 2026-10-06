"""Favorita source-layout discovery and safe outer-ZIP extraction.

No training CSV is decompressed from 7z here. The optional outer ZIP contains
eight sources (usually compressed CSV.7z files). It is preserved, as are all
existing raw files. These helpers have no dependency on the ML stack.
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

SOURCE_STEMS = ("train", "items", "stores", "transactions", "oil", "holidays_events", "test", "sample_submission")


def source_paths(raw_dir):
    """Prefer CSV.7z when both forms exist; return all missing logical sources."""
    directory = Path(raw_dir)
    selected, missing = {}, []
    for stem in SOURCE_STEMS:
        alternatives = (directory / f"{stem}.csv.7z", directory / f"{stem}.csv")
        found = next((path for path in alternatives if path.is_file() and path.stat().st_size > 0), None)
        if found is None:
            missing.append(f"{stem}.csv.7z OR {stem}.csv")
        else:
            selected[stem] = found
    return selected, missing


def dataset_layout_error(raw_dir, missing):
    directory = Path(raw_dir)
    present = sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []
    nested = [path.name for path in directory.iterdir() if path.is_dir()] if directory.is_dir() else []
    return FileNotFoundError(
        f"Favorita sources are incomplete in {directory}.\n"
        f"Missing: {', '.join(missing)}\n"
        f"Present: {', '.join(present) or '(folder absent or empty)'}\n"
        f"Subfolders: {', '.join(nested) or '(none)'}\n"
        "Upload the original favorita-grocery-sales-forecasting.zip to this folder and rerun "
        "prepare_raw_sources, or set RAW_DIR to the folder containing all eight sources. "
        "If the ZIP is elsewhere in Drive, set SOURCE_ZIP to its full path."
    )


def _crc(path):
    value = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def extract_outer_zip(zip_path, raw_dir):
    """Flatten only verified source members; never overwrite differing files."""
    destination = Path(raw_dir)
    with zipfile.ZipFile(zip_path) as archive:
        candidates = {}
        allowed = {f"{stem}.csv{suffix}" for stem in SOURCE_STEMS for suffix in ("", ".7z")}
        for info in archive.infolist():
            member = PurePosixPath(info.filename.replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe ZIP member path: {info.filename}")
            if info.is_dir() or member.name not in allowed:
                continue
            if member.name in candidates:
                raise ValueError(f"Ambiguous duplicate source name in ZIP: {member.name}")
            if info.file_size <= 0:
                raise ValueError(f"Empty source member in ZIP: {info.filename}")
            candidates[member.name] = info
        missing = [stem for stem in SOURCE_STEMS if not any(f"{stem}.csv{suffix}" in candidates for suffix in (".7z", ""))]
        if missing:
            raise ValueError(f"ZIP does not contain the complete Favorita source set: missing {missing}")
        selected = [candidates[next(name for name in (f"{stem}.csv.7z", f"{stem}.csv") if name in candidates)] for stem in SOURCE_STEMS]
        destination.mkdir(parents=True, exist_ok=True)
        pending = []
        # Verify every existing destination before writing any new file.
        for info in selected:
            target = destination / PurePosixPath(info.filename.replace("\\", "/")).name
            if target.exists():
                if not target.is_file() or target.stat().st_size != info.file_size or _crc(target) != info.CRC:
                    raise FileExistsError(f"Existing source differs from ZIP: {target}; preserved without overwrite")
            else:
                pending.append((info, target))
        if shutil.disk_usage(destination).free < sum(info.file_size for info, _ in pending) * 1.05:
            raise RuntimeError("Not enough space to extract the outer ZIP; no source file was overwritten")
        for info, target in pending:
            with tempfile.NamedTemporaryFile(prefix=".source-", suffix=".partial", dir=destination, delete=False) as handle:
                temporary = Path(handle.name)
                try:
                    with archive.open(info) as source:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                except BaseException:
                    # Partial data is preserved for inspection, never accepted as source.
                    raise
            if temporary.stat().st_size != info.file_size or _crc(temporary) != info.CRC:
                raise ValueError(f"ZIP extraction failed integrity check: {temporary}")
            temporary.rename(target)
    return destination


def prepare_raw_sources(raw_dir, zip_path=None):
    """Resolve sources in the folder/one subfolder, or expand the outer ZIP.

    Explicit zip_path avoids searching unrelated Drive directories. If there
    are multiple ZIP candidates/subfolders, require the user to choose one.
    """
    directory = Path(raw_dir)
    _, missing = source_paths(directory)
    if not missing:
        return directory
    if zip_path is not None:
        archive = Path(zip_path)
        if not archive.is_file():
            raise FileNotFoundError(f"SOURCE_ZIP is absent: {archive}")
        extract_outer_zip(archive, directory)
        return directory
    if directory.is_dir():
        complete_folders = [path for path in directory.iterdir() if path.is_dir() and not source_paths(path)[1]]
        zip_candidates = sorted(directory.glob("*.zip"))
        if len(complete_folders) == 1 and not zip_candidates:
            return complete_folders[0]
        if len(zip_candidates) == 1 and not complete_folders:
            extract_outer_zip(zip_candidates[0], directory)
            return directory
        if len(complete_folders) + len(zip_candidates) > 1:
            raise ValueError("Multiple dataset folders/ZIPs found. Set RAW_DIR or SOURCE_ZIP explicitly.")
    raise dataset_layout_error(directory, missing)
