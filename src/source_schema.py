"""Validate every Favorita CSV schema, with bounded 7z header-only decoding."""
from __future__ import annotations

import csv
import io
from pathlib import Path

SCHEMAS = {
    "train": {"id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"},
    "test": {"id", "date", "store_nbr", "item_nbr", "onpromotion"},
    "items": {"item_nbr", "family", "class", "perishable"},
    "stores": {"store_nbr", "city", "state", "type", "cluster"},
    "transactions": {"date", "store_nbr", "transactions"},
    "oil": {"date", "dcoilwtico"},
    "holidays_events": {"date", "type", "locale", "locale_name", "description", "transferred"},
    "sample_submission": {"id", "unit_sales"},
}
HEADER_LIMIT = 64 * 1024


def validate_columns(stem, columns):
    if not columns or len(columns) != len(set(columns)):
        raise ValueError(f"{stem}.csv has empty/duplicate column names")
    missing = SCHEMAS[stem] - set(columns)
    if missing:
        raise ValueError(f"{stem}.csv schema mismatch; missing {sorted(missing)}, found {columns}")
    return columns


def csv_columns(path):
    with Path(path).open("rb") as handle:
        header = handle.readline(HEADER_LIMIT + 1)
    if len(header) > HEADER_LIMIT or not header:
        raise ValueError("CSV header is empty or exceeds the bounded header limit")
    return next(csv.reader([header.decode("utf-8-sig").rstrip("\r\n")]))


class _HeaderRead(Exception):
    """Intentional stop after the header; this is not an archive integrity scan."""


def archive_columns(path):
    import py7zr
    from py7zr.io import Py7zIO, WriterFactory

    class HeaderSink(Py7zIO):
        def __init__(self):
            self.buffer = io.BytesIO()

        def write(self, data):
            remaining = HEADER_LIMIT + 1 - self.size()
            self.buffer.write(data[:remaining])
            if b"\n" in self.buffer.getvalue():
                raise _HeaderRead()
            if self.size() > HEADER_LIMIT:
                raise ValueError("Archive CSV header exceeds the bounded header limit")
            return len(data)

        def read(self, size=None):
            return self.buffer.read(-1 if size is None else size)

        def seek(self, offset, whence=0):
            return self.buffer.seek(offset, whence)

        def flush(self):
            pass

        def size(self):
            return len(self.buffer.getvalue())

    sink = HeaderSink()

    class HeaderFactory(WriterFactory):
        def create(self, filename):
            return sink

    # A file object disables py7zr parallel extraction, ensuring the deliberate
    # stop propagates to this caller. Only a bounded prefix is kept in memory.
    with Path(path).open("rb") as source:
        with py7zr.SevenZipFile(source, "r", blocksize=HEADER_LIMIT) as archive:
            members = archive.list()
            expected = Path(path).name[:-3]
            if len(members) != 1 or members[0].filename != expected:
                raise ValueError(f"Unexpected members in {path}")
            try:
                archive.extractall(factory=HeaderFactory())
            except _HeaderRead:
                pass
    header = sink.buffer.getvalue().split(b"\n", 1)[0]
    if not header or len(header) > HEADER_LIMIT:
        raise ValueError(f"Invalid CSV header in {path}")
    return next(csv.reader([header.decode("utf-8-sig").rstrip("\r")]))


def validate_source_schema(path, stem):
    path = Path(path)
    columns = archive_columns(path) if path.name.endswith(".7z") else csv_columns(path)
    return validate_columns(stem, columns)
