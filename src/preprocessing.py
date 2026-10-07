"""Chunked Favorita sales aggregation with metadata-only graph construction.

The source is a sparse daily sales ledger. Under the explicit ``zero`` policy,
absent aggregate node/day cells are zero recorded sales, not imputed demand.
No dates after the last labeled observation are synthesized. State is used as
Region. Family nodes are global totals shared by stores: the graph is a typed
aggregation network, not a strict summation tree at the store-family level.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .integrity import atomic_json, fingerprint
    from .source_schema import validate_source_schema
else:
    from integrity import atomic_json, fingerprint
    from source_schema import validate_source_schema


def slug(value):
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper() or "UNKNOWN"


def materialize_csv(raw_dir, filename, cache_dir=None):
    raw_dir = Path(raw_dir)
    original = raw_dir / filename
    archive_path = raw_dir / f"{filename}.7z"
    # Match preflight's source preference when CSV and 7z are both present.
    if original.is_file() and not archive_path.is_file():
        validate_source_schema(original, filename[:-4])
        return original
    cache = Path(cache_dir) if cache_dir else raw_dir
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    import py7zr
    archive_hash = fingerprint(archive_path)["sha256"]
    # Source-content namespace avoids reusing a CSV from a different archive.
    cache = cache / ".verified-csv" / archive_hash
    cache.mkdir(parents=True, exist_ok=True)
    extracted = cache / filename
    checksum_path = cache / f"{filename}.integrity.json"
    active_path = cache / "active.json"
    if active_path.exists():
        relative = Path(json.loads(active_path.read_text(encoding="utf-8"))["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.name != filename:
            raise ValueError("Unsafe cache pointer")
        extracted = cache / relative
        checksum_path = extracted.with_name(f"{filename}.integrity.json")
    with py7zr.SevenZipFile(archive_path, "r") as archive:
        members = archive.list()
        if len(members) != 1 or members[0].filename != filename:
            raise ValueError(f"Unexpected archive members: {archive_path}")
        if extracted.is_file():
            if checksum_path.exists():
                try:
                    recorded = json.loads(checksum_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeError):
                    recorded = None
                actual = fingerprint(extracted)
                if (isinstance(recorded, dict) and recorded.get("archive_sha256") == archive_hash
                        and recorded.get("csv") == actual
                        and actual["bytes"] == members[0].uncompressed
                        and (members[0].crc32 is None or actual["crc32"] == members[0].crc32)):
                    validate_source_schema(extracted, filename[:-4])
                    return extracted
            # Preserve suspect/orphaned cache versions. The new verified
            # extraction below gets its own unique directory instead.
            extracted = None
        if shutil.disk_usage(cache).free < members[0].uncompressed * 1.1:
            raise RuntimeError(f"Insufficient free space to extract {filename} into {cache}")
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=cache) as folder:
            staging = Path(folder)
            archive.extractall(path=staging)
            candidate = staging / filename
            actual = fingerprint(candidate)
            if actual["bytes"] != members[0].uncompressed:
                raise ValueError(f"Incomplete archive extraction: {candidate}")
            if members[0].crc32 is not None and actual["crc32"] != members[0].crc32:
                raise ValueError(f"Archive CRC mismatch: {candidate}")
            validate_source_schema(candidate, filename[:-4])
            atomic_json(staging / f"{filename}.integrity.json", {"archive_sha256": archive_hash, "csv": actual})
            if extracted is not None and not extracted.exists():
                # Promote the file atomically, then its checksum. An interrupted
                # promotion without a checksum is never trusted on the next run.
                candidate.replace(extracted)
                (staging / f"{filename}.integrity.json").replace(checksum_path)
                return extracted
            # Recover around a suspect existing cache without deleting it.
            recovered = cache / f"verified-{staging.name.removeprefix('.extract-')}"
            staging.rename(recovered)
            atomic_json(active_path, {"path": str((recovered / filename).relative_to(cache))})
            return recovered / filename


def load_metadata(raw_dir, cache_dir=None):
    items = pd.read_csv(materialize_csv(raw_dir, "items.csv", cache_dir),
                        usecols=["item_nbr", "family"], dtype={"item_nbr": "int32", "family": "string"})
    stores = pd.read_csv(materialize_csv(raw_dir, "stores.csv", cache_dir),
                         usecols=["store_nbr", "city", "state"],
                         dtype={"store_nbr": "int16", "city": "string", "state": "string"})
    for frame, key in ((items, "item_nbr"), (stores, "store_nbr")):
        if frame.isna().any().any() or frame[key].duplicated().any():
            raise ValueError(f"Missing or duplicate metadata key: {key}")
    for values in (items["family"], stores["state"]):
        originals = set(values.astype(str))
        if len({slug(value) for value in originals}) != len(originals):
            raise ValueError("Node slug collision; do not silently merge distinct metadata labels")
    city_keys = {(str(row.state), str(row.city)) for row in stores.itertuples(index=False)}
    if len({(slug(state), slug(city)) for state, city in city_keys}) != len(city_keys):
        raise ValueError("City node ID collision")
    return items, stores


def build_metadata_graph(items, stores):
    """Use metadata, never validation/test sales, to choose nodes or edges."""
    node_rows = {}
    edges = set()
    def add(node_id, kind, label, parent="", **metadata):
        node_rows[node_id] = {"node_id": node_id, "node_type": kind, "label": label,
                              "parent_id": parent, **metadata}
    families = sorted(set(items["family"].astype(str)))
    for family in families:
        add(f"FAMILY__{slug(family)}", "Family", family, family=family)
    for row in stores.itertuples(index=False):
        region = f"REGION__{slug(row.state)}"
        city = f"CITY__{slug(row.state)}__{slug(row.city)}"
        store = f"STORE__{int(row.store_nbr):04d}"
        add(region, "Region", str(row.state), region=str(row.state))
        add(city, "City", str(row.city), region, region=str(row.state), city=str(row.city))
        add(store, "Store", str(row.store_nbr), city, region=str(row.state), city=str(row.city), store_nbr=int(row.store_nbr))
        edges.add((region, city, "region_city"))
        edges.add((city, store, "city_store"))
        for family in families:
            edges.add((store, f"FAMILY__{slug(family)}", "store_family"))
    nodes = pd.DataFrame([node_rows[key] for key in sorted(node_rows)])
    return nodes, pd.DataFrame(sorted(edges), columns=["source", "target", "edge_type"])


def preprocess(raw_dir, processed_dir, start_date, end_date, chunksize=500000,
               missing_policy="zero", cache_dir=None):
    if missing_policy not in {"zero", "error"}:
        raise ValueError("missing_policy must be zero or error")
    start, requested_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if start > requested_end or chunksize <= 0:
        raise ValueError("Invalid date range or chunksize")
    processed_dir = Path(processed_dir)
    if any((processed_dir / name).exists() for name in ("nodes.csv", "edges.csv", "sales.csv")):
        raise FileExistsError("Processed outputs already exist; choose a versioned directory")
    items, stores = load_metadata(raw_dir, cache_dir)
    nodes, edges = build_metadata_graph(items, stores)
    dates = pd.date_range(start, requested_end, freq="D")
    node_ids = nodes["node_id"].tolist()
    index = {node: i for i, node in enumerate(node_ids)}
    totals = np.zeros((len(dates), len(nodes)), dtype=np.float64)
    observed = np.zeros_like(totals, dtype=bool)
    family_map = dict(zip(items["item_nbr"], items["family"]))
    store_map = {int(row.store_nbr): (str(row.state), str(row.city)) for row in stores.itertuples(index=False)}
    report = {"source_rows_in_range": 0, "negative_sales_clipped": 0, "duplicate_business_key_rows_aggregated": 0,
              "duplicate_id_policy": "reject non-increasing source IDs, including across chunks",
              "business_key_policy": "sum ledger rows; duplicates inside each chunk counted; no deduplication by sales value"}
    last_id = None
    last_date = None
    latest_observed_date = None
    train_path = materialize_csv(raw_dir, "train.csv", cache_dir)
    for chunk in pd.read_csv(train_path, usecols=["id", "date", "store_nbr", "item_nbr", "unit_sales"],
                             dtype={"id": "int64", "store_nbr": "int16", "item_nbr": "int32"}, chunksize=chunksize):
        identifiers = chunk["id"].to_numpy()
        if (np.diff(identifiers) <= 0).any() or (last_id is not None and identifiers[0] <= last_id):
            raise ValueError("Source row IDs are duplicate/unordered; inspect raw data instead of double-counting")
        last_id = int(identifiers[-1])
        parsed = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="coerce")
        if parsed.isna().any():
            raise ValueError("Invalid raw dates")
        if not parsed.is_monotonic_increasing or (last_date is not None and parsed.iloc[0] < last_date):
            raise ValueError("Source dates are unordered")
        last_date = parsed.iloc[-1]
        chunk["date"] = parsed
        chunk = chunk.loc[(parsed >= start) & (parsed <= requested_end)].copy()
        if chunk.empty:
            continue
        values = pd.to_numeric(chunk["unit_sales"], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Invalid/missing raw unit_sales; no silent zero replacement")
        report["source_rows_in_range"] += len(chunk)
        report["negative_sales_clipped"] += int((values < 0).sum())
        report["duplicate_business_key_rows_aggregated"] += int(chunk.duplicated(["date", "store_nbr", "item_nbr"], keep=False).sum())
        chunk["unit_sales"] = np.maximum(values, 0)
        chunk["family"] = chunk["item_nbr"].map(family_map)
        if chunk["family"].isna().any() or not set(chunk["store_nbr"]).issubset(store_map):
            raise ValueError("Source rows reference missing store/item metadata")
        grouped = chunk.groupby(["date", "store_nbr", "family"], observed=True, sort=False)["unit_sales"].sum()
        for (day, store_number, family), value in grouped.items():
            day_index = (day - start).days
            state, city = store_map[int(store_number)]
            for node in (f"REGION__{slug(state)}", f"CITY__{slug(state)}__{slug(city)}",
                         f"STORE__{int(store_number):04d}", f"FAMILY__{slug(family)}"):
                totals[day_index, index[node]] += float(value)
                observed[day_index, index[node]] = True
        latest_observed_date = chunk["date"].max()
    if latest_observed_date is None:
        raise ValueError("No labeled observations in the requested range")
    length = (latest_observed_date - start).days + 1
    dates, totals, observed = dates[:length], totals[:length], observed[:length]
    missing_cells = int((~observed).sum())
    if missing_policy == "error" and missing_cells:
        raise ValueError(f"{missing_cells} absent aggregate node/date observations; error policy rejects them")
    sales = pd.DataFrame(totals, columns=node_ids)
    sales.insert(0, "Date", dates)
    report.update(dataset="Corporación Favorita Grocery Sales Forecasting", date_start=str(start.date()),
                  date_end=str(latest_observed_date.date()), requested_end=str(requested_end.date()),
                  missing_observation_policy=missing_policy, missing_observations_filled_zero=missing_cells,
                  node_count=len(nodes), edge_count=len(edges), time_steps=len(dates),
                  node_types=nodes["node_type"].value_counts().to_dict(),
                  graph_policy="metadata-only typed region-city-store/global-family graph; no target-dependent pruning",
                  zero_semantics="absence under the explicit ledger policy is zero recorded sales, not imputed demand")
    processed_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(processed_dir / "nodes.csv", index=False)
    edges.to_csv(processed_dir / "edges.csv", index=False)
    sales.to_csv(processed_dir / "sales.csv", index=False, date_format="%Y-%m-%d")
    (processed_dir / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_outputs(processed_dir)
    return report


def validate_outputs(processed_dir):
    path = Path(processed_dir)
    nodes, edges = pd.read_csv(path / "nodes.csv"), pd.read_csv(path / "edges.csv")
    sales = pd.read_csv(path / "sales.csv")
    node_ids = set(nodes["node_id"])
    checks = [len(nodes) > 0, len(edges) > 0, len(sales) > 0, len(node_ids) == len(nodes),
              set(edges["source"]).issubset(node_ids), set(edges["target"]).issubset(node_ids),
              not (edges["source"] == edges["target"]).any(), not edges.duplicated(["source", "target"]).any(),
              set(sales.columns) == {"Date", *node_ids}]
    dates = pd.to_datetime(sales["Date"], errors="coerce")
    values = sales[list(node_ids)].to_numpy(dtype=np.float64)
    checks.extend([dates.notna().all(), dates.is_monotonic_increasing, not dates.duplicated().any(),
                   dates.diff().dropna().eq(pd.Timedelta(days=1)).all(), np.isfinite(values).all(), (values >= 0).all()])
    if not all(checks):
        raise ValueError("Processed node/edge/series/date/demand validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2017-08-15")
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--missing-policy", choices=["zero", "error"], default="zero")
    args = parser.parse_args()
    print(json.dumps(preprocess(args.raw_dir, args.processed_dir, args.start_date, args.end_date,
                               args.chunksize, args.missing_policy, args.cache_dir), indent=2))
