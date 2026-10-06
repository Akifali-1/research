"""Memory-aware preprocessing for the Corporación Favorita dataset.

The archive distributed by Kaggle contains ``*.csv.7z`` members.  This script
expects those archives, or the extracted CSV files, in ``--raw-dir``.  When
``py7zr`` is installed it can materialize a missing CSV on demand.  The large
raw files are never committed to Git.

The Favorita source has no explicit region column.  Its ``state`` field is
used as the geographic region level:

    State/Region -> City -> Store -> Product Family

``unit_sales`` is the demand quantity.  Missing rows are treated as zero only
after aggregation because the Favorita train table is a daily sales ledger:
recorded in ``data_quality.json`` and can be changed to ``error``.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


RAW_FILES = {
    "train": "train.csv",
    "items": "items.csv",
    "stores": "stores.csv",
}


def slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()
    return text or "UNKNOWN"


def materialize_csv(raw_dir: Path, filename: str) -> Path:
    """Return a CSV path, extracting ``filename.7z`` only when necessary."""
    csv_path = raw_dir / filename
    if csv_path.exists():
        return csv_path
    archive_path = raw_dir / f"{filename}.7z"
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Required {filename} or {archive_path.name} was not found in {raw_dir}"
        )
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            f"{archive_path.name} is present but {filename} is not extracted. "
            "Install py7zr or extract the archive with 7-Zip before running preprocessing."
        ) from exc
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extract(path=raw_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Extraction did not produce {filename}")
    return csv_path


def load_metadata(raw_dir: Path) -> Tuple[Dict[int, str], Dict[int, Tuple[str, str]]]:
    items_path = materialize_csv(raw_dir, RAW_FILES["items"])
    stores_path = materialize_csv(raw_dir, RAW_FILES["stores"])
    items = pd.read_csv(
        items_path,
        usecols=["item_nbr", "family"],
        dtype={"item_nbr": "int32", "family": "string"},
    ).drop_duplicates("item_nbr")
    stores = pd.read_csv(
        stores_path,
        usecols=["store_nbr", "city", "state"],
        dtype={"store_nbr": "int16", "city": "string", "state": "string"},
    ).drop_duplicates("store_nbr")
    item_to_family = {
        int(row.item_nbr): str(row.family)
        for row in items.itertuples(index=False)
        if pd.notna(row.family)
    }
    store_to_location = {
        int(row.store_nbr): (str(row.city), str(row.state))
        for row in stores.itertuples(index=False)
        if pd.notna(row.city) and pd.notna(row.state)
    }
    if not item_to_family or not store_to_location:
        raise ValueError("Metadata files did not contain valid item or store mappings")
    return item_to_family, store_to_location


def _add_grouped(
    grouped: pd.Series,
    node_builder,
    sales_accumulator: Dict[str, Dict[str, float]],
) -> None:
    for key, value in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        date = pd.Timestamp(key[0]).strftime("%Y-%m-%d")
        node_id = node_builder(*key[1:])
        sales_accumulator[node_id][date] += float(value)


def _aggregate_chunk(
    chunk: pd.DataFrame,
    item_to_family: Dict[int, str],
    store_to_location: Dict[int, Tuple[str, str]],
    sales_accumulator: Dict[str, Dict[str, float]],
    observed_store_families: set[Tuple[int, str]],
    observed_nodes: set[str],
    counters: Dict[str, int],
) -> None:
    chunk = chunk.copy()
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    counters["invalid_dates"] += int(chunk["date"].isna().sum())
    chunk = chunk.loc[chunk["date"].notna()].copy()
    chunk["unit_sales"] = pd.to_numeric(chunk["unit_sales"], errors="coerce")
    counters["invalid_sales"] += int(chunk["unit_sales"].isna().sum())
    chunk = chunk.loc[chunk["unit_sales"].notna()].copy()
    negative_count = int((chunk["unit_sales"] < 0).sum())
    counters["negative_sales"] += negative_count
    # Returns/corrections are retained in the raw data but demand is non-negative.
    chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0.0)

    chunk["family"] = chunk["item_nbr"].map(item_to_family)
    locations = chunk["store_nbr"].map(store_to_location)
    chunk["city"] = locations.map(lambda value: value[0] if isinstance(value, tuple) else np.nan)
    chunk["state"] = locations.map(lambda value: value[1] if isinstance(value, tuple) else np.nan)
    missing_metadata = chunk[["family", "city", "state"]].isna().any(axis=1)
    counters["missing_metadata_rows"] += int(missing_metadata.sum())
    chunk = chunk.loc[~missing_metadata].copy()
    if chunk.empty:
        return

    chunk["region_node"] = chunk["state"].map(lambda value: f"REGION__{slug(value)}")
    chunk["city_node"] = chunk.apply(
        lambda row: f"CITY__{slug(row['state'])}__{slug(row['city'])}", axis=1
    )
    chunk["store_node"] = chunk["store_nbr"].map(lambda value: f"STORE__{int(value):04d}")
    chunk["family_node"] = chunk["family"].map(lambda value: f"FAMILY__{slug(value)}")

    for row in chunk[["store_nbr", "family"]].drop_duplicates().itertuples(index=False):
        observed_store_families.add((int(row.store_nbr), str(row.family)))
    observed_nodes.update(chunk["region_node"].astype(str))
    observed_nodes.update(chunk["city_node"].astype(str))
    observed_nodes.update(chunk["store_node"].astype(str))
    observed_nodes.update(chunk["family_node"].astype(str))

    for column in ["region_node", "city_node", "store_node", "family_node"]:
        grouped = chunk.groupby(["date", column], observed=True, sort=False)["unit_sales"].sum()
        _add_grouped(grouped, lambda node, _node=node: node, sales_accumulator)


def preprocess(
    raw_dir: Path,
    processed_dir: Path,
    start_date: str,
    end_date: str,
    chunksize: int,
    missing_policy: str,
) -> Dict[str, object]:
    if missing_policy not in {"zero", "error"}:
        raise ValueError("missing_policy must be 'zero' or 'error'")
    train_path = materialize_csv(raw_dir, RAW_FILES["train"])
    item_to_family, store_to_location = load_metadata(raw_dir)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    accumulator: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    observed_store_families: set[Tuple[int, str]] = set()
    observed_nodes: set[str] = set()
    counters = defaultdict(int)
    usecols = ["id", "date", "store_nbr", "item_nbr", "unit_sales"]
    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        chunksize=chunksize,
        dtype={"id": "int64", "store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"},
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        counters["invalid_dates"] += int(chunk["date"].isna().sum())
        chunk = chunk.loc[(chunk["date"] >= start) & (chunk["date"] <= end)]
        if chunk.empty:
            continue
        counters["source_rows_in_range"] += len(chunk)
        _aggregate_chunk(
            chunk,
            item_to_family,
            store_to_location,
            accumulator,
            observed_store_families,
            observed_nodes,
            counters,
        )

    if not accumulator:
        raise RuntimeError("No usable rows were found in the requested date range")
    dates = pd.date_range(start=min(pd.Timestamp(value) for values in accumulator.values() for value in values), end=end, freq="D")
    # Restrict to the actual observed range instead of adding leading empty days.
    first_date = min(pd.Timestamp(value) for values in accumulator.values() for value in values)
    dates = pd.date_range(first_date, end, freq="D")
    ordered_nodes = sorted(observed_nodes, key=lambda node: (node.split("__", 1)[0], node))
    sales = pd.DataFrame({"Date": dates})
    date_strings = dates.strftime("%Y-%m-%d")
    for node_id in ordered_nodes:
        values = accumulator[node_id]
        sales[node_id] = [float(values.get(date, 0.0)) for date in date_strings]
    missing_cells = sum(
        1 for node_id in ordered_nodes for date in date_strings if date not in accumulator[node_id]
    )
    if missing_policy == "error" and missing_cells:
        raise ValueError("Missing date/node observations were found under missing_policy=error")

    metadata: Dict[str, Dict[str, object]] = {}
    edge_rows = []
    state_node = {}
    city_node = {}
    store_node = {}
    family_node = {}
    for node_id in ordered_nodes:
        kind, remainder = node_id.split("__", 1)
        metadata[node_id] = {
            "node_id": node_id,
            "node_type": {"REGION": "Region", "CITY": "City", "STORE": "Store", "FAMILY": "Family"}[kind],
            "label": remainder,
            "parent_id": "",
            "region": "",
            "city": "",
            "store_nbr": "",
            "family": "",
        }
        if kind == "REGION":
            state_node[remainder] = node_id
            metadata[node_id]["region"] = remainder
        elif kind == "CITY":
            pieces = remainder.split("__", 1)
            metadata[node_id]["region"], metadata[node_id]["city"] = pieces
            state_node_id = state_node.get(pieces[0], f"REGION__{pieces[0]}")
            metadata[node_id]["parent_id"] = state_node_id
            city_node[remainder] = node_id
            edge_rows.append((state_node_id, node_id, "region_city"))
        elif kind == "STORE":
            store_number = int(remainder)
            city, state = store_to_location[store_number]
            parent = f"CITY__{slug(state)}__{slug(city)}"
            metadata[node_id].update({"region": slug(state), "city": slug(city), "store_nbr": store_number, "parent_id": parent})
            store_node[store_number] = node_id
            edge_rows.append((parent, node_id, "city_store"))
        else:
            metadata[node_id]["family"] = remainder
            family_node[remainder] = node_id

    for store_number, family in sorted(observed_store_families):
        source = store_node.get(store_number)
        target = family_node.get(slug(family))
        if source and target:
            edge_rows.append((source, target, "store_family"))
    edges = pd.DataFrame(edge_rows, columns=["source", "target", "edge_type"]).drop_duplicates()
    edges = edges.loc[edges["source"] != edges["target"]].reset_index(drop=True)
    nodes = pd.DataFrame([metadata[node_id] for node_id in ordered_nodes])

    processed_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(processed_dir / "nodes.csv", index=False)
    edges.to_csv(processed_dir / "edges.csv", index=False)
    sales.to_csv(processed_dir / "sales.csv", index=False, date_format="%Y-%m-%d")
    report = {
        "dataset": "Corporación Favorita Grocery Sales Forecasting",
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "source_rows_in_range": counters["source_rows_in_range"],
        "invalid_dates": counters["invalid_dates"],
        "invalid_sales": counters["invalid_sales"],
        "negative_sales_clipped": counters["negative_sales"],
        "missing_metadata_rows": counters["missing_metadata_rows"],
        "missing_observation_policy": missing_policy,
        "missing_observations_filled_zero": int(missing_cells),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "time_steps": len(sales),
        "node_types": nodes["node_type"].value_counts().to_dict(),
        "edge_types": edges["edge_type"].value_counts().to_dict(),
    }
    with (processed_dir / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    validate_outputs(processed_dir)
    return report


def validate_outputs(processed_dir: Path) -> None:
    nodes = pd.read_csv(processed_dir / "nodes.csv")
    edges = pd.read_csv(processed_dir / "edges.csv")
    sales = pd.read_csv(processed_dir / "sales.csv")
    node_ids = set(nodes["node_id"].astype(str))
    assert len(node_ids) == len(nodes), "Node IDs must be unique"
    assert set(edges["source"]).issubset(node_ids)
    assert set(edges["target"]).issubset(node_ids)
    assert not (edges["source"] == edges["target"]).any()
    assert not edges.duplicated(["source", "target", "edge_type"]).any()
    assert set(sales.columns[1:]) == node_ids
    assert pd.to_datetime(sales["Date"]).is_monotonic_increasing
    assert not sales["Date"].duplicated().any()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2017-12-31")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--missing-policy", choices=["zero", "error"], default="zero")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = preprocess(
        arguments.raw_dir,
        arguments.processed_dir,
        arguments.start_date,
        arguments.end_date,
        arguments.chunksize,
        arguments.missing_policy,
    )
    print(json.dumps(result, indent=2, default=str))
