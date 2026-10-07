"""Convert an ISOMORPH rollout to the existing graph/time-series interface.

The sample's customer demand is recorded at NewYork only. To preserve a
meaningful node-level forecasting task, this adapter uses aggregate on-hand
inventory across the 50 items as the target state for every physical node.
Demand remains available in the raw rollout and is not copied to upstream
nodes. The downstream STGT/training/metric code is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _read_edges(path: Path) -> pd.DataFrame:
    return pq.ParquetFile(path).read().to_pandas()


def _aggregate_history(path: Path, days: int, node_to_index: dict[str, int], value_column: str) -> np.ndarray:
    result = np.zeros((days, len(node_to_index)), dtype=np.float64)
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["day", "node", value_column], batch_size=1_000_000):
        day = batch.column("day").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        node = np.asarray(batch.column("node").to_pylist(), dtype=object)
        value = batch.column(value_column).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        node_index = np.fromiter((node_to_index[str(name)] for name in node), dtype=np.int64, count=len(node))
        np.add.at(result, (day, node_index), value)
    return result


def process_rollout(raw_root: Path, output_root: Path, start_date: str = "2000-01-01") -> dict:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    scenario = json.loads((raw_root / "scenario.json").read_text(encoding="utf-8"))
    days = int(scenario["days"])
    edge_frame = _read_edges(raw_root / "edge_list.parquet")
    nodes = sorted(set(edge_frame["from"].astype(str)) | set(edge_frame["to"].astype(str)))
    node_to_index = {name: index for index, name in enumerate(nodes)}
    in_degree = {name: 0 for name in nodes}
    out_degree = {name: 0 for name in nodes}
    for row in edge_frame.itertuples(index=False):
        out_degree[str(row[1])] += 1
        in_degree[str(row[2])] += 1
    node_rows = []
    for node in nodes:
        role = "source" if in_degree[node] == 0 else "destination" if out_degree[node] == 0 else "warehouse"
        # Preserve the existing four-category node-type interface without
        # changing model definitions: source/warehouse/destination map to the
        # established region/city/store embedding IDs respectively.
        interface_type = {"source": "region", "warehouse": "city", "destination": "store"}[role]
        node_rows.append({"node_id": node, "node_type": interface_type, "role": role, "label": node, "parent_id": ""})
    node_frame = pd.DataFrame(node_rows)
    edges = edge_frame.rename(columns={"from": "source", "to": "target"})[["source", "target"]].copy()
    edges["edge_type"] = "directed_route"
    inventory = _aggregate_history(raw_root / "inventory_history.parquet", days, node_to_index, "on_hand")
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    sales = pd.DataFrame(inventory, columns=nodes)
    sales.insert(0, "Date", dates)
    output_root.mkdir(parents=True, exist_ok=True)
    node_frame.to_csv(output_root / "nodes.csv", index=False)
    edges.to_csv(output_root / "edges.csv", index=False)
    sales.to_csv(output_root / "sales.csv", index=False, date_format="%Y-%m-%d")
    target_definition = {
        "target": "aggregate_on_hand_inventory",
        "aggregation": "sum of on_hand across all 50 items for each physical node and day",
        "reason": "ISOMORPH customer demand is recorded at NewYork only; aggregate inventory is the available all-node state target without replicating demand to upstream nodes",
        "raw_demand_source": "daily_records.parquet and demand_signals.npy remain auxiliary and are not copied into upstream node targets",
        "node_count": len(nodes),
        "item_count": int(scenario["n_items"]),
        "time_steps": days,
        "start_date_anchor": start_date,
        "directed_edges": True,
        "edge_count": len(edges),
    }
    (output_root / "target_definition.json").write_text(json.dumps(target_definition, indent=2), encoding="utf-8")
    quality = {
        "dataset": "ISOMORPH sample baseline_item50",
        "source_scenario": scenario,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "time_steps": days,
        "node_types": node_frame["node_type"].value_counts().to_dict(),
        "target": target_definition,
        "min_target": float(inventory.min()),
        "max_target": float(inventory.max()),
        "mean_target": float(inventory.mean()),
        "all_finite_nonnegative": bool(np.isfinite(inventory).all() and (inventory >= 0).all()),
    }
    (output_root / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return quality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(process_rollout(args.raw_root, args.output_root), indent=2))
