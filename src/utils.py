"""Shared configuration, reproducibility, graph, scaling, and I/O helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import RobustScaler, StandardScaler


def load_yaml(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_json(payload: Mapping[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=default)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all practical random seeds without claiming perfect GPU determinism."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def read_processed_data(data_dir: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    directory = Path(data_dir)
    nodes = pd.read_csv(directory / "nodes.csv")
    edges = pd.read_csv(directory / "edges.csv")
    sales = pd.read_csv(directory / "sales.csv", parse_dates=["Date"])

    required_nodes = {"node_id", "node_type"}
    required_edges = {"source", "target"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"nodes.csv must contain {sorted(required_nodes)}")
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"edges.csv must contain {sorted(required_edges)}")
    if "Date" not in sales.columns:
        raise ValueError("sales.csv must contain a Date column")
    if sales["Date"].duplicated().any():
        raise ValueError("sales.csv contains duplicate dates")
    if not sales["Date"].is_monotonic_increasing:
        raise ValueError("sales.csv must be sorted in chronological order")
    if sales["Date"].isna().any():
        raise ValueError("sales.csv has invalid dates")
    if not sales["Date"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("sales.csv must have a continuous daily timeline")

    node_ids = nodes["node_id"].astype(str).tolist()
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("nodes.csv contains duplicate node_id values")
    missing_sales = sorted(set(node_ids) - set(sales.columns))
    if missing_sales:
        raise ValueError(f"Nodes missing from sales.csv columns: {missing_sales[:10]}")
    unknown_sales = sorted(set(sales.columns) - {"Date", *node_ids})
    if unknown_sales:
        raise ValueError(f"sales.csv contains columns not present in nodes.csv: {unknown_sales[:10]}")
    values = sales[node_ids].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Processed demand must be finite and nonnegative")
    known = set(node_ids)
    if not set(edges["source"].astype(str)).issubset(known):
        raise ValueError("edges.csv contains an unknown source node")
    if not set(edges["target"].astype(str)).issubset(known):
        raise ValueError("edges.csv contains an unknown target node")
    if (edges["source"].astype(str) == edges["target"].astype(str)).any():
        raise ValueError("edges.csv contains self-loops; self-loops are added by the model if needed")
    return nodes, edges, sales


def build_edge_index(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    undirected: bool = True,
) -> torch.Tensor:
    """Build a validated, de-duplicated edge index from processed CSV files."""
    index = {str(node_id): i for i, node_id in enumerate(nodes["node_id"].astype(str))}
    pairs = set()
    for row in edges.itertuples(index=False):
        source = str(getattr(row, "source"))
        target = str(getattr(row, "target"))
        if source == target:
            continue
        if source not in index or target not in index:
            raise ValueError(f"Unknown edge endpoint: {source}, {target}")
        pairs.add((index[source], index[target]))
        if undirected:
            pairs.add((index[target], index[source]))
    if not pairs:
        raise ValueError("No valid graph edges were found")
    ordered = sorted(pairs)
    return torch.tensor(ordered, dtype=torch.long).t().contiguous()


def node_type_ids(nodes: pd.DataFrame) -> torch.Tensor:
    mapping = {"region": 0, "state": 0, "city": 1, "store": 2, "family": 3}
    values = []
    for value in nodes["node_type"].astype(str):
        key = value.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported node_type {value!r}")
        values.append(mapping[key])
    return torch.tensor(values, dtype=torch.long)


def repeat_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """Repeat a single graph for a dense batch of independent graph windows."""
    if batch_size == 1:
        return edge_index
    pieces = []
    for batch_index in range(batch_size):
        pieces.append(edge_index + batch_index * num_nodes)
    return torch.cat(pieces, dim=1)


def split_target_ranges(
    num_steps: int,
    lookback: int,
    horizon: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> Dict[str, Tuple[int, int]]:
    """Return target-index ranges with historical context crossing split boundaries."""
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive")
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Train and validation ratios must be positive and leave test data")
    train_end = int(num_steps * train_ratio)
    validation_end = train_end + int(num_steps * validation_ratio)
    if train_end <= lookback + horizon - 1:
        raise ValueError("Training split is too short for the requested lookback and horizon")
    if validation_end - train_end < horizon or num_steps - validation_end < horizon:
        raise ValueError("Validation or test split is too short for the requested horizon")
    return {
        "train": (lookback, train_end - horizon + 1),
        "validation": (train_end, validation_end - horizon + 1),
        "test": (validation_end, num_steps - horizon + 1),
        "train_end": (train_end, train_end),
        "validation_end": (validation_end, validation_end),
    }


def fit_scale_matrix(
    series: np.ndarray,
    train_end: int,
    kind: str = "standard",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit one scaler per node using only the training prefix."""
    if kind not in {"standard", "robust"}:
        raise ValueError("scaler kind must be 'standard' or 'robust'")
    scaled = np.empty_like(series, dtype=np.float32)
    state: Dict[str, Any] = {"kind": kind, "center": [], "scale": []}
    for node_index in range(series.shape[0]):
        values = series[node_index].astype(np.float64).reshape(-1, 1)
        scaler = StandardScaler() if kind == "standard" else RobustScaler()
        scaler.fit(values[:train_end])
        transformed = scaler.transform(values).reshape(-1)
        scaled[node_index] = transformed.astype(np.float32)
        center = getattr(scaler, "mean_", getattr(scaler, "center_", np.array([0.0])))
        scale = getattr(scaler, "scale_", np.array([1.0]))
        state["center"].append(float(center[0]))
        state["scale"].append(float(scale[0]) if float(scale[0]) != 0 else 1.0)
    return scaled, state


def inverse_scale(values: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    center = np.asarray(state["center"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    return values * scale.reshape(1, -1, 1) + center.reshape(1, -1, 1)


def scale_values(values: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    center = np.asarray(state["center"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    return ((values - center.reshape(-1, 1)) / scale.reshape(-1, 1)).astype(np.float32)
