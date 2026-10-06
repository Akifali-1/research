"""Lightweight tests only; no real dataset is loaded or trained."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import forecasting_metrics  # noqa: E402
from preprocessing import validate_outputs  # noqa: E402


def test_synthetic_processed_files_validate(tmp_path):
    nodes = pd.DataFrame(
        {
            "node_id": ["REGION__A", "CITY__A__X", "STORE__0001", "FAMILY__FOOD"],
            "node_type": ["Region", "City", "Store", "Family"],
        }
    )
    edges = pd.DataFrame(
        {
            "source": ["REGION__A", "CITY__A__X", "STORE__0001"],
            "target": ["CITY__A__X", "STORE__0001", "FAMILY__FOOD"],
            "edge_type": ["region_city", "city_store", "store_family"],
        }
    )
    sales = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=5),
            "REGION__A": [1, 2, 3, 4, 5],
            "CITY__A__X": [1, 2, 3, 4, 5],
            "STORE__0001": [1, 2, 3, 4, 5],
            "FAMILY__FOOD": [1, 2, 3, 4, 5],
        }
    )
    nodes.to_csv(tmp_path / "nodes.csv", index=False)
    edges.to_csv(tmp_path / "edges.csv", index=False)
    sales.to_csv(tmp_path / "sales.csv", index=False)
    validate_outputs(tmp_path)
    pytest.importorskip("torch")
    from utils import build_edge_index, node_type_ids

    edge_index = build_edge_index(nodes, edges)
    assert edge_index.shape == (2, 6)
    assert node_type_ids(nodes).tolist() == [0, 1, 2, 3]


def test_window_shapes_and_chronological_ranges():
    pytest.importorskip("torch")
    from dataset import GraphWindowDataset, collate_graph_windows
    from utils import split_target_ranges

    series = np.arange(4 * 40, dtype=np.float32).reshape(4, 40)
    ranges = split_target_ranges(40, lookback=5, horizon=3)
    dataset = GraphWindowDataset(series, ranges["validation"], 5, 3)
    sample = dataset[0]
    assert tuple(sample.x.shape) == (4, 5, 1)
    assert tuple(sample.y.shape) == (4, 3)
    batch = collate_graph_windows([dataset[0], dataset[1]])
    assert tuple(batch["x"].shape) == (2, 4, 5, 1)
    assert tuple(batch["y"].shape) == (2, 4, 3)


def test_metrics_zero_policy_is_explicit():
    metrics = forecasting_metrics(np.array([0.0, 10.0]), np.array([2.0, 8.0]))
    assert metrics["MAE"] == pytest.approx(2.0)
    assert metrics["MAPE_valid_fraction"] == pytest.approx(0.5)
    assert np.isfinite(metrics["sMAPE"])


def test_model_forward_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gat_lstm import AdaptiveFusionGATLSTM
    from stgt import ReferenceSTGT

    x = torch.randn(2, 4, 5, 1)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    node_types = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    stgt = ReferenceSTGT(lookback=5, horizon=3, d_model=16, temporal_heads=2, spatial_heads=2)
    gat_lstm = AdaptiveFusionGATLSTM(
        lookback=5,
        horizon=3,
        gat_hidden=8,
        gat_heads=2,
        lstm_hidden=8,
        gat_layers=2,
        lstm_layers=2,
    )
    stgt.eval()
    gat_lstm.eval()
    with torch.no_grad():
        assert tuple(stgt(x, edges, node_types).shape) == (2, 4, 3)
        assert tuple(gat_lstm(x, edges).shape) == (2, 4, 3)
