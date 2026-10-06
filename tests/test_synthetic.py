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
from preprocessing import preprocess, validate_outputs  # noqa: E402


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


def test_chunked_preprocessing_uses_tiny_csvs_only(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame({"item_nbr": [1, 2], "family": ["FOOD", "DRINK"]}).to_csv(raw / "items.csv", index=False)
    pd.DataFrame({"store_nbr": [1], "state": ["A"], "city": ["X"]}).to_csv(raw / "stores.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3, 4], "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
                  "store_nbr": [1, 1, 1, 1], "item_nbr": [1, 2, 1, 2], "unit_sales": [10, 5, -1, 2]}).to_csv(raw / "train.csv", index=False)
    output = tmp_path / "processed"
    report = preprocess(raw, output, "2020-01-01", "2020-01-04", chunksize=1)
    sales = pd.read_csv(output / "sales.csv")
    assert len(sales) == 2  # no manufactured future zero labels
    assert sales["STORE__0001"].tolist() == [15.0, 2.0]
    assert sales["FAMILY__FOOD"].tolist() == [10.0, 0.0]
    assert report["negative_sales_clipped"] == 1
    assert report["edge_count"] == 4
    validate_outputs(output)


def test_raw_validator_accepts_csvs_and_checks_train_schema(tmp_path):
    from orchestrator import validate_archives
    from src.raw_data import SOURCE_STEMS
    for stem in SOURCE_STEMS:
        (tmp_path / f"{stem}.csv").write_text("fixture\n")
    (tmp_path / "items.csv").write_text("item_nbr,family,class,perishable\n1,FOOD,1,0\n")
    (tmp_path / "stores.csv").write_text("store_nbr,city,state,type,cluster\n1,X,A,D,1\n")
    (tmp_path / "train.csv").write_text("id,date,store_nbr,item_nbr,unit_sales\n1,2020-01-01,1,1,5\n")
    manifest = validate_archives(tmp_path)
    assert len(manifest["files"]) == 8
    assert all(entry["name"].endswith(".csv") for entry in manifest["files"])
    (tmp_path / "train.csv").write_text("wrong,headers\n")
    with pytest.raises(ValueError, match="train.csv"):
        validate_archives(tmp_path)


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
    assert ranges["train"][1] - 1 + 3 <= ranges["train_end"][0]
    assert ranges["validation"][1] - 1 + 3 <= ranges["validation_end"][0]
    assert ranges["test"][0] == ranges["validation_end"][0]


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
        prediction = stgt(x, edges, node_types)
        assert tuple(prediction.shape) == (2, 4, 3)
        assert torch.isfinite(torch.nn.functional.huber_loss(prediction, torch.zeros_like(prediction)))
        separate = torch.cat([stgt(x[i:i+1], edges, node_types) for i in range(2)])
        assert torch.allclose(prediction, separate, atol=1e-5)
        gat_prediction = gat_lstm(x, edges)
        assert tuple(gat_prediction.shape) == (2, 4, 3)
        separate = torch.cat([gat_lstm(x[i:i+1], edges) for i in range(2)])
        assert torch.allclose(gat_prediction, separate, atol=1e-5)


def test_atomic_checkpoint_and_rng_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from checkpointing import atomic_torch_save, capture_rng_state, restore_rng_state
    state = capture_rng_state()
    expected = torch.rand(3)
    atomic_torch_save({"epoch": 2, "rng_state": state, "tensor": expected}, tmp_path / "latest.pt")
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    restore_rng_state(saved["rng_state"])
    assert torch.equal(torch.rand(3), expected)
    assert saved["epoch"] == 2


def test_scaler_uses_training_prefix_only():
    pytest.importorskip("torch")
    from utils import fit_scale_matrix
    series = np.array([[1., 3., 1000., 2000.]])
    _, state = fit_scale_matrix(series, train_end=2)
    assert state["center"] == [2.0]
    assert state["scale"] == [1.0]
