"""Train one model under the shared chronological experiment protocol.

This script performs real training only when explicitly invoked by the user in
Colab. Importing the module does not load data or start computation.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import GraphWindowDataset, make_loader
from gat_lstm import AdaptiveFusionGATLSTM
from metrics import forecasting_metrics, metrics_by_horizon, metrics_by_node_type
from stgt import ReferenceSTGT
from utils import (
    build_edge_index,
    fit_scale_matrix,
    inverse_scale,
    load_yaml,
    model_parameter_count,
    node_type_ids,
    read_processed_data,
    save_json,
    set_seed,
    split_target_ranges,
)


class ExperimentData:
    """Load and scale processed data once for either model."""

    def __init__(self, config: Dict[str, Any], project_root: Path):
        data_config = config["data"]
        data_dir = Path(data_config["processed_dir"])
        if not data_dir.is_absolute():
            data_dir = project_root / data_dir
        self.nodes, self.edges, self.sales = read_processed_data(data_dir)
        self.node_names = self.nodes["node_id"].astype(str).tolist()
        self.node_type_ids = node_type_ids(self.nodes)
        self.edge_index = build_edge_index(self.nodes, self.edges, data_config.get("undirected_graph", True))
        self.dates = pd.to_datetime(self.sales["Date"]).dt.strftime("%Y-%m-%d").tolist()
        raw_series = self.sales[self.node_names].to_numpy(dtype=np.float32).T
        self.num_nodes, self.num_steps = raw_series.shape
        self.lookback = int(data_config["lookback"])
        self.horizon = int(data_config["horizon"])
        self.ranges = split_target_ranges(
            self.num_steps,
            self.lookback,
            self.horizon,
            float(data_config.get("train_ratio", 0.70)),
            float(data_config.get("validation_ratio", 0.15)),
        )
        self.train_end = self.ranges["train_end"][0]
        self.scaled_series, self.scaler_state = fit_scale_matrix(
            raw_series, self.train_end, str(data_config.get("scaler", "standard"))
        )
        self.baseline_scaled = None
        if bool(data_config.get("use_residual_baseline", False)):
            if self.horizon != 1:
                raise ValueError("The causal residual adapter currently supports horizon=1 only")
            baseline = np.zeros_like(raw_series, dtype=np.float32)
            baseline[:, 0] = raw_series[:, 0]
            alpha = float(data_config.get("baseline_alpha", 2.0 / 8.0))
            for step in range(1, self.num_steps):
                baseline[:, step] = alpha * raw_series[:, step - 1] + (1 - alpha) * baseline[:, step - 1]
            _, baseline_state = fit_scale_matrix(baseline, self.train_end, str(data_config.get("scaler", "standard")))
            # Use the demand scaler so residuals and predictions invert consistently.
            center = np.asarray(self.scaler_state["center"]).reshape(-1, 1)
            scale = np.asarray(self.scaler_state["scale"]).reshape(-1, 1)
            self.baseline_scaled = ((baseline - center) / scale).astype(np.float32)

        self.datasets = {
            name: GraphWindowDataset(
                self.scaled_series,
                self.ranges[name],
                self.lookback,
                self.horizon,
                self.baseline_scaled,
            )
            for name in ("train", "validation", "test")
        }

    def loaders(self, batch_size: int):
        return {
            "train": make_loader(self.datasets["train"], batch_size, shuffle=True),
            "validation": make_loader(self.datasets["validation"], batch_size, shuffle=False),
            "test": make_loader(self.datasets["test"], batch_size, shuffle=False),
        }


def build_model(model_name: str, config: Dict[str, Any], data: ExperimentData) -> torch.nn.Module:
    model_config = dict(config["models"][model_name])
    common = {"lookback": data.lookback, "horizon": data.horizon}
    if model_name == "stgt":
        return ReferenceSTGT(**common, **model_config)
    if model_name == "gat_lstm":
        return AdaptiveFusionGATLSTM(**common, **model_config)
    raise ValueError(f"Unsupported model: {model_name}")


def _call_model(model_name: str, model: torch.nn.Module, batch: Dict[str, torch.Tensor], edge_index, node_types):
    if model_name == "stgt":
        return model(batch["x"], edge_index, node_types)
    return model(batch["x"], edge_index)


def _absolute_arrays(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    scaler_state: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if "baseline_y" in batch:
        baseline_np = batch["baseline_y"].detach().cpu().numpy()
        prediction_np = prediction_np + baseline_np
        target_np = target_np + baseline_np
    return inverse_scale(target_np, scaler_state), inverse_scale(prediction_np, scaler_state)


@torch.no_grad()
def evaluate_loader(
    model_name: str,
    model: torch.nn.Module,
    loader,
    data: ExperimentData,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    true_batches = []
    prediction_batches = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = _call_model(model_name, model, batch, data.edge_index.to(device), data.node_type_ids.to(device))
        true_np, prediction_np = _absolute_arrays(prediction, batch["y"], batch, data.scaler_state)
        true_batches.append(true_np)
        prediction_batches.append(prediction_np)
    if not true_batches:
        raise RuntimeError("Evaluation loader is empty")
    y_true = np.concatenate(true_batches, axis=0)
    y_pred = np.concatenate(prediction_batches, axis=0)
    return {
        "overall": forecasting_metrics(y_true, y_pred),
        "by_horizon": metrics_by_horizon(y_true, y_pred),
        "by_node_type": metrics_by_node_type(y_true, y_pred, data.node_type_ids.numpy()),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_model(model_name: str, config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ExperimentData(config, project_root)
    loaders = data.loaders(int(config["training"]["batch_size"]))
    model = build_model(model_name, config, data).to(device)
    training_config = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config["scheduler_factor"]),
        patience=int(training_config["scheduler_patience"]),
    )
    epochs = int(training_config["epochs"])
    patience = int(training_config["patience"])
    clip_norm = float(training_config["gradient_clip_norm"])
    delta = float(training_config.get("huber_delta", 1.0))
    best_validation = float("inf")
    best_state = None
    wait = 0
    history = []
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = _call_model(model_name, model, batch, data.edge_index.to(device), data.node_type_ids.to(device))
            loss = F.huber_loss(prediction, batch["y"], delta=delta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        train_loss = running_loss / max(1, len(loaders["train"]))

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in loaders["validation"]:
                batch = {key: value.to(device) for key, value in batch.items()}
                prediction = _call_model(model_name, model, batch, data.edge_index.to(device), data.node_type_ids.to(device))
                validation_loss += float(F.huber_loss(prediction, batch["y"], delta=delta).cpu())
        validation_loss /= max(1, len(loaders["validation"]))
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - start_time
    evaluation_start = time.perf_counter()
    evaluation = evaluate_loader(model_name, model, loaders["test"], data, device)
    inference_seconds = time.perf_counter() - evaluation_start
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    outputs = config["outputs"]
    checkpoint_dir = project_root / outputs["checkpoint_dir"]
    metrics_dir = project_root / outputs["metrics_dir"]
    predictions_dir = project_root / outputs["predictions_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_name": model_name,
        "model_config": config["models"][model_name],
        "data_config": config["data"],
        "state_dict": best_state,
        "node_names": data.node_names,
        "node_type_ids": data.node_type_ids,
        "edge_index": data.edge_index,
        "scaler_state": data.scaler_state,
        "split_ranges": data.ranges,
        "history": history,
    }
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"
    torch.save(checkpoint, checkpoint_path)
    np.savez_compressed(predictions_dir / f"{model_name}_test.npz", y_true=evaluation["y_true"], y_pred=evaluation["y_pred"])
    report = {
        "model": model_name,
        "device": str(device),
        "parameter_count": model_parameter_count(model),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_bytes": int(peak_memory),
        "epochs_completed": len(history),
        "best_validation_loss": best_validation,
        "test": evaluation["overall"],
        "test_by_horizon": evaluation["by_horizon"],
        "test_by_node_type": evaluation["by_node_type"],
        "checkpoint": str(checkpoint_path),
    }
    save_json(report, metrics_dir / f"{model_name}.json")
    save_json({"model": model_name, "history": history}, project_root / outputs["logs_dir"] / f"{model_name}_history.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--model", choices=["stgt", "gat_lstm"], required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    root = arguments.project_root.resolve()
    configuration = load_yaml(root / arguments.config)
    report = train_model(arguments.model, configuration, root)
    print(report)
