"""Train one model under the shared chronological experiment protocol.

This script performs real training only when explicitly invoked by the user in
Colab. Importing the module does not load data or start computation.
"""

from __future__ import annotations

import argparse
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
from stgt import AdaptiveSTGT, ReferenceSTGT
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
from checkpointing import atomic_torch_save, capture_rng_state, restore_rng_state


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
    if model_name == "adaptive_stgt":
        return AdaptiveSTGT(**common, **model_config)
    if model_name == "gat_lstm":
        return AdaptiveFusionGATLSTM(**common, **model_config)
    raise ValueError(f"Unsupported model: {model_name}")


def _call_model(model_name: str, model: torch.nn.Module, batch: Dict[str, torch.Tensor], edge_index, node_types):
    if model_name in {"stgt", "adaptive_stgt"}:
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
    target_indices = []
    edge_index = data.edge_index.to(device)
    types = data.node_type_ids.to(device)
    inference_seconds = 0.0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = _call_model(model_name, model, batch, edge_index, types)
        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds += time.perf_counter() - started
        true_np, prediction_np = _absolute_arrays(prediction, batch["y"], batch, data.scaler_state)
        true_batches.append(true_np)
        prediction_batches.append(prediction_np)
        target_indices.extend(batch["target_index"].cpu().tolist())
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
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "inference_seconds": inference_seconds,
    }


def train_model(model_name: str, config: Dict[str, Any], project_root: Path, resume: Path | None = None) -> Dict[str, Any]:
    """Validation-only training. Test evaluation is an explicit later stage.

    Interrupted attempts recover from the last fully committed epoch. Any
    partial epoch is replayed in an explicit, budget-counted retry.
    """
    if not config.get("run", {}).get("experiment_id"):
        raise RuntimeError("Training must be launched by the budget controller with a registered run")
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
    start_epoch = 1
    prior_seconds = 0.0
    outputs = config["outputs"]
    checkpoint_dir = project_root / outputs["checkpoint_dir"]
    metrics_dir = project_root / outputs["metrics_dir"]
    predictions_dir = project_root / outputs["predictions_dir"]
    logs_dir = project_root / outputs["logs_dir"]
    for directory in (checkpoint_dir, metrics_dir, predictions_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"
    run = config["run"]
    metadata = {
        "model_name": model_name,
        "configuration": config,
        "signature": run["signature"],
        "node_names": data.node_names,
        "node_type_ids": data.node_type_ids,
        "edge_index": data.edge_index,
        "scaler_state": data.scaler_state,
        "split_ranges": data.ranges,
        "dates": data.dates,
    }
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["signature"] != run["signature"] or saved["node_names"] != data.node_names:
            raise ValueError("Resume checkpoint does not match code/config/dataset signature or node order")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        best_state = saved["best_state"]
        best_validation = saved["best_validation_loss"]
        wait = saved["wait"]
        history = saved["history"]
        start_epoch = int(saved["epoch"]) + 1
        prior_seconds = float(saved["training_seconds"])
        restore_rng_state(saved["rng_state"])
        atomic_torch_save({**metadata, "state_dict": best_state}, best_path)
        # A retry finishing after the last epoch still needs its own durable
        # latest checkpoint and history, rather than pointing only at an old attempt.
        atomic_torch_save({**saved, **metadata}, latest_path)
        save_json({"history": history}, logs_dir / "history.json")
        print(f"Resuming after committed epoch {saved['epoch']}; this retry consumes another budget slot", flush=True)
    start_time = time.perf_counter()
    runtime_limit = float(config["experiment"]["max_job_seconds"])
    edge_index = data.edge_index.to(device)
    types = data.node_type_ids.to(device)

    def check_time():
        if time.perf_counter() - start_time >= runtime_limit:
            raise TimeoutError("Training runtime limit reached; committed epoch checkpoints are preserved")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print(f"Training {model_name} on {device}: nodes={data.num_nodes}, steps={data.num_steps}, "
          f"train_windows={len(data.datasets['train'])}, val_windows={len(data.datasets['validation'])}, "
          f"epochs={epochs}, batch_size={training_config['batch_size']}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        if wait >= patience:
            break
        check_time()
        model.train()
        running_loss = 0.0
        sample_count = 0
        epoch_started = time.perf_counter()
        for step, batch in enumerate(loaders["train"], start=1):
            check_time()
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = _call_model(model_name, model, batch, edge_index, types)
            loss = F.huber_loss(prediction, batch["y"], delta=delta)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            count = batch["x"].shape[0]
            running_loss += float(loss.detach().cpu()) * count
            sample_count += count
            if step == 1 or step % 25 == 0 or step == len(loaders["train"]):
                print(f"[{model_name}] Epoch {epoch}/{epochs} | batch {step}/{len(loaders['train'])} "
                      f"| mean_train_loss={running_loss / sample_count:.6f}", flush=True)
        train_loss = running_loss / sample_count

        model.eval()
        validation_loss = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in loaders["validation"]:
                check_time()
                batch = {key: value.to(device) for key, value in batch.items()}
                prediction = _call_model(model_name, model, batch, edge_index, types)
                count = batch["x"].shape[0]
                validation_loss += float(F.huber_loss(prediction, batch["y"], delta=delta).cpu()) * count
                validation_count += count
        validation_loss /= validation_count
        if not np.isfinite(validation_loss):
            raise ValueError("Nonfinite validation loss")
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
            atomic_torch_save({**metadata, "state_dict": best_state}, best_path)
            wait = 0
        else:
            wait += 1
        latest = {
            **metadata, "state_dict": model.state_dict(), "epoch": epoch,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_state": best_state, "best_validation_loss": best_validation,
            "wait": wait, "history": history, "rng_state": capture_rng_state(),
            "training_seconds": prior_seconds + time.perf_counter() - start_time,
        }
        atomic_torch_save(latest, latest_path)
        save_json({"history": history}, logs_dir / "history.json")
        print(f"[{model_name}] Epoch {epoch}/{epochs} complete | train={train_loss:.6f} "
              f"| val={validation_loss:.6f} | best_val={best_validation:.6f} "
              f"| patience={wait}/{patience} | lr={optimizer.param_groups[0]['lr']:.2g} "
              f"| elapsed={time.perf_counter() - epoch_started:.1f}s | checkpoint={latest_path}", flush=True)
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    training_seconds = prior_seconds + time.perf_counter() - start_time
    evaluation = evaluate_loader(model_name, model, loaders["validation"], data, device)
    inference_seconds = evaluation["inference_seconds"]
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    np.savez_compressed(predictions_dir / "validation.npz", y_true=evaluation["y_true"], y_pred=evaluation["y_pred"],
                        target_indices=evaluation["target_indices"], node_names=data.node_names, dates=data.dates)
    report = {
        "model": model_name,
        "device": str(device),
        "parameter_count": model_parameter_count(model),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_bytes": int(peak_memory),
        "epochs_completed": len(history),
        "best_validation_loss": best_validation,
        "experiment_id": run["experiment_id"],
        "signature": run["signature"],
        "git_commit": run["git_commit"],
        "validation": evaluation["overall"],
        "validation_by_horizon": evaluation["by_horizon"],
        "validation_by_node_type": evaluation["by_node_type"],
        "checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    save_json(report, metrics_dir / "validation.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--model", choices=["stgt", "adaptive_stgt", "gat_lstm"], required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume", type=Path, default=None)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    root = arguments.project_root.resolve()
    configuration = load_yaml(root / arguments.config)
    report = train_model(arguments.model, configuration, root, arguments.resume)
    print(report)
