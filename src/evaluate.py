"""Shared final-test evaluator for a frozen pair of registered checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train import ExperimentData, build_model, evaluate_loader
from utils import load_yaml, model_parameter_count, save_json, set_seed


def load_checkpoint_model(path, name, configuration, data, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = saved["configuration"]
    # Architecture and preprocessing must come from the selected checkpoint.
    if saved["model_name"] != name or saved["node_names"] != data.node_names:
        raise ValueError("Checkpoint model name/node order does not match evaluation")
    if saved_config["data"] != configuration["data"]:
        raise ValueError("Checkpoint preprocessing differs from the frozen comparison protocol")
    if saved["scaler_state"] != data.scaler_state or saved["split_ranges"] != data.ranges:
        raise ValueError("Checkpoint scaler or chronological split differs from evaluation")
    if not torch.equal(saved["edge_index"], data.edge_index):
        raise ValueError("Checkpoint graph differs from evaluation")
    model = build_model(name, saved_config, data).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, saved


def compare_checkpoints(configuration, root, paths):
    """One untouched-test evaluation, followed by tables, report, and plots.

    Overall metrics are micro-averaged over forecast-origin/node/horizon
    entries. Overlapping horizon targets remain separate forecast tasks.
    """
    set_seed(int(configuration["seed"]), bool(configuration.get("deterministic", True)))
    output = root / configuration["outputs"]["metrics_dir"]
    prediction_dir = root / configuration["outputs"]["predictions_dir"]
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if (output / "model_comparison.csv").exists():
        raise FileExistsError("Comparison already exists; it must be verified, not overwritten")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ExperimentData(configuration, root)
    loader = data.loaders(int(configuration["training"]["batch_size"]))["test"]
    rows, breakdowns = [], {}
    reference_targets = None
    for name, path in paths.items():
        model, saved = load_checkpoint_model(path, name, configuration, data, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        evaluation = evaluate_loader(name, model, loader, data, device)
        if reference_targets is None:
            reference_targets = evaluation["y_true"]
        elif not np.array_equal(reference_targets, evaluation["y_true"]):
            raise ValueError("Models were not evaluated on identical original-scale targets")
        attempt = Path(path).parent.parent
        training_report = json.loads((attempt / "metrics/validation.json").read_text())
        rows.append({"Model": name, **evaluation["overall"], "Parameter_count": model_parameter_count(model),
                     "Training_seconds": training_report["training_seconds"],
                     "Inference_seconds": evaluation["inference_seconds"],
                     "Training_peak_GPU_bytes": training_report["peak_gpu_memory_bytes"],
                     "Inference_peak_GPU_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0})
        breakdowns[name] = {"by_horizon": evaluation["by_horizon"], "by_node_type": evaluation["by_node_type"],
                            "experiment_id": saved["configuration"]["run"]["experiment_id"]}
        np.savez_compressed(prediction_dir / f"{name}_test.npz", y_true=evaluation["y_true"], y_pred=evaluation["y_pred"],
                            target_indices=evaluation["target_indices"], node_names=data.node_names, dates=data.dates)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["Model"] == "stgt"].iloc[0]
    for metric in ("MAE", "MSE", "RMSE", "WAPE", "sMAPE", "R2", "MAPE_masked"):
        value = float(baseline[metric])
        frame[f"{metric}_absolute_difference_vs_stgt"] = frame[metric] - value
        if np.isfinite(value) and value != 0:
            frame[f"{metric}_percent_difference_vs_stgt"] = (frame[metric] - value) / abs(value) * 100
    frame.to_csv(output / "model_comparison.csv", index=False)
    save_json(breakdowns, output / "model_comparison_breakdowns.json")
    lines = ["# Frozen test comparison", "", "Configurations selected using validation only.", "",
             "Overall metrics micro-average over forecast-origin × node × horizon; series at different hierarchy levels overlap.",
             "WAPE = sum(|error|)/sum(|actual|). sMAPE averages 2|error|/(|actual|+|prediction|), with 0/0=0.",
             "MAPE uses actual > 0.1; valid fraction is reported. WAPE/MAPE/R² are NaN when undefined.",
             "Percent metrics are ratios, not percentages. Negative predictions are clipped to zero for all metrics.",
             "Inference timing measures synchronized forward passes only; training timing includes completed epochs across retries.", "",
             "```csv", frame.to_csv(index=False), "```", "No automatic superiority claim is made."]
    (output / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.bar(frame["Model"], frame[metric])
        axis.set_title(f"Test {metric} (original units)")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--stgt-checkpoint", type=Path, required=True)
    parser.add_argument("--gat-lstm-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if not config.get("final_evaluation", {}).get("selection_frozen"):
        raise SystemExit("Test gate: use orchestrator --finalize-test after freezing validation-selected experiment IDs")
    print(compare_checkpoints(config, args.project_root.resolve(),
                              {"stgt": args.stgt_checkpoint, "gat_lstm": args.gat_lstm_checkpoint}).to_string(index=False))
