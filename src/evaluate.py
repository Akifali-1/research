"""Shared final-test evaluator with immutable, checksum-verified recovery stages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Executable scripts and package imports must share the same support modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation_recovery import EvaluationWorkspace
from src.integrity import fingerprint
from train import ExperimentData, build_model, evaluate_loader
from utils import load_yaml, model_parameter_count, save_json, set_seed


def load_checkpoint_model(path, name, configuration, data, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved_config = saved["configuration"]
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


def _report(folder, results):
    rows, breakdowns = [], {}
    for stage_name, directory in results.items():
        result = json.loads((directory / "result.json").read_text())
        rows.append(result["row"])
        breakdowns[stage_name] = result["breakdowns"]
    frame = pd.DataFrame(rows).sort_values(["Model", "Seed"]).reset_index(drop=True)
    baseline = frame.loc[frame["Model"] == "stgt"].set_index("Seed")
    for metric in ("MAE", "MSE", "RMSE", "WAPE", "sMAPE", "R2", "MAPE_masked"):
        differences = []
        percentages = []
        for _, row in frame.iterrows():
            value = float(baseline.loc[row["Seed"], metric])
            difference = float(row[metric]) - value
            differences.append(difference)
            percentages.append(difference / abs(value) * 100 if np.isfinite(value) and value != 0 else np.nan)
        frame[f"{metric}_absolute_difference_vs_stgt"] = differences
        frame[f"{metric}_percent_difference_vs_stgt"] = percentages
    summary_names = (
        "MAE", "MSE", "RMSE", "WAPE", "sMAPE", "MAPE_masked", "R2",
        "Parameter_count", "Training_seconds", "Inference_seconds",
    )
    summary = frame.groupby("Model", as_index=False)[list(summary_names)].agg(["mean", "std"])
    summary.columns = [
        "Model" if index == ("Model", "") else f"{index[0]}_{index[1]}"
        for index in summary.columns.to_flat_index()
    ]
    frame.to_csv(folder / "model_comparison.csv", index=False)
    summary.to_csv(folder / "model_comparison_summary.csv", index=False)
    save_json(breakdowns, folder / "model_comparison_breakdowns.json")
    lines = ["# Frozen test comparison", "", "Configurations selected using validation only.", "",
             "Overall metrics micro-average over forecast-origin × node × horizon; hierarchy aggregates overlap.",
             "WAPE = sum(|error|)/sum(|actual|). sMAPE averages 2|error|/(|actual|+|prediction|), with 0/0=0.",
             "MAPE uses actual > 0.1; valid fraction is reported. WAPE/MAPE/R² are NaN when undefined.",
             "Ratio metrics are ratios, not percentages. Negative predictions are clipped to zero for all metrics.",
             "Inference time is synchronized forward-pass time; training time includes committed epochs across retries.",
             "Different parameter counts/epochs do not establish equal FLOPs. Per-seed rows precede mean/std summaries.",
             "STGT is a project-reference baseline, not a verified published-model reproduction.", "",
             "## Per-seed results", "", "```csv", frame.to_csv(index=False), "```", "",
             "## Mean and standard deviation across seeds", "", "```csv", summary.to_csv(index=False), "```",
             "No automatic superiority claim is made."]
    (folder / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("MAE", "RMSE")):
        axis.bar(frame["Model"], frame[metric])
        axis.set_title(f"Test {metric} (original units)")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=150)
    plt.close(fig)


def compare_checkpoints(configuration, root, paths):
    """Resume only missing model/report stages of the same frozen selection."""
    frozen = configuration["final_evaluation"]
    directory = Path(frozen["directory"])
    workspace = EvaluationWorkspace(directory, frozen["experiment_ids"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ExperimentData(configuration, root)
    loader = data.loaders(int(configuration["training"]["batch_size"]))["test"]
    metadata = {"data": configuration["data"], "seed": configuration["seed"],
                "code_fingerprint": frozen["code_fingerprint"], "packages": frozen["packages"]}
    results = {}
    reference_targets = reference_origins = None
    for stage_name, (name, path) in paths.items():
        path = Path(path)
        checkpoint_hash = fingerprint(path)["sha256"]
        def compute_stage(folder):
            model, saved = load_checkpoint_model(path, name, configuration, data, device)
            saved_seed = int(saved["configuration"]["seed"])
            set_seed(saved_seed, bool(configuration.get("deterministic", True)))
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            evaluation = evaluate_loader(name, model, loader, data, device)
            attempt = path.parent.parent
            training_report = json.loads((attempt / "metrics/validation.json").read_text())
            selected_epoch = training_report.get("selected_epoch")
            if selected_epoch is None:
                history = json.loads((attempt / "logs/history.json").read_text())["history"]
                selected_epoch = min(history, key=lambda item: item["validation_loss"])["epoch"]
            row = {"Model": name, **evaluation["overall"], "Parameter_count": model_parameter_count(model),
                   "Training_seconds": training_report["training_seconds"], "Inference_seconds": evaluation["inference_seconds"],
                   "Training_peak_GPU_bytes": training_report["peak_gpu_memory_bytes"],
                   "Inference_peak_GPU_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                   "Inference_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                   "Selected_epoch": selected_epoch, "Seed": saved_seed}
            result = {"row": row, "breakdowns": {"by_horizon": evaluation["by_horizon"], "by_node_type": evaluation["by_node_type"],
                      "experiment_id": saved["configuration"]["run"]["experiment_id"]}}
            save_json(result, folder / "result.json")
            np.savez_compressed(folder / "predictions.npz", y_true=evaluation["y_true"], y_pred=evaluation["y_pred"],
                                target_indices=evaluation["target_indices"], node_names=data.node_names, dates=data.dates)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        stage = workspace.stage(stage_name, {**metadata, "model_name": name,
                                             "seed": int(configuration["seed"]),
                                             "checkpoint_sha256": checkpoint_hash}, compute_stage,
                                ["result.json", "predictions.npz"])
        with np.load(stage / "predictions.npz", allow_pickle=False) as saved:
            targets, origins = saved["y_true"], saved["target_indices"]
            if reference_targets is None:
                reference_targets, reference_origins = targets, origins
            elif not np.array_equal(reference_targets, targets) or not np.array_equal(reference_origins, origins):
                raise ValueError("Models were not evaluated on identical targets and forecast origins")
        results[stage_name] = stage
        print(f"Verified final-test model stage: {stage}", flush=True)
    report = workspace.stage(
        "report",
        metadata,
        lambda folder: _report(folder, results),
        ["model_comparison.csv", "model_comparison_summary.csv",
         "model_comparison_breakdowns.json", "comparison_report.md", "comparison.png"],
    )
    return pd.read_csv(report / "model_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint specifications of the form MODEL_seedSEED=/path/to/best.pt",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    if not config.get("final_evaluation", {}).get("selection_frozen"):
        raise SystemExit("Use orchestrator --finalize-test with frozen validation-selected IDs")
    paths = {}
    for specification in args.checkpoints:
        try:
            stage_name, path = specification.split("=", 1)
            model_name, seed_text = stage_name.rsplit("_seed", 1)
            int(seed_text)
        except ValueError as exc:
            raise SystemExit(f"Invalid checkpoint specification: {specification}") from exc
        if stage_name in paths or model_name not in {"stgt", "adaptive_stgt", "gat_lstm"}:
            raise SystemExit(f"Invalid or duplicate checkpoint stage: {stage_name}")
        paths[stage_name] = (model_name, Path(path))
    print(compare_checkpoints(config, args.project_root.resolve(), paths).to_string(index=False))
