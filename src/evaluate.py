"""Evaluate saved model checkpoints and produce a comparison CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from train import ExperimentData, _call_model, evaluate_loader, build_model
from utils import load_yaml, save_json, set_seed


def load_checkpoint_model(checkpoint_path: Path, model_name: str, config: Dict[str, Any], data: ExperimentData, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(model_name, config, data).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def compare_checkpoints(
    config: Dict[str, Any],
    project_root: Path,
    checkpoint_paths: Dict[str, Path],
) -> pd.DataFrame:
    set_seed(int(config["seed"]), bool(config.get("deterministic", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ExperimentData(config, project_root)
    loaders = data.loaders(int(config["training"]["batch_size"]))
    rows = []
    reports = {}
    for model_name, checkpoint_path in checkpoint_paths.items():
        model = load_checkpoint_model(checkpoint_path, model_name, config, data, device)
        evaluation = evaluate_loader(model_name, model, loaders["test"], data, device)
        row = {"Model": model_name, **evaluation["overall"]}
        rows.append(row)
        reports[model_name] = {"overall": evaluation["overall"], "by_horizon": evaluation["by_horizon"], "by_node_type": evaluation["by_node_type"]}
    frame = pd.DataFrame(rows)
    if len(frame) == 2:
        reference = frame.iloc[0]
        for metric in ["MAE", "MSE", "RMSE", "WAPE", "sMAPE", "R2", "MAPE_masked"]:
            frame[f"{metric}_difference_vs_{reference['Model']}"] = frame[metric] - float(reference[metric])
            if float(reference[metric]) != 0:
                frame[f"{metric}_percent_difference_vs_{reference['Model']}"] = (
                    (frame[metric] - float(reference[metric])) / abs(float(reference[metric])) * 100.0
                )
    output_dir = project_root / config["outputs"]["metrics_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "model_comparison.csv", index=False)
    save_json(reports, output_dir / "model_comparison_breakdowns.json")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--stgt-checkpoint", type=Path, default=None)
    parser.add_argument("--gat-lstm-checkpoint", type=Path, default=None)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    root = arguments.project_root.resolve()
    configuration = load_yaml(root / arguments.config)
    checkpoints = {}
    if arguments.stgt_checkpoint:
        checkpoints["stgt"] = arguments.stgt_checkpoint
    if arguments.gat_lstm_checkpoint:
        checkpoints["gat_lstm"] = arguments.gat_lstm_checkpoint
    if len(checkpoints) < 1:
        raise SystemExit("Provide at least one checkpoint")
    print(compare_checkpoints(configuration, root, checkpoints).to_string(index=False))
