"""Consistent original-scale forecasting metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray, zero_threshold: float = 0.1) -> Dict[str, float]:
    """Compute metrics on original units.

    MAPE is reported only on observations whose actual value is greater than
    zero_threshold. WAPE and sMAPE retain all observations and remain defined
    when actual demand is zero.
    """
    true = _flatten(y_true)
    pred = np.maximum(_flatten(y_pred), 0.0)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-empty shape")
    absolute_error = np.abs(true - pred)
    denominator = np.abs(true) + np.abs(pred)
    mask = np.abs(true) > zero_threshold
    wape_denominator = np.sum(np.abs(true))
    r2 = float(r2_score(true, pred)) if np.unique(true).size > 1 else 0.0
    return {
        "MAE": float(mean_absolute_error(true, pred)),
        "MSE": float(mean_squared_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "WAPE": float(np.sum(absolute_error) / wape_denominator) if wape_denominator > 0 else 0.0,
        "sMAPE": float(np.mean(2.0 * absolute_error / np.maximum(denominator, 1e-8))),
        "R2": r2,
        "MAPE_masked": float(np.mean(absolute_error[mask] / np.abs(true[mask]))) if np.any(mask) else 0.0,
        "MAPE_valid_fraction": float(np.mean(mask)),
    }


def metrics_by_horizon(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    if y_true.ndim != 3:
        raise ValueError("Expected [samples, nodes, horizon]")
    return {
        f"horizon_{index + 1}": forecasting_metrics(y_true[:, :, index], y_pred[:, :, index])
        for index in range(y_true.shape[2])
    }


def metrics_by_node_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    node_type_ids: np.ndarray,
    type_names: Dict[int, str] | None = None,
) -> Dict[str, Dict[str, float]]:
    if y_true.ndim != 3 or len(node_type_ids) != y_true.shape[1]:
        raise ValueError("Node type IDs must align with the node dimension")
    names = type_names or {0: "region", 1: "city", 2: "store", 3: "family"}
    result = {}
    for type_id in sorted(set(int(value) for value in node_type_ids)):
        mask = np.asarray(node_type_ids) == type_id
        result[names.get(type_id, str(type_id))] = forecasting_metrics(y_true[:, mask, :], y_pred[:, mask, :])
    return result
