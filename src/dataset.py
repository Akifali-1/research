"""Leakage-safe chronological graph-window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class GraphWindow:
    x: torch.Tensor
    y: torch.Tensor
    target_index: int
    baseline_y: torch.Tensor | None = None


class GraphWindowDataset(Dataset):
    """Dense windows over one fixed graph.

    A sample contains all nodes for one target start time. This avoids treating
    nodes from unrelated graphs as independent examples while retaining simple
    batching over time windows.
    """

    def __init__(
        self,
        series: np.ndarray,
        target_range: Tuple[int, int],
        lookback: int,
        horizon: int,
        baseline: np.ndarray | None = None,
    ) -> None:
        if series.ndim != 2:
            raise ValueError("series must have shape [num_nodes, num_time_steps]")
        self.series = series.astype(np.float32, copy=False)
        self.start, self.end = target_range
        self.lookback = lookback
        self.horizon = horizon
        self.baseline = baseline.astype(np.float32, copy=False) if baseline is not None else None
        if self.start < lookback or self.end <= self.start:
            raise ValueError("Invalid target range for lookback")
        # ``end`` is exclusive; the final target starts at end - 1.
        if self.end + horizon - 2 >= self.series.shape[1]:
            raise ValueError("Target range exceeds available time steps")
        if self.baseline is not None and self.baseline.shape != self.series.shape:
            raise ValueError("baseline must have the same shape as series")

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, index: int) -> GraphWindow:
        target = self.start + index
        x = self.series[:, target - self.lookback : target]
        y = self.series[:, target : target + self.horizon]
        baseline_y = None
        if self.baseline is not None:
            x = x - self.baseline[:, target - self.lookback : target]
            baseline_y = torch.from_numpy(self.baseline[:, target : target + self.horizon])
            y = y - self.baseline[:, target : target + self.horizon]
        x_tensor = torch.from_numpy(x.copy()).unsqueeze(-1)  # [N, L, 1]
        y_tensor = torch.from_numpy(y.copy())  # [N, H]
        return GraphWindow(x_tensor, y_tensor, target, baseline_y)


def collate_graph_windows(samples: list[GraphWindow]) -> Dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    if any(sample.baseline_y is not None for sample in samples) and not all(
        sample.baseline_y is not None for sample in samples
    ):
        raise ValueError("Either every sample or no sample may use a baseline")
    result: Dict[str, torch.Tensor] = {
        "x": torch.stack([sample.x for sample in samples]),
        "y": torch.stack([sample.y for sample in samples]),
        "target_index": torch.tensor([sample.target_index for sample in samples], dtype=torch.long),
    }
    if samples[0].baseline_y is not None:
        result["baseline_y"] = torch.stack([sample.baseline_y for sample in samples])
    return result


def make_loader(dataset: GraphWindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_graph_windows,
        pin_memory=torch.cuda.is_available(),
    )
