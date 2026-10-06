"""Adaptive Fusion GAT-LSTM proposed model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from utils import repeat_edge_index


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(sequence), dim=1)
        return torch.sum(sequence * weights, dim=1)


class AdaptiveAttentionFusion(nn.Module):
    """Learn a per-node gate between temporal and spatial representations."""

    def __init__(self, temporal_dim: int, spatial_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.temporal_projection = nn.Linear(temporal_dim, hidden_dim)
        self.spatial_projection = nn.Linear(spatial_dim, hidden_dim)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(1, hidden_dim // 2), 1),
            nn.Sigmoid(),
        )

    def forward(self, temporal: torch.Tensor, spatial: torch.Tensor):
        temporal = F.elu(self.temporal_norm(self.temporal_projection(temporal)))
        spatial = F.elu(self.spatial_norm(self.spatial_projection(spatial)))
        temporal_weight = self.gate(torch.cat([temporal, spatial], dim=-1)).squeeze(-1)
        spatial_weight = 1.0 - temporal_weight
        fused = temporal_weight.unsqueeze(-1) * temporal + spatial_weight.unsqueeze(-1) * spatial
        return fused, temporal_weight, spatial_weight


class MultiScaleGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        if layers < 1:
            raise ValueError("GAT layers must be at least one")
        convolutions = []
        norms = []
        current = input_dim
        for layer_index in range(layers):
            final_layer = layer_index == layers - 1
            convolution = GATConv(
                current,
                hidden_dim,
                heads=1 if final_layer else heads,
                concat=not final_layer,
                dropout=dropout,
            )
            convolutions.append(convolution)
            output_dim = hidden_dim if final_layer else hidden_dim * heads
            if not final_layer:
                norms.append(nn.BatchNorm1d(output_dim))
            current = output_dim
        self.convolutions = nn.ModuleList(convolutions)
        self.norms = nn.ModuleList(norms)
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.dropout = dropout

    def forward(self, hidden: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        original = hidden
        norm_index = 0
        for layer_index, convolution in enumerate(self.convolutions):
            hidden = convolution(hidden, edge_index)
            if layer_index < len(self.convolutions) - 1:
                hidden = self.norms[norm_index](hidden)
                norm_index += 1
                hidden = F.elu(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden + self.residual(original)


class AdaptiveFusionGATLSTM(nn.Module):
    """Adaptive Fusion GAT-LSTM with configurable multi-step output.

    Input:  x [batch, nodes, lookback, channels]
    Output: y [batch, nodes, horizon]
    """

    def __init__(
        self,
        lookback: int = 20,
        horizon: int = 1,
        gat_hidden: int = 48,
        gat_heads: int = 3,
        lstm_hidden: int = 64,
        dropout: float = 0.25,
        gat_layers: int = 2,
        lstm_layers: int = 3,
        input_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.lstm_hidden = lstm_hidden
        self.dropout_rate = dropout
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        temporal_dim = 2 * lstm_hidden
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.temporal_attention = TemporalAttention(temporal_dim)
        self.gat = MultiScaleGAT(temporal_dim, gat_hidden, gat_heads, gat_layers, dropout)
        self.fusion = AdaptiveAttentionFusion(temporal_dim, gat_hidden, gat_hidden, dropout)
        self.prediction_head = nn.Sequential(
            nn.Linear(gat_hidden, gat_hidden * 3),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(gat_hidden * 3),
            nn.Linear(gat_hidden * 3, gat_hidden * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(gat_hidden * 2),
            nn.Linear(gat_hidden * 2, gat_hidden),
            nn.ELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(gat_hidden, horizon),
        )
        self.input_noise = input_noise

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ):
        if x.ndim != 4:
            raise ValueError("GAT-LSTM expects x with shape [batch, nodes, lookback, channels]")
        batch_size, num_nodes, lookback, _ = x.shape
        if lookback != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, received {lookback}")
        if self.training and self.input_noise > 0:
            x = x + torch.randn_like(x) * self.input_noise
        sequence = x.reshape(batch_size * num_nodes, lookback, -1)
        sequence, _ = self.lstm(sequence)
        temporal = self.temporal_attention(sequence)
        temporal = self.temporal_norm(temporal)
        batched_edges = repeat_edge_index(edge_index, batch_size, num_nodes)
        spatial = self.gat(temporal, batched_edges)
        fused, temporal_weight, spatial_weight = self.fusion(temporal, spatial)
        output = self.prediction_head(fused).reshape(batch_size, num_nodes, self.horizon)
        if return_attention:
            return output, temporal_weight.reshape(batch_size, num_nodes), spatial_weight.reshape(batch_size, num_nodes)
        return output


OptimizedGATLSTM = AdaptiveFusionGATLSTM
