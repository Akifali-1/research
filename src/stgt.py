"""Reference STGT baseline from the supplied project implementation.

This is explicitly named a project-reference STGT baseline. STGT is not a
single universally standardized architecture; this implementation follows the
provided reference: temporal Transformer self-attention, node-type embeddings,
and two TransformerConv spatial layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from utils import repeat_edge_index


class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, d_model: int, max_timesteps: int, nhead: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.input_projection = nn.Linear(in_channels, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_timesteps, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_pooling = nn.Linear(max_timesteps, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x) + self.pos_encoder[:, : x.shape[1]]
        x = self.transformer_encoder(x)
        return self.output_pooling(x.transpose(1, 2)).squeeze(-1)


class ReferenceSTGT(nn.Module):
    """Project-reference Spatio-Temporal Graph Transformer baseline.

    Input:  x [batch, nodes, lookback, channels]
    Output: y [batch, nodes, horizon]
    """

    def __init__(
        self,
        lookback: int = 14,
        horizon: int = 1,
        d_model: int = 64,
        temporal_heads: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.3,
        node_type_count: int = 4,
    ) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.temporal_heads = temporal_heads
        self.spatial_heads = spatial_heads
        self.dropout_rate = dropout
        self.temporal_encoder = TemporalAttention(1, d_model, lookback, temporal_heads, dropout)
        self.type_embedding = nn.Embedding(node_type_count, d_model)
        if d_model % spatial_heads != 0:
            raise ValueError("d_model must be divisible by spatial_heads")
        channels = d_model // spatial_heads
        self.conv1 = TransformerConv(d_model, channels, heads=spatial_heads, dropout=dropout)
        self.conv2 = TransformerConv(d_model, channels, heads=spatial_heads, dropout=dropout)
        self.fc = nn.Linear(d_model, horizon)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("STGT expects x with shape [batch, nodes, lookback, channels]")
        batch_size, num_nodes, lookback, _ = x.shape
        if lookback != self.lookback:
            raise ValueError(f"Expected lookback={self.lookback}, received {lookback}")
        if node_types.shape[0] != num_nodes:
            raise ValueError("node_types must contain one value per graph node")

        temporal = self.temporal_encoder(x.reshape(batch_size * num_nodes, lookback, -1))
        types = self.type_embedding(node_types).repeat(batch_size, 1)
        hidden = temporal + types
        batched_edges = repeat_edge_index(edge_index, batch_size, num_nodes)
        hidden = F.elu(self.conv1(hidden, batched_edges))
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        hidden = F.elu(self.conv2(hidden, batched_edges))
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        output = self.fc(hidden)
        return output.reshape(batch_size, num_nodes, self.horizon)


# Backward-compatible descriptive alias for experiment configuration.
STGTModel = ReferenceSTGT
