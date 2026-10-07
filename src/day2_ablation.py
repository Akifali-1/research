"""Validation-only STGT ablation variants for the independent Day 2 campaign."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from stgt import ReferenceSTGT, AdaptiveSTGTFusion, TemporalAttention
from utils import repeat_edge_index


DAY2_VARIANTS = (
    "temporal_only",
    "spatial_only",
    "fixed_fusion",
    "stgt",
    "adaptive_stgt",
)


class TemporalOnlySTGT(nn.Module):
    """STGT temporal path plus node metadata, without graph message passing."""

    def __init__(self, lookback=14, horizon=7, d_model=64, temporal_heads=2,
                 spatial_heads=4, dropout=0.3, node_type_count=4):
        super().__init__()
        if d_model % temporal_heads:
            raise ValueError("d_model must be divisible by temporal_heads")
        self.lookback = lookback
        self.horizon = horizon
        self.temporal_encoder = TemporalAttention(1, d_model, lookback, temporal_heads, dropout)
        self.type_embedding = nn.Embedding(node_type_count, d_model)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x, edge_index, node_types):
        if x.ndim != 4:
            raise ValueError("Temporal-only STGT expects [batch, nodes, lookback, channels]")
        batch_size, num_nodes, lookback, _ = x.shape
        if lookback != self.lookback or node_types.shape[0] != num_nodes:
            raise ValueError("Temporal-only input shape or node metadata does not match configuration")
        temporal = self.temporal_encoder(x.reshape(batch_size * num_nodes, lookback, -1))
        hidden = temporal + self.type_embedding(node_types).repeat(batch_size, 1)
        return self.fc(hidden).reshape(batch_size, num_nodes, self.horizon)


class SpatialOnlySTGT(nn.Module):
    """Graph path driven by the current observed value, without temporal attention."""

    def __init__(self, lookback=14, horizon=7, d_model=64, temporal_heads=2,
                 spatial_heads=4, dropout=0.3, node_type_count=4):
        super().__init__()
        if d_model % spatial_heads:
            raise ValueError("d_model must be divisible by spatial_heads")
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.dropout_rate = dropout
        self.input_projection = nn.Linear(1, d_model)
        self.type_embedding = nn.Embedding(node_type_count, d_model)
        channels = d_model // spatial_heads
        self.conv1 = TransformerConv(d_model, channels, heads=spatial_heads, dropout=dropout)
        self.conv2 = TransformerConv(d_model, channels, heads=spatial_heads, dropout=dropout)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x, edge_index, node_types):
        if x.ndim != 4:
            raise ValueError("Spatial-only STGT expects [batch, nodes, lookback, channels]")
        batch_size, num_nodes, lookback, _ = x.shape
        if lookback != self.lookback or node_types.shape[0] != num_nodes:
            raise ValueError("Spatial-only input shape or node metadata does not match configuration")
        # Use only the latest observed value; no temporal encoder or sequence pooling.
        hidden = self.input_projection(x[:, :, -1, :]).reshape(batch_size * num_nodes, self.d_model)
        hidden = hidden + self.type_embedding(node_types).repeat(batch_size, 1)
        batched_edges = repeat_edge_index(edge_index, batch_size, num_nodes)
        hidden = F.elu(self.conv1(hidden, batched_edges))
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        hidden = F.elu(self.conv2(hidden, batched_edges))
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        return self.fc(hidden).reshape(batch_size, num_nodes, self.horizon)


class FixedSTGTFusion(nn.Module):
    def __init__(self, d_model, alpha=0.5):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.spatial_norm = nn.LayerNorm(d_model)
        self.alpha = float(alpha)

    def forward(self, temporal, spatial):
        if temporal.shape != spatial.shape:
            raise ValueError("Fixed-fusion branches must have identical shapes")
        temporal = self.temporal_norm(temporal)
        spatial = self.spatial_norm(spatial)
        return self.alpha * temporal + (1.0 - self.alpha) * spatial


class FixedFusionSTGT(ReferenceSTGT):
    """Reference STGT branches with a fixed 0.5/0.5 normalized fusion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion = FixedSTGTFusion(self.d_model, alpha=0.5)

    def forward(self, x, edge_index, node_types):
        if x.ndim != 4:
            raise ValueError("Fixed-fusion STGT expects [batch, nodes, lookback, channels]")
        batch_size, num_nodes, lookback, _ = x.shape
        if lookback != self.lookback or node_types.shape[0] != num_nodes:
            raise ValueError("Fixed-fusion input shape or node metadata does not match configuration")
        temporal = self.temporal_encoder(x.reshape(batch_size * num_nodes, lookback, -1))
        temporal = temporal + self.type_embedding(node_types).repeat(batch_size, 1)
        batched_edges = repeat_edge_index(edge_index, batch_size, num_nodes)
        spatial = F.elu(self.conv1(temporal, batched_edges))
        spatial = F.dropout(spatial, p=self.dropout_rate, training=self.training)
        spatial = F.elu(self.conv2(spatial, batched_edges))
        spatial = F.dropout(spatial, p=self.dropout_rate, training=self.training)
        fused = self.fusion(temporal, spatial)
        return self.fc(fused).reshape(batch_size, num_nodes, self.horizon)
