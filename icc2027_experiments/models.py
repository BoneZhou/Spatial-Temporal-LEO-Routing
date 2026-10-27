from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS: Dict[str, Dict[str, bool]] = {
    "dqn": {"use_gat": False, "use_lstm": False},
    "lstm_dqn": {"use_gat": False, "use_lstm": True},
    "gat_dqn": {"use_gat": True, "use_lstm": False},
    "gat_lstm_dqn": {"use_gat": True, "use_lstm": True},
}


class DenseGraphAttention(nn.Module):
    """Multi-head dense GAT layer implemented with stock PyTorch."""

    def __init__(self, input_dim: int, output_dim: int, heads: int = 4):
        super().__init__()
        if output_dim % heads != 0:
            raise ValueError("output_dim must be divisible by the number of attention heads")
        self.heads = heads
        self.head_dim = output_dim // heads
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.source_score = nn.Parameter(torch.empty(heads, self.head_dim))
        self.target_score = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.source_score)
        nn.init.xavier_uniform_(self.target_score)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = node_features.shape
        projected = self.projection(node_features).view(batch, nodes, self.heads, self.head_dim)
        source = (projected * self.source_score).sum(dim=-1).permute(0, 2, 1)
        target = (projected * self.target_score).sum(dim=-1).permute(0, 2, 1)
        scores = source.unsqueeze(-1) + target.unsqueeze(-2)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0).expand(node_features.size(0), -1, -1)
        scores = scores.masked_fill(~adjacency.unsqueeze(1), torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        values = projected.permute(0, 2, 1, 3)
        attended = torch.matmul(attention, values)
        return attended.permute(0, 2, 1, 3).contiguous().view(batch, nodes, self.output_dim)


class UnifiedRoutingQNetwork(nn.Module):
    """Common Q-network whose GAT and LSTM switches define the ablations."""

    def __init__(
        self,
        variant: str,
        feature_dim: int,
        hidden_dim: int,
        lstm_hidden_dim: int,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        switches = VARIANTS[variant]
        self.variant = variant
        self.use_gat = switches["use_gat"]
        self.use_lstm = switches["use_lstm"]

        self.encoder = nn.Linear(feature_dim, hidden_dim)
        self.gat = DenseGraphAttention(hidden_dim, hidden_dim, heads=4) if self.use_gat else None
        representation_dim = hidden_dim
        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            representation_dim = lstm_hidden_dim
        else:
            self.lstm = None
        self.q_head = nn.Sequential(
            nn.Linear(representation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_sequence: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # state_sequence: [batch, time, nodes, features]
        batch, time_steps, nodes, _ = state_sequence.shape
        if self.use_lstm:
            encoded = F.relu(self.encoder(state_sequence))
            if self.use_gat:
                flattened = encoded.view(batch * time_steps, nodes, -1)
                if adjacency.dim() == 3:
                    graph_mask = adjacency.repeat_interleave(time_steps, dim=0)
                else:
                    graph_mask = adjacency
                encoded = F.elu(self.gat(flattened, graph_mask)).view(
                    batch, time_steps, nodes, -1
                )
            sequence_by_node = encoded.permute(0, 2, 1, 3).contiguous()
            sequence_by_node = sequence_by_node.view(batch * nodes, time_steps, -1)
            temporal, _ = self.lstm(sequence_by_node)
            representation = temporal[:, -1].view(batch, nodes, -1)
        else:
            representation = F.relu(self.encoder(state_sequence[:, -1]))
            if self.use_gat:
                representation = F.elu(self.gat(representation, adjacency))
        return self.q_head(representation).squeeze(-1)

