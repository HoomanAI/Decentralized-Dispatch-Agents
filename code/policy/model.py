"""Graph-aware shared actor and centralized critic for Stage 6."""

from __future__ import annotations

import math

import torch
from torch import nn


class BeliefGraphEncoder(nn.Module):
    """Encode nodes using belief mean, uncertainty, length, and road class edges."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_input = nn.Linear(node_dim, hidden_dim)
        self.edge_input = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        node_state = torch.relu(self.node_input(node_features))
        edge_state = self.edge_input(edge_features)
        source, target = edge_index
        messages = node_state[source] + edge_state
        aggregate = torch.zeros_like(node_state)
        aggregate.index_add_(0, target, messages)
        aggregate.index_add_(0, source, node_state[target] + edge_state)
        degree = torch.zeros(node_state.shape[0], device=node_state.device)
        degree.index_add_(0, source, torch.ones_like(source, dtype=node_state.dtype))
        degree.index_add_(0, target, torch.ones_like(target, dtype=node_state.dtype))
        aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
        return self.update(torch.cat([node_state, aggregate], dim=-1))


class ExploitationActorCritic(nn.Module):
    """Score compatible vehicles for one pending patient with shared parameters."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        vehicle_dim: int,
        patient_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.graph_encoder = BeliefGraphEncoder(node_dim, edge_dim, hidden_dim)
        self.vehicle_encoder = nn.Sequential(
            nn.Linear(vehicle_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.patient_encoder = nn.Sequential(
            nn.Linear(patient_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        vehicle_features: torch.Tensor,
        vehicle_nodes: torch.Tensor,
        patient_features: torch.Tensor,
        patient_node: torch.Tensor,
        travel_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_state = self.graph_encoder(node_features, edge_index, edge_features)
        vehicle_state = self.vehicle_encoder(
            torch.cat([vehicle_features, node_state[vehicle_nodes]], dim=-1)
        )
        patient_state = self.patient_encoder(
            torch.cat([patient_features, node_state[patient_node]], dim=-1)
        )
        logits = (
            vehicle_state @ patient_state.transpose(0, 1)
        ).squeeze(-1) / math.sqrt(self.hidden_dim)
        logits = logits + travel_bias
        value = self.critic(
            torch.cat(
                [
                    node_state.mean(dim=0),
                    vehicle_state.mean(dim=0),
                    patient_state.mean(dim=0),
                ]
            )
        ).squeeze(-1)
        return logits, value
