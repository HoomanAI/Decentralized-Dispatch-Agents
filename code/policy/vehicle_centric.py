"""Shared vehicle-centric actor and action-value critic."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from code.env.vehicle_centric import Candidate


class VehicleActorQCritic(nn.Module):
    """Score patient actions and estimate an action value for every feasible action."""

    def __init__(self, edge_count: int, hidden: int = 32, max_vehicles: int = 16) -> None:
        super().__init__()
        self.vehicle_embedding = nn.Embedding(max_vehicles, 4)
        self.belief_encoder = nn.Sequential(nn.Linear(2 * edge_count, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.pair_encoder = nn.Sequential(nn.Linear(hidden + 4 + 4 + 7, hidden), nn.ReLU())
        self.actor_head = nn.Linear(hidden, 1)
        self.q_head = nn.Linear(hidden + 3, 1)

    def encode(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        belief = self.belief_encoder(torch.cat([state["belief_mean"], state["belief_std"]]))
        count = state["candidate_features"].shape[0]
        vehicle = torch.cat([self.vehicle_embedding(state["vehicle_index"]), state["vehicle_features"]])
        common = torch.cat([belief, vehicle]).unsqueeze(0).repeat(count, 1)
        return self.pair_encoder(torch.cat([common, state["candidate_features"]], dim=1))

    def forward(self, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.encode(state)
        logits = self.actor_head(pair).squeeze(-1) + state["arrival_bias"]
        other = state.get("other_action_summary", torch.zeros(3)).unsqueeze(0).repeat(len(pair), 1)
        q_values = self.q_head(torch.cat([pair, other], dim=1)).squeeze(-1)
        return logits, q_values


class ResidualVehiclePolicy(nn.Module):
    """Frozen cloned logits plus a zero-initialized learned correction."""

    def __init__(self, clone: VehicleActorQCritic) -> None:
        super().__init__()
        self.clone = clone
        for parameter in self.clone.parameters():
            parameter.requires_grad = False
        hidden = clone.actor_head.in_features
        self.residual_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.q_head = nn.Linear(hidden + 3, 1)

    def forward(self, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pair = self.clone.encode(state)
            base = self.clone.actor_head(pair).squeeze(-1) + state["arrival_bias"]
        logits = base + self.residual_head(pair).squeeze(-1)
        other = state.get("other_action_summary", torch.zeros(3)).unsqueeze(0).repeat(len(pair), 1)
        q_values = self.q_head(torch.cat([pair, other], dim=1)).squeeze(-1)
        return logits, q_values


class LookaheadResidualPolicy(nn.Module):
    """Greedy-logit warm start plus shared lookahead trunk and triage-specific heads."""

    def __init__(
        self,
        hidden: int = 32,
        max_vehicles: int = 16,
        lookahead_dim: int = 16,
        base_logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.vehicle_embedding = nn.Embedding(max_vehicles, 4)
        self.shared_trunk = nn.Sequential(
            nn.Linear(4 + 4 + 7 + lookahead_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.triage_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])
        self.q_head = nn.Linear(hidden + 3, 1)
        self.register_buffer("base_logit_scale", torch.tensor(float(base_logit_scale)))
        for head in self.triage_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def encode(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        count = state["candidate_features"].shape[0]
        vehicle = torch.cat(
            [self.vehicle_embedding(state["vehicle_index"]), state["vehicle_features"]]
        )
        common = torch.cat([vehicle, state["lookahead_features"]]).unsqueeze(0).repeat(count, 1)
        return self.shared_trunk(torch.cat([common, state["candidate_features"]], dim=1))

    def forward(self, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(state)
        candidate = state["candidate_features"]
        triage = torch.argmax(candidate[:, :3], dim=1)
        is_noop = candidate[:, 6] > 0.5
        residual = torch.zeros(len(encoded), dtype=encoded.dtype, device=encoded.device)
        for index, head in enumerate(self.triage_heads):
            mask = (triage == index) & ~is_noop
            if bool(mask.any()):
                residual[mask] = head(encoded[mask]).squeeze(-1)
        logits = self.base_logit_scale * state["arrival_bias"] + residual
        other = state.get("other_action_summary", torch.zeros(3)).unsqueeze(0).repeat(len(encoded), 1)
        q_values = self.q_head(torch.cat([encoded, other], dim=1)).squeeze(-1)
        return logits, q_values


class RouteActorQCritic(nn.Module):
    """Per-agent route selector with a no-detour residual warm start."""

    def __init__(
        self,
        hidden: int = 32,
        max_vehicles: int = 12,
        route_feature_dim: int = 11,
        alternative_logit_penalty: float = 4.0,
    ) -> None:
        super().__init__()
        self.vehicle_embedding = nn.Embedding(max_vehicles, 4)
        self.route_encoder = nn.Sequential(
            nn.Linear(4 + route_feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, 1)
        self.q_head = nn.Linear(hidden, 1)
        self.register_buffer(
            "alternative_logit_penalty",
            torch.tensor(float(alternative_logit_penalty)),
        )
        nn.init.zeros_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)

    def forward(self, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        count = state["route_features"].shape[0]
        vehicle = self.vehicle_embedding(state["vehicle_index"]).unsqueeze(0).repeat(count, 1)
        encoded = self.route_encoder(torch.cat([vehicle, state["route_features"]], dim=1))
        base = -self.alternative_logit_penalty * (state["route_rank"] > 0).float()
        logits = base + self.actor_head(encoded).squeeze(-1)
        return logits, self.q_head(encoded).squeeze(-1)


def route_state(
    context: dict[str, Any],
    route_indices: list[int],
    edge_lookup: dict[int, int],
) -> dict[str, torch.Tensor]:
    """Encode only current beliefs, route costs, triage, and isolation history."""
    belief = context["belief"]
    total = belief.a + belief.b
    width = np.sqrt(
        belief.a * belief.b / (total**2 * (total + 1.0))
    )
    features: list[list[float]] = []
    ranks: list[int] = []
    for index in route_indices:
        candidate = context["candidates"][index]
        if candidate.patient is None or candidate.inbound is None:
            raise ValueError("route state requires a patient route")
        triage = int(candidate.patient["triage"])
        edge_indices = [edge_lookup[int(edge_id)] for edge_id in candidate.inbound[0]]
        route_width = float(np.mean(width[edge_indices])) if edge_indices else 0.0
        features.append(
            [
                route_width,
                float(candidate.information_value),
                float(candidate.exploration_cost),
                float(triage == 1),
                float(triage == 2),
                float(triage == 3),
                min(float(context.get("isolation_epochs", 0)) / 100.0, 1.0),
                float(candidate.inbound[1]) / 720.0,
                float(candidate.detour_distance_km) / 20.0,
                float(candidate.route_rank) / 3.0,
                float(candidate.immediate_survival) / 3.0,
            ]
        )
        ranks.append(int(candidate.route_rank))
    return {
        "vehicle_index": torch.tensor(int(context["vehicle_index"]), dtype=torch.long),
        "route_features": torch.tensor(features, dtype=torch.float32),
        "route_rank": torch.tensor(ranks, dtype=torch.long),
    }


class NeuralRouteController:
    """Keep protocol-myopic patient choice fixed and learn only route choice."""

    def __init__(
        self,
        model: RouteActorQCritic,
        edge_lookup: dict[int, int],
        greedy: bool = False,
    ) -> None:
        self.model = model
        self.edge_lookup = edge_lookup
        self.greedy = greedy
        self.pending: tuple[
            dict[str, torch.Tensor], int, torch.Tensor, int, float
        ] | None = None
        self.transitions: list[dict[str, Any]] = []
        self.normalized_entropies: list[float] = []
        self.route_set_sizes: list[int] = []
        self.detour_choices: list[bool] = []

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        allowed = list(context["allowed_indices"])
        patient_groups: dict[str, list[int]] = {}
        noop: list[int] = []
        for index in allowed:
            candidate = context["candidates"][index]
            if candidate.patient_id is None:
                noop.append(index)
            else:
                patient_groups.setdefault(candidate.patient_id, []).append(index)
        if not patient_groups:
            self.pending = None
            choice = min(noop)
            return choice, 0.0
        selected_patient = max(
            patient_groups,
            key=lambda patient_id: max(
                context["candidates"][index].immediate_survival
                for index in patient_groups[patient_id]
            ),
        )
        route_indices = patient_groups[selected_patient]
        state = route_state(context, route_indices, self.edge_lookup)
        logits, q_values = self.model(state)
        distribution = Categorical(logits=logits)
        local_action = torch.argmax(logits) if self.greedy else distribution.sample()
        chosen_index = route_indices[int(local_action.item())]
        candidate = context["candidates"][chosen_index]
        self.route_set_sizes.append(len(route_indices))
        entropy = float(distribution.entropy().item())
        if len(route_indices) > 1:
            self.normalized_entropies.append(entropy / float(np.log(len(route_indices))))
        self.detour_choices.append(bool(candidate.route_rank > 0))
        self.pending = (
            state,
            int(local_action.item()),
            distribution.log_prob(local_action).detach(),
            int(candidate.patient["triage"]),
            float(candidate.exploration_cost),
        )
        return chosen_index, float(candidate.immediate_survival)

    def observe(self, reward: float, terminal: bool = False) -> None:
        if self.pending is None:
            return
        state, action, old_logp, triage, cost = self.pending
        self.transitions.append(
            {
                "state": state,
                "action": action,
                "old_logp": old_logp,
                "reward": float(reward),
                "terminal": terminal,
                "triage": triage,
                "exploration_cost": cost,
            }
        )
        self.pending = None

    def set_other_actions(self, summary: np.ndarray) -> None:
        del summary


def vehicle_state(context: dict[str, Any], edge_count: int) -> dict[str, torch.Tensor]:
    belief = context["belief"]
    candidates: list[Candidate] = context["candidates"]
    now = float(context.get("now", 0.0))
    features = []
    arrival_bias = []
    for candidate in candidates:
        if candidate.patient is None:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            arrival_bias.append(0.0)
        else:
            triage = int(candidate.patient["triage"])
            elapsed = max(now - float(candidate.patient["onset_minute"]), 0.0) / 720.0
            features.append([float(triage == 1), float(triage == 2), float(triage == 3), elapsed, candidate.immediate_survival / 3.0, candidate.estimated_arrival / 720.0, 0.0])
            arrival_bias.append(candidate.immediate_survival)
    kind = str(context["vehicle"]["kind"])
    total = belief.a + belief.b
    return {
        "belief_mean": torch.tensor(belief.mean[:edge_count], dtype=torch.float32),
        "belief_std": torch.tensor(np.sqrt(belief.a[:edge_count] * belief.b[:edge_count] / (total[:edge_count] ** 2 * (total[:edge_count] + 1.0))), dtype=torch.float32),
        "vehicle_index": torch.tensor(int(context["vehicle_index"]), dtype=torch.long),
        "vehicle_features": torch.tensor([float(kind == "A"), float(kind == "B"), float(kind == "C"), float(context["vehicle"]["available"]) / 720.0], dtype=torch.float32),
        "candidate_features": torch.tensor(features, dtype=torch.float32),
        "arrival_bias": torch.tensor(arrival_bias, dtype=torch.float32),
        "lookahead_features": torch.tensor(
            context.get("lookahead_features", np.zeros(16, dtype=float)), dtype=torch.float32
        ),
    }


class NeuralVehicleController:
    def __init__(self, model: nn.Module, edge_count: int, greedy: bool = False, reference_model: nn.Module | None = None, temperature: float = 1.0) -> None:
        self.model, self.edge_count, self.greedy = model, edge_count, greedy
        self.reference_model = reference_model
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.pending: tuple[dict[str, torch.Tensor], int, torch.Tensor, torch.Tensor] | None = None
        self.transitions: list[dict[str, Any]] = []
        self.entropies: list[float] = []
        self.normalized_entropies: list[float] = []
        self.nontrivial_normalized_entropies: list[float] = []
        self.allowed_action_counts: list[int] = []
        self.logit_margins: list[float] = []
        self.myopic_agreements: list[bool] = []
        self.reference_kls: list[float] = []
        self.reference_argmax_agreements: list[bool] = []

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        state = vehicle_state(context, self.edge_count)
        allowed = context["allowed_indices"]
        logits, q = self.model(state)
        masked = torch.full_like(logits, -torch.inf); masked[allowed] = logits[allowed]
        distribution = Categorical(logits=masked / self.temperature)
        action = torch.argmax(masked) if self.greedy else distribution.sample()
        state["allowed_mask"] = torch.tensor(
            [index in allowed for index in range(len(logits))], dtype=torch.bool
        )
        state["behavior_temperature"] = torch.tensor(self.temperature, dtype=torch.float32)
        self.pending = (state, int(action.item()), distribution.log_prob(action).detach(), q.detach())
        entropy = float(distribution.entropy().item())
        self.entropies.append(entropy)
        self.allowed_action_counts.append(len(allowed))
        self.normalized_entropies.append(
            entropy / float(np.log(len(allowed))) if len(allowed) > 1 else 0.0
        )
        if len(allowed) > 1:
            self.nontrivial_normalized_entropies.append(
                entropy / float(np.log(len(allowed)))
            )
            ordered_logits = torch.sort(masked[allowed], descending=True).values
            self.logit_margins.append(float((ordered_logits[0] - ordered_logits[1]).item()))
        myopic = max(
            allowed,
            key=lambda index: context["candidates"][index].immediate_survival,
        )
        self.myopic_agreements.append(int(action.item()) == myopic)
        if self.reference_model is not None:
            with torch.no_grad():
                reference_logits, _ = self.reference_model(state)
                reference_masked = torch.full_like(reference_logits, -torch.inf)
                reference_masked[allowed] = reference_logits[allowed]
                reference_distribution = Categorical(logits=reference_masked)
                self.reference_kls.append(
                    float(torch.distributions.kl_divergence(distribution, reference_distribution).item())
                )
                self.reference_argmax_agreements.append(
                    int(torch.argmax(masked).item()) == int(torch.argmax(reference_masked).item())
                )
        return int(action.item()), float(distribution.probs[action].item())

    def observe(self, reward: float, terminal: bool = False) -> None:
        if self.pending is not None:
            state, action, logp, q = self.pending
            self.transitions.append({"state": state, "action": action, "old_logp": logp, "old_q": q, "reward": float(reward), "terminal": terminal})
            self.pending = None

    def set_other_actions(self, summary: np.ndarray) -> None:
        """Attach earlier commitments and later proposals to the pending critic state."""
        if self.pending is not None:
            state, action, logp, q = self.pending
            state["other_action_summary"] = torch.tensor(summary, dtype=torch.float32)
            self.pending = (state, action, logp, q)
