"""Shared-actor PPO utilities for exploitation-only Stage 6 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch.distributions import Categorical

from code.belief.beta import BetaBelief
from code.env.dispatch import _survival
from code.policy.model import ExploitationActorCritic


@dataclass
class Transition:
    """One policy decision and its realized survival reward."""

    state: dict[str, torch.Tensor]
    action: int
    old_log_probability: float
    old_value: float
    reward: float = 0.0
    terminal: bool = False


def generalized_advantage_estimate(
    rewards: np.ndarray,
    values: np.ndarray,
    terminals: np.ndarray,
    discount: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns for one rollout sequence."""
    advantages = np.zeros_like(rewards, dtype=float)
    next_advantage = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 1.0 - float(terminals[index])
        delta = rewards[index] + discount * next_value * continuation - values[index]
        next_advantage = delta + discount * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values


class PPOController:
    """Select one feasible vehicle and retain on-policy transitions for PPO."""

    def __init__(
        self,
        model: ExploitationActorCritic,
        edge_lookup: dict[int, int],
        road_classes: list[str],
        training: bool,
        seed: int,
        evaluation_mode: str = "greedy",
    ) -> None:
        self.model = model
        self.edge_lookup = edge_lookup
        self.road_classes = road_classes
        self.training = training
        if evaluation_mode not in {"greedy", "sample"}:
            raise ValueError(f"Unknown evaluation mode {evaluation_mode}")
        self.evaluation_mode = evaluation_mode
        self.generator = torch.Generator().manual_seed(seed)
        self.transitions: list[Transition] = []
        self.decision_entropies: list[float] = []
        self.decision_normalized_entropies: list[float] = []
        self.myopic_agreements: list[bool] = []
        self.decision_records: list[dict[str, Any]] = []
        self._pending: Transition | None = None
        self._static_graph: dict[str, Any] | None = None

    def _state(
        self,
        graph: nx.Graph,
        belief: BetaBelief,
        patient: dict[str, Any],
        options: list[tuple[Any, ...]],
        operational_minutes: float,
        survival_parameters: dict[str, float],
    ) -> dict[str, torch.Tensor]:
        if self._static_graph is None:
            nodes = list(graph.nodes())
            node_position = {node: index for index, node in enumerate(nodes)}
            xs = np.array([graph.nodes[node]["x"] for node in nodes], dtype=float)
            ys = np.array([graph.nodes[node]["y"] for node in nodes], dtype=float)
            x_scale = max(float(np.ptp(xs)), 1.0e-9)
            y_scale = max(float(np.ptp(ys)), 1.0e-9)
            node_features = np.column_stack(
                [
                    (xs - xs.min()) / x_scale,
                    (ys - ys.min()) / y_scale,
                    [float(graph.nodes[node].get("hospital", False)) for node in nodes],
                ]
            )
            edges = list(graph.edges(data=True))
            edge_index = np.array(
                [
                    [node_position[u] for u, _, _ in edges],
                    [node_position[v] for _, v, _ in edges],
                ],
                dtype=np.int64,
            )
            class_position = {name: index for index, name in enumerate(self.road_classes)}
            static_edge_features = []
            edge_belief_indices = []
            for _, _, attributes in edges:
                edge_id = int(attributes["edge_id"])
                edge_belief_indices.append(self.edge_lookup[edge_id])
                one_hot = np.zeros(len(self.road_classes), dtype=float)
                one_hot[class_position[attributes["fclass"]]] = 1.0
                static_edge_features.append(
                    [float(attributes["length_km"]) / 10.0, *one_hot]
                )
            self._static_graph = {
                "node_position": node_position,
                "node_features": torch.tensor(node_features, dtype=torch.float32),
                "edge_index": torch.tensor(edge_index, dtype=torch.long),
                "static_edge_features": np.asarray(static_edge_features),
                "edge_belief_indices": np.asarray(edge_belief_indices, dtype=int),
            }
        static = self._static_graph
        node_position = static["node_position"]
        belief_indices = static["edge_belief_indices"]
        total = belief.a + belief.b
        belief_std = np.sqrt(belief.a * belief.b / (total * total * (total + 1.0)))
        edge_features = np.column_stack(
            [
                belief.mean[belief_indices],
                belief_std[belief_indices],
                static["static_edge_features"],
            ]
        )
        vehicle_features = []
        vehicle_nodes = []
        travel_bias = []
        onset = float(patient["onset_minute"])
        for arrival, vehicle, inbound, outbound, _ in options:
            kind = str(vehicle["kind"])
            vehicle_features.append(
                [
                    max(float(vehicle["available"]) - onset, 0.0) / operational_minutes,
                    float(inbound[1]) / operational_minutes,
                    float(outbound[1]) / operational_minutes,
                    float(kind == "A"),
                    float(kind == "B"),
                    float(kind == "C"),
                ]
            )
            vehicle_nodes.append(node_position[int(vehicle["node"])])
            travel_bias.append(-(float(arrival) - onset) / operational_minutes)
        triage = int(patient["triage"])
        zero_survival = _survival(0.0, survival_parameters)
        current_delay = max(min(float(option[0]) for option in options) - onset, 0.0)
        if current_delay == 0.0 and survival_parameters["c"] > 1.0:
            survival_derivative = 0.0
        else:
            survival_derivative = (
                survival_parameters["a"]
                * np.exp(
                    survival_parameters["b"]
                    * current_delay ** survival_parameters["c"]
                )
                * survival_parameters["b"]
                * survival_parameters["c"]
                * current_delay ** (survival_parameters["c"] - 1.0)
            )
        patient_features = [[
            onset / operational_minutes,
            float(triage == 1),
            float(triage == 2),
            float(triage == 3),
            zero_survival,
            survival_derivative,
        ]]
        return {
            "node_features": static["node_features"],
            "edge_index": static["edge_index"],
            "edge_features": torch.tensor(np.asarray(edge_features), dtype=torch.float32),
            "vehicle_features": torch.tensor(vehicle_features, dtype=torch.float32),
            "vehicle_nodes": torch.tensor(vehicle_nodes, dtype=torch.long),
            "patient_features": torch.tensor(patient_features, dtype=torch.float32),
            "patient_node": torch.tensor([node_position[int(patient["node"])]], dtype=torch.long),
            "travel_bias": torch.tensor(travel_bias, dtype=torch.float32),
        }

    def select(self, **context: Any) -> int:
        """Sample during training and take the highest score during evaluation."""
        state = self._state(**context)
        with torch.no_grad():
            logits, value = self.model(**state)
            distribution = Categorical(logits=logits)
            if self.training or self.evaluation_mode == "sample":
                action = distribution.sample()
            else:
                action = torch.argmax(logits)
            myopic_action = min(
                range(len(context["options"])),
                key=lambda index: context["options"][index][0],
            )
            entropy = float(distribution.entropy().item())
            maximum_entropy = float(np.log(len(context["options"])))
            self.decision_entropies.append(entropy)
            self.decision_normalized_entropies.append(
                entropy / maximum_entropy if maximum_entropy > 0.0 else 0.0
            )
            self.myopic_agreements.append(int(action.item()) == myopic_action)
            probabilities = torch.softmax(logits, dim=0)
            top_probabilities = torch.topk(
                probabilities, k=min(2, len(probabilities))
            ).values
            margin = float(top_probabilities[0].item())
            if len(top_probabilities) == 2:
                margin -= float(top_probabilities[1].item())
            self.decision_records.append(
                {
                    "patient_id": str(context["patient"]["patient_id"]),
                    "feasible_action_count": len(context["options"]),
                    "agrees_with_myopic": int(action.item()) == myopic_action,
                    "top1_top2_margin": margin,
                    "selected_probability": float(probabilities[action].item()),
                }
            )
            transition = Transition(
                state=state,
                action=int(action.item()),
                old_log_probability=float(distribution.log_prob(action).item()),
                old_value=float(value.item()),
            )
        self._pending = transition
        return transition.action

    def observe(self, reward: float) -> None:
        """Attach realized survival to the last selected action."""
        if self._pending is None:
            raise RuntimeError("Controller received a reward without a pending action")
        self._pending.reward = reward
        self.decision_records[-1]["reward"] = float(reward)
        self.transitions.append(self._pending)
        self._pending = None

    def finish_episode(self) -> None:
        """Mark the final stored transition as terminal."""
        if self.transitions:
            self.transitions[-1].terminal = True


def ppo_update(
    model: ExploitationActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    discount: float,
    gae_lambda: float,
    clip_ratio: float,
    value_weight: float,
    entropy_weight: float,
    epochs: int,
    minibatch_size: int | None = None,
) -> dict[str, float]:
    """Apply clipped PPO with a centralized value loss."""
    if not transitions:
        return {
            "loss": float("nan"),
            "policy_loss": float("nan"),
            "value_loss": float("nan"),
            "entropy": float("nan"),
            "optimizer_step": False,
        }
    rewards = np.array([item.reward for item in transitions], dtype=float)
    values = np.array([item.old_value for item in transitions], dtype=float)
    terminals = np.array([item.terminal for item in transitions], dtype=bool)
    advantages, returns = generalized_advantage_estimate(
        rewards, values, terminals, discount, gae_lambda
    )
    advantage_std_before_normalization = float(advantages.std())
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    last_metrics = {}
    for _ in range(epochs):
        indices = np.arange(len(transitions))
        if minibatch_size is not None and len(indices) > minibatch_size:
            indices = np.random.choice(indices, size=minibatch_size, replace=False)
        policy_losses = []
        value_losses = []
        entropies = []
        for index in indices:
            transition = transitions[int(index)]
            logits, value = model(**transition.state)
            distribution = Categorical(logits=logits)
            action = torch.tensor(transition.action)
            log_probability = distribution.log_prob(action)
            ratio = torch.exp(log_probability - transition.old_log_probability)
            advantage = torch.tensor(advantages[index], dtype=torch.float32)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            policy_losses.append(-torch.minimum(unclipped, clipped))
            target = torch.tensor(returns[index], dtype=torch.float32)
            value_losses.append((value - target) ** 2)
            entropies.append(distribution.entropy())
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "advantage_std_before_normalization": advantage_std_before_normalization,
            "optimizer_step": True,
        }
    return last_metrics
