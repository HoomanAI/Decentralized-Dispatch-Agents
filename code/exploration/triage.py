"""Fixed-policy evaluation of triage-rationed route exploration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExplorationLedger:
    budgets: tuple[float, float, float]
    costs: dict[int, float] = field(default_factory=lambda: {1: 0.0, 2: 0.0, 3: 0.0})
    detour_distance: dict[str, float] = field(default_factory=lambda: {"A": 0.0, "B": 0.0, "C": 0.0})


class InformationRoutePolicy:
    """Select patient-route pairs by survival plus fleet information value."""

    def __init__(self, ledger: ExplorationLedger, information_weight: float) -> None:
        self.ledger = ledger
        self.information_weight = float(information_weight)
        self.pending = None

    def propose(self, context: dict[str, Any]) -> tuple[int, float]:
        allowed = list(context["allowed_indices"])
        feasible = []
        for index in allowed:
            candidate = context["candidates"][index]
            if candidate.patient is None:
                feasible.append(index)
                continue
            triage = int(candidate.patient["triage"])
            budget = self.ledger.budgets[triage - 1]
            if self.ledger.costs[triage] + candidate.exploration_cost <= budget + 1.0e-12:
                feasible.append(index)
        if not feasible:
            feasible = [index for index in allowed if context["candidates"][index].patient is None]
        patient_groups: dict[str, list[int]] = {}
        noop = []
        for index in feasible:
            candidate = context["candidates"][index]
            if candidate.patient_id is None:
                noop.append(index)
            else:
                patient_groups.setdefault(candidate.patient_id, []).append(index)
        if patient_groups:
            selected_patient = max(
                patient_groups,
                key=lambda patient_id: max(
                    context["candidates"][index].immediate_survival
                    for index in patient_groups[patient_id]
                ),
            )
            route_choices = patient_groups[selected_patient]
        else:
            route_choices = noop
        choice = max(
            route_choices,
            key=lambda index: (
                context["candidates"][index].immediate_survival
                + self.information_weight * context["candidates"][index].information_value,
                -context["candidates"][index].exploration_cost,
                -index,
            ),
        )
        self.pending = (context["candidates"][choice], str(context["vehicle"]["kind"]))
        score = context["candidates"][choice].immediate_survival + self.information_weight * context["candidates"][choice].information_value
        return choice, float(score)

    def observe(self, reward: float, terminal: bool = False) -> None:
        del reward, terminal
        if self.pending is None:
            return
        candidate, vehicle_class = self.pending
        if candidate.patient is not None:
            triage = int(candidate.patient["triage"])
            self.ledger.costs[triage] += float(candidate.exploration_cost)
            self.ledger.detour_distance[vehicle_class] += float(candidate.detour_distance_km)
        self.pending = None

    def set_other_actions(self, summary: np.ndarray) -> None:
        del summary
