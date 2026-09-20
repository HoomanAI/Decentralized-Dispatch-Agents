from __future__ import annotations

import numpy as np

from code.belief.beta import BetaBelief


def test_traversal_changes_belief() -> None:
    belief = BetaBelief.from_reliability(
        np.array([0.5, 0.7]), prior_strength=8.0, discount=0.95, pooling=0.0
    )
    before = belief.mean.copy()
    belief.update(0, True)
    assert not np.array_equal(before, belief.mean)
    assert belief.mean[0] > before[0]
