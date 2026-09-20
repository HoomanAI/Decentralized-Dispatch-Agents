# CAV Agents and Decentralized dispatch on a wildfire-degraded road network

Simulation code for a study of autonomous emergency medical dispatch when the road network
is being degraded by a wildfire and vehicles cannot always talk to each other.

Each vehicle keeps its own Beta belief about which road arcs are passable, updates it only
from arcs it actually drives, and reconciles with other vehicles when they are in contact.
Because finding out whether a road is open costs a patient time, route-level exploration is
constrained by a per-day survival-risk budget that depends on triage class. The case study
uses the exposure sequence of the January 2025 Palisades Fire.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put the four input files in `data/` first. `data/README.md` says what they are and where
to get them; two are available from the authors on request.

## Running

```bash
python -m code.run_all --stage loaders       # validate and clean the inputs
python -m code.run_all --stage network       # build the routable network
python -m code.run_all --stage environment   # dispatch simulation
python -m code.run_all --stage bound         # hindsight reference
python -m code.run_all --stage policy        # policy training
```

Individual experiments live in `code/eval/` and run the same way, for example:

```bash
python -m code.eval.minor_resilience_ablation      # information-architecture ablation
python -m code.eval.stage11_component_b            # learned route-exploration policy
```

Figures are rebuilt from saved results without rerunning anything:

```bash
python -m code.reporting.stage10_load_bearing
python -m code.reporting.figure_style_upgrade
```

Tests:

```bash
pytest code/tests
```

## Layout

```
code/belief         Beta posterior, evidence discounting, reconciliation
code/env            dispatch simulators, centralized and vehicle-centric
code/exploration    triage-rationed route exploration
code/policy         policy networks and PPO training
code/reliability    damage recursion and the estimation machinery
code/network        synthetic routable network generator
code/baselines      metaheuristic and published comparators
code/eval           experiment scripts, one per reported result
code/reporting      figure and table builders
tables              derived summary tables used by the figures
```

The pipeline creates `results/` and `figures/` on first run. Both are untracked.

## Notes

Results depend on the seeds in `code/config/default.yaml`. The evaluation runs use 30 held-out
environment seeds; policy training uses a single training seed, so the learned results are not
bounded across independent training runs.

The routable network is generated rather than observed. The generator is matched to the
district on summary statistics, so nothing here is a map of real streets.
