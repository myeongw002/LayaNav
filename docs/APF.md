# Artificial Potential Field (APF) classical baseline

LayaNav includes a **discrete Artificial Potential Field** (APF) navigation
policy that operates on exactly the same public six-channel belief observation
as Online A*, CNN-BC, PPO and Laya. The algorithm does **not** receive the
simulator's privileged ground-truth obstacle map.

## Definition

For a robot at grid cell `q=(x,y)`, goal `g`, and the set of currently
known obstacles `O` (original map walls + discovered extra obstacles):

```text
U_att(q) = 0.5 * k_att * ||q - g||²

d(q) = min_{o in O} ||q - o||

U_rep(q) = 0.5 * k_rep * (1/d(q) - 1/rho0)²     when 0 < d(q) < rho0
         = 0                                    otherwise

U(q) = U_att(q) + U_rep(q) + lambda_unknown * I(q is not observed)
```

The final unknown penalty is **optional**, is zero by default, and is a
heuristic—not a calibrated occupancy probability or risk estimate.
All distances are in grid-cell units. Obstacles are excluded as motion
candidates before scoring.

At each step, evaluate the potential of four 4-connected neighbors and
choose the lowest-cost free cell **only if the potential strictly decreases**.
If there is no strictly lower neighboring free cell, output action `STAY`
and label the diagnostic state `local_minimum` or `blocked`. This avoids
unbounded two-cell oscillations, but **does not solve the local minimum
problem**. APF does not invoke A* or attempt a random escape. That choice
keeps the baseline a controlled local reactive method.

The repulsion uses Euclidean distance to the *nearest observed obstacle*.
It does not sum forces from every obstacle, and does not include walls
beyond map boundaries unless those cells appear in the prior wall layer.

## Suggested comparison

```bash
python -m pip install -e ".[dev]"

# Headless A* and APF on the same seed range:
python -m layanav --policy astar --episodes 5 --seed 42
python -m layanav --policy apf --episodes 5 --seed 42

# Paired-seed classical planning benchmark:
python -m layanav.evaluation.cli \
  --methods astar apf \
  --families sparse standard dense \
  --sensor raycast --episodes 100 --eval-seed 200000 \
  --output-dir results/astar_vs_apf
```

No GPU or neural-model checkpoint is required for this comparison.

APF parameters can be changed for controlled sensitivity experiments:

```bash
python -m layanav.evaluation.cli \
  --methods astar apf --families standard \
  --apf-attractive-gain 1.0 \
  --apf-repulsive-gain 8.0 \
  --apf-influence-radius 3.0 \
  --apf-unknown-penalty 0.0 \
  --episodes 100 --output-dir results/apf_params
```

The applied APF configuration is recorded in `summary.json`. Do not tune
parameters on the final test seeds; separate tuning and evaluation map sets
to avoid optimistic generalization claims.

## GUI

Start `python -m layanav --gui` and select **APF** in the policy dropdown.
`Play` and `Step` then use APF instead of A*. A*'s planned-path overlay
is cleared while APF operates because the latter returns only a local
action. The robot's actual trajectory is still shown and recorded as an
episode. When APF returns `STAY` at a local minimum, automatic playback
stops and the status bar explains why. The environment remains live for
manual inspection or reset.

## Expected limitations

- A barrier may have a valid global detour but still trap APF at a
  local minimum. See `tests/test_apf.py` for a reproducible example.
- Only observed obstacles exert repulsion; unobserved hidden obstacles
  cannot be anticipated.
- APF gives no completeness, reachability, or shortest-path guarantee.
- Strict descent makes the controller deterministic and eliminates
  trivial potential oscillations, but may stop where a non-monotonic
  detour is essential.
- APF parameter selection is heuristic, not theoretically optimal.
- Comparisons should focus on success, collisions, moves and inference
  latency on **paired maps**, not assume APF is competitive with A* in
  every obstacle layout.

A* remains the classical global planner. APF is included as a
meaningfully different **local/reactive** baseline.
