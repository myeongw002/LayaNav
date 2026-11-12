# LayaNav v0.3 — Evaluation Protocol

## Research question

Can a learned navigation policy handle previously unknown, stationary obstacles with only a known prior map, limited-radius sensing and a persistent belief map? The comparison supports **Online A***, **Artificial Potential Field (APF)**, **CNN behavior cloning (BC)**, **CNN PPO**, and experimental **Laya-spatial BC**, under the same evaluation conditions.

This benchmark does not establish that a language-pretrained Decision Model transfers to spatial navigation. Laya-spatial integration is available as an experimental adapter, but no real-pretrained-checkpoint performance has been measured.

## Experimental environment

- Agent input: six-channel policy-safe float32 map, including prior walls, discovered hidden obstacles, unknown cells, current position, goal and capped visit counts.
- True hidden obstacles remain simulator-private. The policy and A* baseline receive only the policy observation.
- Actions: configurable 4-way = four cardinal moves + stay (5 choices); 8-way = the same five action IDs plus four diagonal actions (9 choices). Attempted collisions do not change position.
- Diagonal motion is forbidden if either adjacent orthogonal cell contains an obstacle, including hidden simulator-only blockers.
- Cardinal travel costs 1 grid unit, diagonal travel costs `sqrt(2)`. A* uses an Octile heuristic in 8-way mode. Path efficiency is computed with geometric distance, **not** move count.
- Models must be trained for the same action-space size as the evaluation environment; 5-action weights are not silently expanded into 9-action weights.
- Fixed timeout, reward and collision logic per method within a benchmark. Preserve movement mode within an evaluation group; 4-way and 8-way have different feasible path spaces and cannot be compared as identical tasks.
- Episode-level random map generation is seeded by `eval_seed + episode_index`; every method uses the **same seed for each map family**.
- Map families differ in hidden obstacle count and known wall segments; all use the same grid size and action space. Family densities scale with grid size.
- `raycast` senses only cells with discrete Bresenham line-of-sight; `circle` is an ideal (no-occlusion) sensor.
- APF is a local, strictly potential-descending controller on the belief map;
  it may stop at a local minimum even when the environment is traversable.
  APF parameters are saved in `summary.json` and must be tuned using
  seeds disjoint from the final evaluation set.

## Structured topology evaluation and failure analysis

Use `--topologies random maze corridor rooms u_shape cluttered` to
evaluate each static topology as a separate map distribution. The legacy
`random` default is unchanged, retaining historical seed-to-map samples.
For structured topologies, a reachability check guarantees a path exists
under the selected 4/8-way motion contract. Maze generation uses randomized
DFS corridor carving; rooms have door openings, corridor maps use connected
bands, U-shaped maps intentionally place the start inside the concavity,
and cluttered layouts contain multi-cell obstacle rectangles.

Non-random layouts use topology-specific wall geometry. Consequently, a
`dense` Maze and a `dense` Random map are **not** controlled to have
the same wall occupancy. Compare policies **within** each
topology/family/movement-mode combination before attempting broad claims.

Failure diagnostics distinguish the environment termination condition
(`goal_reached` or `time_limit`) from an inferred failure attribution
(`local_minimum`, `blocked`, `no_known_path`,
`persistent_stay`, or `time_limit`). These categories are diagnostics
based on final policy status and actions, not ground-truth causal proofs.
Revisit counts exclude STAY actions; attempted collisions remain separate.
Failed rollouts are saved as verified deterministic traces in `failures/`
by default; disable using `--no-save-failures`. These JSON files contain
**privileged full obstacle truth** for replay and are not training data.

## Benchmark v2: statistical and difficulty definitions

Summary schema `2` preserves historical method/topology keys, with new
metrics alongside old outputs. Success `mean` is the observed Bernoulli
proportion; its two-sided 95% interval uses the **Wilson score method**.
A run with 20/20 successes has an interval narrower than `[0,1]` but
**not** the degenerate interval `[1,1]`. Continuous metrics use the
existing independent-episode percentile bootstrap (not a paired-difference
CI); outputs include their non-null sample count `n`.

For each episode with an oracle path, SPL is:

```text
SPL_i = success_i * L_i / max(P_i, L_i)
```

where `L_i` is the true-map shortest **geometric** route distance
(cardinal 1, diagonal sqrt(2)) and `P_i` is actual distance traveled.
A failed episode contributes zero to SPL. The mean uses **all episodes**.
`oracle_efficiency` remains successful-only, while
`successful_steps`, `successful_moves` and
`successful_distance_traveled` summarize successful episodes only.
When no episodes succeed, these summaries are null.

`--oracle-diagnostic` is mandatory to calculate SPL and descriptive
map complexity. Without it, these fields remain null. This privileged
computation never feeds model inputs or action selection.

Map metrics are evaluated once per episode from initial truth and the
full-map optimal path; **they are not an experimentally validated scalar
difficulty score**:

| Key | Definition |
| --- | --- |
| `obstacle_density` | fraction blocked over full grid, *including perimeter* |
| `interior_obstacle_density` | blocked fraction excluding perimeter |
| `initial_unknown_ratio` | initially unobserved cells / cells not known as prior walls |
| `straight_line_distance` | Euclidean distance between initial start and goal |
| `path_tortuosity` | shortest geometric path length / straight-line distance |
| `minimum_clearance` | smallest center-to-center distance from shortest-path cells to any blocked cell (not physical clearance) |
| `dead_end_count` | count of graph-degree-one cells in the start-reachable true-map component, under active movement rules |
| `reachable_free_cells` | size of that start-reachable component |
| `map_sha256` | fingerprint of known and hidden map layers, start, goal, movement mode |

Topologies are intentionally distributionally different. Do **not**
interpret changes in success rate across topologies as the isolated
effect of any one complexity metric. The U-Shape generator now varies
start and goal positions across the closed barrier; prior U-Shape maps
with identical seeds are not comparable across code revisions.

## Structured generator v2: constrained map sampling

The generator samples start/goal on the prior topology before hidden blockers.
For each structured map instance, hidden cells are selected by removing leaves
from a cardinal spanning tree rooted at start (while preserving the goal).
The remaining prior-reachable component stays connected under 4-way and 8-way
(no diagonal corner cutting) motion. With fixed prior geometry/start/goal and
nested obstacle placements, increasing hidden occupancy cannot reduce shortest
true-map path length. The diagnostic `prior_layout_sha256` hashes the
known-wall layer and start/goal (with an empty hidden layer), allowing
cross-density verification of the common prior scenario, while
`map_sha256` identifies the complete physical map.

**Important limitations:**

- A fully tree-structured Maze may become *less extensive* when leaf branches
  are blocked, but the start-goal route can stay unchanged. Increasing hidden
  density is therefore not equivalent to increasing navigation difficulty.
- Cluttered map structural walls depend on family wall-segment counts and do
  not share a single underlying prior map.
- Custom `--max-route-steps` may require different generated instances across
  family densities, preventing same-seed nesting.
- Generator v2 changes previous structured seed layouts. Report the generator
  version, seed, config and code revision for all comparisons. Never compare
  v1 and v2 same-seed structured maps as though identical.

Route bounds `min_route_steps` and `max_route_steps` are **integer
shortest-path action counts** under the active 4/8-way graph, not weighted
geometric distance. For structured maps the pair is chosen on the prior;
the final map must still meet any explicitly requested upper bound.
Impossible constraints throw an error, rather than relaxing counts.

## Tortuosity-based evaluation groups

With `--oracle-diagnostic`, `episodes.csv` contains a descriptive
`tortuosity_bin`. The new `tortuosity_groups.csv` reports
method/topology/family-specific success rates with Wilson CIs and SPL
bootstrap CIs for three fixed, **heuristic** ranges:

- `direct`: optimal-path tortuosity < 1.25.
- `moderate_detour`: 1.25 ≤ tortuosity < 2.
- `large_detour`: tortuosity ≥ 2.

Empty groups have `episodes = 0` and null statistics. The bin is defined
using true-map shortest route geometry, **not** the executed policy path,
and is never a policy input. Small-group intervals will be broad. The
additional `mean_path_clearance` is mean Euclidean cell-center distance
to nearest blocked cell along the optimal route; it does **not** represent
physical robot clearance.

## Separation and fairness

Use a disjoint set of map seeds for BC train, BC validation and benchmark. PPO stores the seed of its training RNG; evaluation uses a far-removed, explicitly recorded seed sequence. This does **not** mathematically guarantee distinct generated layouts, especially for small grids; choose nontrivial map families and consider checksums of generated maps in a future protocol extension.

Keep the observation radius, sensor, grid size, maximum steps, initial map generation, reward, and collision semantics identical across methods. Compare inference latency on the same machine with consistent device and concurrency conditions; the local CI smoke test is not a latency benchmark.

BC and PPO are **not equally budgeted** by default:
- BC uses supervised expert labels from Online A*.
- PPO samples environment transitions, including failed episodes.
- BC's human-specified dataset/epochs and PPO's environment timesteps must be reported separately.

Future fairer comparisons should constrain wall time, compute, parameter counts, training maps and samples, in addition to test-time observations.

## Reported metrics

Per episode (`episodes.csv`): family, method, seed, success, steps, actual moved steps, attempted collisions, accumulated return, mean policy inference latency in milliseconds. When enabled, add the oracle shortest move count, oracle shortest geometric distance and success-only geometric path efficiency (oracle distance / actual distance).

In `summary.json`: mean and 95% percentile-bootstrap interval for each metric over episode/map seeds, sample count, environment config, seed range, checkpoint paths+SHA-256, and PPO model training metadata. Bootstrap samples **episodes**, never consecutive timesteps. Confidence intervals for path efficiency include only successful episodes and are null when there are none.

**Oracle A*** can see the full truth map but is used exclusively for the lower-bound diagnostic; it never participates in action selection, dataset creation or PPO rewards. It is not a fair competitor to Online A* or the neural policies.

## Instructions

```bash
# Install CPU/CUDA PyTorch first, then:
python -m pip install -e ".[train,ppo]"

python -m layanav.training.ppo_cli \
    --output checkpoints/ppo_standard.zip --timesteps 100000 \
    --seed 42 --n-envs 4 --n-steps 128 --batch-size 128 \
    --device cuda --sensor raycast

python -m layanav.evaluation.cli \
    --methods astar ppo --families sparse standard dense \
    --ppo-checkpoint checkpoints/ppo_standard.zip \
    --episodes 100 --eval-seed 200000 --grid-size 32 --radius 5 \
    --sensor raycast --max-steps 256 --oracle-diagnostic \
    --device cuda --output-dir results/ppo_comparison
```

Run `python -m layanav.evaluation.cli --methods astar ...` without PyTorch. Add `bc` and `--bc-checkpoint` if the BC model from the earlier milestone is available.

## Limitations

- The current task has no moving obstacles, odometry drift, probabilistic occupancy, exploration bonus or formal safety guarantee.
- Sparse terminal rewards can prevent PPO from learning useful behavior, even when code and unit tests work.
- The feed-forward CNN and current spatial Laya adapter cannot represent arbitrary action history beyond belief map and visit count; compare recurrent PPO or a planning-memory module in follow-up work.
- CI only verifies small CPU episodes, gradient flow, saving/loading and contract invariants. It does **not** provide empirical navigation performance on the user GPU.
