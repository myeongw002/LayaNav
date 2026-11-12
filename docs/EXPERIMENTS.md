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

## Separation and fairness

Use a disjoint set of map seeds for BC train, BC validation and benchmark. PPO stores the seed of its training RNG; evaluation uses a far-removed, explicitly recorded seed sequence. This does **not** mathematically guarantee distinct generated layouts, especially for small grids; choose nontrivial map families and consider checksums of generated maps in a future protocol extension.

Keep the observation radius, sensor, grid size, maximum steps, initial map generation, reward, and collision semantics identical across methods. Compare inference latency on the same machine with consistent device and concurrency conditions; the local CI smoke test is not a latency benchmark.

BC and PPO are **not equally budgeted** by default:
- BC uses supervised expert labels from Online A*.
- PPO samples environment transitions, including failed episodes.
- BC's human-specified dataset/epochs and PPO's environment timesteps must be reported separately.

Future fairer comparisons should constrain wall time, compute, parameter counts, training maps and samples, in addition to test-time observations.

## Reported metrics

Per episode (`episodes.csv`): family, method, seed, success, steps, actual moved steps, attempted collisions, accumulated return, mean policy inference latency in milliseconds. When enabled, add the oracle shortest number of moves and a success-only path efficiency ratio (oracle moves / actual moved moves).

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
