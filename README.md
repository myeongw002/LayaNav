# LayaNav

An experimental platform for **partially observable grid navigation** with online map updates, unknown static obstacles, interactive simulation, and future decision-model policies.

## Current milestone: v0.4 experimental Laya spatial adapter

- Gymnasium-compatible deterministic GridWorld; 32×32 by default
- Prior static map, simulator-private unknown obstacles, and robot belief map
- Selectable ideal circular or occlusion-aware Bresenham raycast sensor (radius 5 by default)
- Configurable **4-way (5 actions)** or **8-way (9 actions)** navigation. Legacy action IDs 0–4 unchanged; diagonals are 5–8.
- Diagonal corner-cutting is forbidden; diagonal geometric travel distance is `sqrt(2)`.
- Online A* and discrete Artificial Potential Field (APF) using the **belief map**, not privileged ground truth
- PySide6 desktop map editor with sensor selection, JSON map import/export, planned A* path, visited trail and adjustable playback speed
- Deterministic episode recording and replay with observation/reward consistency checks
- PPO baseline with a custom spatial CNN extractor and checkpoint metadata
- Paired-seed evaluation of A*, APF, BC, PPO and experimental spatial Laya across controlled map families with 95% bootstrap confidence intervals
- Tests for reproducibility, limited sensing, mapping, collision behavior, and replanning

**The GUI's ground-truth panel is privileged debug visualization; policies must never receive it.**

## Installation (Ubuntu 24.04 / Python 3.12)

```bash
git clone https://github.com/myeongw002/LayaNav.git
cd LayaNav
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[gui,dev]"
pytest -q
python -m layanav --gui
```

Without a GUI:

```bash
python -m pip install -e ".[dev]"
python -m layanav --episodes 5 --seed 42
```

The simulator does **not** require PyTorch. For future RTX 5070 training, install a supported CUDA-enabled PyTorch wheel separately and verify `torch.cuda.is_available()` before installing Laya.

## Motion modes (4-way and 8-way)

The default remains **4-way**, maintaining compatibility with existing
training datasets, models and recorded episodes. Specify `--movement-mode 8`
for experiments or choose **8-way** in the GUI. Both modes always include
a `STAY` action:

| Action | Direction | Grid delta |
|---|---|---|
| 0 | UP | (0,-1) |
| 1 | DOWN | (0,+1) |
| 2 | LEFT | (-1,0) |
| 3 | RIGHT | (+1,0) |
| 4 | STAY | (0,0) |
| 5 (8-way only) | UP_LEFT | (-1,-1) |
| 6 (8-way only) | UP_RIGHT | (+1,-1) |
| 7 (8-way only) | DOWN_LEFT | (-1,+1) |
| 8 (8-way only) | DOWN_RIGHT | (+1,+1) |

A diagonal attempt is rejected if **either** adjoining orthogonal cell
is blocked, including still-undiscovered obstacles (physics enforces this;
a planner can only reason about observed blockers). The 8-way A* uses
**Octile Distance** and weighted edge costs: cardinal 1, diagonal `sqrt(2)`.
The environment reports both step count and geometric distance traveled.
The benchmark's oracle path efficiency uses geometric length, not step
count. Reward semantics remain unchanged (step-based).

```bash
# CLI: same map seeds, different action spaces
python -m layanav --policy astar --movement-mode 4 --seed 42
python -m layanav --policy astar --movement-mode 8 --seed 42

# Paired-seed A* and APF for 8-way; compare separately with 4-way.
python -m layanav.evaluation.cli \
  --methods astar apf --movement-mode 8 \
  --families sparse standard dense --sensor raycast \
  --episodes 100 --eval-seed 200000 \
  --output-dir results/astar_apf_8way
```

GUI: select **4-way / 8-way**. Manual control uses arrow keys in either
mode and **Q/E/Z/C** for diagonals when 8-way is active. Switching modes
resets the episode without changing the map. Replays store the movement
mode; old traces without a mode remain 4-way.

CNN/BC, PPO and Laya training use the action count of the environment or
training dataset. **Existing 5-output model checkpoints only work in 4-way
evaluations**; to run 8-way learned policies, collect a new 8-way teacher
dataset and retrain. The benchmark explicitly rejects mismatched action
counts to prevent invalid comparisons.

## Policy observation (v0.2)

Float32 tensor `(6, H, W)`, with channels:
1. Known prior walls
2. Discovered unknown obstacles
3. Unknown/unobserved cells
4. Robot position
5. Goal position
6. Capped normalized visit counts

Unknown static obstacles are sampled at episode reset, never move during the episode, and only appear in the observation after entering sensor range. The simulator-only `debug_snapshot()` method explicitly exposes ground truth for GUI/debugging and must not be used by a policy or teacher.

## Map file format and sensing

The GUI can save and load a JSON map containing the entire simulator layout: a schema version, square map size, known static walls, **privileged** hidden obstacle positions, initial robot position and goal. The file is for environment construction only; do not pass this file or its full obstacle contents into the policy or teacher. The `load_map` and `save_map` APIs validate dimensions, positions, binary cells and schema version.

The `circle` sensor sees every cell inside its radius, regardless of walls. The `raycast` sensor uses center-to-center Bresenham rays blocked by known walls **and** hidden obstacles: the obstacle surface is visible, but cells behind it remain unobserved. This is a discrete geometric approximation, not a full LiDAR or physics model. All discovered static obstacle cells remain in the belief map after leaving sensor range. Switching sensor modes resets the episode without replacing its layout.

## Episode recording and replay

In LayaNav Studio, select **Online A*** or **APF** and click **Play/Step** to navigate. The yellow line is the latest A* belief-map path, and the purple trail shows visited positions. **Save episode** writes a replayable JSON trace containing actions, rewards, observation SHA-256 digests, sensor settings and the initial map layout. **Load replay** reconstructs the environment and lets you replay/verify steps; its action sequence cannot be modified during replay. Speed is adjustable in steps per second.

**Security and research isolation:** Episode JSON stores the simulator's *complete obstacle truth* for deterministic replays. It is privileged debug data, not a policy observation or training input. The 6-channel policy observation remains separate. The trace records hashes rather than embedding every full observation; the same observations are regenerated and verified at replay time.

## Classical APF Baseline

APF is a deterministic **local reactive controller** with goal attraction and
repulsion from the nearest *known* obstacle. It selects only potential-decreasing
four-connected moves. If every valid neighbor has equal or higher potential,
it outputs STAY at a local minimum, **even when A* could find a detour**.
No global A* recovery is used, to keep the baseline distinct.

```bash
# Both policies use the same generated maps for each seed:
python -m layanav.evaluation.cli \
  --methods astar apf --families sparse standard dense \
  --episodes 100 --eval-seed 200000 --sensor raycast \
  --output-dir results/astar_vs_apf

# GUI: select APF from the policy dropdown.
python -m layanav --gui
```

No PyTorch is needed for APF. The optional unknown-cell penalty is
heuristic and defaults to zero. The benchmark records APF parameters.
See [docs/APF.md](docs/APF.md) for equations and limitations.

## CNN + Behavior Cloning baseline (v0.3)

The CNN policy is an **experimental baseline**, not a trained or calibrated model included in the repository. It consumes exactly the six-channel policy observation. The teacher is online A* operating only on the same belief map (it does **not** see hidden obstacle truth).

For a CUDA GPU, first install a PyTorch build compatible with RTX 5070 using the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install `python -m pip install -e ".[train]"`. For CPU-only testing, install a supported CPU wheel.

```bash
# Collect disjoint train/validation map seeds
python -m layanav.training.cli collect \
  --train datasets/teacher_train.npz \
  --validation datasets/teacher_val.npz \
  --train-episodes 100 --val-episodes 20 \
  --sensor raycast

# Behavior cloning using GPU if PyTorch CUDA is available
python -m layanav.training.cli train \
  --train datasets/teacher_train.npz \
  --validation datasets/teacher_val.npz \
  --output checkpoints/cnn_bc.pt \
  --epochs 10 --device cuda

# Hold-out evaluation: no overlap with train/val seed ranges
python -m layanav.training.cli evaluate \
  --checkpoint checkpoints/cnn_bc.pt \
  --episodes 30 --seed 200000 --device cuda --sensor raycast
```

A dataset stores 6-channel observations as uint8 values in `[0,10]` (scaled by 10 to preserve visit-count increments) and action labels. It deliberately omits simulator-private hidden obstacle truth. Saved checkpoints use PyTorch weights-only-compatible dictionaries. Reported accuracy is imitation-label accuracy; success rate must be measured separately through evaluation. Data collection excludes failed teacher episodes and reports their count.

## PPO baseline and comparable evaluation (Issue #6)

Install the PyTorch wheel matching your NVIDIA RTX 5070 and CUDA runtime **before** installing the PPO optional dependency (see [PyTorch install selector](https://pytorch.org/get-started/locally/)). For a CPU smoke test, the default PyTorch CPU build is sufficient. Stable-Baselines3 and the policy share the same 6-channel float32 observation; **image normalization is explicitly disabled**. The default sparse goal reward makes PPO a challenging baseline, so initial training success is not guaranteed.

```bash
# After verifying torch.cuda.is_available() on your GPU:
python -m pip install -e ".[train,ppo]"

# Collect a specified training budget in 4 environments.
python -m layanav.training.ppo_cli \
  --output checkpoints/ppo_standard.zip \
  --timesteps 100000 --n-envs 4 --n-steps 128 \
  --batch-size 128 --sensor raycast --device cuda

# Same map seeds and observation rules for all evaluation policies:
python -m layanav.evaluation.cli \
  --methods astar bc ppo --families sparse standard dense \
  --bc-checkpoint checkpoints/cnn_bc.pt \
  --ppo-checkpoint checkpoints/ppo_standard.zip \
  --episodes 100 --eval-seed 200000 --sensor raycast \
  --grid-size 32 --radius 5 --max-steps 256 \
  --oracle-diagnostic --device cuda --output-dir results/bench_v03
```

Outputs: `episodes.csv` (per-map results) and `summary.json` (paired seed range, exact environment configurations, checkpoint SHA-256 hashes, PPO training metadata, and percentile bootstrap 95% CIs over independent map seeds). The optional Oracle A* shortest-path diagnostic uses **privileged simulator truth only for reporting a lower bound**—not for training or choosing actions.

**Comparability limits:** identical evaluation map seeds, actions, timeout and sensor settings do not make BC and PPO training budgets equivalent. BC receives expert demonstrations; PPO consumes environment interactions. If performance differs, training data and computation must also be controlled before concluding a model is superior. This v0.3 pipeline is implemented and smoke-tested; it is **not** a measured performance claim.

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the evaluation protocol and research limitations.

## Experimental Laya spatial adapter (v0.4)

The original Laya model uses a pretrained ModernBERT text encoder plus a text-specific decision head. This branch reuses **only the pretrained encoder** and adds a six-channel spatial tokenizer, trainable positions, an actor head and a critic head. It is not the standard `laya.Agent.predict()` API.

For the RTX 5070, install a Blackwell-compatible CUDA PyTorch wheel first, then `python -m pip install -e ".[train,laya]"` (Laya pinned to 0.4.31). Train against the previously collected belief-only A* demonstrations:

```bash
python -m layanav.training.laya_cli \
  --train datasets/teacher_train.npz \
  --validation datasets/teacher_val.npz \
  --output checkpoints/laya_spatial_bc.pt \
  --model-id convaiinnovations/laya \
  --device cuda --batch-size 2 --accumulation-steps 4

python -m layanav.evaluation.cli \
  --methods astar bc ppo laya \
  --bc-checkpoint checkpoints/cnn_bc.pt \
  --ppo-checkpoint checkpoints/ppo_standard.zip \
  --laya-checkpoint checkpoints/laya_spatial_bc.pt \
  --families sparse standard dense --eval-seed 200000 \
  --episodes 100 --device cuda --output-dir results/laya_comparison
```

For reproducible pretrained weights, pass an **exact model Hub commit** with `--revision`; a default floating revision is not recommended for research. Only the small adapter/head state is saved; the pretrained backbone is loaded again on evaluation. The value head is **not** trained by Behavior Cloning.

Run a **single real pretrained-weight GPU smoke check** on your Ubuntu machine before attempting training:

```bash
python -m layanav.training.laya_smoke \
  --model-id convaiinnovations/laya --device cuda \
  --batch-size 1 --token-grid 8 --grid-size 32
```

This reports the resolved checkpoint revision, forward/backward latency, actor/critic shapes, nonzero adapter gradient, and peak CUDA allocation/reservation. It **does not** measure navigation performance.

**Validation limits:** unit tests use a tiny fake backbone. The 421M-parameter pretrained Laya checkpoint has **not** been executed on the target RTX 5070 here. There are no trained navigation results yet. See [docs/LAYA_ADAPTER.md](docs/LAYA_ADAPTER.md) for implementation details and rigorous baseline requirements.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). The **Laya spatial adapter and frozen-encoder BC training path** are implemented but not validated on the real pretrained checkpoint; full fine-tuning, PPO adaptation, and controlled transfer ablations remain unimplemented. v0.3 PPO/benchmark support is an experimental foundation, not evidence that PPO or Laya performs well.

## License

No license has been selected. Repository remains private.
