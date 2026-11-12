# LayaNav

An experimental platform for **partially observable grid navigation** with online map updates, unknown static obstacles, interactive simulation, and future decision-model policies.

## Current milestone: v0.3 (CNN/BC + PPO + benchmark)

- Gymnasium-compatible deterministic GridWorld; 32×32 by default
- Prior static map, simulator-private unknown obstacles, and robot belief map
- Selectable ideal circular or occlusion-aware Bresenham raycast sensor (radius 5 by default)
- Five discrete actions: up, down, left, right, stay
- Online A* that replans on the **belief map**, not privileged ground truth
- PySide6 desktop map editor with sensor selection, JSON map import/export, planned A* path, visited trail and adjustable playback speed
- Deterministic episode recording and replay with observation/reward consistency checks
- PPO baseline with a custom spatial CNN extractor and checkpoint metadata
- Paired-seed evaluation of A*, BC and PPO across controlled map families with 95% bootstrap confidence intervals
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

In LayaNav Studio, select **Online A*** and click **Play/Step** to navigate. The yellow line is the latest A* belief-map path, and the purple trail shows visited positions. **Save episode** writes a replayable JSON trace containing actions, rewards, observation SHA-256 digests, sensor settings and the initial map layout. **Load replay** reconstructs the environment and lets you replay/verify steps; its action sequence cannot be modified during replay. Speed is adjustable in steps per second.

**Security and research isolation:** Episode JSON stores the simulator's *complete obstacle truth* for deterministic replays. It is privileged debug data, not a policy observation or training input. The 6-channel policy observation remains separate. The trace records hashes rather than embedding every full observation; the same observations are regenerated and verified at replay time.

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

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). The **Laya spatial adapter, fine-tuning, and controlled transfer ablations** remain unimplemented. v0.3 PPO/benchmark support is an experimental foundation, not evidence that PPO or Laya performs well.

## License

No license has been selected. Repository remains private.
