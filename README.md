# LayaNav

An experimental platform for **partially observable grid navigation** with online map updates, unknown static obstacles, interactive simulation, and future decision-model policies.

## Current milestone: v0.1 (environment + GUI + A* baseline)

- Gymnasium-compatible deterministic GridWorld; 32×32 by default
- Prior static map, simulator-private unknown obstacles, and robot belief map
- Ideal circular sensor (radius 5 by default; no occlusion yet)
- Five discrete actions: up, down, left, right, stay
- Online A* that replans on the **belief map**, not privileged ground truth
- Optional PySide6 desktop map viewer/editor with manual and A* modes
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

## Policy observation (v0.1)

Float32 tensor `(6, H, W)`, with channels:
1. Known prior walls
2. Discovered unknown obstacles
3. Unknown/unobserved cells
4. Robot position
5. Goal position
6. Capped normalized visit counts

Unknown static obstacles are sampled at episode reset, never move during the episode, and only appear in the observation after entering sensor range. The simulator-only `debug_snapshot()` method explicitly exposes ground truth for GUI/debugging and must not be used by a policy or teacher.

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). The **Laya adapter, PPO training, and model benchmark are not implemented in v0.1**; these require separate, testable milestones.

## License

No license has been selected. Repository remains private.
