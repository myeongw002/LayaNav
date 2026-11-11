# LayaNav

An experimental platform for **partially observable grid navigation** with online map updates, unknown static obstacles, interactive simulation, and future decision-model policies.

## Current milestone: v0.2 (sensor occlusion + JSON maps)

- Gymnasium-compatible deterministic GridWorld; 32×32 by default
- Prior static map, simulator-private unknown obstacles, and robot belief map
- Selectable ideal circular or occlusion-aware Bresenham raycast sensor (radius 5 by default)
- Five discrete actions: up, down, left, right, stay
- Online A* that replans on the **belief map**, not privileged ground truth
- Optional PySide6 desktop map viewer/editor with manual and A* modes, sensor selection, JSON map import/export
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

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md). The **Laya adapter, PPO training, and model benchmark are not implemented in v0.2**; these require separate, testable milestones.

## License

No license has been selected. Repository remains private.
