# Implementation roadmap

## Principles

1. Keep the environment, policy, and GUI independent.
2. Never expose hidden obstacle coordinates through `reset()`, `step()`, or observations.
3. Keep training and evaluation map seeds disjoint, and track seeds/configuration.
4. Compare learned policies against belief-map A* and equal-capacity random-init backbones.
5. Don't assume text-pretrained Laya transfers to numerical spatial inputs.

## Milestone v0.1 — environment foundation (this PR)

- [x] Packaging, tests and CI
- [x] GridWorldEnv (Gymnasium API)
- [x] Full prior map and separate unknown obstacle truth
- [x] Radius-limited ideal sensor + persistent belief map
- [x] Configurable maps/start/goal, deterministic seeds, collision handling
- [x] Online A* using only policy observations
- [x] PySide6 viewer/editor; simulation playback and manual actions
- [ ] Confirm end-to-end GUI appearance on user's Ubuntu 24.04 desktop
- [ ] Confirm PyTorch and Laya compatibility on user's RTX 5070 hardware

## Milestone v0.2 — environment quality

- Ray-cast occlusion sensor; document partial-observability assumptions
- Map JSON load/save and map-generation presets
- GUI path overlays, richer runtime diagnostics, episode recording/replay
- Robust evaluation suite across controlled map difficulty and seeds

## Milestone v0.3 — classical and neural baselines

- Online A* evaluation/benchmark runner
- CNN+MLP imitation-learning policy
- CNN+PPO policy; optional recurrent baseline
- Ensure teacher sees the same belief map, not simulator ground truth
- Isolate trajectories by map seed and record wall-clock latency

## Milestone v0.4 — Laya spatial adaptation

- Verify upstream Laya model structure/weights and licensing
- Numeric/spatial encoder: 6-channel grid → spatial tokens
- Adapter into pretrained Laya encoder and custom 5-action head
- Frozen-encoder behavior cloning, logprobs, checkpointing
- Test for gradient flow and absence of privileged state leakage

## Milestone v0.5 — training and research evaluation

- PPO fine-tuning with suitable critic
- Frozen / partial-unfreeze / random-init ablations
- Compare success, path length, collisions, speed and memory
- Hold-out map families, different observation radii and sensor models
- Dynamic obstacles only after stable static-obstacle baseline

## Non-goals for v0.1

Real robot control, dynamic-obstacle prediction, full Laya fine-tuning, optimality or safety guarantees.
