# Implementation roadmap

## Principles

1. Keep the environment, policy, and GUI independent.
2. Never expose hidden obstacle coordinates through `reset()`, `step()`, or observations.
3. Keep training and evaluation map seeds disjoint, and track seeds/configuration.
4. Compare learned policies against belief-map A* and equal-capacity random-init backbones.
5. Don't assume text-pretrained Laya transfers to numerical spatial inputs.

## Milestone v0.1 — environment foundation (merged)

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

- [x] Ray-cast occlusion sensor; document partial-observability assumptions
- [x] Versioned JSON map load/save with validation (including GUI)
- [ ] Map-generation presets
- [x] GUI path overlay, visited trajectory and adjustable simulation speed
- [x] Deterministic episode JSON recorder and verified replay in GUI
- [ ] Richer diagnostics, replay seek/scrub and map-generation presets
- [x] Paired-seed bootstrap benchmark across controlled map families and held-out seeds

## Milestone v0.3 — classical and neural baselines

- [x] Online A* / BC / PPO paired-seed benchmark and CSV/JSON reports
- [x] CNN+MLP imitation-learning policy, compact datasets, CLI and checkpoints
- [x] CNN+PPO policy with Gymnasium-compatible custom spatial extractor
- [ ] Optional recurrent PPO baseline
- [x] Ensure teacher sees the same belief map, not simulator ground truth
- [x] Split trajectories by map seed and measure step-level inference latency
- [ ] Add DAgger for recovery from policy-induced distribution shift

## Milestone v0.4 — Laya spatial adaptation

- [x] Verify upstream Laya API, ModernBERT encoder access, package version and model license
- [x] Numeric/spatial encoder: 6-channel grid → spatial tokens
- [x] Adapter into Laya encoder interface and custom 5-action/critic heads (mock-backbone CI only)
- [x] Frozen-encoder behavior cloning and adapter-only checkpointing
- [ ] Real checkpoint RTX 5070 forward/backward, GPU memory/latency measurement
- [ ] PPO logprobs and actor/critic training with spatial Laya
- [x] Gradient tests and belief-only data path with tiny stand-in backbone

## Milestone v0.5 — training and research evaluation

- PPO fine-tuning with suitable critic
- [ ] Frozen / partial-unfreeze / random-init ablations
- Compare success, path length, collisions, speed and memory
- Hold-out map families, different observation radii and sensor models
- Dynamic obstacles only after stable static-obstacle baseline

## Non-goals for v0.1

Real robot control, dynamic-obstacle prediction, full Laya fine-tuning, optimality or safety guarantees.
