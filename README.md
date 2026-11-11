# LayaNav

Research platform for partially observable grid navigation, online mapping, interactive simulation, and experimental decision policies.

## Project status

Initial implementation in progress. The first milestone is a deterministic Gymnasium-compatible GridWorld with a limited-radius sensor, persistent belief map, online A* replanning, and a PySide6 desktop viewer.

## Research constraints

- The policy only observes the static prior map, current position, goal and acquired obstacle observations.
- Hidden obstacle positions are simulator-private and must never leak into policy observations.
- Model-specific learning (including modified Laya) is a later, independently evaluated stage.

See `docs/ROADMAP.md` after the first implementation milestone.
