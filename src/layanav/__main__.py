"""CLI entry point: headless A* benchmark or optional GUI."""

from __future__ import annotations

import argparse

from layanav.envs import MAP_TOPOLOGIES, GridConfig, GridWorldEnv
from layanav.planners import ArtificialPotentialFieldPolicy, OnlineAStarPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="LayaNav grid navigation")
    parser.add_argument("--gui", action="store_true", help="launch optional PySide6 viewer")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=["astar", "apf"], default="astar")
    parser.add_argument("--movement-mode", type=int, choices=[4, 8], default=4)
    parser.add_argument("--topology", choices=MAP_TOPOLOGIES, default="random")
    args = parser.parse_args()
    if args.gui:
        try:
            from layanav.gui.app import launch
        except ImportError as exc:
            raise SystemExit("GUI requires: pip install -e '.[gui]'") from exc
        launch()
        return
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    env = GridWorldEnv(GridConfig(
        movement_mode=args.movement_mode, topology=args.topology,
    ))
    successes = 0
    for i in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + i)
        policy = OnlineAStarPolicy(args.movement_mode) if args.policy == "astar" else ArtificialPotentialFieldPolicy(movement_mode=args.movement_mode)
        finished = False
        last_info = None
        while not finished:
            obs, _, terminated, truncated, last_info = env.step(policy.act(obs))
            finished = terminated or truncated
        successes += int(last_info["success"])
        print(f"episode={i} success={last_info['success']} steps={last_info['steps']}")
    print(f"policy={args.policy} success_rate={successes / args.episodes:.3f}")
    env.close()


if __name__ == "__main__":
    main()
