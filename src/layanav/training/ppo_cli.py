"""CLI for training a PPO spatial CNN baseline with a fixed training budget."""

from __future__ import annotations

import argparse
import json

from layanav.envs import MAP_TOPOLOGIES, GridConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="LayaNav PPO training")
    parser.add_argument("--output", required=True, help="Checkpoint path ending in .zip")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--movement-mode", type=int, choices=[4, 8], default=4)
    parser.add_argument("--topology", choices=MAP_TOPOLOGIES, default="random")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--sensor", choices=["circle", "raycast"], default="raycast")
    parser.add_argument("--hidden-obstacles", type=int, default=24)
    parser.add_argument("--wall-segments", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256)
    args = parser.parse_args()

    from layanav.training.ppo import PPOSettings, train_ppo

    env_config = GridConfig(
        size=args.grid_size,
        observation_radius=args.radius,
        sensor_mode=args.sensor,
        movement_mode=args.movement_mode,
        topology=args.topology,
        hidden_obstacle_count=args.hidden_obstacles,
        known_wall_segments=args.wall_segments,
        max_steps=args.max_steps,
    )
    settings = PPOSettings(
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        channels=args.channels,
        device=args.device,
    )
    metadata = train_ppo(output=args.output, env_config=env_config, settings=settings)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
