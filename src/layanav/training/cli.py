"""CLI for A* imitation demonstrations, CNN behavior cloning and evaluation."""

from __future__ import annotations

import argparse
import json

from layanav.envs import MAP_TOPOLOGIES, GridConfig


def _environment(args: argparse.Namespace) -> GridConfig:
    return GridConfig(
        size=args.grid_size,
        observation_radius=args.radius,
        sensor_mode=args.sensor,
        movement_mode=args.movement_mode,
        topology=args.topology,
        min_route_steps=args.min_route_steps,
        max_route_steps=args.max_route_steps,
        hidden_obstacle_count=args.hidden_obstacles,
        max_steps=args.max_steps,
        known_wall_segments=args.wall_segments,
    )


def _environment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--movement-mode", type=int, choices=[4, 8], default=4)
    parser.add_argument("--topology", choices=MAP_TOPOLOGIES, default="random")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--sensor", choices=["circle", "raycast"], default="raycast")
    parser.add_argument("--hidden-obstacles", type=int, default=24)
    parser.add_argument("--wall-segments", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--min-route-steps", type=int)
    parser.add_argument("--max-route-steps", type=int)


def main() -> None:
    parser = argparse.ArgumentParser(description="LayaNav imitation learning")
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect online A* teacher data")
    collect.add_argument("--train", required=True)
    collect.add_argument("--validation", required=True)
    collect.add_argument("--train-episodes", type=int, default=100)
    collect.add_argument("--val-episodes", type=int, default=20)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--val-seed", type=int, default=100000)
    _environment_options(collect)

    train = commands.add_parser("train", help="train CNN behavior-cloning policy")
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--channels", type=int, default=32)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=42)

    evaluate = commands.add_parser("evaluate", help="evaluate CNN checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--episodes", type=int, default=30)
    evaluate.add_argument("--seed", type=int, default=200000)
    evaluate.add_argument("--device", default="cpu")
    _environment_options(evaluate)

    args = parser.parse_args()
    if args.command == "collect":
        from layanav.training.dataset import collect_expert_dataset, save_dataset

        if args.seed <= args.val_seed + args.val_episodes - 1 and (
            args.val_seed <= args.seed + args.train_episodes - 1
        ):
            parser.error("Train/validation map seed ranges must be disjoint")
        config = _environment(args)
        train_set, train_stats = collect_expert_dataset(
            episodes=args.train_episodes, seed_start=args.seed, config=config
        )
        val_set, val_stats = collect_expert_dataset(
            episodes=args.val_episodes, seed_start=args.val_seed, config=config
        )
        save_dataset(args.train, train_set)
        save_dataset(args.validation, val_set)
        print(json.dumps({"train": train_stats, "validation": val_stats}, indent=2))
    elif args.command == "train":
        from layanav.training.behavior_cloning import train_behavior_cloning

        train_behavior_cloning(
            train_data=args.train,
            validation_data=args.validation,
            output_checkpoint=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            channels=args.channels,
            device=args.device,
            seed=args.seed,
        )
    else:
        from layanav.training.evaluation import evaluate_policy

        evaluate_policy(
            args.checkpoint,
            episodes=args.episodes,
            seed_start=args.seed,
            config=_environment(args),
            device=args.device,
        )


if __name__ == "__main__":
    main()
