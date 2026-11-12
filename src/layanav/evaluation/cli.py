"""Run A*, APF, CNN-BC and PPO on identical held-out maps."""

from __future__ import annotations

import argparse
import json

from layanav.evaluation.benchmark import FAMILIES, benchmark
from layanav.planners import APFConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-seed navigation benchmark")
    parser.add_argument("--methods", nargs="+", default=["astar"])
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--bc-checkpoint")
    parser.add_argument("--ppo-checkpoint")
    parser.add_argument("--laya-checkpoint")
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=200_000)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--sensor", choices=["circle", "raycast"], default="raycast")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--apf-attractive-gain", type=float, default=1.0)
    parser.add_argument("--apf-repulsive-gain", type=float, default=8.0)
    parser.add_argument("--apf-influence-radius", type=float, default=3.0)
    parser.add_argument("--apf-unknown-penalty", type=float, default=0.0)
    parser.add_argument("--oracle-diagnostic", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    try:
        result = benchmark(
            methods=tuple(args.methods),
            families=tuple(args.families),
            episodes=args.episodes,
            eval_seed=args.eval_seed,
            size=args.grid_size,
            radius=args.radius,
            sensor=args.sensor,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            bc_checkpoint=args.bc_checkpoint,
            ppo_checkpoint=args.ppo_checkpoint,
            laya_checkpoint=args.laya_checkpoint,
            apf_config=APFConfig(
                attractive_gain=args.apf_attractive_gain,
                repulsive_gain=args.apf_repulsive_gain,
                influence_radius=args.apf_influence_radius,
                unknown_penalty=args.apf_unknown_penalty,
            ),
            device=args.device,
            oracle_diagnostic=args.oracle_diagnostic,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
