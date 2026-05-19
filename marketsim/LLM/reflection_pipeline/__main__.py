from __future__ import annotations

import argparse

from marketsim.LLM.reflection_pipeline.config import build_default_config_pool
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the reflection pipeline")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--episodes-per-config", type=int, default=2)
    parser.add_argument("--ucb-constant", type=float, default=1.0)
    parser.add_argument("--max-debug-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--storage-dir", type=str, default="llm_calls_reflection_pipeline")
    parser.add_argument("--output", type=str, default="results/reflection_pipeline_summary.json")
    parser.add_argument("--reward-mode", type=str, default="advantage", choices=["advantage", "surplus", "consumer_surplus"])
    parser.add_argument("--strategy-output-mode", type=str, default="x_cap", choices=["x_cap", "shade"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--disable-critic", action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_pool = build_default_config_pool()
    fixed_strategy = None

    pipeline = ReflectionPipeline(
        config_pool=config_pool,
        batch_size=1 if args.smoke else args.batch_size,
        steps=1 if args.smoke else args.steps,
        episodes_per_config=1 if args.smoke else args.episodes_per_config,
        exploration_constant=args.ucb_constant,
        max_debug_retries=1 if args.smoke else args.max_debug_retries,
        storage_dir=args.storage_dir,
        output_path=args.output,
        seed=args.seed,
        strategy_output_mode=args.strategy_output_mode,
        reward_mode=args.reward_mode,
        fixed_strategy_code=fixed_strategy,
        critic_enabled=not args.disable_critic and not args.smoke,
        run_dir=args.run_dir,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
