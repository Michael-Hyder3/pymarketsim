#!/usr/bin/env python
"""
Test script for the multi-state RL feedback loop pipeline.

Architecture:
  - State space: 1 MarketConfig (baseline only)
  - Policy: LLM generates N=3 candidate strategies per episode
  - Reward: LLM surplus from a single diagnostic sim (fixed seed per state/episode)
  - Value function: ValueTable tracks best surplus seen per market config
  - Context: raw JSON (offline ZI data → last diagnostic JSON + value table)
"""

import sys
import json
import argparse
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from marketsim.LLM.feedback_loop import FeedbackLoopPipeline
from marketsim.LLM.market_config import MarketConfig
from marketsim.fourheap.constants import BUY


FIXED_PRIVATE_VALUES = [
    631.0449829101562,
    423.5244445800781,
    332.5553894042969,
    254.97203063964844,
    216.13760375976562,
    141.75466918945312,
    118.5871810913086,
    46.538047790527344,
    -25.70964813232422,
    -236.3279571533203,
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LLM feedback-loop baseline test.")
    parser.add_argument(
        "--pv-mode",
        choices=["fixed", "variable"],
        default="fixed",
        help=(
            "Private-value mode: 'fixed' uses a hardcoded PV vector, "
            "'variable' samples PVs from RNG (seed-dependent)."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of feedback-loop episodes to run (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base diagnostic seed (default: 100).",
    )
    return parser.parse_args()


def main(args):
    """Run feedback loop test — 5 episodes on baseline config only."""

    # Single-state run: only the baseline config, repeated 5 times.
    # This lets us observe how the strategy improves episode-over-episode
    # on the same market without the distraction of config switching.
    private_values = FIXED_PRIVATE_VALUES if args.pv_mode == "fixed" else None

    baseline = MarketConfig(
        num_zi_buy=5, num_zi_sell=5, timesteps=1000,
        q_max=10, pv_var=1e5, shade_range=[0, 500], label="baseline",
        fixed_private_values_relative=private_values,
        fixed_seed=args.seed,
    )

    print(f"PV mode: {args.pv_mode} | fixed_seed: {args.seed} | episodes: {args.episodes}")

    pipeline = FeedbackLoopPipeline(
        llm_order_type=BUY,
        market_configs=[baseline],
        num_candidates=3,
        buffer_size=5,
        max_inner_iterations=3,
        storage_dir=str(REPO_ROOT / "llm_calls"),
        offline_data_path=str(REPO_ROOT / "results" / "offline_zi_baseline.json"),
    )

    # 10 episodes — all on the same baseline config
    results = pipeline.run_feedback_loop(num_episodes=args.episodes)

    # Save results
    output_path = REPO_ROOT / "results" / "baseline_only_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    cli_args = parse_args()
    results = main(cli_args)
