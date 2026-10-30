#!/usr/bin/env python3
"""HBL ceiling on the stratified homeset-3 markets (seed 123).

Runs HBL (L=6) with the same zi_advantage evaluation stack as the LOB
reflection pipeline: 3 homeset markets × N episodes, 21 ZI shade caps.

Example:
  .venv/bin/python -m marketsim.LLM.reflection_pipeline.eval_hbl_homeset_ceiling \\
    --episodes-per-config 100 --seed 123 --out-dir .../runs/hbl_homeset3_ceiling_...
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

from marketsim.LLM.reflection_pipeline.config import (
    build_default_config_space,
    sample_stratified_home_set,
)
from marketsim.LLM.reflection_pipeline.metrics import mean_and_variance
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
    evaluate_batch_with_zi_benchmarks,
    mean_llm_surplus_from_batch_data,
)


def _dummy_strategy(*_args: Any, **_kwargs: Any) -> float:
    """Unused when special_agent_kind='hbl'."""
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--home-set-size", type=int, default=3)
    parser.add_argument("--episodes-per-config", type=int, default=100)
    parser.add_argument("--hbl-l", type=int, default=6)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Directory for summary JSON (created if missing).",
    )
    args = parser.parse_args()

    space = build_default_config_space()
    home = sample_stratified_home_set(
        space, random.Random(int(args.seed)), size=int(args.home_set_size)
    )
    print("[hbl_ceiling] homeset:", flush=True)
    for cfg in home:
        print(
            f"  {cfg.label}: buy={cfg.num_zi_buy} sell={cfg.num_zi_sell} "
            f"t={cfg.timesteps} q_max={cfg.q_max} shade={cfg.shade_range} "
            f"arrival={cfg.arrival_rate}",
            flush=True,
        )

    result = evaluate_batch_with_zi_benchmarks(
        strategy_fn=_dummy_strategy,
        batch_configs=home,
        episodes_per_config=int(args.episodes_per_config),
        step_index=0,
        base_seed=int(args.seed),
        reward_mode="advantage",
        max_events_per_episode=0,
        strategy_output_mode="shade",
        strategy_type="lob",
        include_batch_data=True,
        q_reward_mode="zi_advantage",
        special_agent_kind="hbl",
        hbl_l_param=int(args.hbl_l),
    )

    surplus = mean_llm_surplus_from_batch_data(result.batch_data or {})
    comparison = result.zi_benchmark_comparison or {}
    summary: Dict[str, Any] = {
        "agent": "hbl",
        "hbl_l": int(args.hbl_l),
        "seed": int(args.seed),
        "episodes_per_config": int(args.episodes_per_config),
        "home_set": [cfg.observable_state() for cfg in home],
        "mean_advantage": float(result.reward),
        "advantage_variance": float(result.reward_variance),
        "mean_surplus": float(surplus),
        "per_config_advantage": dict(result.per_config_rewards or {}),
        "zi_beat": f"{comparison.get('beat_count', 0)}/{comparison.get('total_benchmarks', 0)}",
        "zi_beat_fraction": float(comparison.get("beat_fraction", 0.0)),
        "zi_benchmark_comparison": comparison,
    }

    print(
        f"[hbl_ceiling] advantage={summary['mean_advantage']:.4f} "
        f"surplus={summary['mean_surplus']:.4f} "
        f"zi_beat={summary['zi_beat']}",
        flush=True,
    )
    for label, adv in (result.per_config_rewards or {}).items():
        print(f"[hbl_ceiling]   {label}: advantage={adv:.4f}", flush=True)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "hbl_homeset_ceiling_summary.json"
        path.write_text(json.dumps(summary, indent=2))
        print(f"[hbl_ceiling] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
