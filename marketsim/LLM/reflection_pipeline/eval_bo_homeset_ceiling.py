#!/usr/bin/env python3
"""Per-config fixed-shade BO oracle on the stratified homeset-3 (seed 123).

Trains s*_c on each homeset market (grid over that market's shade_range,
train-only seeds), then freezes those shades and evaluates with the same
zi_advantage stack as HBL / ZIP / PV ceilings.

Example:
  .venv/bin/python -m marketsim.LLM.reflection_pipeline.eval_bo_homeset_ceiling \\
    --episodes-per-config 100 --bo-runs 500 --seed 123 \\
    --out-dir .../runs/bo_homeset3_ceiling_...
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
from marketsim.LLM.reflection_pipeline.run_holdout import (
    BO_TRAIN_STEP_INDEX,
    _train_bo_fixed_shade_per_config,
)
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
    evaluate_batch_with_zi_benchmarks,
    mean_llm_surplus_from_batch_data,
)


def _dummy_strategy(*_args: Any, **_kwargs: Any) -> float:
    """Unused when special_agent_kind='bo' with frozen fixed shade."""
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--home-set-size", type=int, default=3)
    parser.add_argument("--episodes-per-config", type=int, default=100)
    parser.add_argument(
        "--bo-runs",
        type=int,
        default=500,
        help="Approx train episodes per market (split equally across 21 grid points).",
    )
    parser.add_argument(
        "--bo-train-step-index",
        type=int,
        default=BO_TRAIN_STEP_INDEX,
        help="Train seed namespace (must differ from eval step_index=0).",
    )
    parser.add_argument(
        "--eval-step-index",
        type=int,
        default=0,
        help="Eval seed namespace (matches HBL/ZIP/PV ceiling eval).",
    )
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    if int(args.bo_train_step_index) == int(args.eval_step_index):
        raise SystemExit(
            "--bo-train-step-index must differ from --eval-step-index "
            "(train/eval seed separation)."
        )

    space = build_default_config_space()
    home = sample_stratified_home_set(
        space, random.Random(int(args.seed)), size=int(args.home_set_size)
    )
    print("[bo_ceiling] homeset:", flush=True)
    for cfg in home:
        print(
            f"  {cfg.label}: buy={cfg.num_zi_buy} sell={cfg.num_zi_sell} "
            f"t={cfg.timesteps} q_max={cfg.q_max} shade={cfg.shade_range} "
            f"arrival={cfg.arrival_rate}",
            flush=True,
        )

    print(
        f"[bo_ceiling] training per-config fixed-shade oracle "
        f"(~{int(args.bo_runs)} eps/config)...",
        flush=True,
    )
    train = _train_bo_fixed_shade_per_config(
        holdout_configs=home,
        runs_per_config=int(args.bo_runs),
        base_seed=int(args.seed),
        train_step_index=int(args.bo_train_step_index),
    )
    s_map = dict(train.s_star_by_label)
    print(
        "[bo_ceiling] frozen s*: "
        + ", ".join(f"{lab}={s:.6f}" for lab, s in s_map.items()),
        flush=True,
    )

    result = evaluate_batch_with_zi_benchmarks(
        strategy_fn=_dummy_strategy,
        batch_configs=home,
        episodes_per_config=int(args.episodes_per_config),
        step_index=int(args.eval_step_index),
        base_seed=int(args.seed),
        reward_mode="advantage",
        max_events_per_episode=0,
        strategy_output_mode="shade",
        strategy_type="lob",
        include_batch_data=True,
        q_reward_mode="zi_advantage",
        special_agent_kind="bo",
        bo_fixed_x_by_label=s_map,
    )

    surplus = mean_llm_surplus_from_batch_data(result.batch_data or {})
    comparison = result.zi_benchmark_comparison or {}
    summary: Dict[str, Any] = {
        "agent": "bo_fixed_shade_per_config",
        "seed": int(args.seed),
        "episodes_per_config": int(args.episodes_per_config),
        "bo_runs": int(args.bo_runs),
        "bo_train_step_index": int(args.bo_train_step_index),
        "eval_step_index": int(args.eval_step_index),
        "home_set": [cfg.observable_state() for cfg in home],
        "trained_bo_x_by_label": s_map,
        "train_mean_surplus_by_label": dict(train.train_mean_surplus_by_label),
        "search_bounds_by_label": {
            k: list(v) for k, v in train.search_bounds_by_label.items()
        },
        "mean_advantage": float(result.reward),
        "advantage_variance": float(result.reward_variance),
        "mean_surplus": float(surplus),
        "per_config_advantage": dict(result.per_config_rewards or {}),
        "zi_beat": f"{comparison.get('beat_count', 0)}/{comparison.get('total_benchmarks', 0)}",
        "zi_beat_fraction": float(comparison.get("beat_fraction", 0.0)),
        "zi_benchmark_comparison": comparison,
    }

    print(
        f"[bo_ceiling] advantage={summary['mean_advantage']:.4f} "
        f"surplus={summary['mean_surplus']:.4f} "
        f"zi_beat={summary['zi_beat']}",
        flush=True,
    )
    for label, adv in (result.per_config_rewards or {}).items():
        s_star = s_map.get(label, float("nan"))
        print(
            f"[bo_ceiling]   {label}: advantage={adv:.4f} s*={s_star:.6f}",
            flush=True,
        )

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "bo_homeset_ceiling_summary.json"
        path.write_text(json.dumps(summary, indent=2))
        print(f"[bo_ceiling] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
