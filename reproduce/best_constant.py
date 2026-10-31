"""Select the best constant shade C* on fresh training markets, then score it on both holdouts.

usage: python reproduce/best_constant.py {pv,sp,lob} [--episodes N] [--out DIR]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import EPISODES, REGIMES, RESULTS, holdout_configs, run_seat, write_json
from marketsim.LLM.reflection_pipeline.config import (
    build_id_holdout_config_space,
    sample_id_holdout_episode_configs,
)
from marketsim.LLM.reflection_pipeline.run_holdout import BO_TRAIN_STEP_INDEX

GRID = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.02, 0.03, 0.05]
N_TRAIN = 2000
TRAIN_SAMPLE_SEED = 777
TRAIN_BASE_SEED = 24680


def constant_seat(configs, shade):
    return {
        "special_agent_kind": "bo",
        "bo_optimizer_strategy": "ucb",
        "bo_action_mode": "fixed_shade",
        "bo_fixed_x_by_label": {c.label: shade for c in configs},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("regime", choices=sorted(REGIMES))
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()
    bg = REGIMES[args.regime]["background"]

    train = sample_id_holdout_episode_configs(
        N_TRAIN, random.Random(TRAIN_SAMPLE_SEED), space=build_id_holdout_config_space()
    )
    for c in train:
        c.label = "train_" + c.label

    curve = {}
    for shade in GRID:
        res = run_seat(train, bg, seat=constant_seat(train, shade),
                       step=BO_TRAIN_STEP_INDEX, base_seed=TRAIN_BASE_SEED)
        per = res.per_config_rewards
        curve[shade] = sum(per.values()) / len(per)
        print(f"{args.regime} train shade={shade}: {curve[shade]:.2f}")
    best = max(curve, key=curve.get)
    print(f"{args.regime} C* = {best}")

    holdout = {}
    for h in ("ood", "id"):
        configs = holdout_configs(h, args.episodes)
        holdout[h] = dict(run_seat(configs, bg, seat=constant_seat(configs, best)).per_config_rewards)
    write_json(
        args.out / f"cs_{args.regime}.json",
        {"regime": args.regime, "grid": GRID, "train_curve": curve, "best": best, "per_episode": holdout},
    )


if __name__ == "__main__":
    main()
