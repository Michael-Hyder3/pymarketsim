"""Score the reported program and the ZI, ZIP and HBL seats on a holdout set.

usage: python reproduce/holdout_eval.py {pv,sp,lob} {ood,id} [--episodes N] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from common import EPISODES, PROGRAMS, REGIMES, RESULTS, holdout_configs, mean_se, run_seat, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("regime", choices=sorted(REGIMES))
    ap.add_argument("holdout", choices=("ood", "id"))
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    spec = REGIMES[args.regime]
    configs = holdout_configs(args.holdout, args.episodes)
    per_episode = {}
    res = run_seat(configs, spec["background"], program=PROGRAMS / f"{args.regime}.py")
    per_episode["Ours"] = dict(res.per_config_rewards)
    baseline = float(res.baseline_surplus)
    for name, seat in spec["seats"].items():
        per_episode[name] = dict(run_seat(configs, spec["background"], seat=seat).per_config_rewards)

    for name, vals in per_episode.items():
        m, se = mean_se(list(vals.values()))
        print(f"{args.regime} {args.holdout} {name:5s} {m:9.2f} +- {se:.2f}")
    print(f"clearing baseline B = {baseline:.2f}")
    write_json(
        args.out / f"{args.regime}_{args.holdout}.json",
        {"regime": args.regime, "holdout": args.holdout, "n": len(configs),
         "baseline_B": baseline, "per_episode": per_episode},
    )


if __name__ == "__main__":
    main()
