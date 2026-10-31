"""Rerun the first N holdout episodes and compare them episode by episode with the released results.

usage: python reproduce/check_subset.py [--episodes N]
"""
from __future__ import annotations

import argparse
import json

from common import PROGRAMS, REGIMES, RESULTS, holdout_configs, run_seat
from best_constant import constant_seat


def compare(new, old, tag):
    worst = max(abs(v - old[k]) for k, v in new.items())
    print(f"{tag:16s} {len(new)} episodes, max |diff| = {worst:.3g}")
    return worst < 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    ok = True
    for regime, spec in REGIMES.items():
        cs = json.loads((RESULTS / f"cs_{regime}.json").read_text())
        for h in ("ood", "id"):
            old = json.loads((RESULTS / f"{regime}_{h}.json").read_text())["per_episode"]
            configs = holdout_configs(h, args.episodes)
            bg = spec["background"]
            new = {"Ours": run_seat(configs, bg, program=PROGRAMS / f"{regime}.py").per_config_rewards}
            for name, seat in spec["seats"].items():
                new[name] = run_seat(configs, bg, seat=seat).per_config_rewards
            for name, vals in new.items():
                ok &= compare(vals, old[name], f"{regime} {h} {name}")
            vals = run_seat(configs, bg, seat=constant_seat(configs, cs["best"])).per_config_rewards
            ok &= compare(vals, cs["per_episode"][h], f"{regime} {h} C*")
    print("all episodes match" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
