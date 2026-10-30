"""
Phase 1: LLM-call futility / sufficiency analysis.

Analyzes finished reflection_pipeline_summary.json runs (or launches multi-seed
recipe replicates) to decide whether a frozen pipeline recipe is still improving
or has plateaued below a pre-registered success criterion.

Success is pre-registered. Learning curves track best-so-far metrics every N
generator steps (default 25). A futility rule stops extending budget when
replicates show no improvement and no successes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class SuccessCriterion:
    """Pre-registered 'positive result' definition for a pipeline run."""

    min_zi_beat_count: Optional[int] = None
    min_reward: Optional[float] = None
    require_positive_reward: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_zi_beat_count": self.min_zi_beat_count,
            "min_reward": self.min_reward,
            "require_positive_reward": self.require_positive_reward,
        }

    def is_met(self, *, reward: Optional[float], beat_count: Optional[int]) -> bool:
        if self.require_positive_reward:
            if reward is None or float(reward) <= 0.0:
                return False
        if self.min_reward is not None:
            if reward is None or float(reward) < float(self.min_reward):
                return False
        if self.min_zi_beat_count is not None:
            if beat_count is None or int(beat_count) < int(self.min_zi_beat_count):
                return False
        if (
            self.min_zi_beat_count is None
            and self.min_reward is None
            and not self.require_positive_reward
        ):
            # Default: any positive reward is success when nothing else is set.
            return reward is not None and float(reward) > 0.0
        return True


def _step_reward(step: Dict[str, Any]) -> Optional[float]:
    if not step.get("success", True) and "reward" not in step:
        return None
    if "reward" not in step:
        return None
    return float(step["reward"])


def _step_beat_count(step: Dict[str, Any]) -> Optional[int]:
    comparison = step.get("zi_benchmark_comparison")
    if not isinstance(comparison, dict):
        return None
    if "beat_count" not in comparison:
        return None
    return int(comparison["beat_count"])


def build_learning_curve(
    summary_steps: Sequence[Dict[str, Any]],
    *,
    checkpoint_every: int = 25,
    success: Optional[SuccessCriterion] = None,
) -> Dict[str, Any]:
    """Build best-so-far curves and checkpoint snapshots from a run summary."""
    success = success or SuccessCriterion()
    checkpoint_every = max(1, int(checkpoint_every))

    best_reward = -float("inf")
    best_beat = -1
    points: List[Dict[str, Any]] = []
    successes_at_step: List[int] = []

    successful_steps = [
        step for step in summary_steps if step.get("success", True) and "step" in step
    ]
    for index, step in enumerate(successful_steps, start=1):
        reward = _step_reward(step)
        beat = _step_beat_count(step)
        if reward is not None and reward > best_reward:
            best_reward = reward
        if beat is not None and beat > best_beat:
            best_beat = beat
        met = success.is_met(
            reward=None if best_reward == -float("inf") else best_reward,
            beat_count=None if best_beat < 0 else best_beat,
        )
        if met:
            successes_at_step.append(int(step.get("step", index)))
        if index % checkpoint_every == 0 or index == len(successful_steps):
            points.append(
                {
                    "generator_calls": index,
                    "step": int(step.get("step", index)),
                    "best_reward_so_far": (
                        None if best_reward == -float("inf") else float(best_reward)
                    ),
                    "best_zi_beat_count_so_far": (
                        None if best_beat < 0 else int(best_beat)
                    ),
                    "success_so_far": met,
                }
            )

    return {
        "n_generator_calls": len(successful_steps),
        "checkpoint_every": checkpoint_every,
        "first_success_step": successes_at_step[0] if successes_at_step else None,
        "success_by_end": bool(successes_at_step),
        "checkpoints": points,
    }


def improvement_slope(
    checkpoints: Sequence[Dict[str, Any]],
    *,
    window: int = 20,
    metric: str = "best_reward_so_far",
) -> Optional[float]:
    """Simple finite-difference slope of best-so-far metric over the last window."""
    values = [
        (int(point["generator_calls"]), point.get(metric))
        for point in checkpoints
        if point.get(metric) is not None
    ]
    if len(values) < 2:
        return None
    # Use last points whose generator_calls span roughly `window` calls.
    end_x, end_y = values[-1]
    start = values[0]
    for item in reversed(values[:-1]):
        if end_x - item[0] >= window:
            start = item
            break
        start = item
    start_x, start_y = start
    if end_x == start_x:
        return 0.0
    return (float(end_y) - float(start_y)) / float(end_x - start_x)


def analyze_run_summary(
    summary_path: Path,
    *,
    checkpoint_every: int = 25,
    success: Optional[SuccessCriterion] = None,
    slope_window: int = 20,
) -> Dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    params = data.get("params", {})
    curve = build_learning_curve(
        data.get("summary", []),
        checkpoint_every=checkpoint_every,
        success=success,
    )
    slope = improvement_slope(
        curve["checkpoints"],
        window=slope_window,
        metric="best_reward_so_far",
    )
    beat_slope = improvement_slope(
        curve["checkpoints"],
        window=slope_window,
        metric="best_zi_beat_count_so_far",
    )
    return {
        "summary_path": str(summary_path),
        "run_dir": data.get("run_dir"),
        "params": {
            "seed": params.get("seed"),
            "steps": params.get("steps"),
            "episodes_per_config": params.get("episodes_per_config"),
            "q_reward_mode": params.get("q_reward_mode"),
            "parent_selection_mode": params.get("parent_selection_mode"),
            "resolved_parent_selection_mode": params.get(
                "resolved_parent_selection_mode"
            ),
            "recipe_id": params.get("recipe_id"),
        },
        "learning_curve": curve,
        "reward_slope_last_window": slope,
        "zi_beat_slope_last_window": beat_slope,
        "success_criterion": (success or SuccessCriterion()).to_dict(),
    }


def futility_decision(
    replicate_analyses: Sequence[Dict[str, Any]],
    *,
    slope_epsilon: float = 1e-9,
) -> Dict[str, Any]:
    """
    Aggregate replicate analyses into continue / need_more_time / futile.

    - futile: 0 successes and all reward slopes ~0 or missing
    - need_more_time: 0 successes but at least one positive slope
    - continue: at least one success, or mixed improving signals
    """
    n = len(replicate_analyses)
    n_success = sum(
        1
        for item in replicate_analyses
        if item.get("learning_curve", {}).get("success_by_end")
    )
    slopes = [
        item.get("reward_slope_last_window")
        for item in replicate_analyses
        if item.get("reward_slope_last_window") is not None
    ]
    any_improving = any(float(s) > slope_epsilon for s in slopes)
    all_flat = bool(slopes) and all(abs(float(s)) <= slope_epsilon for s in slopes)

    if n_success > 0:
        decision = "continue"
        rationale = f"{n_success}/{n} replicates met the success criterion."
    elif any_improving:
        decision = "need_more_time"
        rationale = (
            f"0/{n} successes but at least one replicate still has positive "
            "best-so-far reward slope."
        )
    elif n == 0:
        decision = "insufficient_data"
        rationale = "No replicate analyses provided."
    else:
        decision = "futile"
        rationale = (
            f"0/{n} successes and no positive reward slope "
            f"(flat={all_flat}, slopes_observed={len(slopes)})."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "n_replicates": n,
        "n_successes": n_success,
        "any_improving": any_improving,
        "all_flat": all_flat,
    }


def analyze_summaries(
    summary_paths: Sequence[Path],
    *,
    checkpoint_every: int = 25,
    success: Optional[SuccessCriterion] = None,
    slope_window: int = 20,
) -> Dict[str, Any]:
    analyses = [
        analyze_run_summary(
            path,
            checkpoint_every=checkpoint_every,
            success=success,
            slope_window=slope_window,
        )
        for path in summary_paths
    ]
    decision = futility_decision(analyses)
    return {
        "success_criterion": (success or SuccessCriterion()).to_dict(),
        "checkpoint_every": checkpoint_every,
        "slope_window": slope_window,
        "replicates": analyses,
        "futility": decision,
    }


def launch_recipe_replicate(
    *,
    recipe_id: str,
    seed: int,
    output_root: Path,
    steps: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Spawn one pipeline process for a frozen recipe + seed."""
    from marketsim.LLM.reflection_pipeline.recipes import get_recipe

    recipe = get_recipe(recipe_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{recipe_id}_seed{seed}_{stamp}"
    cmd = [
        sys.executable,
        "-m",
        "marketsim.LLM.reflection_pipeline",
        "--recipe",
        recipe_id,
        "--seed",
        str(int(seed)),
        "--run-dir",
        str(run_dir),
    ]
    if steps is not None:
        cmd.extend(["--steps", str(int(steps))])
    elif recipe.steps:
        cmd.extend(["--steps", str(int(recipe.steps))])

    info = {"command": cmd, "run_dir": str(run_dir), "seed": int(seed), "recipe_id": recipe_id}
    if dry_run:
        info["launched"] = False
        return info
    subprocess.Popen(cmd)
    info["launched"] = True
    return info


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pipeline learning curves for LLM-call futility / sufficiency."
        )
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Path to reflection_pipeline_summary.json (repeatable)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Generator-call spacing for best-so-far checkpoints",
    )
    parser.add_argument("--slope-window", type=int, default=20)
    parser.add_argument(
        "--min-zi-beats",
        type=int,
        default=None,
        help="Success if best ZI beat_count reaches this value",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help="Success if best-so-far reward reaches this value",
    )
    parser.add_argument(
        "--require-positive-reward",
        action="store_true",
        help="Success requires best-so-far reward > 0",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="marketsim/LLM/reflection_pipeline/runs/futility_analysis.json",
    )
    parser.add_argument(
        "--launch-recipe",
        type=str,
        default=None,
        help="Optional recipe_id to launch multi-seed replicates",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="123,456,789",
        help="Comma-separated seeds for --launch-recipe",
    )
    parser.add_argument(
        "--launch-root",
        type=str,
        default="marketsim/LLM/reflection_pipeline/runs/futility_replicates",
    )
    parser.add_argument("--launch-steps", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print launch commands without starting processes",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    success = SuccessCriterion(
        min_zi_beat_count=args.min_zi_beats,
        min_reward=args.min_reward,
        require_positive_reward=bool(args.require_positive_reward),
    )

    if args.launch_recipe:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        launches = [
            launch_recipe_replicate(
                recipe_id=args.launch_recipe,
                seed=seed,
                output_root=Path(args.launch_root),
                steps=args.launch_steps,
                dry_run=bool(args.dry_run),
            )
            for seed in seeds
        ]
        print(json.dumps({"launches": launches}, indent=2))
        if not args.summary:
            return

    paths = [Path(p) for p in args.summary]
    if not paths:
        raise SystemExit("Provide at least one --summary path (or only --launch-recipe).")

    report = analyze_summaries(
        paths,
        checkpoint_every=int(args.checkpoint_every),
        success=success,
        slope_window=int(args.slope_window),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    decision = report["futility"]
    print(
        f"[futility] decision={decision['decision']} "
        f"({decision['n_successes']}/{decision['n_replicates']} successes) "
        f"— {decision['rationale']}"
    )
    print(f"[futility] wrote {output}")


if __name__ == "__main__":
    main()
