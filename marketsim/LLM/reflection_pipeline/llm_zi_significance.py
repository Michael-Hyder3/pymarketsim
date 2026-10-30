"""
Paired significance tests: LLM strategies vs ZI benchmarks.

For each market config, runs each LLM strategy once and each unique ZI shade
once on the shared episode grid (identical seeds / setup as zi_benchmark_eval).
Per-episode differences (llm_surplus - zi_surplus) feed paired t-tests to check
whether 50 episodes are enough to detect LLM vs ZI differences.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from marketsim.LLM.reflection_pipeline.config import (
    DEFAULT_ZI_BENCHMARK_SHADE_RANGES,
    ReflectionMarketConfig,
    build_default_config_pool,
)
from marketsim.LLM.reflection_pipeline.evaluation import _build_episode_jobs
from marketsim.LLM.reflection_pipeline.metrics import mean_and_variance
from marketsim.LLM.reflection_pipeline.run_holdout import (
    infer_strategy_type,
    load_strategy_from_file,
)
from marketsim.LLM.reflection_pipeline.zi_pairwise_power import (
    analyze_paired_differences,
    run_shade_episode_runs,
    shade_range_key,
    unique_shade_ranges,
    write_power_study_report,
)


def _run_llm_episodes(
    *,
    config: ReflectionMarketConfig,
    strategy_fn: Callable,
    strategy_type: str,
    episodes_per_config: int,
    base_seed: int,
    step_index: int = 0,
    strategy_output_mode: str = "x_cap",
    parallel_workers: int = 4,
    lob_input_mode: str = "raw",
) -> List[Dict[str, Any]]:
    """Run LLM once per episode on the shared seed grid; return per-episode rows."""
    from marketsim.LLM.episode_runner import run_episode

    jobs = _build_episode_jobs(
        batch_configs=[config],
        episodes_per_config=episodes_per_config,
        step_index=step_index,
        base_seed=base_seed,
        episode_seeds=None,
        episode_private_values=None,
    )
    episode_kwargs = dict(config.to_run_episode_kwargs())

    def _one(job):
        kwargs = dict(episode_kwargs)
        if job.episode_private_values is not None:
            kwargs["llm_private_values"] = job.episode_private_values
        result = run_episode(
            llm_strategy_func=strategy_fn,
            random_seed=job.seed,
            capture_detail=False,
            llm_action_mode=strategy_output_mode,
            strategy_type=strategy_type,
            special_agent_kind="llm",
            lob_input_mode=lob_input_mode,
            **kwargs,
        )
        return {
            "episode_index": int(job.episode_index),
            "seed": int(job.seed),
            "surplus": float(result.get("llm_surplus", 0.0)),
        }

    workers = max(1, min(int(parallel_workers), len(jobs), os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_one, jobs))
    rows.sort(key=lambda item: int(item["episode_index"]))
    return rows


def paired_points_llm_vs_zi(
    *,
    llm_rows: Sequence[Dict[str, Any]],
    zi_rows: Sequence[Dict[str, Any]],
) -> List[float]:
    by_llm = {int(row["episode_index"]): float(row["surplus"]) for row in llm_rows}
    by_zi = {int(row["episode_index"]): float(row["surplus"]) for row in zi_rows}
    indices = sorted(set(by_llm) & set(by_zi))
    return [by_llm[i] - by_zi[i] for i in indices]


def evaluate_strategy_vs_zi(
    *,
    strategy_label: str,
    strategy_path: Path,
    configs: Sequence[ReflectionMarketConfig],
    shade_ranges: Sequence[Sequence[float]],
    episodes_per_config: int,
    base_seed: int = 123,
    step_index: int = 0,
    parallel_workers: int = 4,
    strategy_output_mode: str = "x_cap",
    bootstrap_samples: int = 200,
    lob_input_mode: str = "raw",
) -> Dict[str, Any]:
    strategy_fn = load_strategy_from_file(strategy_path)
    strategy_type = infer_strategy_type(strategy_fn)
    unique_shades = unique_shade_ranges(shade_ranges)
    comparisons: List[Dict[str, Any]] = []

    for config in configs:
        print(
            f"[llm_zi_significance] {strategy_label} | {config.label} | "
            f"LLM + {len(unique_shades)} ZI shades × {episodes_per_config} eps...",
            flush=True,
        )
        llm_rows = _run_llm_episodes(
            config=config,
            strategy_fn=strategy_fn,
            strategy_type=strategy_type,
            episodes_per_config=episodes_per_config,
            base_seed=base_seed,
            step_index=step_index,
            strategy_output_mode=strategy_output_mode,
            parallel_workers=parallel_workers,
            lob_input_mode=lob_input_mode,
        )
        zi_runs = run_shade_episode_runs(
            config=config,
            shade_ranges=unique_shades,
            episodes_per_config=episodes_per_config,
            base_seed=base_seed,
            step_index=step_index,
            parallel_workers=parallel_workers,
            strategy_type=strategy_type,
        )
        llm_values = [float(row["surplus"]) for row in llm_rows]
        llm_mean, llm_var = mean_and_variance(llm_values)
        llm_std = math.sqrt(llm_var) if llm_var > 0.0 else 0.0

        for shade in unique_shades:
            key = shade_range_key(shade)
            zi_rows = zi_runs[key]
            differences = paired_points_llm_vs_zi(llm_rows=llm_rows, zi_rows=zi_rows)
            analysis = analyze_paired_differences(
                differences,
                bootstrap_samples=bootstrap_samples,
            )
            zi_values = [float(row["surplus"]) for row in zi_rows]
            zi_mean, zi_var = mean_and_variance(zi_values)
            zi_std = math.sqrt(zi_var) if zi_var > 0.0 else 0.0
            comparisons.append(
                {
                    "strategy_label": strategy_label,
                    "strategy_path": str(strategy_path),
                    "strategy_type": strategy_type,
                    "config_label": config.label,
                    "shade_range": list(shade),
                    "shade_hi": float(shade[1]) if len(shade) > 1 else float(shade[0]),
                    "n_episodes": len(differences),
                    "mean_surplus_llm": float(llm_mean),
                    "std_surplus_llm": float(llm_std),
                    "mean_surplus_zi": float(zi_mean),
                    "std_surplus_zi": float(zi_std),
                    "mean_difference": analysis["mean_difference"],
                    "std_difference": analysis["std_difference"],
                    "se_difference": analysis["se_difference"],
                    "paired_t_statistic": analysis["paired_t_statistic"],
                    "paired_t_pvalue": analysis["paired_t_pvalue"],
                    "significant_at_05": (
                        analysis["paired_t_pvalue"] == analysis["paired_t_pvalue"]
                        and float(analysis["paired_t_pvalue"]) < 0.05
                    ),
                    "llm_beats_zi_mean": float(llm_mean) > float(zi_mean),
                    "n_for_power_80": analysis["n_for_power_80"],
                }
            )

    n_sig = sum(1 for row in comparisons if row["significant_at_05"])
    n_beat = sum(1 for row in comparisons if row["llm_beats_zi_mean"])
    return {
        "strategy_label": strategy_label,
        "strategy_path": str(strategy_path),
        "n_comparisons": len(comparisons),
        "n_significant_at_05": n_sig,
        "n_llm_beats_zi_mean": n_beat,
        "comparisons": comparisons,
    }


def write_significance_tables(
    report: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for strategy_block in report.get("strategies", []):
        rows.extend(strategy_block.get("comparisons", []))

    csv_path = output_dir / "llm_zi_significance_ttests.csv"
    headers = [
        "strategy_label",
        "config_label",
        "shade_hi",
        "episodes",
        "mean_llm",
        "std_llm",
        "mean_zi",
        "std_zi",
        "mean_diff",
        "std_diff",
        "se_diff",
        "paired_t",
        "paired_p",
        "significant_at_05",
        "llm_beats_zi_mean",
        "n_for_power_80",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "strategy_label": row["strategy_label"],
                    "config_label": row["config_label"],
                    "shade_hi": row["shade_hi"],
                    "episodes": row["n_episodes"],
                    "mean_llm": row["mean_surplus_llm"],
                    "std_llm": row["std_surplus_llm"],
                    "mean_zi": row["mean_surplus_zi"],
                    "std_zi": row["std_surplus_zi"],
                    "mean_diff": row["mean_difference"],
                    "std_diff": row["std_difference"],
                    "se_diff": row["se_difference"],
                    "paired_t": row["paired_t_statistic"],
                    "paired_p": row["paired_t_pvalue"],
                    "significant_at_05": row["significant_at_05"],
                    "llm_beats_zi_mean": row["llm_beats_zi_mean"],
                    "n_for_power_80": row["n_for_power_80"],
                }
            )

    md_path = output_dir / "llm_zi_significance_ttests.md"

    def fmt(x: Any) -> str:
        try:
            val = float(x)
        except (TypeError, ValueError):
            return str(x)
        if val != val:
            return "nan"
        return f"{val:.6g}"

    lines = [
        "| strategy | config | shade hi | eps | mean LLM | std LLM | mean ZI | std ZI | mean diff | std diff | se diff | paired t | paired p | sig@0.05 | LLM>ZI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy_label']} | {row['config_label']} | {fmt(row['shade_hi'])} | "
            f"{row['n_episodes']} | {fmt(row['mean_surplus_llm'])} | {fmt(row['std_surplus_llm'])} | "
            f"{fmt(row['mean_surplus_zi'])} | {fmt(row['std_surplus_zi'])} | "
            f"{fmt(row['mean_difference'])} | {fmt(row['std_difference'])} | "
            f"{fmt(row['se_difference'])} | "
            f"{fmt(row['paired_t_statistic'])} | {fmt(row['paired_t_pvalue'])} | "
            f"{str(bool(row['significant_at_05'])).lower()} | "
            f"{str(bool(row['llm_beats_zi_mean'])).lower()} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired LLM vs ZI significance study on shared episode seeds "
            "(tests whether 50 episodes detect LLM–ZI differences)."
        )
    )
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Strategy as label=/path/to/agent_strategy.py (repeatable)",
    )
    parser.add_argument("--episodes-per-config", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--parallel-workers", type=int, default=8)
    parser.add_argument("--strategy-output-mode", type=str, default="x_cap")
    parser.add_argument(
        "--lob-input-mode",
        type=str,
        default="raw",
        choices=["raw", "features"],
        help="LOB observation mode (features required for lob_features strategies)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="marketsim/LLM/reflection_pipeline/runs/llm_zi_significance",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One config, first 2 ZI shades, 2 episodes",
    )
    return parser.parse_args(argv)


def _default_best_strategies() -> List[tuple[str, Path]]:
    root = Path(
        "marketsim/LLM/reflection_pipeline/runs/PV_zi_adv_critic_on20260629_100818/llm_calls"
    )
    return [
        ("pv_adv_best_beat_step71", root / "iteration_71" / "agent_strategy.py"),
        ("pv_adv_best_reward_step73", root / "iteration_73" / "agent_strategy.py"),
        ("pv_adv_high_surplus_step75", root / "iteration_75" / "agent_strategy.py"),
    ]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configs = build_default_config_pool()
    shade_ranges = list(DEFAULT_ZI_BENCHMARK_SHADE_RANGES)
    episodes = int(args.episodes_per_config)

    strategies: List[tuple[str, Path]] = []
    for item in args.strategy:
        if "=" not in item:
            raise SystemExit(f"--strategy must be LABEL=PATH, got: {item}")
        label, path_str = item.split("=", 1)
        strategies.append((label.strip(), Path(path_str.strip())))
    if not strategies:
        strategies = _default_best_strategies()

    if args.smoke:
        configs = configs[:1]
        shade_ranges = shade_ranges[:2]
        episodes = 2
        strategies = strategies[:1]

    for _, path in strategies:
        if not path.exists():
            raise SystemExit(f"Strategy not found: {path}")

    print(
        f"[llm_zi_significance] {len(strategies)} strateg(ies), "
        f"{len(configs)} config(s), {len(unique_shade_ranges(shade_ranges))} ZI shade(s), "
        f"{episodes} episode(s)...",
        flush=True,
    )

    strategy_reports = []
    for label, path in strategies:
        strategy_reports.append(
            evaluate_strategy_vs_zi(
                strategy_label=label,
                strategy_path=path,
                configs=configs,
                shade_ranges=shade_ranges,
                episodes_per_config=episodes,
                base_seed=int(args.base_seed),
                step_index=int(args.step_index),
                parallel_workers=int(args.parallel_workers),
                strategy_output_mode=str(args.strategy_output_mode),
                bootstrap_samples=int(args.bootstrap_samples),
                lob_input_mode=str(args.lob_input_mode),
            )
        )

    report = {
        "params": {
            "episodes_per_config": episodes,
            "base_seed": int(args.base_seed),
            "step_index": int(args.step_index),
            "parallel_workers": int(args.parallel_workers),
            "strategy_output_mode": str(args.strategy_output_mode),
            "lob_input_mode": str(args.lob_input_mode),
            "config_labels": [cfg.label for cfg in configs],
            "zi_shade_ranges": [list(s) for s in unique_shade_ranges(shade_ranges)],
            "strategies": [
                {"label": label, "path": str(path)} for label, path in strategies
            ],
        },
        "strategies": strategy_reports,
    }

    output_dir = Path(args.output_dir)
    write_power_study_report(report, output_dir / "llm_zi_significance_report.json")
    write_significance_tables(report, output_dir)

    for block in strategy_reports:
        print(
            f"[llm_zi_significance] {block['strategy_label']}: "
            f"{block['n_significant_at_05']}/{block['n_comparisons']} significant, "
            f"{block['n_llm_beats_zi_mean']}/{block['n_comparisons']} LLM>ZI mean",
            flush=True,
        )
    print(f"[llm_zi_significance] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
