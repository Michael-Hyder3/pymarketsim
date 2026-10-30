"""
Pairwise statistical power study for ZI benchmark strategies.

For each market config, runs each unique ZI shade cap once on the shared episode
grid (same conventions as zi_benchmark_eval). Pairwise comparisons are computed
from aligned per-episode surpluses without re-running simulations.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy import stats

from marketsim.LLM.reflection_pipeline.config import (
    DEFAULT_ZI_BENCHMARK_SHADE_RANGES,
    ReflectionMarketConfig,
    build_default_config_pool,
)
from marketsim.LLM.reflection_pipeline.evaluation import _build_episode_jobs
from marketsim.LLM.reflection_pipeline.metrics import mean_and_variance
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
    _zi_episode_worker,
    _zi_process_context,
)


def unique_shade_ranges(
    shade_ranges: Sequence[Sequence[float]],
) -> List[List[float]]:
    """Deduplicate shade ranges while preserving order (matches benchmark grid)."""
    unique: List[List[float]] = []
    seen: set[Tuple[float, float]] = set()
    for item in shade_ranges:
        lo = float(item[0]) if len(item) > 0 else 0.0
        hi = float(item[1]) if len(item) > 1 else float(item[0])
        key = (lo, hi)
        if key in seen:
            continue
        seen.add(key)
        unique.append([lo, hi])
    return unique


def shade_ranges_equal(
    left: Sequence[float],
    right: Sequence[float],
) -> bool:
    lo_a = float(left[0]) if len(left) > 0 else 0.0
    hi_a = float(left[1]) if len(left) > 1 else lo_a
    lo_b = float(right[0]) if len(right) > 0 else 0.0
    hi_b = float(right[1]) if len(right) > 1 else lo_b
    return lo_a == lo_b and hi_a == hi_b


def iter_shade_range_pairs(
    shade_ranges: Sequence[Sequence[float]],
    *,
    include_self_pairs: bool = True,
) -> List[Tuple[List[float], List[float]]]:
    """Unordered shade comparisons; self-pairs first when enabled (sanity checks)."""
    unique = unique_shade_ranges(shade_ranges)
    pairs: List[Tuple[List[float], List[float]]] = []
    if include_self_pairs:
        pairs.extend((list(shade), list(shade)) for shade in unique)
    pairs.extend((list(a), list(b)) for a, b in combinations(unique, 2))
    return pairs


def shade_range_key(shade_range: Sequence[float]) -> str:
    lo = float(shade_range[0]) if len(shade_range) > 0 else 0.0
    hi = float(shade_range[1]) if len(shade_range) > 1 else lo
    return f"[{lo:g},{hi:g}]"


@dataclass
class PairedEpisodePoint:
    """One paired observation: surplus difference on a single shared episode."""

    config_label: str
    shade_range_a: List[float]
    shade_range_b: List[float]
    episode_index: int
    seed: int
    surplus_a: float
    surplus_b: float

    @property
    def difference(self) -> float:
        return float(self.surplus_a) - float(self.surplus_b)


@dataclass
class PairwiseTestResult:
    config_label: str
    shade_range_a: List[float]
    shade_range_b: List[float]
    n_episodes: int
    mean_difference: float
    std_difference: float
    se_difference: float
    paired_t_statistic: float
    paired_t_pvalue: float
    wilcoxon_statistic: Optional[float]
    wilcoxon_pvalue: Optional[float]
    cohens_dz: float
    n_for_power_80: Optional[int]
    is_self_pair: bool = False
    empirical_power: Dict[int, float] = field(default_factory=dict)
    episode_points: List[PairedEpisodePoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        surplus_a_values = [float(point.surplus_a) for point in self.episode_points]
        surplus_b_values = [float(point.surplus_b) for point in self.episode_points]
        mean_surplus_a, var_surplus_a = mean_and_variance(surplus_a_values)
        mean_surplus_b, var_surplus_b = mean_and_variance(surplus_b_values)
        return {
            "config_label": self.config_label,
            "is_self_pair": self.is_self_pair,
            "shade_range_a": list(self.shade_range_a),
            "shade_range_b": list(self.shade_range_b),
            "shade_hi_a": float(self.shade_range_a[1]) if len(self.shade_range_a) > 1 else float(self.shade_range_a[0]),
            "shade_hi_b": float(self.shade_range_b[1]) if len(self.shade_range_b) > 1 else float(self.shade_range_b[0]),
            "n_episodes": self.n_episodes,
            "mean_surplus_a": float(mean_surplus_a),
            "std_surplus_a": math.sqrt(var_surplus_a) if var_surplus_a > 0.0 else 0.0,
            "mean_surplus_b": float(mean_surplus_b),
            "std_surplus_b": math.sqrt(var_surplus_b) if var_surplus_b > 0.0 else 0.0,
            "mean_difference": self.mean_difference,
            "std_difference": self.std_difference,
            "se_difference": self.se_difference,
            "paired_t_statistic": self.paired_t_statistic,
            "paired_t_pvalue": self.paired_t_pvalue,
            "wilcoxon_statistic": self.wilcoxon_statistic,
            "wilcoxon_pvalue": self.wilcoxon_pvalue,
            "cohens_dz": self.cohens_dz,
            "n_for_power_80": self.n_for_power_80,
            "empirical_power": dict(self.empirical_power),
        }


def analyze_paired_differences(
    differences: Sequence[float],
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
    bootstrap_samples: int = 500,
    power_ns: Optional[Sequence[int]] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Run paired tests and sample-size estimates on per-episode differences."""
    values = [float(v) for v in differences]
    n = len(values)
    if n < 2:
        return {
            "n_episodes": n,
            "mean_difference": float(values[0]) if n == 1 else 0.0,
            "std_difference": 0.0,
            "se_difference": 0.0,
            "paired_t_statistic": float("nan"),
            "paired_t_pvalue": float("nan"),
            "wilcoxon_statistic": None,
            "wilcoxon_pvalue": None,
            "cohens_dz": 0.0,
            "n_for_power_80": None,
            "empirical_power": {},
        }

    mean_diff, var_diff = mean_and_variance(values)
    std_diff = math.sqrt(var_diff) if var_diff > 0.0 else 0.0
    se_diff = std_diff / math.sqrt(n) if n > 0 else 0.0
    t_stat, t_pvalue = stats.ttest_1samp(values, 0.0)
    cohens_dz = mean_diff / std_diff if std_diff > 0.0 else 0.0

    wilcoxon_stat = None
    wilcoxon_pvalue = None
    if n >= 5 and any(v != 0.0 for v in values):
        try:
            w_stat, w_pvalue = stats.wilcoxon(values, alternative="two-sided")
            wilcoxon_stat = float(w_stat)
            wilcoxon_pvalue = float(w_pvalue)
        except ValueError:
            pass

    n_for_power = sample_size_for_power(
        std_diff,
        abs(mean_diff) if abs(mean_diff) > 0.0 else None,
        alpha=alpha,
        power=target_power,
    )

    rng = rng or random.Random(0)
    if power_ns is None:
        power_ns = _default_power_ns(n)
    empirical = empirical_power_curve(
        values,
        ns=power_ns,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )

    return {
        "n_episodes": n,
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "se_difference": float(se_diff),
        "paired_t_statistic": float(t_stat),
        "paired_t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cohens_dz": float(cohens_dz),
        "n_for_power_80": n_for_power,
        "empirical_power": empirical,
    }


def sample_size_for_power(
    std_difference: float,
    effect_size: Optional[float],
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> Optional[int]:
    """Analytic paired-sample size from pilot std and detectable mean difference."""
    if effect_size is None or effect_size <= 0.0 or std_difference <= 0.0:
        return None
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_beta = float(stats.norm.ppf(power))
    n = ((z_alpha + z_beta) * std_difference / effect_size) ** 2
    return max(2, int(math.ceil(n)))


def empirical_power_curve(
    differences: Sequence[float],
    *,
    ns: Sequence[int],
    alpha: float = 0.05,
    bootstrap_samples: int = 500,
    rng: random.Random,
) -> Dict[int, float]:
    """Fraction of bootstrap subsamples that reject H0: mean difference = 0."""
    values = list(differences)
    total = len(values)
    if total < 2:
        return {}

    curve: Dict[int, float] = {}
    for n in ns:
        n_eff = int(n)
        if n_eff < 2 or n_eff > total:
            continue
        reject = 0
        for _ in range(bootstrap_samples):
            sample = rng.sample(values, n_eff)
            _, pvalue = stats.ttest_1samp(sample, 0.0)
            if float(pvalue) < alpha:
                reject += 1
        curve[n_eff] = reject / float(bootstrap_samples)
    return curve


def _default_power_ns(n_episodes: int) -> List[int]:
    candidates = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    return [c for c in candidates if c <= n_episodes]


def run_shade_episode_runs(
    *,
    config: ReflectionMarketConfig,
    shade_ranges: Sequence[Sequence[float]],
    episodes_per_config: int,
    base_seed: int,
    step_index: int = 0,
    parallel_workers: int = 1,
    strategy_type: str = "full",
) -> Dict[str, List[Dict[str, Any]]]:
    """Run each unique shade once per config; return surpluses keyed by shade_range_key."""
    unique_shades = unique_shade_ranges(shade_ranges)
    jobs = _build_episode_jobs(
        batch_configs=[config],
        episodes_per_config=episodes_per_config,
        step_index=step_index,
        base_seed=base_seed,
        episode_seeds=None,
        episode_private_values=None,
    )
    episode_kwargs = dict(config.to_run_episode_kwargs())
    payloads: List[Dict[str, Any]] = []
    for shade_index, shade_range in enumerate(unique_shades):
        for job in jobs:
            payloads.append(
                {
                    "shade_key": shade_range_key(shade_range),
                    "shade_range": list(shade_range),
                    "shade_index": shade_index,
                    "episode_index": job.episode_index,
                    "seed": job.seed,
                    "episode_kwargs": (
                        {**episode_kwargs, "llm_private_values": job.episode_private_values}
                        if job.episode_private_values is not None
                        else episode_kwargs
                    ),
                    "strategy_type": strategy_type,
                }
            )

    workers = max(1, min(int(parallel_workers), len(payloads), os.cpu_count() or 1))
    mp_context = _zi_process_context()
    worker_payloads = _shade_payload_to_worker_payloads(payloads)
    raw_results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
        worker_results = list(pool.map(_zi_episode_worker, worker_payloads))
    for payload, result in zip(payloads, worker_results):
        raw_results.append(
            {
                "shade_key": str(payload["shade_key"]),
                "shade_range": list(payload["shade_range"]),
                "episode_index": int(payload["episode_index"]),
                "seed": int(payload["seed"]),
                "surplus": float(result["llm_surplus"]),
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in raw_results:
        grouped.setdefault(item["shade_key"], []).append(item)
    for episodes in grouped.values():
        episodes.sort(key=lambda row: int(row["episode_index"]))
    return grouped


def _shade_payload_to_worker_payloads(
    payloads: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "benchmark_index": int(payload["shade_index"]),
            "shade_range": list(payload["shade_range"]),
            "seed": int(payload["seed"]),
            "episode_kwargs": payload["episode_kwargs"],
            "strategy_type": str(payload["strategy_type"]),
        }
        for payload in payloads
    ]


def paired_points_from_shade_runs(
    *,
    config_label: str,
    shade_range_a: List[float],
    shade_range_b: List[float],
    shade_runs: Dict[str, List[Dict[str, Any]]],
) -> List[PairedEpisodePoint]:
    """Build aligned paired episode points from cached per-shade episode runs."""
    key_a = shade_range_key(shade_range_a)
    key_b = shade_range_key(shade_range_b)
    if key_a not in shade_runs or key_b not in shade_runs:
        raise KeyError(f"missing shade run for comparison {key_a} vs {key_b}")

    by_index_a = {
        int(item["episode_index"]): item for item in shade_runs[key_a]
    }
    by_index_b = {
        int(item["episode_index"]): item for item in shade_runs[key_b]
    }
    episode_indices = sorted(set(by_index_a) & set(by_index_b))
    return [
        PairedEpisodePoint(
            config_label=config_label,
            shade_range_a=list(shade_range_a),
            shade_range_b=list(shade_range_b),
            episode_index=episode_index,
            seed=int(by_index_a[episode_index]["seed"]),
            surplus_a=float(by_index_a[episode_index]["surplus"]),
            surplus_b=float(by_index_b[episode_index]["surplus"]),
        )
        for episode_index in episode_indices
    ]


def run_pairwise_power_study(
    *,
    configs: Sequence[ReflectionMarketConfig],
    shade_ranges: Sequence[Sequence[float]],
    episodes_per_config: int,
    base_seed: int = 123,
    step_index: int = 0,
    parallel_workers: int = 4,
    alpha: float = 0.05,
    target_power: float = 0.8,
    bootstrap_samples: int = 500,
    power_ns: Optional[Sequence[int]] = None,
    strategy_type: str = "full",
    include_self_pairs: bool = True,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    unique_shades = unique_shade_ranges(shade_ranges)
    pairs = iter_shade_range_pairs(shade_ranges, include_self_pairs=include_self_pairs)
    all_results: List[PairwiseTestResult] = []

    for config in configs:
        shade_runs = run_shade_episode_runs(
            config=config,
            shade_ranges=unique_shades,
            episodes_per_config=episodes_per_config,
            base_seed=base_seed,
            step_index=step_index,
            parallel_workers=parallel_workers,
            strategy_type=strategy_type,
        )
        for shade_a, shade_b in pairs:
            is_self_pair = shade_ranges_equal(shade_a, shade_b)
            points = paired_points_from_shade_runs(
                config_label=config.label,
                shade_range_a=shade_a,
                shade_range_b=shade_b,
                shade_runs=shade_runs,
            )
            differences = [point.difference for point in points]
            analysis = analyze_paired_differences(
                differences,
                alpha=alpha,
                target_power=target_power,
                bootstrap_samples=bootstrap_samples,
                power_ns=power_ns,
                rng=rng,
            )
            all_results.append(
                PairwiseTestResult(
                    config_label=config.label,
                    shade_range_a=list(shade_a),
                    shade_range_b=list(shade_b),
                    is_self_pair=is_self_pair,
                    episode_points=points,
                    **analysis,
                )
            )

    conservative_n = _conservative_episodes_recommendation(
        [item for item in all_results if not item.is_self_pair]
    )
    return {
        "params": {
            "episodes_per_config": episodes_per_config,
            "base_seed": base_seed,
            "step_index": step_index,
            "alpha": alpha,
            "target_power": target_power,
            "bootstrap_samples": bootstrap_samples,
            "parallel_workers": parallel_workers,
            "include_self_pairs": include_self_pairs,
            "unique_shade_ranges": unique_shades,
            "config_labels": [cfg.label for cfg in configs],
            "shade_range_pairs": [
                {"a": a, "b": b, "is_self_pair": shade_ranges_equal(a, b)}
                for a, b in pairs
            ],
        },
        "recommended_episodes_per_config": conservative_n,
        "pairwise_results": [item.to_dict() for item in all_results],
    }


def _conservative_episodes_recommendation(
    results: Sequence[PairwiseTestResult],
) -> Optional[int]:
    """Max analytic n_for_power_80 across cells; fallback to empirical n at target power."""
    analytic: List[int] = []
    empirical: List[int] = []
    for item in results:
        if item.n_for_power_80 is not None:
            analytic.append(int(item.n_for_power_80))
        for n, power in sorted(item.empirical_power.items()):
            if power >= 0.8:
                empirical.append(int(n))
                break
    if analytic:
        return max(analytic)
    if empirical:
        return max(empirical)
    return None


def write_power_study_report(report: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pairwise ZI power study: run each shade once, then compare aligned "
            "per-episode surpluses."
        )
    )
    parser.add_argument("--episodes-per-config", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument(
        "--output",
        type=str,
        default="marketsim/LLM/reflection_pipeline/runs/zi_pairwise_power_report.json",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One config, one shade pair, two episodes (quick sanity check)",
    )
    parser.add_argument(
        "--no-self-pairs",
        action="store_true",
        help="Skip shade-vs-itself comparisons (sanity checks expecting zero difference)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configs = build_default_config_pool()
    shade_ranges = DEFAULT_ZI_BENCHMARK_SHADE_RANGES

    episodes = 2 if args.smoke else int(args.episodes_per_config)
    if args.smoke:
        configs = configs[:1]
        shade_ranges = shade_ranges[:2]

    unique_count = len(unique_shade_ranges(shade_ranges))
    pair_count = len(
        iter_shade_range_pairs(shade_ranges, include_self_pairs=not args.no_self_pairs)
    )
    print(
        f"[zi_pairwise_power] {len(configs)} config(s), "
        f"{unique_count} unique shade(s), {pair_count} comparison(s), "
        f"{episodes} episode(s) per shade...",
        flush=True,
    )

    report = run_pairwise_power_study(
        configs=configs,
        shade_ranges=shade_ranges,
        episodes_per_config=episodes,
        base_seed=int(args.base_seed),
        step_index=int(args.step_index),
        parallel_workers=int(args.parallel_workers),
        alpha=float(args.alpha),
        target_power=float(args.target_power),
        bootstrap_samples=int(args.bootstrap_samples),
        include_self_pairs=not args.no_self_pairs,
    )

    output_path = Path(args.output)
    write_power_study_report(report, output_path)
    rec = report.get("recommended_episodes_per_config")
    print(
        f"[zi_pairwise_power] done — recommended episodes_per_config={rec} "
        f"report={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
