from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Mapping

from marketsim.LLM.reflection_pipeline.config import (
    DEFAULT_HBL_BENCHMARK_L_VALUES,
    DEFAULT_LADDER_PARALLEL_WORKERS,
    DEFAULT_ZI_BENCHMARK_SHADE_RANGES,
    DEFAULT_ZIP_BENCHMARK_SHADE_RANGES,
    ZI_BENCHMARK_PARALLEL_WORKERS,
    ReflectionMarketConfig,
)
from marketsim.LLM.reflection_pipeline.evaluation import (
    BatchEvaluationResult,
    _aggregate_episode_outcomes,
    _build_episode_jobs,
    _run_episode_job,
)
from marketsim.LLM.reflection_pipeline.metrics import (
    mean_and_variance,
    mean_llm_requested_x_cap_from_batch_data,
    mean_llm_shade_from_batch_data,
)
from marketsim.LLM.reflection_pipeline.zi_benchmark_feedback import (
    enrich_zi_comparison_with_benchmark_tests,
)


def _dummy_tracked_strategy(*_args: Any, **_kwargs: Any) -> float:
    return 0.0


def mean_llm_surplus_from_batch_data(batch_data: Dict[str, Any]) -> float:
    values: List[float] = []
    for cfg_data in batch_data.get("configs", []):
        for episode in cfg_data.get("episodes", []):
            values.append(float(episode.get("llm_surplus", 0.0)))
    if not values:
        return 0.0
    mean, _ = mean_and_variance(values)
    return float(mean)


def build_zi_beat_comparison(
    *,
    llm_mean_surplus: float,
    benchmark_results: List[Dict[str, Any]],
    llm_mean_shade: Optional[float] = None,
    llm_mean_requested_x_cap: Optional[float] = None,
) -> Dict[str, Any]:
    comparisons: List[Dict[str, Any]] = []
    beat_count = 0
    for item in benchmark_results:
        beat = llm_mean_surplus > float(item["mean_surplus"])
        if beat:
            beat_count += 1
        entry: Dict[str, Any] = {
            "benchmark_index": int(item["benchmark_index"]),
            "shade_range": list(item.get("shade_range") or []),
            "mean_surplus": float(item["mean_surplus"]),
            "beat": beat,
        }
        if item.get("ladder_mode") is not None:
            entry["ladder_mode"] = str(item["ladder_mode"])
        if item.get("label") is not None:
            entry["label"] = str(item["label"])
        if item.get("hbl_l") is not None:
            entry["hbl_l"] = int(item["hbl_l"])
        comparisons.append(entry)
    total = len(benchmark_results)
    beat_fraction = float(beat_count) / float(total) if total else 0.0
    comparison: Dict[str, Any] = {
        "llm_mean_surplus": float(llm_mean_surplus),
        "beat_count": beat_count,
        "total_benchmarks": total,
        "beat_fraction": beat_fraction,
        "benchmarks": comparisons,
        "beaten_shade_ranges": [
            c["shade_range"] for c in comparisons if c["beat"]
        ],
        "not_beaten_shade_ranges": [
            c["shade_range"] for c in comparisons if not c["beat"]
        ],
    }
    if llm_mean_shade is not None:
        comparison["llm_mean_shade"] = float(llm_mean_shade)
    if llm_mean_requested_x_cap is not None:
        comparison["llm_mean_requested_x_cap"] = float(llm_mean_requested_x_cap)
    return enrich_zi_comparison_with_benchmark_tests(comparison)


def apply_zi_q_reward_from_comparison(
    llm_result: BatchEvaluationResult,
    comparison: Dict[str, Any],
    *,
    q_reward_mode: str,
) -> BatchEvaluationResult:
    beat_fraction = float(comparison["beat_fraction"])
    comparison["beat_fraction_diagnostic"] = beat_fraction
    llm_result.zi_benchmark_comparison = comparison
    if q_reward_mode == "zi_advantage":
        llm_result.scalar_reward = beat_fraction
        return llm_result

    _, beat_variance = mean_and_variance(
        [1.0 if item["beat"] else 0.0 for item in comparison["benchmarks"]]
    )
    llm_result.scalar_reward = float(llm_result.reward)
    llm_result.reward = beat_fraction
    llm_result.reward_variance = float(beat_variance)
    return llm_result


def _zi_episode_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    from marketsim.LLM.episode_runner import run_episode

    tracked_kind = str(payload.get("tracked_agent_kind") or "zi")
    ladder_mode = str(payload.get("ladder_mode") or "fixed_mu")
    episode_kwargs = dict(payload["episode_kwargs"])
    shade_range = list(payload.get("shade_range") or [])
    if tracked_kind in {"zip", "zip_noisy"} and ladder_mode == "standard":
        special_shade: Optional[List[float]] = None
    else:
        special_shade = shade_range
    run_kwargs: Dict[str, Any] = {
        "llm_strategy_func": _dummy_tracked_strategy,
        "random_seed": int(payload["seed"]),
        "capture_detail": False,
        "strategy_type": str(payload["strategy_type"]),
        "special_agent_kind": tracked_kind,
        "special_agent_shade_range": special_shade,
        **episode_kwargs,
    }
    if tracked_kind == "zi":
        run_kwargs["special_agent_shade_sampling"] = "x_cap"
    if tracked_kind in {"hbl", "hbl_fast"} and payload.get("hbl_l") is not None:
        run_kwargs["special_agent_hbl_l"] = int(payload["hbl_l"])
    result = run_episode(**run_kwargs)
    out: Dict[str, Any] = {
        "benchmark_index": int(payload["benchmark_index"]),
        "shade_range": shade_range,
        "llm_surplus": float(result.get("llm_surplus", 0.0)),
        "ladder_mode": ladder_mode,
    }
    if payload.get("label") is not None:
        out["label"] = str(payload["label"])
    if payload.get("hbl_l") is not None:
        out["hbl_l"] = int(payload["hbl_l"])
    return out


def _collect_zi_futures(futures: List[Future]) -> List[Dict[str, Any]]:
    return [future.result() for future in futures]


def _zi_process_context() -> mp.context.BaseContext:
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context("spawn")


def aggregate_zi_episode_results(
    episode_results: List[Dict[str, Any]],
    ladder_specs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    surplus_by_benchmark: Dict[int, List[float]] = defaultdict(list)
    for item in episode_results:
        surplus_by_benchmark[int(item["benchmark_index"])].append(
            float(item["llm_surplus"])
        )

    benchmark_results: List[Dict[str, Any]] = []
    for benchmark_index, spec in enumerate(ladder_specs):
        values = surplus_by_benchmark.get(benchmark_index, [])
        mean_surplus, _ = mean_and_variance(values) if values else (0.0, 0.0)
        entry: Dict[str, Any] = {
            "benchmark_index": benchmark_index,
            "shade_range": list(spec.get("shade_range") or []),
            "mean_surplus": float(mean_surplus),
            "ladder_mode": str(spec.get("ladder_mode") or "zi"),
        }
        if spec.get("label") is not None:
            entry["label"] = str(spec["label"])
        if spec.get("hbl_l") is not None:
            entry["hbl_l"] = int(spec["hbl_l"])
        benchmark_results.append(entry)
    return benchmark_results


def _tracked_ladder_kind(background_agent_kind: str) -> str:
    bg = str(background_agent_kind or "zi").strip().lower()
    if bg in {"zip_noisy", "zip_fixed"}:
        return "zip_noisy"
    if bg == "zip":
        return "zip"
    if bg in {"hbl", "hbl_fast"}:
        return "hbl_fast"
    return "zi"


def _ladder_label(tracked_ladder_kind: str) -> str:
    kind = str(tracked_ladder_kind or "zi").strip().lower()
    if kind in {"zip", "zip_noisy"}:
        return "ZIP"
    if kind in {"hbl", "hbl_fast"}:
        return "HBL"
    return "ZI"


def build_ladder_specs(
    *,
    tracked_ladder_kind: str,
    shade_ranges: Optional[List[List[float]]] = None,
    hbl_l_values: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    kind = str(tracked_ladder_kind or "zi").strip().lower()
    if kind in {"zip", "zip_noisy"}:
        fixed = shade_ranges if shade_ranges is not None else DEFAULT_ZIP_BENCHMARK_SHADE_RANGES
        specs: List[Dict[str, Any]] = [
            {
                "shade_range": list(r),
                "ladder_mode": "fixed_mu",
                "label": f"fixed_mu={float(r[-1]):g}",
            }
            for r in fixed
        ]
        specs.append(
            {
                "shade_range": [],
                "ladder_mode": "standard",
                "label": "standard_zip",
            }
        )
        return specs
    if kind in {"hbl", "hbl_fast"}:
        ls = (
            [int(x) for x in hbl_l_values]
            if hbl_l_values is not None
            else [int(x) for x in DEFAULT_HBL_BENCHMARK_L_VALUES]
        )
        return [
            {
                "shade_range": [],
                "ladder_mode": "hbl_L",
                "label": f"hbl_L={int(L)}",
                "hbl_l": int(L),
            }
            for L in ls
        ]
    fixed = shade_ranges if shade_ranges is not None else DEFAULT_ZI_BENCHMARK_SHADE_RANGES
    return [
        {
            "shade_range": list(r),
            "ladder_mode": "zi",
            "label": f"zi_cap={float(r[-1]):g}",
        }
        for r in fixed
    ]


def _build_zi_cohort_payloads(
    *,
    job,
    episode_kwargs: Dict[str, Any],
    ladder_specs: List[Dict[str, Any]],
    strategy_type: str,
    tracked_agent_kind: str = "zi",
) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for benchmark_index, spec in enumerate(ladder_specs):
        payload: Dict[str, Any] = {
            "benchmark_index": benchmark_index,
            "shade_range": list(spec.get("shade_range") or []),
            "ladder_mode": str(spec.get("ladder_mode") or "zi"),
            "label": spec.get("label"),
            "seed": job.seed,
            "episode_kwargs": episode_kwargs,
            "strategy_type": strategy_type,
            "tracked_agent_kind": tracked_agent_kind,
        }
        if spec.get("hbl_l") is not None:
            payload["hbl_l"] = int(spec["hbl_l"])
        payloads.append(payload)
    return payloads


def evaluate_batch_with_zi_benchmarks(
    *,
    strategy_fn: Callable,
    batch_configs: List[ReflectionMarketConfig],
    episodes_per_config: int,
    step_index: int,
    base_seed: int,
    reward_mode: str = "advantage",
    max_events_per_episode: int = 20,
    strategy_output_mode: str = "x_cap",
    strategy_type: str = "full",
    episode_seeds: Optional[List[List[int]]] = None,
    episode_private_values: Optional[List[List[List[float]]]] = None,
    include_batch_data: bool = True,
    zi_benchmark_shade_ranges: Optional[List[List[float]]] = None,
    parallel_workers: int = ZI_BENCHMARK_PARALLEL_WORKERS,
    q_reward_mode: str = "zi_beat_fraction",
    lob_input_mode: str = "raw",
    special_agent_kind: str = "llm",
    hbl_l_param: int = 6,
    bo_fixed_x: Optional[float] = None,
    bo_fixed_x_by_label: Optional[Mapping[str, float]] = None,
    spread_input_mode: str = "classic",
    spread_omit_fundamental: bool = False,
    background_agent_kind: str = "zi",
    background_zip_fixed_mu: Optional[float] = None,
    background_zip_initial_mu: Optional[float] = None,
    background_zip_zi_warmup_steps: int = 25,
    parallel_episodes: bool = False,
    parallel_episode_workers: Optional[int] = None,
    strategy_code: Optional[str] = None,
    suite_aggregate: str = "mean",
    suite_aggregate_lambda: float = 0.5,
) -> BatchEvaluationResult:
    tracked_ladder_kind = _tracked_ladder_kind(background_agent_kind)
    ladder_shade_ranges = (
        None
        if tracked_ladder_kind in {"hbl", "hbl_fast"}
        else zi_benchmark_shade_ranges
    )
    ladder_specs = build_ladder_specs(
        tracked_ladder_kind=tracked_ladder_kind,
        shade_ranges=ladder_shade_ranges,
    )
    ladder_label = _ladder_label(tracked_ladder_kind)
    jobs = _build_episode_jobs(
        batch_configs=batch_configs,
        episodes_per_config=episodes_per_config,
        step_index=step_index,
        base_seed=base_seed,
        episode_seeds=episode_seeds,
        episode_private_values=episode_private_values,
    )
    requested_workers = max(
        int(parallel_workers),
        int(parallel_episode_workers)
        if parallel_episode_workers is not None
        else 0,
        int(DEFAULT_LADDER_PARALLEL_WORKERS),
    )
    cpu_cap = os.cpu_count() or 1
    if parallel_episodes:
        total_ladder_jobs = max(1, len(jobs) * len(ladder_specs))
        zi_workers = max(1, min(total_ladder_jobs, requested_workers, cpu_cap))
    else:
        zi_workers = max(
            1, min(len(ladder_specs), requested_workers, cpu_cap)
        )

    if parallel_episodes:
        return _evaluate_batch_with_zi_staged(
            strategy_fn=strategy_fn,
            batch_configs=batch_configs,
            episodes_per_config=episodes_per_config,
            step_index=step_index,
            base_seed=base_seed,
            reward_mode=reward_mode,
            max_events_per_episode=max_events_per_episode,
            strategy_output_mode=strategy_output_mode,
            strategy_type=strategy_type,
            episode_seeds=episode_seeds,
            episode_private_values=episode_private_values,
            include_batch_data=include_batch_data,
            ladder_specs=ladder_specs,
            zi_workers=zi_workers,
            q_reward_mode=q_reward_mode,
            lob_input_mode=lob_input_mode,
            special_agent_kind=special_agent_kind,
            hbl_l_param=hbl_l_param,
            bo_fixed_x=bo_fixed_x,
            bo_fixed_x_by_label=bo_fixed_x_by_label,
            spread_input_mode=spread_input_mode,
            spread_omit_fundamental=spread_omit_fundamental,
            background_agent_kind=background_agent_kind,
            background_zip_fixed_mu=background_zip_fixed_mu,
            background_zip_initial_mu=background_zip_initial_mu,
            background_zip_zi_warmup_steps=background_zip_zi_warmup_steps,
            parallel_episode_workers=parallel_episode_workers,
            strategy_code=strategy_code,
            jobs=jobs,
            tracked_ladder_kind=tracked_ladder_kind,
            suite_aggregate=suite_aggregate,
            suite_aggregate_lambda=suite_aggregate_lambda,
        )

    print(
        f"[{q_reward_mode}] {len(jobs)} LLM episodes; each overlaps with "
        f"{len(ladder_specs)} {ladder_label} benchmarks ({zi_workers} process workers)...",
        flush=True,
    )

    llm_outcomes = []
    zi_episode_results: List[Dict[str, Any]] = []
    pending_zi_futures: Optional[List[Future]] = None

    mp_context = _zi_process_context()
    with ProcessPoolExecutor(max_workers=zi_workers, mp_context=mp_context) as zi_pool:
        with ThreadPoolExecutor(max_workers=1) as helper:
            for job in jobs:
                config = batch_configs[job.config_index]
                episode_kwargs = dict(config.to_run_episode_kwargs())
                if job.episode_private_values is not None:
                    episode_kwargs["llm_private_values"] = job.episode_private_values
                episode_kwargs["background_agent_kind"] = background_agent_kind
                if background_zip_fixed_mu is not None:
                    episode_kwargs["background_zip_fixed_mu"] = float(
                        background_zip_fixed_mu
                    )
                if background_zip_initial_mu is not None:
                    episode_kwargs["background_zip_initial_mu"] = float(
                        background_zip_initial_mu
                    )
                episode_kwargs["background_zip_zi_warmup_steps"] = int(
                    background_zip_zi_warmup_steps
                )

                current_zi_futures = [
                    zi_pool.submit(
                        _zi_episode_worker,
                        payload,
                    )
                    for payload in _build_zi_cohort_payloads(
                        job=job,
                        episode_kwargs=episode_kwargs,
                        ladder_specs=ladder_specs,
                        strategy_type=strategy_type,
                        tracked_agent_kind=tracked_ladder_kind,
                    )
                ]

                prev_collect_future = None
                if pending_zi_futures is not None:
                    prev_collect_future = helper.submit(
                        _collect_zi_futures,
                        pending_zi_futures,
                    )

                resolved_bo_x = bo_fixed_x
                if bo_fixed_x_by_label is not None and config.label in bo_fixed_x_by_label:
                    resolved_bo_x = float(bo_fixed_x_by_label[config.label])
                llm_outcomes.append(
                    _run_episode_job(
                        job,
                        config=config,
                        strategy_fn=strategy_fn,
                        reward_mode=reward_mode,
                        max_events_per_episode=max_events_per_episode,
                        strategy_output_mode=strategy_output_mode,
                        strategy_type=strategy_type,
                        include_batch_data=include_batch_data,
                        special_agent_kind=special_agent_kind,
                        special_agent_shade_range=None,
                        bo_optimizer_strategy="ucb",
                        bo_action_mode="fixed_shade",
                        bo_fixed_x=resolved_bo_x,
                        hbl_l_param=hbl_l_param,
                        lob_input_mode=lob_input_mode,
                        spread_input_mode=spread_input_mode,
                        spread_omit_fundamental=spread_omit_fundamental,
                        background_agent_kind=background_agent_kind,
                        background_zip_fixed_mu=background_zip_fixed_mu,
                        background_zip_initial_mu=background_zip_initial_mu,
                        background_zip_zi_warmup_steps=background_zip_zi_warmup_steps,
                    )
                )

                if prev_collect_future is not None:
                    zi_episode_results.extend(prev_collect_future.result())

                pending_zi_futures = current_zi_futures

            if pending_zi_futures is not None:
                zi_episode_results.extend(_collect_zi_futures(pending_zi_futures))

    return _finalize_zi_benchmark_result(
        llm_outcomes=llm_outcomes,
        zi_episode_results=zi_episode_results,
        batch_configs=batch_configs,
        episodes_per_config=episodes_per_config,
        include_batch_data=include_batch_data,
        max_events_per_episode=max_events_per_episode,
        reward_mode=reward_mode,
        ladder_specs=ladder_specs,
        q_reward_mode=q_reward_mode,
        ladder_kind=tracked_ladder_kind,
        suite_aggregate=suite_aggregate,
        suite_aggregate_lambda=suite_aggregate_lambda,
    )


def _finalize_zi_benchmark_result(
    *,
    llm_outcomes: List,
    zi_episode_results: List[Dict[str, Any]],
    batch_configs: List[ReflectionMarketConfig],
    episodes_per_config: int,
    include_batch_data: bool,
    max_events_per_episode: int,
    reward_mode: str,
    ladder_specs: List[Dict[str, Any]],
    q_reward_mode: str,
    ladder_kind: str = "zi",
    suite_aggregate: str = "mean",
    suite_aggregate_lambda: float = 0.5,
) -> BatchEvaluationResult:
    llm_result = _aggregate_episode_outcomes(
        outcomes=llm_outcomes,
        batch_configs=batch_configs,
        episodes_per_config=episodes_per_config,
        include_batch_data=include_batch_data,
        max_events_per_episode=max_events_per_episode,
        reward_mode=reward_mode,
        suite_aggregate=suite_aggregate,
        suite_aggregate_lambda=suite_aggregate_lambda,
    )
    benchmark_results = aggregate_zi_episode_results(zi_episode_results, ladder_specs)
    if llm_result.llm_mean_surplus is not None:
        llm_mean_surplus = float(llm_result.llm_mean_surplus)
    else:
        llm_mean_surplus = mean_llm_surplus_from_batch_data(llm_result.batch_data)
    llm_mean_shade = mean_llm_shade_from_batch_data(llm_result.batch_data)
    llm_mean_requested_x_cap = mean_llm_requested_x_cap_from_batch_data(
        llm_result.batch_data
    )
    comparison = build_zi_beat_comparison(
        llm_mean_surplus=llm_mean_surplus,
        benchmark_results=benchmark_results,
        llm_mean_shade=llm_mean_shade,
        llm_mean_requested_x_cap=llm_mean_requested_x_cap,
    )
    comparison["ladder_kind"] = str(ladder_kind)
    comparison["ladder_label"] = _ladder_label(ladder_kind)

    if llm_result.batch_data is not None:
        llm_result.batch_data["zi_benchmark_comparison"] = comparison

    return apply_zi_q_reward_from_comparison(
        llm_result,
        comparison,
        q_reward_mode=q_reward_mode,
    )


def _evaluate_batch_with_zi_staged(
    *,
    strategy_fn: Callable,
    batch_configs: List[ReflectionMarketConfig],
    episodes_per_config: int,
    step_index: int,
    base_seed: int,
    reward_mode: str,
    max_events_per_episode: int,
    strategy_output_mode: str,
    strategy_type: str,
    episode_seeds: Optional[List[List[int]]],
    episode_private_values: Optional[List[List[List[float]]]],
    include_batch_data: bool,
    ladder_specs: List[Dict[str, Any]],
    zi_workers: int,
    q_reward_mode: str,
    lob_input_mode: str,
    special_agent_kind: str,
    hbl_l_param: int,
    bo_fixed_x: Optional[float],
    bo_fixed_x_by_label: Optional[Mapping[str, float]],
    spread_input_mode: str,
    spread_omit_fundamental: bool,
    background_agent_kind: str,
    background_zip_fixed_mu: Optional[float] = None,
    background_zip_initial_mu: Optional[float] = None,
    background_zip_zi_warmup_steps: int = 25,
    parallel_episode_workers: Optional[int],
    strategy_code: Optional[str],
    jobs,
    tracked_ladder_kind: str = "zi",
    suite_aggregate: str = "mean",
    suite_aggregate_lambda: float = 0.5,
) -> BatchEvaluationResult:
    from marketsim.LLM.reflection_pipeline.evaluation import evaluate_batch

    ladder_label = _ladder_label(tracked_ladder_kind)
    print(
        f"[{q_reward_mode}] staged eval: {len(jobs)} LLM episodes first "
        f"(parallel_episodes), then {len(ladder_specs)} {ladder_label} seats × "
        f"{len(jobs)} episodes ({zi_workers} workers)...",
        flush=True,
    )

    llm_result = evaluate_batch(
        strategy_fn=strategy_fn,
        batch_configs=batch_configs,
        episodes_per_config=episodes_per_config,
        step_index=step_index,
        base_seed=base_seed,
        reward_mode=reward_mode,
        max_events_per_episode=max_events_per_episode,
        strategy_output_mode=strategy_output_mode,
        strategy_type=strategy_type,
        episode_seeds=episode_seeds,
        episode_private_values=episode_private_values,
        include_batch_data=include_batch_data,
        special_agent_kind=special_agent_kind,
        hbl_l_param=hbl_l_param,
        bo_fixed_x=bo_fixed_x,
        bo_fixed_x_by_label=(
            dict(bo_fixed_x_by_label) if bo_fixed_x_by_label is not None else None
        ),
        parallel_episodes=True,
        parallel_workers=parallel_episode_workers,
        strategy_code=strategy_code,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
        spread_omit_fundamental=spread_omit_fundamental,
        background_agent_kind=background_agent_kind,
        background_zip_fixed_mu=background_zip_fixed_mu,
        background_zip_initial_mu=background_zip_initial_mu,
        background_zip_zi_warmup_steps=background_zip_zi_warmup_steps,
        suite_aggregate=suite_aggregate,
        suite_aggregate_lambda=suite_aggregate_lambda,
    )

    zi_payloads: List[Dict[str, Any]] = []
    for job in jobs:
        config = batch_configs[job.config_index]
        episode_kwargs = dict(config.to_run_episode_kwargs())
        if job.episode_private_values is not None:
            episode_kwargs["llm_private_values"] = job.episode_private_values
        episode_kwargs["background_agent_kind"] = background_agent_kind
        if background_zip_fixed_mu is not None:
            episode_kwargs["background_zip_fixed_mu"] = float(
                background_zip_fixed_mu
            )
        if background_zip_initial_mu is not None:
            episode_kwargs["background_zip_initial_mu"] = float(
                background_zip_initial_mu
            )
        episode_kwargs["background_zip_zi_warmup_steps"] = int(
            background_zip_zi_warmup_steps
        )
        zi_payloads.extend(
            _build_zi_cohort_payloads(
                job=job,
                episode_kwargs=episode_kwargs,
                ladder_specs=ladder_specs,
                strategy_type=strategy_type,
                tracked_agent_kind=tracked_ladder_kind,
            )
        )

    mp_context = _zi_process_context()
    print(
        f"[{q_reward_mode}] running {len(zi_payloads)} {ladder_label} benchmark episodes "
        f"({zi_workers} process workers)...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=zi_workers, mp_context=mp_context) as zi_pool:
        zi_episode_results = list(zi_pool.map(_zi_episode_worker, zi_payloads))

    benchmark_results = aggregate_zi_episode_results(zi_episode_results, ladder_specs)
    if llm_result.llm_mean_surplus is not None:
        llm_mean_surplus = float(llm_result.llm_mean_surplus)
    else:
        llm_mean_surplus = mean_llm_surplus_from_batch_data(llm_result.batch_data)
    llm_mean_shade = mean_llm_shade_from_batch_data(llm_result.batch_data)
    llm_mean_requested_x_cap = mean_llm_requested_x_cap_from_batch_data(
        llm_result.batch_data
    )
    comparison = build_zi_beat_comparison(
        llm_mean_surplus=llm_mean_surplus,
        benchmark_results=benchmark_results,
        llm_mean_shade=llm_mean_shade,
        llm_mean_requested_x_cap=llm_mean_requested_x_cap,
    )
    comparison["ladder_kind"] = str(tracked_ladder_kind)
    comparison["ladder_label"] = ladder_label

    if llm_result.batch_data is not None:
        llm_result.batch_data["zi_benchmark_comparison"] = comparison

    return apply_zi_q_reward_from_comparison(
        llm_result,
        comparison,
        q_reward_mode=q_reward_mode,
    )
