from __future__ import annotations

import multiprocessing as mp
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple

from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.reflection_pipeline.config import (
    DEFAULT_PARALLEL_EPISODE_WORKERS,
    ReflectionMarketConfig,
)
from marketsim.LLM.reflection_pipeline.metrics import mean_and_variance


_FORK_STRATEGY_FN: Optional[Callable] = None
_FORK_BACKGROUND_STRATEGY_FN: Optional[Callable] = None

SUITE_AGGREGATE_MODES = ("mean", "mean_min", "min")
DEFAULT_SUITE_AGGREGATE = "mean"
DEFAULT_SUITE_AGGREGATE_LAMBDA = 0.5


def aggregate_suite_reward(
    per_config_rewards: Dict[str, float],
    *,
    mode: str = DEFAULT_SUITE_AGGREGATE,
    lam: float = DEFAULT_SUITE_AGGREGATE_LAMBDA,
) -> float:
    values = [float(v) for v in per_config_rewards.values()]
    if not values:
        return 0.0
    mean_v = sum(values) / len(values)
    min_v = min(values)
    mode_key = str(mode or DEFAULT_SUITE_AGGREGATE).strip().lower()
    if mode_key == "min":
        return float(min_v)
    if mode_key == "mean_min":
        weight = min(1.0, max(0.0, float(lam)))
        return float((1.0 - weight) * mean_v + weight * min_v)
    if mode_key != "mean":
        raise ValueError(
            f"Unsupported suite_aggregate mode {mode!r}; "
            f"expected one of {SUITE_AGGREGATE_MODES}"
        )
    return float(mean_v)


def _episode_process_context() -> Optional[mp.context.BaseContext]:
    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    if "spawn" in mp.get_all_start_methods():
        return mp.get_context("spawn")
    return None


def _compile_strategy_from_code(strategy_code: str) -> Callable:
    namespace: Dict[str, Any] = {}
    exec(strategy_code, namespace)
    fn = namespace.get("agent_strategy")
    if fn is None or not callable(fn):
        raise ValueError("agent_strategy not found in strategy_code")
    return fn


def _process_episode_worker(
    payload: Dict[str, Any],
) -> Tuple[int, int, Dict[str, Any], float]:
    strategy_code = payload.get("strategy_code")
    if strategy_code is not None:
        strategy_fn = _compile_strategy_from_code(str(strategy_code))
    else:
        strategy_fn = _FORK_STRATEGY_FN
        if strategy_fn is None:
            raise RuntimeError(
                "process episode worker missing strategy_code and fork strategy_fn"
            )

    background_strategy_code = payload.get("background_strategy_code")
    if background_strategy_code is not None:
        background_strategy_fn = _compile_strategy_from_code(
            str(background_strategy_code)
        )
    else:
        background_strategy_fn = _FORK_BACKGROUND_STRATEGY_FN

    config = ReflectionMarketConfig(**payload["config"])
    job = _EpisodeJob(
        config_index=int(payload["config_index"]),
        episode_index=int(payload["episode_index"]),
        seed=int(payload["seed"]),
        episode_private_values=payload.get("episode_private_values"),
    )
    from marketsim.LLM.spread_input import set_spread_omit_fundamental
    set_spread_omit_fundamental(bool(payload.get("spread_omit_fundamental", False)))
    return _run_episode_job(
        job,
        config=config,
        strategy_fn=strategy_fn,
        reward_mode=str(payload["reward_mode"]),
        max_events_per_episode=int(payload["max_events_per_episode"]),
        strategy_output_mode=str(payload["strategy_output_mode"]),
        strategy_type=str(payload["strategy_type"]),
        include_batch_data=bool(payload["include_batch_data"]),
        special_agent_kind=str(payload["special_agent_kind"]),
        special_agent_shade_range=payload.get("special_agent_shade_range"),
        bo_optimizer_strategy=str(payload["bo_optimizer_strategy"]),
        bo_action_mode=str(payload["bo_action_mode"]),
        bo_fixed_x=payload.get("bo_fixed_x"),
        hbl_l_param=int(payload["hbl_l_param"]),
        special_agent_hbl_l=payload.get("special_agent_hbl_l"),
        lob_input_mode=str(payload["lob_input_mode"]),
        spread_input_mode=str(payload["spread_input_mode"]),
        spread_omit_fundamental=bool(payload.get("spread_omit_fundamental", False)),
        background_agent_kind=str(payload["background_agent_kind"]),
        background_zip_fixed_mu=payload.get("background_zip_fixed_mu"),
        background_zip_initial_mu=payload.get("background_zip_initial_mu"),
        background_zip_zi_warmup_steps=int(
            payload.get("background_zip_zi_warmup_steps", 25)
        ),
        background_strategy_fn=background_strategy_fn,
        background_strategy_type=str(
            payload.get("background_strategy_type") or "pv"
        ),
        background_strategy_output_mode=str(
            payload.get("background_strategy_output_mode") or "shade"
        ),
    )


@dataclass
class BatchEvaluationResult:
    reward: float
    batch_data: Dict[str, Any]
    per_config_rewards: Dict[str, float]
    baseline_surplus: float = 0.0
    per_config_baseline_surplus: Optional[Dict[str, float]] = None
    reward_variance: float = 0.0
    per_config_reward_variances: Dict[str, float] = field(default_factory=dict)
    zi_benchmark_comparison: Optional[Dict[str, Any]] = None
    scalar_reward: Optional[float] = None
    llm_mean_surplus: Optional[float] = None
























def _summarize_episode(
    result: Dict[str, Any],
    max_events: int,
    *,
    reward_mode: str,
) -> Dict[str, Any]:
    summary = {
        "llm_surplus": float(result.get("llm_surplus", 0.0)),
        "baseline_surplus": float(result.get("baseline_surplus", 0.0)),
        "avg_zi_same_surplus": float(result.get("avg_zi_same_surplus", 0.0)),
        "consumer_surplus": float(result.get("consumer_surplus", 0.0)),
        "background_surplus_total": float(result.get("background_surplus_total", 0.0)),
        "background_surplus_mean": float(result.get("background_surplus_mean", 0.0)),
        "background_surplus_mean_buy": float(
            result.get("background_surplus_mean_buy", 0.0)
        ),
        "background_surplus_mean_sell": float(
            result.get("background_surplus_mean_sell", 0.0)
        ),
        "background_surplus_n": int(result.get("background_surplus_n", 0) or 0),
        "background_surplus_n_buy": int(result.get("background_surplus_n_buy", 0) or 0),
        "background_surplus_n_sell": int(
            result.get("background_surplus_n_sell", 0) or 0
        ),
        "llm_advantage": float(result.get("llm_advantage", 0.0)),
        "background_advantage_total": float(
            result.get("background_advantage_total", 0.0)
        ),
        "background_advantage_mean": float(
            result.get("background_advantage_mean", 0.0)
        ),
        "background_advantage_mean_buy": float(
            result.get("background_advantage_mean_buy", 0.0)
        ),
        "background_advantage_mean_sell": float(
            result.get("background_advantage_mean_sell", 0.0)
        ),
        "reward": float(_compute_reward(result, reward_mode)),
        "episode_stats": result.get("episode_stats", {}),
    }
    if result.get("avg_shade") is not None:
        summary["avg_shade"] = float(result["avg_shade"])
    if result.get("avg_requested_x_cap") is not None:
        summary["avg_requested_x_cap"] = float(result["avg_requested_x_cap"])
    diagnostic = result.get("diagnostic_json")
    if diagnostic:
        summary["diagnostic_json"] = dict(diagnostic)
    return summary


def _compute_reward(result: Dict[str, Any], mode: str) -> float:
    llm_surplus = float(result.get("llm_surplus", 0.0))
    avg_zi = float(result.get("avg_zi_same_surplus", 0.0))
    consumer_surplus = float(result.get("consumer_surplus", 0.0))
    baseline_surplus = float(result.get("baseline_surplus", 0.0))
    if mode == "advantage":
        return llm_surplus - baseline_surplus
    if mode == "baseline_surplus":
        return baseline_surplus
    if mode == "consumer_surplus":
        return llm_surplus
    return llm_surplus


class _EpisodeJob(NamedTuple):
    config_index: int
    episode_index: int
    seed: int
    episode_private_values: Optional[List[float]]


def _episode_seed_for_batch(
    *,
    config: ReflectionMarketConfig,
    config_index: int,
    episode_index: int,
    step_index: int,
    base_seed: int,
    episode_seeds: Optional[List[List[int]]],
) -> int:
    if episode_seeds is not None:
        return int(episode_seeds[config_index][episode_index])
    return config.seed_for(
        base_seed=base_seed,
        step=step_index,
        config_index=config_index,
        episode_index=episode_index,
    )


def _build_episode_jobs(
    *,
    batch_configs: List[ReflectionMarketConfig],
    episodes_per_config: int,
    step_index: int,
    base_seed: int,
    episode_seeds: Optional[List[List[int]]],
    episode_private_values: Optional[List[List[List[float]]]],
) -> List[_EpisodeJob]:
    jobs: List[_EpisodeJob] = []
    for config_index, config in enumerate(batch_configs):
        for episode_index in range(episodes_per_config):
            private_values = None
            if episode_private_values is not None:
                private_values = episode_private_values[config_index][episode_index]
            jobs.append(
                _EpisodeJob(
                    config_index=config_index,
                    episode_index=episode_index,
                    seed=_episode_seed_for_batch(
                        config=config,
                        config_index=config_index,
                        episode_index=episode_index,
                        step_index=step_index,
                        base_seed=base_seed,
                        episode_seeds=episode_seeds,
                    ),
                    episode_private_values=private_values,
                )
            )
    return jobs


def _run_episode_job(
    job: _EpisodeJob,
    *,
    config: ReflectionMarketConfig,
    strategy_fn: Callable,
    reward_mode: str,
    max_events_per_episode: int,
    strategy_output_mode: str,
    strategy_type: str,
    include_batch_data: bool,
    special_agent_kind: str,
    special_agent_shade_range: Optional[List[float]],
    bo_optimizer_strategy: str,
    bo_action_mode: str,
    bo_fixed_x: Optional[float],
    hbl_l_param: int,
    special_agent_hbl_l: Optional[int] = None,
    lob_input_mode: str = "raw",
    spread_input_mode: str = "classic",
    spread_omit_fundamental: bool = False,
    background_agent_kind: str = "zi",
    background_zip_fixed_mu: Optional[float] = None,
    background_zip_initial_mu: Optional[float] = None,
    background_zip_zi_warmup_steps: int = 25,
    background_strategy_fn: Optional[Callable] = None,
    background_strategy_type: str = "pv",
    background_strategy_output_mode: str = "shade",
) -> Tuple[int, int, Dict[str, Any], float]:
    episode_kwargs = config.to_run_episode_kwargs()
    if job.episode_private_values is not None:
        episode_kwargs["llm_private_values"] = job.episode_private_values
    result = run_episode(
        llm_strategy_func=strategy_fn,
        random_seed=job.seed,
        capture_detail=include_batch_data,
        llm_action_mode=strategy_output_mode,
        strategy_type=strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
        spread_omit_fundamental=spread_omit_fundamental,
        background_agent_kind=background_agent_kind,
        background_zip_fixed_mu=background_zip_fixed_mu,
        background_zip_initial_mu=background_zip_initial_mu,
        background_zip_zi_warmup_steps=background_zip_zi_warmup_steps,
        background_strategy_func=background_strategy_fn,
        background_strategy_type=background_strategy_type,
        background_strategy_output_mode=background_strategy_output_mode,
        special_agent_kind=special_agent_kind,
        special_agent_shade_range=special_agent_shade_range,
        special_agent_hbl_l=special_agent_hbl_l,
        bo_optimizer_strategy=bo_optimizer_strategy,
        bo_action_mode=bo_action_mode,
        bo_fixed_x=bo_fixed_x,
        hbl_l_param=hbl_l_param,
        **episode_kwargs,
    )
    reward = _compute_reward(result, reward_mode)
    return job.config_index, job.episode_index, result, reward


def _aggregate_episode_outcomes(
    *,
    outcomes: List[Tuple[int, int, Dict[str, Any], float]],
    batch_configs: List[ReflectionMarketConfig],
    episodes_per_config: int,
    include_batch_data: bool,
    max_events_per_episode: int,
    reward_mode: str,
    suite_aggregate: str = DEFAULT_SUITE_AGGREGATE,
    suite_aggregate_lambda: float = DEFAULT_SUITE_AGGREGATE_LAMBDA,
    apply_suite_aggregate: bool = True,
) -> BatchEvaluationResult:
    outcomes.sort(key=lambda item: (item[0], item[1]))
    per_config_episode_rewards: Dict[int, List[float]] = defaultdict(list)
    per_config_baseline_sums: Dict[int, float] = defaultdict(float)
    per_config_episodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_episode_rewards: List[float] = []
    all_llm_surpluses: List[float] = []

    for config_index, _episode_index, result, reward in outcomes:
        per_config_episode_rewards[config_index].append(reward)
        per_config_baseline_sums[config_index] += float(result.get("baseline_surplus", 0.0))
        all_episode_rewards.append(reward)
        all_llm_surpluses.append(float(result.get("llm_surplus", 0.0)))
        if include_batch_data:
            per_config_episodes[config_index].append(
                _summarize_episode(
                    result,
                    max_events_per_episode,
                    reward_mode=reward_mode,
                )
            )

    batch_baseline = 0.0
    per_config_rewards: Dict[str, float] = {}
    per_config_baseline: Dict[str, float] = {}
    per_config_reward_variances: Dict[str, float] = {}
    batch_data: Dict[str, Any] = {"configs": []}

    for config_index, config in enumerate(batch_configs):
        config_episode_rewards = per_config_episode_rewards.get(config_index, [])
        config_reward, config_reward_variance = mean_and_variance(config_episode_rewards)
        config_baseline = per_config_baseline_sums[config_index] / max(1, episodes_per_config)
        per_config_rewards[config.label] = float(config_reward)
        per_config_reward_variances[config.label] = float(config_reward_variance)
        per_config_baseline[config.label] = float(config_baseline)
        batch_baseline += config_baseline
        if include_batch_data:
            batch_data["configs"].append(
                {
                    "config": config.observable_state(),
                    "episodes": per_config_episodes.get(config_index, []),
                    "mean_reward": float(config_reward),
                    "reward_variance": float(config_reward_variance),
                }
            )

    reward = aggregate_suite_reward(
        per_config_rewards,
        mode=suite_aggregate if apply_suite_aggregate else DEFAULT_SUITE_AGGREGATE,
        lam=suite_aggregate_lambda,
    )
    _, batch_reward_variance = mean_and_variance(all_episode_rewards)
    baseline_surplus = batch_baseline / max(1, len(batch_configs))
    llm_mean_surplus, _ = mean_and_variance(all_llm_surpluses) if all_llm_surpluses else (None, 0.0)
    return BatchEvaluationResult(
        reward=float(reward),
        batch_data=batch_data,
        per_config_rewards=per_config_rewards,
        baseline_surplus=float(baseline_surplus),
        per_config_baseline_surplus=per_config_baseline,
        reward_variance=float(batch_reward_variance),
        per_config_reward_variances=per_config_reward_variances,
        llm_mean_surplus=(
            float(llm_mean_surplus) if llm_mean_surplus is not None else None
        ),
    )


def evaluate_batch(
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
    special_agent_kind: str = "llm",
    special_agent_shade_range: Optional[List[float]] = None,
    bo_optimizer_strategy: str = "ucb",
    bo_action_mode: str = "fixed_shade",
    bo_fixed_x: Optional[float] = None,
    bo_fixed_x_by_label: Optional[Dict[str, float]] = None,
    hbl_l_param: int = 6,
    special_agent_hbl_l: Optional[int] = None,
    parallel_episodes: bool = False,
    parallel_workers: Optional[int] = None,
    strategy_code: Optional[str] = None,
    lob_input_mode: str = "raw",
    spread_input_mode: str = "classic",
    spread_omit_fundamental: bool = False,
    background_agent_kind: str = "zi",
    background_zip_fixed_mu: Optional[float] = None,
    background_zip_initial_mu: Optional[float] = None,
    background_zip_zi_warmup_steps: int = 25,
    background_strategy_fn: Optional[Callable] = None,
    background_strategy_code: Optional[str] = None,
    background_strategy_type: str = "pv",
    background_strategy_output_mode: str = "shade",
    suite_aggregate: str = DEFAULT_SUITE_AGGREGATE,
    suite_aggregate_lambda: float = DEFAULT_SUITE_AGGREGATE_LAMBDA,
) -> BatchEvaluationResult:
    jobs = _build_episode_jobs(
        batch_configs=batch_configs,
        episodes_per_config=episodes_per_config,
        step_index=step_index,
        base_seed=base_seed,
        episode_seeds=episode_seeds,
        episode_private_values=episode_private_values,
    )

    def _resolve_bo_fixed_x(config: ReflectionMarketConfig) -> Optional[float]:
        if bo_fixed_x_by_label is not None and config.label in bo_fixed_x_by_label:
            return float(bo_fixed_x_by_label[config.label])
        return bo_fixed_x

    if parallel_episodes and len(jobs) > 1:
        workers = max(
            1,
            min(
                int(
                    parallel_workers
                    if parallel_workers is not None
                    else DEFAULT_PARALLEL_EPISODE_WORKERS
                ),
                int(os.cpu_count() or 1),
                len(jobs),
            ),
        )
        mp_context = _episode_process_context()
        can_process = mp_context is not None and (
            strategy_code is not None
            or (mp_context.get_start_method() == "fork")
        )
        if can_process and workers > 1:
            payloads: List[Dict[str, Any]] = []
            for job in jobs:
                config = batch_configs[job.config_index]
                payloads.append(
                    {
                        "config_index": job.config_index,
                        "episode_index": job.episode_index,
                        "seed": job.seed,
                        "episode_private_values": job.episode_private_values,
                        "config": asdict(config),
                        "strategy_code": strategy_code,
                        "reward_mode": reward_mode,
                        "max_events_per_episode": max_events_per_episode,
                        "strategy_output_mode": strategy_output_mode,
                        "strategy_type": strategy_type,
                        "include_batch_data": include_batch_data,
                        "special_agent_kind": special_agent_kind,
                        "special_agent_shade_range": special_agent_shade_range,
                        "bo_optimizer_strategy": bo_optimizer_strategy,
                        "bo_action_mode": bo_action_mode,
                        "bo_fixed_x": _resolve_bo_fixed_x(config),
                        "hbl_l_param": hbl_l_param,
                        "special_agent_hbl_l": special_agent_hbl_l,
                        "lob_input_mode": lob_input_mode,
                        "spread_input_mode": spread_input_mode,
                        "spread_omit_fundamental": spread_omit_fundamental,
                        "background_agent_kind": background_agent_kind,
                        "background_zip_fixed_mu": background_zip_fixed_mu,
                        "background_zip_initial_mu": background_zip_initial_mu,
                        "background_zip_zi_warmup_steps": (
                            background_zip_zi_warmup_steps
                        ),
                        "background_strategy_code": background_strategy_code,
                        "background_strategy_type": background_strategy_type,
                        "background_strategy_output_mode": (
                            background_strategy_output_mode
                        ),
                    }
                )
            global _FORK_STRATEGY_FN, _FORK_BACKGROUND_STRATEGY_FN
            previous_fn = _FORK_STRATEGY_FN
            previous_bg = _FORK_BACKGROUND_STRATEGY_FN
            if strategy_code is None:
                _FORK_STRATEGY_FN = strategy_fn
            if background_strategy_code is None and background_strategy_fn is not None:
                _FORK_BACKGROUND_STRATEGY_FN = background_strategy_fn
            try:
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=mp_context
                ) as executor:
                    outcomes = list(executor.map(_process_episode_worker, payloads))
            finally:
                _FORK_STRATEGY_FN = previous_fn
                _FORK_BACKGROUND_STRATEGY_FN = previous_bg
            return _aggregate_episode_outcomes(
                outcomes=outcomes,
                batch_configs=batch_configs,
                episodes_per_config=episodes_per_config,
                include_batch_data=include_batch_data,
                max_events_per_episode=max_events_per_episode,
                reward_mode=reward_mode,
                suite_aggregate=suite_aggregate,
                suite_aggregate_lambda=suite_aggregate_lambda,
                apply_suite_aggregate=(str(special_agent_kind) == "llm"),
            )

    outcomes: List[Tuple[int, int, Dict[str, Any], float]] = []
    for job in jobs:
        config = batch_configs[job.config_index]
        outcomes.append(
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
                special_agent_shade_range=special_agent_shade_range,
                bo_optimizer_strategy=bo_optimizer_strategy,
                bo_action_mode=bo_action_mode,
                bo_fixed_x=_resolve_bo_fixed_x(config),
                hbl_l_param=hbl_l_param,
                special_agent_hbl_l=special_agent_hbl_l,
                lob_input_mode=lob_input_mode,
                spread_input_mode=spread_input_mode,
                spread_omit_fundamental=spread_omit_fundamental,
                background_agent_kind=background_agent_kind,
                background_zip_fixed_mu=background_zip_fixed_mu,
                background_zip_initial_mu=background_zip_initial_mu,
                background_zip_zi_warmup_steps=background_zip_zi_warmup_steps,
                background_strategy_fn=background_strategy_fn,
                background_strategy_type=background_strategy_type,
                background_strategy_output_mode=background_strategy_output_mode,
            )
        )
    return _aggregate_episode_outcomes(
        outcomes=outcomes,
        batch_configs=batch_configs,
        episodes_per_config=episodes_per_config,
        include_batch_data=include_batch_data,
        max_events_per_episode=max_events_per_episode,
        reward_mode=reward_mode,
        suite_aggregate=suite_aggregate,
        suite_aggregate_lambda=suite_aggregate_lambda,
        apply_suite_aggregate=(str(special_agent_kind) == "llm"),
    )



def _optional_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _config_label(config_state: Any) -> str:
    if isinstance(config_state, dict):
        return str(config_state.get("label", "unknown"))
    return "unknown"


def _config_q_max(config_state: Any) -> Optional[float]:
    if not isinstance(config_state, dict):
        return None
    raw = config_state.get("q_max")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _derive_compact_shout(
    event: Dict[str, Any],
    *,
    q_max: Optional[float],
) -> Dict[str, Any]:
    pre = event.get("pre_lob") or {}
    action = event.get("agent_action") or {}
    bid = pre.get("best_bid_relative")
    ask = pre.get("best_ask_relative")
    pv = action.get("current_pv")
    is_buy = bool(action.get("agent_bid", True))
    spread = None
    if bid is not None and ask is not None:
        spread = float(ask) - float(bid)
    gap = None
    if pv is not None:
        if is_buy and ask is not None:
            gap = float(pv) - float(ask)
        elif (not is_buy) and bid is not None:
            gap = float(bid) - float(pv)
    inv_frac = None
    position = action.get("position")
    if position is not None and q_max:
        inv_frac = abs(float(position)) / q_max
    shade = action.get("shade")
    shout: Dict[str, Any] = {
        "t": event.get("time"),
        "side": "buy" if is_buy else "sell",
        "filled": bool(event.get("transacted")),
    }
    if shade is not None:
        shout["shade"] = round(float(shade), 4)
    if spread is not None:
        shout["spread"] = round(float(spread), 4)
    if inv_frac is not None:
        shout["inv_frac"] = round(float(inv_frac), 3)
    if gap is not None:
        shout["gap"] = round(float(gap), 3)
    surplus = event.get("surplus_earned")
    if surplus is not None:
        shout["surplus"] = round(float(surplus), 2)
    return shout


def _llm_order_events(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostic = episode.get("diagnostic_json") or {}
    events = diagnostic.get("events") or []
    return [event for event in events if event.get("event_type") == "llm_order"]


def _select_tape_events(
    events: List[Dict[str, Any]],
    max_interactions: int,
) -> List[Dict[str, Any]]:
    n = max(0, int(max_interactions))
    if n <= 0 or len(events) <= n:
        return list(events[:n] if n else [])
    head = max(1, n // 2)
    tail = n - head
    if tail <= 0:
        return list(events[:n])
    return list(events[:head]) + list(events[-tail:])


def _episode_side_stats(
    episode: Dict[str, Any],
    *,
    q_max: Optional[float],
) -> Dict[str, Any]:
    events = _llm_order_events(episode)
    shade_filled: List[float] = []
    shade_miss: List[float] = []
    buy_surplus: List[float] = []
    sell_surplus: List[float] = []
    buy_fills = 0
    sell_fills = 0
    pos_gap = 0
    pos_gap_miss = 0
    for event in events:
        shout = _derive_compact_shout(event, q_max=q_max)
        shade = shout.get("shade")
        filled = bool(shout.get("filled"))
        side = shout.get("side")
        if shade is not None:
            (shade_filled if filled else shade_miss).append(float(shade))
        surplus = event.get("surplus_earned")
        if side == "buy":
            if filled:
                buy_fills += 1
            if surplus is not None:
                buy_surplus.append(float(surplus))
        else:
            if filled:
                sell_fills += 1
            if surplus is not None:
                sell_surplus.append(float(surplus))
        gap = shout.get("gap")
        if gap is not None and float(gap) > 0:
            pos_gap += 1
            if not filled:
                pos_gap_miss += 1
    stats = episode.get("episode_stats") or {}
    fill_ratio = stats.get("execution_ratio")
    leftover_frac = None
    if fill_ratio is not None:
        leftover_frac = max(0.0, 1.0 - float(fill_ratio))
    return {
        "fill_ratio": None if fill_ratio is None else float(fill_ratio),
        "leftover_frac": leftover_frac,
        "shade_filled": _optional_mean(shade_filled),
        "shade_miss": _optional_mean(shade_miss),
        "buy_surplus": _optional_mean(buy_surplus),
        "sell_surplus": _optional_mean(sell_surplus),
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
        "pos_gap_n": pos_gap,
        "pos_gap_miss_rate": (
            float(pos_gap_miss) / float(pos_gap) if pos_gap else None
        ),
        "n_shouts": len(events),
    }


def _compact_tape(
    *,
    episode: Dict[str, Any],
    label: str,
    role: str,
    max_interactions: int,
    q_max: Optional[float],
) -> Dict[str, Any]:
    stats = _episode_side_stats(episode, q_max=q_max)
    events = _select_tape_events(_llm_order_events(episode), max_interactions)
    return {
        "role": role,
        "label": label,
        "reward": round(float(episode.get("reward", 0.0)), 4),
        "surplus": round(float(episode.get("llm_surplus", 0.0)), 2),
        "fill": None if stats["fill_ratio"] is None else round(stats["fill_ratio"], 3),
        "leftover": (
            None if stats["leftover_frac"] is None else round(stats["leftover_frac"], 3)
        ),
        "shouts": [_derive_compact_shout(ev, q_max=q_max) for ev in events],
    }


def build_case_card_payload(
    batch_data: Dict[str, Any],
    *,
    max_interactions: int = 8,
) -> Dict[str, Any]:
    configs = batch_data.get("configs") or []
    if not configs:
        return {"case_cards": [], "worst_market": None, "tapes": {}}

    cards: List[Dict[str, Any]] = []
    worst_eps: List[Tuple[float, str, Dict[str, Any], Optional[float]]] = []
    best_eps: List[Tuple[float, str, Dict[str, Any], Optional[float]]] = []

    for cfg_data in configs:
        config_state = cfg_data.get("config") or {}
        label = _config_label(config_state)
        q_max = _config_q_max(config_state)
        episodes = list(cfg_data.get("episodes") or [])
        rewards = [float(ep.get("reward", 0.0)) for ep in episodes]
        surpluses = [float(ep.get("llm_surplus", 0.0)) for ep in episodes]
        zi_vals = [
            float(ep["avg_zi_same_surplus"])
            for ep in episodes
            if ep.get("avg_zi_same_surplus") is not None
        ]
        baselines = [float(ep.get("baseline_surplus", 0.0)) for ep in episodes]
        side_rows = [_episode_side_stats(ep, q_max=q_max) for ep in episodes]
        fill_vals = [
            float(row["fill_ratio"])
            for row in side_rows
            if row.get("fill_ratio") is not None
        ]
        leftover_vals = [
            float(row["leftover_frac"])
            for row in side_rows
            if row.get("leftover_frac") is not None
        ]
        shade_f = [
            float(row["shade_filled"])
            for row in side_rows
            if row.get("shade_filled") is not None
        ]
        shade_m = [
            float(row["shade_miss"])
            for row in side_rows
            if row.get("shade_miss") is not None
        ]
        buy_s = [
            float(row["buy_surplus"])
            for row in side_rows
            if row.get("buy_surplus") is not None
        ]
        sell_s = [
            float(row["sell_surplus"])
            for row in side_rows
            if row.get("sell_surplus") is not None
        ]
        gap_rates = [
            float(row["pos_gap_miss_rate"])
            for row in side_rows
            if row.get("pos_gap_miss_rate") is not None
        ]
        adv = _optional_mean(rewards)
        card = {
            "label": label,
            "advantage": None if adv is None else round(adv, 4),
            "surplus": None if not surpluses else round(_optional_mean(surpluses) or 0.0, 2),
            "zi_surplus": None if not zi_vals else round(_optional_mean(zi_vals) or 0.0, 2),
            "baseline": None if not baselines else round(_optional_mean(baselines) or 0.0, 2),
            "fill": None if not fill_vals else round(_optional_mean(fill_vals) or 0.0, 3),
            "leftover": (
                None if not leftover_vals else round(_optional_mean(leftover_vals) or 0.0, 3)
            ),
            "shade_fill": None if not shade_f else round(_optional_mean(shade_f) or 0.0, 4),
            "shade_miss": None if not shade_m else round(_optional_mean(shade_m) or 0.0, 4),
            "buy_surplus": None if not buy_s else round(_optional_mean(buy_s) or 0.0, 2),
            "sell_surplus": None if not sell_s else round(_optional_mean(sell_s) or 0.0, 2),
            "buy_fills": int(sum(row["buy_fills"] for row in side_rows)),
            "sell_fills": int(sum(row["sell_fills"] for row in side_rows)),
            "pos_gap_miss": (
                None if not gap_rates else round(_optional_mean(gap_rates) or 0.0, 3)
            ),
            "n_episodes": len(episodes),
        }
        cards.append(card)
        if episodes:
            worst_ep = min(episodes, key=lambda e: float(e.get("reward", 0.0)))
            best_ep = max(episodes, key=lambda e: float(e.get("reward", 0.0)))
            worst_eps.append(
                (float(worst_ep.get("reward", 0.0)), label, worst_ep, q_max)
            )
            best_eps.append((float(best_ep.get("reward", 0.0)), label, best_ep, q_max))

    worst_market = None
    if cards:
        ranked = [c for c in cards if c.get("advantage") is not None]
        if ranked:
            worst_market = min(ranked, key=lambda c: float(c["advantage"]))["label"]
            for card in cards:
                card["worst"] = card["label"] == worst_market

    tapes: Dict[str, Any] = {}
    n_shouts = max(0, int(max_interactions))
    worst_episode_obj: Optional[Dict[str, Any]] = None
    if worst_market and worst_eps:
        market_worsts = [row for row in worst_eps if row[1] == worst_market]
        pick = min(market_worsts or worst_eps, key=lambda row: row[0])
        worst_episode_obj = pick[2]
        tapes["worst"] = _compact_tape(
            episode=pick[2],
            label=pick[1],
            role="worst episode on worst market",
            max_interactions=n_shouts,
            q_max=pick[3],
        )
    if best_eps:
        other = [row for row in best_eps if row[1] != worst_market]
        pool = other if other else best_eps
        pick_best = max(pool, key=lambda row: row[0])
        if pick_best[2] is worst_episode_obj:
            alt = [row for row in best_eps if row[2] is not worst_episode_obj]
            pick_best = max(alt, key=lambda row: row[0]) if alt else None
        if pick_best is not None:
            tapes["good"] = _compact_tape(
                episode=pick_best[2],
                label=pick_best[1],
                role="better episode (contrast)",
                max_interactions=n_shouts,
                q_max=pick_best[3],
            )

    return {
        "case_cards": cards,
        "worst_market": worst_market,
        "tapes": tapes,
    }
