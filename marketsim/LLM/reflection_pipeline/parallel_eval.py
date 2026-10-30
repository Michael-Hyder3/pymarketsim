from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from marketsim.LLM.lob_features import is_lob_features_mode, is_lob_hbl_parity_mode
from marketsim.LLM.reflection_pipeline.benchmark_suite import FixedEvalSuite


def _strategy_type_params(
    strategy_type: str,
    *,
    lob_input_mode: str = "raw",
    spread_input_mode: str = "classic",
) -> Tuple[int, str]:
    from marketsim.LLM.lob_features import is_lob_book_orders_mode
    from marketsim.LLM.spread_input import (
        is_spread_book_mode,
        is_spread_zip_parity_mode,
    )

    if strategy_type == "pv":
        return 3, "pv"
    requires_single_input = strategy_type in {"spread", "lob"}
    required_args = 1 if requires_single_input else 3
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        input_mode = "lob_features"
    elif strategy_type == "lob" and is_lob_hbl_parity_mode(lob_input_mode):
        input_mode = "lob_hbl_parity"
    elif strategy_type == "lob" and is_lob_book_orders_mode(lob_input_mode):
        input_mode = "lob_book_orders"
    elif strategy_type == "spread" and is_spread_zip_parity_mode(spread_input_mode):
        input_mode = "spread_zip_parity"
    elif strategy_type == "spread" and is_spread_book_mode(spread_input_mode):
        input_mode = "spread_book"
    else:
        input_mode = strategy_type if requires_single_input else "full"
    return required_args, input_mode


def evaluate_node_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    from marketsim.LLM.reflection_pipeline.benchmark_suite import FixedEvalSuite
    from marketsim.LLM.reflection_pipeline.evaluation import evaluate_batch
    from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
    from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
        evaluate_batch_with_zi_benchmarks,
    )

    node_id = int(payload["node_id"])
    code = str(payload["code"])
    strategy_type = str(payload.get("strategy_type", "full"))
    suite = FixedEvalSuite.from_dict(payload["suite"])
    include_batch_data = bool(payload.get("include_batch_data", False))
    q_reward_mode = str(payload.get("q_reward_mode", "advantage"))
    lob_input_mode = str(payload.get("lob_input_mode", "raw"))
    spread_input_mode = str(payload.get("spread_input_mode", "classic"))
    spread_omit_fundamental = bool(payload.get("spread_omit_fundamental", False))
    background_agent_kind = str(payload.get("background_agent_kind", "zi"))
    background_zip_fixed_mu = payload.get("background_zip_fixed_mu")
    background_zip_initial_mu = payload.get("background_zip_initial_mu")
    background_zip_zi_warmup_steps = int(
        payload.get("background_zip_zi_warmup_steps", 25)
    )
    parallel_episodes = bool(payload.get("parallel_episodes", False))
    parallel_episode_workers = payload.get("parallel_episode_workers")
    skip_zi_benchmarks = bool(payload.get("skip_zi_benchmarks", False))

    from marketsim.LLM.spread_input import set_spread_omit_fundamental

    set_spread_omit_fundamental(spread_omit_fundamental)
    required_args, input_mode = _strategy_type_params(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )
    verifier = StrategyVerifier()
    ok, err, strategy_fn = verifier.verify(
        code,
        required_args=required_args,
        input_mode=input_mode,
        spread_omit_fundamental=spread_omit_fundamental,
    )
    if not ok or strategy_fn is None:
        return {
            "node_id": node_id,
            "success": False,
            "error": err,
            "reward": float("-inf"),
            "per_config_rewards": {},
            "batch_data": None,
        }

    common = dict(
        strategy_fn=strategy_fn,
        batch_configs=suite.configs,
        episodes_per_config=int(payload["episodes_per_config"]),
        step_index=-1,
        base_seed=int(payload["base_seed"]),
        reward_mode=str(payload["reward_mode"]),
        max_events_per_episode=20,
        strategy_output_mode=str(payload["strategy_output_mode"]),
        strategy_type=strategy_type,
        episode_seeds=suite.episode_seeds,
        include_batch_data=include_batch_data,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
        spread_omit_fundamental=spread_omit_fundamental,
        background_agent_kind=background_agent_kind,
        background_zip_fixed_mu=background_zip_fixed_mu,
        background_zip_initial_mu=background_zip_initial_mu,
        background_zip_zi_warmup_steps=background_zip_zi_warmup_steps,
        parallel_episodes=parallel_episodes,
        parallel_workers=(
            int(parallel_episode_workers)
            if parallel_episode_workers is not None
            else None
        ),
        strategy_code=code,
        suite_aggregate=str(payload.get("suite_aggregate", "mean")),
        suite_aggregate_lambda=float(payload.get("suite_aggregate_lambda", 0.5)),
    )
    if q_reward_mode == "zi_beat_fraction" and not skip_zi_benchmarks:
        eval_result = evaluate_batch_with_zi_benchmarks(
            **common,
            q_reward_mode=q_reward_mode,
        )
    else:
        eval_result = evaluate_batch(**common, special_agent_kind="llm")
    return {
        "node_id": node_id,
        "success": True,
        "error": None,
        "reward": float(eval_result.reward),
        "reward_variance": float(eval_result.reward_variance),
        "per_config_rewards": eval_result.per_config_rewards,
        "per_config_reward_variances": eval_result.per_config_reward_variances,
        "batch_data": eval_result.batch_data if include_batch_data else None,
        "zi_benchmark_comparison": eval_result.zi_benchmark_comparison,
        "scalar_reward": eval_result.scalar_reward,
        "llm_mean_surplus": (
            float(eval_result.llm_mean_surplus)
            if getattr(eval_result, "llm_mean_surplus", None) is not None
            else None
        ),
    }


def refresh_nodes_parallel(
    *,
    node_specs: List[Tuple[int, str, str]],
    suite: FixedEvalSuite,
    episodes_per_config: int,
    base_seed: int,
    reward_mode: str,
    strategy_output_mode: str,
    parallel_workers: int,
    include_batch_data: bool = False,
    q_reward_mode: str = "advantage",
    lob_input_mode: str = "raw",
    spread_input_mode: str = "classic",
    spread_omit_fundamental: bool = False,
    background_agent_kind: str = "zi",
    background_zip_fixed_mu: Optional[float] = None,
    background_zip_initial_mu: Optional[float] = None,
    background_zip_zi_warmup_steps: int = 25,
    parallel_episodes: bool = False,
    parallel_episode_workers: Optional[int] = None,
    skip_zi_benchmarks: bool = False,
    suite_aggregate: str = "mean",
    suite_aggregate_lambda: float = 0.5,
) -> List[Dict[str, Any]]:
    if not node_specs:
        return []

    workers = max(1, min(int(parallel_workers), len(node_specs), os.cpu_count() or 1))
    suite_payload = suite.to_dict()
    payloads = [
        {
            "node_id": node_id,
            "code": code,
            "strategy_type": strategy_type,
            "suite": suite_payload,
            "episodes_per_config": episodes_per_config,
            "base_seed": base_seed,
            "reward_mode": reward_mode,
            "strategy_output_mode": strategy_output_mode,
            "include_batch_data": include_batch_data,
            "q_reward_mode": q_reward_mode,
            "lob_input_mode": lob_input_mode,
            "spread_input_mode": spread_input_mode,
            "spread_omit_fundamental": spread_omit_fundamental,
            "background_agent_kind": background_agent_kind,
            "background_zip_fixed_mu": background_zip_fixed_mu,
            "background_zip_initial_mu": background_zip_initial_mu,
            "background_zip_zi_warmup_steps": background_zip_zi_warmup_steps,
            "parallel_episodes": parallel_episodes,
            "parallel_episode_workers": parallel_episode_workers,
            "skip_zi_benchmarks": skip_zi_benchmarks,
            "suite_aggregate": suite_aggregate,
            "suite_aggregate_lambda": suite_aggregate_lambda,
        }
        for node_id, code, strategy_type in node_specs
    ]

    if workers == 1:
        return [evaluate_node_worker(payload) for payload in payloads]

    mp_context = mp.get_context("spawn")
    results: List[Optional[Dict[str, Any]]] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
        future_to_index = {
            executor.submit(evaluate_node_worker, payload): index
            for index, payload in enumerate(payloads)
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return [result for result in results if result is not None]
