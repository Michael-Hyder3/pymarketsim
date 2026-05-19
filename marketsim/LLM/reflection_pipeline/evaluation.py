from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig


@dataclass
class BatchEvaluationResult:
    reward: float
    batch_data: Dict[str, Any]
    per_config_rewards: Dict[str, float]


def _trim_events(events: List[Dict[str, Any]], max_events: int) -> List[Dict[str, Any]]:
    if max_events <= 0 or len(events) <= max_events:
        return events
    return events[-max_events:]


def _summarize_episode(result: Dict[str, Any], max_events: int) -> Dict[str, Any]:
    summary = {
        "llm_surplus": float(result.get("llm_surplus", 0.0)),
        "baseline_surplus": float(result.get("baseline_surplus", 0.0)),
        "avg_zi_same_surplus": float(result.get("avg_zi_same_surplus", 0.0)),
        "consumer_surplus": float(result.get("consumer_surplus", 0.0)),
        "episode_stats": result.get("episode_stats", {}),
    }
    diagnostic = result.get("diagnostic_json")
    if diagnostic:
        trimmed = dict(diagnostic)
        events = trimmed.get("events", [])
        trimmed["events"] = _trim_events(events, max_events)
        summary["diagnostic_json"] = trimmed
    return summary


def _compute_reward(result: Dict[str, Any], mode: str) -> float:
    llm_surplus = float(result.get("llm_surplus", 0.0))
    avg_zi = float(result.get("avg_zi_same_surplus", 0.0))
    consumer_surplus = float(result.get("consumer_surplus", 0.0))
    baseline_surplus = float(result.get("baseline_surplus", 0.0))
    if mode == "advantage":
        return llm_surplus - baseline_surplus
    if mode == "consumer_surplus":
        return llm_surplus / consumer_surplus if consumer_surplus > 0 else llm_surplus
    return llm_surplus


def evaluate_batch(
    *,
    strategy_fn: Callable,
    batch_configs: List[ReflectionMarketConfig],
    episodes_per_config: int,
    step_index: int,
    base_seed: int,
    reward_mode: str = "advantage",
    max_events_per_episode: int = 40,
    strategy_output_mode: str = "x_cap",
) -> BatchEvaluationResult:
    batch_reward = 0.0
    per_config_rewards: Dict[str, float] = {}
    batch_data: Dict[str, Any] = {"configs": []}

    for config_index, config in enumerate(batch_configs):
        config_reward_sum = 0.0
        episodes: List[Dict[str, Any]] = []
        for episode_index in range(episodes_per_config):
            seed = config.seed_for(
                base_seed=base_seed,
                step=step_index,
                config_index=config_index,
                episode_index=episode_index,
            )
            result = run_episode(
                llm_strategy_func=strategy_fn,
                random_seed=seed,
                capture_detail=True,
                llm_action_mode=strategy_output_mode,
                **config.to_run_episode_kwargs(),
            )
            reward = _compute_reward(result, reward_mode)
            config_reward_sum += reward
            episodes.append(_summarize_episode(result, max_events_per_episode))
        config_reward = config_reward_sum / max(1, episodes_per_config)
        per_config_rewards[config.label] = float(config_reward)
        batch_reward += config_reward
        batch_data["configs"].append(
            {
                "config": config.observable_state(),
                "episodes": episodes,
                "mean_reward": float(config_reward),
            }
        )

    reward = batch_reward / max(1, len(batch_configs))
    return BatchEvaluationResult(
        reward=float(reward),
        batch_data=batch_data,
        per_config_rewards=per_config_rewards,
    )
