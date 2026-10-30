from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple



def mean_and_variance(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return float(mean), float(variance)


def episode_reward(episode: Dict[str, Any], reward_mode: str) -> float:
    llm_surplus = float(episode.get("llm_surplus", 0.0))
    baseline_surplus = float(episode.get("baseline_surplus", 0.0))
    if reward_mode == "advantage":
        return llm_surplus - baseline_surplus
    if reward_mode == "baseline_surplus":
        return baseline_surplus
    if reward_mode == "consumer_surplus":
        return llm_surplus
    return llm_surplus


def _metric_values(episodes: List[Dict[str, Any]], key: str) -> List[float]:
    return [float(ep.get(key, 0.0)) for ep in episodes]


def _optional_metric_values(episodes: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for ep in episodes:
        raw = ep.get(key)
        if raw is None:
            continue
        values.append(float(raw))
    return values


def _summarize_metric_values(values: List[float]) -> Dict[str, float]:
    mean, variance = mean_and_variance(values)
    return {"mean": mean, "variance": variance}


def summarize_episode_group(
    episodes: List[Dict[str, Any]],
    *,
    reward_mode: str,
) -> Dict[str, Any]:
    llm = _summarize_metric_values(_metric_values(episodes, "llm_surplus"))
    zi = _summarize_metric_values(_metric_values(episodes, "avg_zi_same_surplus"))
    baseline = _summarize_metric_values(_metric_values(episodes, "baseline_surplus"))
    reward_values = [episode_reward(ep, reward_mode) for ep in episodes]
    reward = _summarize_metric_values(reward_values)
    advantage_values = [
        float(ep.get("llm_surplus", 0.0)) - float(ep.get("baseline_surplus", 0.0))
        for ep in episodes
    ]
    advantage = _summarize_metric_values(advantage_values)
    llm_vs_zi_values = [
        float(ep.get("llm_surplus", 0.0)) - float(ep.get("avg_zi_same_surplus", 0.0))
        for ep in episodes
    ]
    llm_vs_zi = _summarize_metric_values(llm_vs_zi_values)
    shade_values = _optional_metric_values(episodes, "avg_shade")
    shade = _summarize_metric_values(shade_values) if shade_values else None
    metrics: Dict[str, Any] = {
        "num_episodes": len(episodes),
        "reward_mean": reward["mean"],
        "reward_variance": reward["variance"],
        "llm_surplus_mean": llm["mean"],
        "llm_surplus_variance": llm["variance"],
        "avg_zi_same_surplus_mean": zi["mean"],
        "avg_zi_same_surplus_variance": zi["variance"],
        "llm_vs_zi_diff_mean": llm_vs_zi["mean"],
        "llm_vs_zi_diff_variance": llm_vs_zi["variance"],
        "baseline_surplus_mean": baseline["mean"],
        "baseline_surplus_variance": baseline["variance"],
        "advantage_mean": advantage["mean"],
        "advantage_variance": advantage["variance"],
    }
    if shade is not None:
        metrics["avg_shade_mean"] = shade["mean"]
        metrics["avg_shade_variance"] = shade["variance"]
        metrics["avg_shade_episode_count"] = len(shade_values)
    x_cap_values = _optional_metric_values(episodes, "avg_requested_x_cap")
    if x_cap_values:
        x_cap = _summarize_metric_values(x_cap_values)
        metrics["avg_requested_x_cap_mean"] = x_cap["mean"]
        metrics["avg_requested_x_cap_variance"] = x_cap["variance"]
        metrics["avg_requested_x_cap_episode_count"] = len(x_cap_values)
    return metrics


def mean_llm_shade_from_batch_data(batch_data: Dict[str, Any]) -> Optional[float]:
    all_episodes: List[Dict[str, Any]] = []
    for cfg_data in batch_data.get("configs", []):
        all_episodes.extend(cfg_data.get("episodes", []))
    shade_values = _optional_metric_values(all_episodes, "avg_shade")
    if not shade_values:
        return None
    mean, _ = mean_and_variance(shade_values)
    return float(mean)


def mean_llm_requested_x_cap_from_batch_data(
    batch_data: Dict[str, Any],
) -> Optional[float]:
    all_episodes: List[Dict[str, Any]] = []
    for cfg_data in batch_data.get("configs", []):
        all_episodes.extend(cfg_data.get("episodes", []))
    values = _optional_metric_values(all_episodes, "avg_requested_x_cap")
    if not values:
        return None
    mean, _ = mean_and_variance(values)
    return float(mean)


def summarize_batch_performance(
    batch_data: Dict[str, Any],
    *,
    reward_mode: str = "advantage",
) -> Dict[str, Any]:
    all_episodes: List[Dict[str, Any]] = []
    for cfg_data in batch_data.get("configs", []):
        all_episodes.extend(cfg_data.get("episodes", []))
    return summarize_episode_group(all_episodes, reward_mode=reward_mode)


def summarize_per_config_performance(
    batch_data: Dict[str, Any],
    *,
    reward_mode: str = "advantage",
) -> Dict[str, Dict[str, Any]]:
    per_config: Dict[str, Dict[str, Any]] = {}
    for cfg_data in batch_data.get("configs", []):
        config_state = cfg_data.get("config", {})
        label = str(config_state.get("label", "unknown"))
        episodes = cfg_data.get("episodes", [])
        metrics = summarize_episode_group(episodes, reward_mode=reward_mode)
        metrics["config"] = config_state
        per_config[label] = metrics
    return per_config


def build_critic_batch_summary(
    *,
    reward: float,
    reward_variance: float,
    per_config_rewards: Dict[str, float],
    per_config_reward_variances: Dict[str, float],
    batch_data: Dict[str, Any],
    reward_mode: str,
    per_config_breakdown: bool,
) -> Dict[str, Any]:
    overall = summarize_batch_performance(batch_data, reward_mode=reward_mode)
    summary: Dict[str, Any] = {
        "reward": float(reward),
        "reward_variance": float(reward_variance),
        "per_config_rewards": dict(per_config_rewards),
        "per_config_reward_variances": dict(per_config_reward_variances),
        "overall": overall,
    }
    has_zi_comparison = bool(batch_data.get("zi_benchmark_comparison"))
    include_per_config = per_config_breakdown or (
        reward_mode == "advantage" and not has_zi_comparison
    )
    if include_per_config:
        summary["per_config_metrics"] = summarize_per_config_performance(
            batch_data,
            reward_mode=reward_mode,
        )
        for label, metrics in summary["per_config_metrics"].items():
            cfg = metrics.get("config")
            if isinstance(cfg, dict):
                metrics["config"] = {"label": cfg.get("label", label)}
    comparison = batch_data.get("zi_benchmark_comparison")
    if comparison:
        summary["zi_benchmark_comparison"] = comparison
    return summary
