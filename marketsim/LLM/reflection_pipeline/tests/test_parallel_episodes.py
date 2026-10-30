"""Process-pool parallel_episodes: identical rewards vs serial; staged before ZI."""

from __future__ import annotations

from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig
from marketsim.LLM.reflection_pipeline.evaluation import evaluate_batch
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
    evaluate_batch_with_zi_benchmarks,
)


STRATEGY_CODE = """
def agent_strategy(market_state):
    return 0.02
"""


def _tiny_configs() -> list:
    return [
        ReflectionMarketConfig(
            label="m0",
            num_zi_buy=2,
            num_zi_sell=2,
            timesteps=40,
            q_max=2,
            shade_range=[0.0, 0.02],
            arrival_rate=0.2,
        ),
        ReflectionMarketConfig(
            label="m1",
            num_zi_buy=2,
            num_zi_sell=2,
            timesteps=40,
            q_max=2,
            shade_range=[0.0, 0.03],
            arrival_rate=0.2,
        ),
    ]


def _strategy_fn():
    ns: dict = {}
    exec(STRATEGY_CODE, ns)
    return ns["agent_strategy"]


def test_parallel_episodes_match_serial_rewards():
    configs = _tiny_configs()
    fn = _strategy_fn()
    common = dict(
        strategy_fn=fn,
        batch_configs=configs,
        episodes_per_config=2,
        step_index=0,
        base_seed=7,
        reward_mode="surplus",
        strategy_output_mode="shade",
        strategy_type="spread",
        include_batch_data=False,
        special_agent_kind="llm",
        background_agent_kind="zi",
        strategy_code=STRATEGY_CODE,
    )
    serial = evaluate_batch(**common, parallel_episodes=False)
    parallel = evaluate_batch(
        **common, parallel_episodes=True, parallel_workers=2
    )
    assert abs(serial.reward - parallel.reward) < 1e-9
    assert serial.per_config_rewards == parallel.per_config_rewards


def test_staged_zi_path_runs_and_matches_serial_llm_mean():
    configs = _tiny_configs()[:1]
    fn = _strategy_fn()
    shade_ranges = [[0.0, 0.01], [0.0, 0.05]]
    common = dict(
        strategy_fn=fn,
        batch_configs=configs,
        episodes_per_config=2,
        step_index=0,
        base_seed=11,
        reward_mode="surplus",
        strategy_output_mode="shade",
        strategy_type="spread",
        include_batch_data=True,
        special_agent_kind="llm",
        background_agent_kind="zi",
        zi_benchmark_shade_ranges=shade_ranges,
        q_reward_mode="zi_advantage",
        strategy_code=STRATEGY_CODE,
        parallel_workers=2,
    )
    overlapped = evaluate_batch_with_zi_benchmarks(
        **common, parallel_episodes=False
    )
    staged = evaluate_batch_with_zi_benchmarks(
        **common,
        parallel_episodes=True,
        parallel_episode_workers=2,
    )
    assert overlapped.zi_benchmark_comparison is not None
    assert staged.zi_benchmark_comparison is not None
    assert abs(
        float(overlapped.zi_benchmark_comparison["llm_mean_surplus"])
        - float(staged.zi_benchmark_comparison["llm_mean_surplus"])
    ) < 1e-9
    assert (
        overlapped.zi_benchmark_comparison["beat_count"]
        == staged.zi_benchmark_comparison["beat_count"]
    )
