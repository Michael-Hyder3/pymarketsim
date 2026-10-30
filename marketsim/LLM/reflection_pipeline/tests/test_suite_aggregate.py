"""Suite scalar aggregation: mean / mean_min / min."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.evaluation import aggregate_suite_reward


class SuiteAggregateTest(unittest.TestCase):
    def test_mean(self):
        r = aggregate_suite_reward({"a": 200.0, "b": -100.0, "c": -100.0}, mode="mean")
        self.assertAlmostEqual(r, 0.0)

    def test_min(self):
        r = aggregate_suite_reward({"a": 200.0, "b": -100.0, "c": -100.0}, mode="min")
        self.assertAlmostEqual(r, -100.0)

    def test_mean_min_lambda_half(self):
        r = aggregate_suite_reward(
            {"a": 200.0, "b": -100.0, "c": -100.0},
            mode="mean_min",
            lam=0.5,
        )
        # mean=0, min=-100 → 0.5*0 + 0.5*(-100) = -50
        self.assertAlmostEqual(r, -50.0)

    def test_mean_min_balanced_unchanged(self):
        r = aggregate_suite_reward(
            {"a": 10.0, "b": 10.0, "c": 10.0},
            mode="mean_min",
            lam=0.5,
        )
        self.assertAlmostEqual(r, 10.0)

    def test_lambda_clamped(self):
        r = aggregate_suite_reward(
            {"a": 200.0, "b": -100.0, "c": -100.0},
            mode="mean_min",
            lam=2.0,
        )
        self.assertAlmostEqual(r, -100.0)


class SuiteAggregateLlmOnlyTest(unittest.TestCase):
    def test_non_llm_ignores_mean_min(self):
        from marketsim.LLM.reflection_pipeline.evaluation import _aggregate_episode_outcomes
        from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig

        configs = [
            ReflectionMarketConfig(label="m0"),
            ReflectionMarketConfig(label="m1"),
            ReflectionMarketConfig(label="m2"),
        ]
        # One episode each: rewards 200, -100, -100 → mean 0, mean_min -50.
        outcomes = [
            (0, 0, {"baseline_surplus": 0.0, "llm_surplus": 200.0}, 200.0),
            (1, 0, {"baseline_surplus": 0.0, "llm_surplus": -100.0}, -100.0),
            (2, 0, {"baseline_surplus": 0.0, "llm_surplus": -100.0}, -100.0),
        ]
        llm = _aggregate_episode_outcomes(
            outcomes=outcomes,
            batch_configs=configs,
            episodes_per_config=1,
            include_batch_data=False,
            max_events_per_episode=5,
            reward_mode="advantage",
            suite_aggregate="mean_min",
            suite_aggregate_lambda=0.5,
            apply_suite_aggregate=True,
        )
        classical = _aggregate_episode_outcomes(
            outcomes=outcomes,
            batch_configs=configs,
            episodes_per_config=1,
            include_batch_data=False,
            max_events_per_episode=5,
            reward_mode="advantage",
            suite_aggregate="mean_min",
            suite_aggregate_lambda=0.5,
            apply_suite_aggregate=False,
        )
        self.assertAlmostEqual(llm.reward, -50.0)
        self.assertAlmostEqual(classical.reward, 0.0)


if __name__ == "__main__":
    unittest.main()
