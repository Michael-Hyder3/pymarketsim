import unittest

from marketsim.LLM.reflection_pipeline.evaluation import BatchEvaluationResult
from marketsim.LLM.reflection_pipeline.prompts import (
    build_reflection_prompt,
    is_zi_q_reward_mode,
)
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
    apply_zi_q_reward_from_comparison,
    build_zi_beat_comparison,
)


class ZiAdvantageModeTest(unittest.TestCase):
    def test_is_zi_q_reward_mode(self):
        self.assertTrue(is_zi_q_reward_mode("zi_beat_fraction"))
        self.assertTrue(is_zi_q_reward_mode("zi_advantage"))
        self.assertFalse(is_zi_q_reward_mode("advantage"))

    def test_apply_zi_q_reward_beat_fraction_overwrites(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=100.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.01], "mean_surplus": 50.0},
            ],
        )
        result = BatchEvaluationResult(
            reward=25.0,
            reward_variance=4.0,
            batch_data={},
            per_config_rewards={},
        )
        out = apply_zi_q_reward_from_comparison(
            result,
            comparison,
            q_reward_mode="zi_beat_fraction",
        )
        self.assertAlmostEqual(out.reward, 1.0)
        self.assertAlmostEqual(out.scalar_reward, 25.0)

    def test_apply_zi_q_reward_zi_advantage_keeps_advantage(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=100.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.01], "mean_surplus": 50.0},
            ],
        )
        result = BatchEvaluationResult(
            reward=-42.5,
            reward_variance=100.0,
            batch_data={},
            per_config_rewards={},
        )
        out = apply_zi_q_reward_from_comparison(
            result,
            comparison,
            q_reward_mode="zi_advantage",
        )
        self.assertAlmostEqual(out.reward, -42.5)
        self.assertAlmostEqual(out.reward_variance, 100.0)
        self.assertAlmostEqual(out.scalar_reward, 1.0)

    def test_reflection_prompt_shows_advantage_and_diagnostic_beat_fraction(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=80.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.01], "mean_surplus": 90.0},
            ],
        )
        prompt = build_reflection_prompt(
            code="def agent_strategy(private_values):\n    return 0.1\n",
            batch_summary={"zi_benchmark_comparison": comparison},
            batch_data={},
            reward=-12.5,
            reward_variance=3.0,
            include_data=False,
            q_reward_mode="zi_advantage",
        )
        self.assertIn("Batch Reward (advantage): -12.5000", prompt)
        self.assertIn("ZI beat fraction (diagnostic)", prompt)
        self.assertIn("diagnostic only", prompt)

    def test_refresh_worker_skips_zi_for_zi_advantage_mode(self):
        """Q refresh passes advantage (or any non-beat mode) → no ZI cohort eval."""
        from unittest import mock

        from marketsim.LLM.reflection_pipeline.benchmark_suite import build_fixed_eval_suite
        from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig
        from marketsim.LLM.reflection_pipeline.evaluation import BatchEvaluationResult
        from marketsim.LLM.reflection_pipeline.parallel_eval import evaluate_node_worker

        config = ReflectionMarketConfig(
            label="fixed",
            timesteps=20,
            q_max=5,
            num_zi_buy=1,
            num_zi_sell=1,
            pv_var=1e4,
            shade_range=[0.0, 0.5],
            arrival_rate=0.2,
        )
        suite = build_fixed_eval_suite(
            configs=[config],
            episodes_per_config=1,
            base_seed=1,
            step_index=0,
        )
        fake = BatchEvaluationResult(
            reward=-10.0,
            reward_variance=1.0,
            batch_data={"configs": []},
            per_config_rewards={"fixed": -10.0},
            per_config_reward_variances={"fixed": 1.0},
        )
        payload = {
            "node_id": 1,
            "code": "def agent_strategy(private_values):\n    return 0.2\n",
            "strategy_type": "pv",
            "suite": suite.to_dict(),
            "episodes_per_config": 1,
            "base_seed": 1,
            "reward_mode": "advantage",
            "strategy_output_mode": "x_cap",
            "include_batch_data": False,
            "q_reward_mode": "advantage",  # pipeline maps zi_advantage → advantage on refresh
        }
        with mock.patch(
            "marketsim.LLM.reflection_pipeline.evaluation.evaluate_batch",
            return_value=fake,
        ) as mock_adv, mock.patch(
            "marketsim.LLM.reflection_pipeline.zi_benchmark_eval.evaluate_batch_with_zi_benchmarks",
        ) as mock_zi:
            out = evaluate_node_worker(payload)
        self.assertTrue(out["success"])
        self.assertAlmostEqual(out["reward"], -10.0)
        mock_adv.assert_called_once()
        mock_zi.assert_not_called()


if __name__ == "__main__":
    unittest.main()
