import unittest

from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
)
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


ZI_COMPARISON = {
    "llm_mean_surplus": 120.0,
    "llm_mean_shade": 0.42,
    "beat_count": 3,
    "total_benchmarks": 10,
    "beat_fraction": 0.3,
    "beaten_shade_ranges": [[0.0, 0.002]],
    "not_beaten_shade_ranges": [[0.0, 0.015], [0.0, 0.025]],
    "benchmarks": [
        {"benchmark_index": 0, "shade_range": [0.0, 0.002], "mean_surplus": 100.0, "beat": True},
        {"benchmark_index": 1, "shade_range": [0.0, 0.015], "mean_surplus": 150.0, "beat": False},
    ],
}


class ZiPromptTest(unittest.TestCase):
    def test_reflection_prompt_five_part_critic(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(private_values):\n    return 0.5\n",
            batch_summary={"zi_benchmark_comparison": ZI_COMPARISON},
            batch_data={},
            reward=0.3,
            include_data=False,
            q_reward_mode="zi_beat_fraction",
        )
        self.assertIn("Mechanism:", prompt)
        self.assertIn("Principle:", prompt)
        self.assertIn("ZI Benchmark Evaluation", prompt)
        self.assertIn("Did not beat these ZI shade ranges", prompt)
        self.assertIn("LLM mean effective shade", prompt)

    def test_generation_prompt_includes_parent_zi_results_and_trajectory(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(private_values):\n    return 0.2\n",
            parent_reflection=(
                "Mechanism: shade 0.9 at t=3, no fill.\n"
                "Principle: lower shade when PV rank is high.\n"
                "Fix: reduce rank scaling."
            ),
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="pv",
            use_reflection=True,
            parent_performance={
                "reward": 0.3,
                "reward_variance": 0.21,
                "zi_benchmark_comparison": ZI_COMPARISON,
            },
            generator_batch_summary={
                "zi_benchmark_comparison": ZI_COMPARISON,
                "overall": {"avg_shade_mean": 0.42},
            },
            generator_batch_data={
                "interaction_count": 1,
                "llm_interactions": [
                    {
                        "time": 3,
                        "transacted": False,
                        "agent_action": {"shade": 0.9, "current_pv": 12.0},
                        "pre_lob": {"best_bid_relative": -1.0, "best_ask_relative": 2.0},
                        "post_lob": {"best_bid_relative": -1.0, "best_ask_relative": 2.0},
                    }
                ],
            },
            q_reward_mode="zi_beat_fraction",
        )
        self.assertIn("Parent ZI Benchmark Results", prompt)
        self.assertIn("DID NOT beat", prompt)
        self.assertIn("ZI Benchmark Evaluation", prompt)
        self.assertIn("Median-Episode LLM Interactions", prompt)
        self.assertIn("Principle", prompt)
        self.assertNotIn("Batch Summary (Aggregated metrics", prompt)

    def test_zi_comparison_includes_mean_shade(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=10.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.01], "mean_surplus": 5.0},
            ],
            llm_mean_shade=0.15,
        )
        self.assertAlmostEqual(comparison["llm_mean_shade"], 0.15)


if __name__ == "__main__":
    unittest.main()
