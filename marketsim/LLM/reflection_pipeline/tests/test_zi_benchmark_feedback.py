import unittest

from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    format_zi_benchmark_test_feedback_lines,
)
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


class ZiBenchmarkFeedbackTest(unittest.TestCase):
    def test_enrich_records_failed_tests_without_hints(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=100.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.002], "mean_surplus": 200.0},
                {"benchmark_index": 1, "shade_range": [0.0, 0.1], "mean_surplus": 50.0},
            ],
            llm_mean_shade=0.4,
        )
        self.assertEqual(len(comparison["failed_tests"]), 1)
        self.assertEqual(comparison["failed_tests"][0]["test_id"], "bench_0")
        self.assertNotIn("hint", comparison["failed_tests"][0])
        self.assertAlmostEqual(comparison["failed_tests"][0]["llm_mean_shade"], 0.4)
        self.assertEqual(len(comparison["passed_tests"]), 1)

    def test_prompt_shows_unit_test_failures_without_hints(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=80.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.003], "mean_surplus": 150.0},
            ],
            llm_mean_shade=0.35,
        )
        lines = format_zi_benchmark_test_feedback_lines(comparison)
        text = "\n".join(lines)
        self.assertIn("FAIL bench_0", text)
        self.assertIn("gap=", text)
        self.assertIn("llm_mean_shade=0.3500", text)
        self.assertNotIn("hint:", text)

        prompt = build_reflection_prompt(
            code="def agent_strategy(private_values):\n    return 0.1\n",
            batch_summary={"zi_benchmark_comparison": comparison},
            batch_data={},
            reward=0.0,
            include_data=False,
            q_reward_mode="zi_beat_fraction",
        )
        self.assertIn("ZI Benchmark Unit Tests", prompt)
        self.assertNotIn("hint:", prompt)

        gen_prompt = build_generation_prompt(
            parent_code="def agent_strategy(private_values):\n    return 0.1\n",
            parent_reflection="Fix: lower shade",
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="pv",
            parent_performance={"zi_benchmark_comparison": comparison},
            q_reward_mode="zi_beat_fraction",
        )
        self.assertIn("diagnosis only", gen_prompt)
        self.assertIn("Do NOT copy the Parent Strategy", gen_prompt)
        self.assertIn(
            "never encode individual ZI benchmarks",
            gen_prompt,
        )
        self.assertNotIn("cite test_id and shade_range in comments", gen_prompt)
        self.assertNotIn("guided hints", gen_prompt)


if __name__ == "__main__":
    unittest.main()
