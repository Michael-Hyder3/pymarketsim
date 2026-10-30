import unittest

from marketsim.LLM.reflection_pipeline.advantage_config_feedback import (
    enrich_advantage_summary_with_config_tests,
)
from marketsim.LLM.reflection_pipeline.metrics import build_critic_batch_summary
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    format_advantage_config_test_feedback_lines,
)


class AdvantageConfigFeedbackTest(unittest.TestCase):
    def test_enrich_records_failed_config_tests(self):
        summary = enrich_advantage_summary_with_config_tests(
            {
                "reward": 1.0,
                "reward_variance": 0.5,
                "per_config_rewards": {"cfg_a": -2.0, "cfg_b": 3.0},
                "per_config_reward_variances": {"cfg_a": 1.0, "cfg_b": 0.25},
            }
        )
        self.assertEqual(summary["config_tests_total"], 2)
        self.assertEqual(summary["config_tests_pass_count"], 1)
        self.assertEqual(len(summary["failed_config_tests"]), 1)
        self.assertEqual(summary["failed_config_tests"][0]["test_id"], "config_0")
        self.assertEqual(summary["failed_config_tests"][0]["config_label"], "cfg_a")
        self.assertNotIn("hint", summary["failed_config_tests"][0])

    def test_build_critic_summary_includes_config_tests_for_advantage(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a", "timesteps": 100},
                    "episodes": [
                        {"llm_surplus": 10.0, "baseline_surplus": 12.0},
                        {"llm_surplus": 8.0, "baseline_surplus": 11.0},
                    ],
                }
            ]
        }
        summary = build_critic_batch_summary(
            reward=-2.5,
            reward_variance=1.0,
            per_config_rewards={"cfg_a": -2.5},
            per_config_reward_variances={"cfg_a": 1.0},
            batch_data=batch_data,
            reward_mode="advantage",
            per_config_breakdown=False,
        )
        self.assertIn("per_config_metrics", summary)
        self.assertIn("config_unit_tests", summary)
        self.assertEqual(summary["config_tests_total"], 1)
        self.assertEqual(len(summary["failed_config_tests"]), 1)

    def test_prompt_shows_config_unit_test_failures(self):
        summary = enrich_advantage_summary_with_config_tests(
            {
                "reward": -1.0,
                "reward_variance": 0.25,
                "per_config_rewards": {"dense": -1.0},
                "per_config_reward_variances": {"dense": 0.25},
                "per_config_metrics": {
                    "dense": {
                        "advantage_mean": -1.0,
                        "llm_surplus_mean": 4.0,
                        "baseline_surplus_mean": 5.0,
                        "avg_shade_mean": 0.42,
                        "config": {"label": "dense"},
                    }
                },
            }
        )
        lines = format_advantage_config_test_feedback_lines(summary)
        text = "\n".join(lines)
        self.assertIn("FAIL config_0", text)
        self.assertIn("advantage=", text)
        self.assertIn("avg_shade=0.4200", text)
        self.assertNotIn("hint:", text)

        prompt = build_reflection_prompt(
            code="def agent_strategy(market_history):\n    return 0.5\n",
            batch_summary=summary,
            batch_data={},
            reward=-1.0,
            include_data=False,
            q_reward_mode="advantage",
        )
        self.assertIn("Mechanism:", prompt)
        self.assertIn("Principle:", prompt)
        self.assertIn("Per-Config Unit Tests", prompt)

        gen_prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_history):\n    return 0.5\n",
            parent_reflection="Fix: lower shade on dense config",
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="lob",
            parent_performance=summary,
            q_reward_mode="advantage",
        )
        self.assertIn("FAILED entry in Per-Config Unit Tests", gen_prompt)
        self.assertIn("Mechanism, Principle, and Fix", gen_prompt)


if __name__ == "__main__":
    unittest.main()
