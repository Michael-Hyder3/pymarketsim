import unittest

from marketsim.LLM.reflection_pipeline.metrics import (
    build_critic_batch_summary,
    mean_and_variance,
    summarize_batch_performance,
    summarize_per_config_performance,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
)


class MetricsTest(unittest.TestCase):
    def test_mean_and_variance(self):
        mean, variance = mean_and_variance([1.0, 3.0, 5.0])
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(variance, 8.0 / 3.0)

    def test_summarize_batch_performance_includes_variances(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a"},
                    "episodes": [
                        {
                            "llm_surplus": 10.0,
                            "baseline_surplus": 4.0,
                            "avg_zi_same_surplus": 8.0,
                        },
                        {
                            "llm_surplus": 14.0,
                            "baseline_surplus": 6.0,
                            "avg_zi_same_surplus": 9.0,
                        },
                    ],
                },
                {
                    "config": {"label": "cfg_b"},
                    "episodes": [
                        {
                            "llm_surplus": 2.0,
                            "baseline_surplus": 1.0,
                            "avg_zi_same_surplus": 3.0,
                        },
                        {
                            "llm_surplus": 6.0,
                            "baseline_surplus": 2.0,
                            "avg_zi_same_surplus": 4.0,
                        },
                    ],
                },
            ]
        }
        overall = summarize_batch_performance(batch_data, reward_mode="advantage")
        self.assertEqual(overall["num_episodes"], 4)
        self.assertIn("llm_surplus_variance", overall)
        self.assertIn("advantage_variance", overall)
        self.assertIn("reward_variance", overall)

    def test_summarize_batch_performance_includes_avg_shade(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a"},
                    "episodes": [
                        {"llm_surplus": 10.0, "baseline_surplus": 4.0, "avg_shade": 0.1},
                        {"llm_surplus": 14.0, "baseline_surplus": 6.0, "avg_shade": 0.3},
                    ],
                }
            ]
        }
        overall = summarize_batch_performance(batch_data, reward_mode="advantage")
        self.assertAlmostEqual(overall["avg_shade_mean"], 0.2)
        self.assertEqual(overall["avg_shade_episode_count"], 2)

    def test_per_config_breakdown_for_critic(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a", "timesteps": 100},
                    "episodes": [
                        {
                            "llm_surplus": 10.0,
                            "baseline_surplus": 4.0,
                            "avg_zi_same_surplus": 8.0,
                        },
                        {
                            "llm_surplus": 12.0,
                            "baseline_surplus": 5.0,
                            "avg_zi_same_surplus": 9.0,
                        },
                    ],
                }
            ]
        }
        summary = build_critic_batch_summary(
            reward=6.5,
            reward_variance=0.25,
            per_config_rewards={"cfg_a": 6.5},
            per_config_reward_variances={"cfg_a": 0.25},
            batch_data=batch_data,
            reward_mode="advantage",
            per_config_breakdown=True,
        )
        self.assertIn("per_config_metrics", summary)
        self.assertIn("cfg_a", summary["per_config_metrics"])
        self.assertIn("overall", summary)

    def test_reflection_prompt_per_config_section(self):
        summary = {
            "reward": 1.0,
            "reward_variance": 0.5,
            "overall": {"reward_mean": 1.0, "reward_variance": 0.5},
            "per_config_metrics": {
                "cfg_a": {
                    "reward_mean": 1.0,
                    "reward_variance": 0.5,
                    "config": {"label": "cfg_a"},
                }
            },
        }
        prompt = build_reflection_prompt(
            code="def agent_strategy(private_values):\n    return 0.5\n",
            batch_summary=summary,
            batch_data={
                "median_episode_reward": 1.0,
                "llm_interactions": [
                    {
                        "pre_lob": {},
                        "agent_action": {},
                        "post_lob": {},
                        "transacted": False,
                    }
                ],
                "interaction_count": 1,
            },
            reward=1.0,
            reward_variance=0.5,
        )
        self.assertIn("Mechanism:", prompt)
        self.assertIn("Principle:", prompt)
        self.assertIn("Median-Episode LLM Interactions", prompt)
        self.assertIn("variance: 0.5000", prompt)

    def test_build_generation_prompt_without_reflection(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.5\n",
            parent_reflection="should not appear",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            use_reflection=False,
            parent_performance={
                "reward": 1.5,
                "reward_variance": 0.25,
                "per_config_rewards": {"cfg_a": 1.5},
            },
        )
        self.assertIn("Parent Performance", prompt)
        self.assertIn("## Parent Batch Reward", prompt)
        self.assertIn("reward=1.5000 (variance=0.2500)", prompt)
        self.assertIn('"reward": 1.5', prompt)
        self.assertNotIn("Parent Reflection", prompt)
        self.assertNotIn("should not appear", prompt)

    def test_build_reflection_prompt_without_data(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(private_values):\n    return 0.5\n",
            batch_summary={"reward": 1.0},
            batch_data={"llm_interactions": [{"pre_lob": {}}]},
            reward=1.0,
            reward_variance=0.5,
            include_data=False,
        )
        self.assertIn("No batch metrics or episode trajectories were provided", prompt)
        self.assertNotIn("Median-Episode LLM Interactions", prompt)
        self.assertNotIn("Batch Summary", prompt)


if __name__ == "__main__":
    unittest.main()
