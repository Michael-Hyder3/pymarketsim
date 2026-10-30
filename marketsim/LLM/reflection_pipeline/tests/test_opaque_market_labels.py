"""Opaque market labels for LLM-facing config identity."""

from __future__ import annotations

import re
import unittest

from marketsim.LLM.reflection_pipeline.advantage_config_feedback import (
    enrich_advantage_summary_with_config_tests,
)
from marketsim.LLM.reflection_pipeline.config import build_default_config_space
from marketsim.LLM.reflection_pipeline.metrics import build_critic_batch_summary


class OpaqueMarketLabelTest(unittest.TestCase):
    def test_sampled_labels_are_opaque(self):
        space = build_default_config_space()
        rng = __import__("random").Random(0)
        for i in range(20):
            cfg = space.sample(rng, batch_index=i)
            self.assertRegex(cfg.label, r"^market_\d{4}$")
            self.assertIsNone(re.search(r"buy|sell|shade|t\d+_q", cfg.label, flags=re.I))
            # Hyperparams still available for logs / humans.
            self.assertIn("num_zi_buy", cfg.observable_state())
            self.assertEqual(cfg.observable_state()["label"], cfg.label)

    def test_critic_summary_strips_config_hyperparams(self):
        batch_data = {
            "configs": [
                {
                    "config": {
                        "label": "market_0007",
                        "num_zi_buy": 12,
                        "num_zi_sell": 5,
                        "timesteps": 400,
                        "q_max": 10,
                        "shade_range": [0.0, 0.02],
                        "arrival_rate": 0.1,
                    },
                    "episodes": [
                        {
                            "reward": -10.0,
                            "baseline_surplus": 100.0,
                            "llm_surplus": 90.0,
                        }
                    ],
                }
            ]
        }
        summary = build_critic_batch_summary(
            reward=-10.0,
            reward_variance=1.0,
            per_config_rewards={"market_0007": -10.0},
            per_config_reward_variances={"market_0007": 1.0},
            batch_data=batch_data,
            reward_mode="advantage",
            per_config_breakdown=True,
        )
        metrics = summary["per_config_metrics"]["market_0007"]
        self.assertEqual(metrics["config"], {"label": "market_0007"})
        self.assertNotIn("num_zi_buy", metrics["config"])

        enriched = enrich_advantage_summary_with_config_tests(
            {
                "per_config_rewards": {"market_0007": -10.0},
                "per_config_reward_variances": {"market_0007": 1.0},
                "per_config_metrics": {
                    "market_0007": {
                        "config": {
                            "label": "market_0007",
                            "num_zi_buy": 12,
                            "timesteps": 400,
                        },
                        "advantage_mean": -10.0,
                    }
                },
            }
        )
        self.assertEqual(
            enriched["config_unit_tests"][0]["config"],
            {"label": "market_0007"},
        )


if __name__ == "__main__":
    unittest.main()
