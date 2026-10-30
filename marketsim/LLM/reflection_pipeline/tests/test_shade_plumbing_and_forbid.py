import unittest

from marketsim.LLM.reflection_pipeline.evaluation import _summarize_episode
from marketsim.LLM.reflection_pipeline.metrics import mean_llm_shade_from_batch_data
from marketsim.LLM.reflection_pipeline.recipes import get_recipe
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier


class AvgShadePlumbingTest(unittest.TestCase):
    def test_summarize_episode_keeps_avg_shade(self):
        summary = _summarize_episode(
            {
                "llm_surplus": 10.0,
                "baseline_surplus": 4.0,
                "avg_zi_same_surplus": 8.0,
                "consumer_surplus": 12.0,
                "avg_shade": 0.25,
                "avg_requested_x_cap": 0.4,
                "episode_stats": {},
            },
            max_events=5,
            reward_mode="advantage",
        )
        self.assertAlmostEqual(summary["avg_shade"], 0.25)
        self.assertAlmostEqual(summary["avg_requested_x_cap"], 0.4)

    def test_mean_llm_shade_from_batch_data(self):
        batch_data = {
            "configs": [
                {
                    "episodes": [
                        {"avg_shade": 0.1, "llm_surplus": 1.0},
                        {"avg_shade": 0.3, "llm_surplus": 2.0},
                    ]
                }
            ]
        }
        self.assertAlmostEqual(mean_llm_shade_from_batch_data(batch_data), 0.2)

    def test_mean_llm_requested_x_cap_from_batch_data(self):
        from marketsim.LLM.reflection_pipeline.metrics import (
            mean_llm_requested_x_cap_from_batch_data,
        )

        batch_data = {
            "configs": [
                {
                    "episodes": [
                        {"avg_requested_x_cap": 0.2, "avg_shade": 0.01},
                        {"avg_requested_x_cap": 0.4, "avg_shade": 0.02},
                    ]
                }
            ]
        }
        self.assertAlmostEqual(mean_llm_requested_x_cap_from_batch_data(batch_data), 0.3)


class ForbidNumpyRandomTest(unittest.TestCase):
    def setUp(self):
        self.verifier = StrategyVerifier()

    def test_recipe_enables_forbid(self):
        recipe = get_recipe("simple_pv_zi_curriculum_newest")
        self.assertTrue(recipe.forbid_numpy_random)

    def test_rejects_numpy_import(self):
        code = (
            "import numpy as np\n"
            "def agent_strategy(private_values, fundamental_estimate, current_pv):\n"
            "    return 0.2\n"
        )
        ok, err, _ = self.verifier.verify(
            code,
            required_args=3,
            input_mode="pv",
            forbid_numpy_random=True,
        )
        self.assertFalse(ok)
        self.assertIn("Forbidden import", err or "")

    def test_rejects_random_import(self):
        code = (
            "import random\n"
            "def agent_strategy(private_values, fundamental_estimate, current_pv):\n"
            "    return random.random()\n"
        )
        ok, err, _ = self.verifier.verify(
            code,
            required_args=3,
            input_mode="pv",
            forbid_numpy_random=True,
        )
        self.assertFalse(ok)
        self.assertIn("Forbidden import", err or "")

    def test_allows_pure_python_when_forbid_on(self):
        code = (
            "def agent_strategy(private_values, fundamental_estimate, current_pv):\n"
            "    if not private_values:\n"
            "        return 0.0\n"
            "    return 0.2\n"
        )
        ok, err, fn = self.verifier.verify(
            code,
            required_args=3,
            input_mode="pv",
            forbid_numpy_random=True,
        )
        self.assertTrue(ok, err)
        self.assertIsNotNone(fn)

    def test_rejects_out_of_range_shade(self):
        high = (
            "def agent_strategy(private_values, fundamental_estimate, current_pv):\n"
            "    return 1.5\n"
        )
        ok, err, _ = self.verifier.verify(high, required_args=3, input_mode="pv")
        self.assertFalse(ok)
        self.assertIn("[0,1]", err or "")

        low = (
            "def agent_strategy(private_values, fundamental_estimate, current_pv):\n"
            "    return -0.1\n"
        )
        ok, err, _ = self.verifier.verify(low, required_args=3, input_mode="pv")
        self.assertFalse(ok)
        self.assertIn("[0,1]", err or "")

    def test_accepts_boundary_shade(self):
        for value in (0.0, 1.0, 0.5):
            code = (
                "def agent_strategy(private_values, fundamental_estimate, current_pv):\n"
                f"    return {value}\n"
            )
            ok, err, fn = self.verifier.verify(code, required_args=3, input_mode="pv")
            self.assertTrue(ok, err)
            self.assertIsNotNone(fn)

    def test_allows_numpy_when_forbid_off(self):
        code = (
            "import numpy as np\n"
            "def agent_strategy(private_values, fundamental_estimate, current_pv):\n"
            "    return float(np.clip(0.2, 0.0, 1.0))\n"
        )
        ok, err, fn = self.verifier.verify(
            code,
            required_args=3,
            input_mode="pv",
            forbid_numpy_random=False,
        )
        self.assertTrue(ok, err)
        self.assertIsNotNone(fn)


if __name__ == "__main__":
    unittest.main()
