import unittest

from marketsim.LLM.reflection_pipeline.config import (
    CURRICULUM_ZI_BENCHMARK_SHADE_RANGES,
    build_curriculum_config_pool,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    format_curriculum_diagnosis_template,
    format_structured_zi_curriculum_facts,
    infer_curriculum_shade_diagnosis,
    validate_structured_curriculum_reflection,
)
from marketsim.LLM.reflection_pipeline.recipes import get_recipe, list_recipes
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


class ZiCurriculumRecipeTest(unittest.TestCase):
    def test_recipe_registered(self):
        self.assertIn("simple_pv_zi_curriculum_newest", list_recipes())
        recipe = get_recipe("simple_pv_zi_curriculum_newest")
        self.assertEqual(recipe.q_reward_mode, "zi_advantage")
        self.assertEqual(recipe.parent_selection_mode, "newest")
        self.assertEqual(recipe.batch_size, 3)
        self.assertEqual(recipe.episodes_per_config, 50)
        self.assertEqual(recipe.steps, 40)
        self.assertEqual(recipe.critic_mode, "structured_zi_curriculum")
        self.assertTrue(recipe.use_curriculum_market)
        self.assertFalse(recipe.inactive_gate_enabled)
        self.assertEqual(recipe.q_eval_mode, "incremental")

    def test_ucb_recipe_registered(self):
        self.assertIn("simple_pv_zi_curriculum_ucb", list_recipes())
        recipe = get_recipe("simple_pv_zi_curriculum_ucb")
        self.assertEqual(recipe.q_reward_mode, "zi_advantage")
        self.assertEqual(recipe.parent_selection_mode, "ucb")
        self.assertEqual(recipe.batch_size, 3)
        self.assertEqual(recipe.episodes_per_config, 50)
        self.assertEqual(recipe.steps, 125)
        self.assertEqual(recipe.critic_mode, "structured_zi_curriculum")
        self.assertTrue(recipe.use_curriculum_market)
        self.assertFalse(recipe.inactive_gate_enabled)
        self.assertEqual(recipe.q_eval_mode, "incremental")
        self.assertTrue(recipe.forbid_numpy_random)

    def test_curriculum_market_and_zi_caps(self):
        pool = build_curriculum_config_pool()
        self.assertEqual(len(pool), 1)
        self.assertEqual(len(CURRICULUM_ZI_BENCHMARK_SHADE_RANGES), 5)


class StructuredZiCurriculumPromptTest(unittest.TestCase):
    def setUp(self):
        self.comparison = build_zi_beat_comparison(
            llm_mean_surplus=100.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.01], "mean_surplus": 50.0},
                {"benchmark_index": 1, "shade_range": [0.0, 0.04], "mean_surplus": 150.0},
            ],
            llm_mean_shade=0.002,
            llm_mean_requested_x_cap=0.5,
        )

    def test_facts_include_beaten_and_failed(self):
        lines = format_structured_zi_curriculum_facts(self.comparison, reward=-100.0)
        text = "\n".join(lines)
        self.assertIn("Highest BEATEN shade_hi (label only): 0.01", text)
        self.assertIn("Lowest FAILED shade_hi (label only): 0.04", text)
        self.assertIn("requested x_cap", text)
        self.assertIn("0.5000", text)
        self.assertIn("realized trade shade", text)
        self.assertIn("Batch advantage (Q reward): -100.0000", text)
        self.assertIn("diagnostic only", text)
        self.assertIn("Suggested shade_error_direction", text)
        self.assertIn("shade_more", text)
        self.assertIn("Suggested error_type", text)
        self.assertIn("Suggested fix_class", text)
        self.assertIn("never x_cap vs shade_hi", text)
        self.assertIn("DIAGNOSTIC HINT", text)
        self.assertIn("GENERAL shading rule", text)
        self.assertNotIn("COMPARE THIS to ZI shade_hi", text)
        self.assertNotIn("LLM mean effective shade", text)

    def test_infer_diagnosis_level_too_low(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=100.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.01], "mean_surplus": 150.0},
                {"benchmark_index": 1, "shade_range": [0.0, 0.04], "mean_surplus": 120.0},
            ],
            llm_mean_shade=0.002,
            llm_mean_requested_x_cap=0.35,
        )
        diag = infer_curriculum_shade_diagnosis(comparison)
        # Surplus gaps + collapsed realized shade → shade_more (not x_cap vs shade_hi).
        self.assertEqual(diag["shade_error_direction"], "shade_more")
        self.assertEqual(diag["error_type"], "level_too_low")
        self.assertEqual(diag["fix_class"], "add_offset")

    def test_reflection_prompt_is_structured(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(private_values):\n    return 0.1\n",
            batch_summary={
                "zi_benchmark_comparison": self.comparison,
                "per_config_rewards": {"cfg_a": -30.0},
                "per_config_reward_variances": {"cfg_a": 5.0},
            },
            batch_data={},
            reward=-50.0,
            include_data=True,
            q_reward_mode="zi_advantage",
            critic_mode="structured_zi_curriculum",
        )
        self.assertIn("DIAGNOSIS:", prompt)
        self.assertIn("shade_error_direction:", prompt)
        self.assertIn("error_type:", prompt)
        self.assertIn("fix_class:", prompt)
        self.assertIn("consensus:", prompt)
        self.assertIn("MECHANISM:", prompt)
        self.assertIn("Curriculum Facts", prompt)
        self.assertIn("Banned topics", prompt)
        self.assertIn("Batch Reward (advantage)", prompt)
        self.assertIn("ZI beat fraction (diagnostic)", prompt)
        self.assertIn("requested x_cap", prompt)
        self.assertIn("Per-Config Advantage", prompt)
        self.assertNotIn("Environment:", prompt)
        self.assertNotIn("what kinds of PV→x_cap strategies", prompt)

    def test_reflection_prompt_multi_config_includes_per_config_directions(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a"},
                    "episodes": [{"avg_requested_x_cap": 0.002, "reward": -1.0}],
                },
                {
                    "config": {"label": "cfg_b"},
                    "episodes": [{"avg_requested_x_cap": 0.5, "reward": -2.0}],
                },
            ]
        }
        prompt = build_reflection_prompt(
            code="def agent_strategy(private_values):\n    return 0.1\n",
            batch_summary={
                "zi_benchmark_comparison": self.comparison,
                "per_config_rewards": {"cfg_a": -30.0, "cfg_b": -70.0},
                "per_config_reward_variances": {"cfg_a": 5.0, "cfg_b": 8.0},
            },
            batch_data=batch_data,
            reward=-50.0,
            include_data=False,
            q_reward_mode="zi_advantage",
            critic_mode="structured_zi_curriculum",
        )
        self.assertIn("per_config_directions:", prompt)
        self.assertIn("consensus:", prompt)

    def test_validate_structured_curriculum_reflection(self):
        good = (
            "DIAGNOSIS:\n"
            "  requested_x_cap: 0.01\n"
            "  shade_error_direction: shade_more\n"
            "  error_type: level_too_low\n"
            "  fix_class: add_offset\n"
            "  consensus: n/a\n"
            "MECHANISM: requested_x_cap 0.01 is below failed bench_1.\n"
            "FIX: add_offset to raise base x_cap.\n"
        )
        self.assertIsNone(validate_structured_curriculum_reflection(good, multi_config=False))
        bad = "Environment: essay\nPerformance: vague\n"
        self.assertIsNotNone(
            validate_structured_curriculum_reflection(bad, multi_config=False)
        )

    def test_generation_prompt_uses_direction_fields(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(private_values):\n    return 0.1\n",
            parent_reflection=(
                "DIAGNOSIS:\n"
                "  shade_error_direction: shade_less\n"
                "  error_type: level_too_high\n"
                "  fix_class: scale_base\n"
                "  consensus: n/a\n"
                "MECHANISM: x_cap too high vs failed caps.\n"
                "FIX: scale_base to lower typical x_cap.\n"
            ),
            strategy_output_mode="x_cap",
            market_description="test",
            strategy_type="pv",
            parent_performance={
                "reward": -50.0,
                "reward_variance": 12.5,
                "per_config_rewards": {"cfg_a": -30.0, "cfg_b": -70.0},
                "per_config_reward_variances": {"cfg_a": 5.0, "cfg_b": 8.0},
                "zi_benchmark_comparison": self.comparison,
            },
            generator_batch_summary={
                "reward": -50.0,
                "zi_benchmark_comparison": self.comparison,
            },
            generator_batch_data={"llm_interactions": [{"time": 1}]},
            q_reward_mode="zi_advantage",
            critic_mode="structured_zi_curriculum",
        )
        self.assertIn("DIAGNOSIS", prompt)
        self.assertIn("shade_error_direction", prompt)
        self.assertIn("fix_class", prompt)
        self.assertIn("Parent Reflection", prompt)
        self.assertIn("Parent Evaluation", prompt)
        self.assertIn("private_values, fundamental_estimate, and current_pv only", prompt)
        self.assertIn("Curriculum Facts", prompt)
        self.assertIn("Parent Per-Config Advantage", prompt)
        self.assertIn("never x_cap vs shade_hi", prompt)
        self.assertNotIn("COMPARE THIS to ZI shade_hi", prompt)
        self.assertIn("ZI Benchmark Unit Tests", prompt)
        self.assertIn("Median-Episode LLM Interactions", prompt)
        self.assertNotIn("CODING step only", prompt)
        self.assertNotIn("reference only", prompt)

    def test_diagnosis_template_mixed_consensus_forces_reshape(self):
        comparison = build_zi_beat_comparison(
            llm_mean_surplus=100.0,
            benchmark_results=[
                {"benchmark_index": 0, "shade_range": [0.0, 0.01], "mean_surplus": 50.0},
                {"benchmark_index": 1, "shade_range": [0.0, 0.04], "mean_surplus": 150.0},
            ],
            llm_mean_requested_x_cap=0.5,
        )
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a"},
                    # Collapsed fills while classical seats ahead → shade_more.
                    "episodes": [
                        {
                            "avg_requested_x_cap": 0.35,
                            "avg_shade": 0.002,
                            "reward": -1.0,
                        }
                    ],
                },
                {
                    "config": {"label": "cfg_b"},
                    # Still losing despite higher realized shade → reshape.
                    "episodes": [
                        {
                            "avg_requested_x_cap": 0.5,
                            "avg_shade": 0.25,
                            "reward": -2.0,
                        }
                    ],
                },
            ]
        }
        lines = format_curriculum_diagnosis_template(
            comparison,
            batch_data=batch_data,
            multi_config=True,
        )
        text = "\n".join(lines)
        self.assertIn("consensus: mixed", text)
        self.assertIn("shade_error_direction: reshape_pv_map", text)
        self.assertIn("cfg_a: shade_more", text)
        self.assertIn("cfg_b: reshape_pv_map", text)


if __name__ == "__main__":
    unittest.main()
