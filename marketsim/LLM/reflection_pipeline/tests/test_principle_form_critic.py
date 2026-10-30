import unittest

from marketsim.LLM.reflection_pipeline.config import CRITIC_MODES
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    format_structured_principle_form_facts,
    validate_structured_principle_form_reflection,
)
from marketsim.LLM.reflection_pipeline.recipes import get_recipe, list_recipes
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


class PrincipleFormRecipeTest(unittest.TestCase):
    def test_mode_registered(self):
        self.assertIn("structured_principle_form", CRITIC_MODES)

    def test_recipe_registered(self):
        self.assertIn("simple_pv_zi_advantage_ucb_principle", list_recipes())
        recipe = get_recipe("simple_pv_zi_advantage_ucb_principle")
        self.assertEqual(recipe.critic_mode, "structured_principle_form")
        self.assertEqual(recipe.q_reward_mode, "zi_advantage")
        self.assertEqual(recipe.parent_selection_mode, "ucb")
        self.assertEqual(recipe.q_eval_mode, "full_refresh")
        self.assertEqual(recipe.q_update_mode, "average")
        self.assertFalse(recipe.inactive_gate_enabled)


class StructuredPrincipleFormPromptTest(unittest.TestCase):
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

    def test_facts_emphasize_advantage_and_shade_hypothesis(self):
        lines = format_structured_principle_form_facts(self.comparison, reward=-100.0)
        text = "\n".join(lines)
        self.assertIn("PRIMARY objective", text)
        self.assertIn("-100.0000", text)
        self.assertIn("0.5000", text)
        self.assertIn("Shade-level hypothesis", text)
        self.assertIn("Suggested shade_error_direction:", text)
        self.assertIn("shade_more", text)
        self.assertIn("Do NOT paste shade_hi", text)
        self.assertNotIn("COMPARE THIS to ZI shade_hi", text)

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
            critic_mode="structured_principle_form",
        )
        self.assertIn("DIAGNOSIS:", prompt)
        self.assertIn("principle:", prompt)
        self.assertIn("form_class:", prompt)
        self.assertIn("features_used:", prompt)
        self.assertIn("current_form:", prompt)
        self.assertIn("why_this_fails:", prompt)
        self.assertIn("proposed_change:", prompt)
        self.assertIn("MECHANISM:", prompt)
        self.assertIn("current_pv_vs_fundamental", prompt)
        self.assertIn("unit_schedule_blend", prompt)
        self.assertIn("signed_power", prompt)
        self.assertIn("Evaluation Facts", prompt)
        self.assertIn("PRIMARY objective", prompt)
        self.assertIn("Per-Config Advantage", prompt)
        self.assertIn("Menu size: 25", prompt)
        self.assertNotIn("Environment:", prompt)
        self.assertIn("shade_error_direction:", prompt)
        self.assertIn("Shade-level hypothesis", prompt)
        self.assertNotIn("Curriculum Facts", prompt)

    def test_validate_structured_principle_form_reflection(self):
        good = (
            "DIAGNOSIS:\n"
            "  principle: pv_rank_or_distance_to_best\n"
            "  form_class: rank\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: rank of abs(pv) among schedule\n"
            "  current_form: pv_mean / pv_max ratio\n"
            "  why_this_fails: flat shade across units; leftover high\n"
            "  action: replace\n"
            "  proposed_change: switch ratio to rank-based map\n"
            "MECHANISM: at t=10 current_pv high but shade low; miss fill later.\n"
            "FIX: implement rank form_class under pv_rank principle.\n"
        )
        self.assertIsNone(validate_structured_principle_form_reflection(good))

        bad_essay = "Environment: essay\nPerformance: vague\n"
        self.assertIsNotNone(validate_structured_principle_form_reflection(bad_essay))

        bad_principle = (
            "DIAGNOSIS:\n"
            "  principle: not_a_real_principle\n"
            "  form_class: rank\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: x\n"
            "  current_form: y\n"
            "  why_this_fails: leftover high\n"
            "  action: replace\n"
            "  proposed_change: w\n"
            "MECHANISM: a\n"
            "FIX: b\n"
        )
        self.assertIsNotNone(validate_structured_principle_form_reflection(bad_principle))

    def test_generation_prompt_mirrors_diagnosis(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(private_values):\n    return 0.1\n",
            parent_reflection=(
                "DIAGNOSIS:\n"
                "  principle: current_pv_in_schedule\n"
                "  form_class: unit_schedule_blend\n"
                "  shade_error_direction: shade_more\n"
                "  features_used: current_pv, private_values, fundamental_estimate\n"
                "  parent_form: constant-ish ratio\n"
                "  why_parent_fails: ignores unit position in schedule\n"
                "  diversity_vs_parent: blend unit rank with schedule/F̃ term\n"
                "MECHANISM: early units overshade.\n"
                "FIX: unit_schedule_blend map.\n"
            ),
            strategy_output_mode="x_cap",
            market_description="test",
            strategy_type="pv",
            parent_performance={
                "reward": -50.0,
                "reward_variance": 12.5,
                "per_config_rewards": {"cfg_a": -30.0},
                "per_config_reward_variances": {"cfg_a": 5.0},
                "zi_benchmark_comparison": self.comparison,
            },
            generator_batch_summary={
                "reward": -50.0,
                "zi_benchmark_comparison": self.comparison,
            },
            generator_batch_data={"llm_interactions": [{"time": 1}]},
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
        )
        self.assertIn("DIAGNOSIS", prompt)
        self.assertIn("principle", prompt)
        self.assertIn("form_class", prompt)
        self.assertIn("Parent Hypothesis", prompt)
        self.assertIn("Parent Evaluation", prompt)
        self.assertIn("unit_schedule_blend", prompt)
        self.assertIn("current_pv", prompt)
        self.assertIn(
            "Prefer using all three when the hypothesis benefits",
            prompt,
        )
        self.assertNotIn("Curriculum Facts", prompt)
        self.assertIn("shade_error_direction", prompt)
        self.assertIn("shade_more", prompt)
        self.assertNotIn("COMPARE THIS to ZI shade_hi", prompt)


if __name__ == "__main__":
    unittest.main()
