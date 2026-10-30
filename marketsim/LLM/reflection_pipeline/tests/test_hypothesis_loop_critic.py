import unittest

from marketsim.LLM.reflection_pipeline.config import CRITIC_MODES
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    format_hypothesis_loop_facts,
    is_hypothesis_loop_critic,
    is_structured_principle_form,
    parse_principle_form_fields,
    validate_hypothesis_loop_reflection,
)
from marketsim.LLM.reflection_pipeline.recipes import get_recipe, list_recipes
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


GOOD_LOOP = """CURRENT_HYPOTHESIS:
  claim: shade from bid-ask width with a piecewise map
  principle: bid_ask_spread_width
  form_class: piecewise_by_sign
  features_used: best_bid, best_ask, current_pv
  current_form: shade = f(spread) with sign branch
EVALUATION:
  what_worked: beat 3/9 ZI seats; some fills when spread wide
  what_failed: leftover high on market_0001; realized trade shade ~0.002
NEXT_HYPOTHESIS:
  claim: raise shade when inventory remaining is high after a reject
  principle: inventory_remaining_fraction
  form_class: piecewise_by_sign
  shade_error_direction: shade_more
  features_used: units_available, q_max, prev_event
  action: replace
  proposed_change: switch primary signal from spread width to inventory remaining
MECHANISM: Current spread map claims high x_cap but fills stay near zero; inventory
should drive aggressiveness when leftover units remain.
FIX: Implement inventory_remaining_fraction + piecewise_by_sign realizing shade_more.
"""


class HypothesisLoopRecipeTest(unittest.TestCase):
    def test_mode_registered(self):
        self.assertIn("structured_hypothesis_loop", CRITIC_MODES)
        self.assertTrue(is_hypothesis_loop_critic("structured_hypothesis_loop"))
        self.assertTrue(is_structured_principle_form("structured_hypothesis_loop"))
        self.assertFalse(is_hypothesis_loop_critic("structured_principle_form"))

    def test_recipes_registered(self):
        self.assertIn("simple_spread_zi_advantage_ucb_hypothesis_loop", list_recipes())
        self.assertIn("simple_pv_zi_advantage_ucb_hypothesis_loop", list_recipes())
        self.assertIn("simple_lob_zi_advantage_ucb_hypothesis_loop", list_recipes())
        recipe = get_recipe("simple_spread_zi_advantage_ucb_hypothesis_loop")
        self.assertEqual(recipe.critic_mode, "structured_hypothesis_loop")
        self.assertEqual(recipe.init_types, ["spread"])


class HypothesisLoopPromptTest(unittest.TestCase):
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

    def test_facts_match_principle_form_inputs(self):
        from marketsim.LLM.reflection_pipeline.prompts import (
            format_structured_principle_form_facts,
        )

        loop = format_hypothesis_loop_facts(self.comparison, reward=-100.0)
        classic = format_structured_principle_form_facts(
            self.comparison, reward=-100.0
        )
        self.assertEqual(loop, classic)

    def test_reflection_prompt_has_three_parts(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(market_state):\n    return 0.1\n",
            batch_summary={
                "zi_benchmark_comparison": self.comparison,
                "per_config_rewards": {"cfg_a": -30.0},
            },
            batch_data={},
            reward=-50.0,
            include_data=True,
            q_reward_mode="zi_advantage",
            critic_mode="structured_hypothesis_loop",
            strategy_type="spread",
            spread_input_mode="book",
        )
        # Same inputs as principle-form critic:
        self.assertIn("Evaluation Facts", prompt)
        self.assertIn("Shade-level hypothesis", prompt)
        self.assertIn("Suggested shade_error_direction:", prompt)
        self.assertIn("Menu size:", prompt)
        # Different required output scaffold:
        self.assertIn("CURRENT_HYPOTHESIS:", prompt)
        self.assertIn("EVALUATION:", prompt)
        self.assertIn("what_worked:", prompt)
        self.assertIn("what_failed:", prompt)
        self.assertIn("NEXT_HYPOTHESIS:", prompt)
        self.assertIn("vocabulary for naming", prompt)
        self.assertIn("Copy Suggested shade_error_direction", prompt)
        self.assertIn("shade_error_direction is required", prompt)

    def test_validate_and_parse_prefers_next(self):
        self.assertIsNone(
            validate_hypothesis_loop_reflection(GOOD_LOOP, strategy_type="spread")
        )
        fields = parse_principle_form_fields(GOOD_LOOP)
        self.assertEqual(fields["principle"], "inventory_remaining_fraction")
        self.assertEqual(fields["form_class"], "piecewise_by_sign")
        self.assertEqual(fields.get("shade_error_direction"), "shade_more")

        bad = GOOD_LOOP.replace("NEXT_HYPOTHESIS:", "NEXT:")
        self.assertIsNotNone(
            validate_hypothesis_loop_reflection(bad, strategy_type="spread")
        )

    def test_shade_direction_required(self):
        no_dir = GOOD_LOOP.replace(
            "  shade_error_direction: shade_more\n",
            "",
        )
        err = validate_hypothesis_loop_reflection(no_dir, strategy_type="spread")
        self.assertIsNotNone(err)
        self.assertIn("shade_error_direction", err.lower())

    def test_generation_reads_next_hypothesis(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection=GOOD_LOOP,
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="spread",
            spread_input_mode="book",
            parent_performance={
                "reward": -50.0,
                "reward_variance": 12.5,
                "zi_benchmark_comparison": self.comparison,
            },
            q_reward_mode="zi_advantage",
            critic_mode="structured_hypothesis_loop",
        )
        self.assertIn("NEXT_HYPOTHESIS", prompt)
        self.assertIn("inventory_remaining_fraction", prompt)
        self.assertIn("Implement the NEXT claim", prompt)
        self.assertIn("shade_error_direction", prompt)
        self.assertIn("shade_more", prompt)


if __name__ == "__main__":
    unittest.main()
