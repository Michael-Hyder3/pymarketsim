import unittest

from marketsim.LLM.reflection_pipeline.pipeline import (
    CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_LOB,
    _critic_system_prompt,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    format_structured_principle_form_facts,
    principle_form_menus,
    validate_structured_principle_form_reflection,
)
from marketsim.LLM.reflection_pipeline.recipes import get_recipe, list_recipes
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


class LobPrincipleRecipeTest(unittest.TestCase):
    def test_recipe_registered(self):
        self.assertIn("simple_lob_zi_advantage_ucb_principle", list_recipes())
        recipe = get_recipe("simple_lob_zi_advantage_ucb_principle")
        self.assertEqual(recipe.critic_mode, "structured_principle_form")
        self.assertEqual(recipe.init_types, ["lob"])
        self.assertEqual(recipe.q_reward_mode, "zi_advantage")
        self.assertEqual(recipe.parent_selection_mode, "ucb")


class LobPrincipleMenusTest(unittest.TestCase):
    def test_lob_menus_are_order_memory(self):
        pv_p, pv_f = principle_form_menus("pv")
        lob_p, lob_f = principle_form_menus("lob")  # raw default: tape-only
        lob_book_p, lob_book_f = principle_form_menus(
            "lob", lob_input_mode="book_orders"
        )
        spread_p, _ = principle_form_menus("spread")
        self.assertIn("current_pv_vs_fundamental", pv_p)
        self.assertIn("current_pv_in_schedule", pv_p)
        self.assertEqual(len(pv_p) * len(pv_f), 25)
        self.assertIn("fill_vs_reject", lob_p)
        self.assertIn("side_pressure", lob_p)
        self.assertNotIn("book_imbalance", lob_p)
        self.assertNotIn("current_pv_vs_flow", lob_p)
        self.assertNotIn("current_pv_vs_fundamental", lob_p)
        # book_orders: resting depth + tape, still 5x5
        self.assertEqual(len(lob_book_p) * len(lob_book_f), 25)
        self.assertIn("book_imbalance", lob_book_p)
        self.assertIn("depth_asymmetry", lob_book_p)
        self.assertIn("fill_vs_reject", lob_book_p)
        self.assertNotIn("current_pv_vs_flow", lob_book_p)
        self.assertNotIn("side_pressure", lob_book_p)
        self.assertNotIn("fill_vs_reject", spread_p)
        self.assertIn("ema_over_window", lob_f)
        self.assertNotIn("ema_over_snapshots", lob_f)
        self.assertNotIn("unit_schedule_blend", lob_f)
        self.assertNotIn("inventory_weighted", lob_f)

    def test_features_mode_keeps_snapshot_menus(self):
        lob_p, lob_f = principle_form_menus("lob", lob_input_mode="features_full")
        self.assertIn("book_imbalance", lob_p)
        self.assertIn("ema_over_snapshots", lob_f)
        self.assertNotIn("fill_vs_reject", lob_p)

    def test_system_prompt_is_lob(self):
        prompt = _critic_system_prompt(
            "zi_advantage",
            "structured_principle_form",
            strategy_type="lob",
        )
        self.assertEqual(prompt, CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_LOB)
        self.assertIn("LOB-only", prompt)
        self.assertIn("orders", prompt)
        self.assertNotIn("PV-only: no LOB", prompt)


class LobPrinciplePromptTest(unittest.TestCase):
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

    def test_facts_allow_lob_signals(self):
        text = "\n".join(
            format_structured_principle_form_facts(
                self.comparison, reward=-50.0, strategy_type="lob"
            )
        )
        self.assertIn("LOB-only", text)
        self.assertIn("orders", text)
        self.assertNotIn("Banned topics: LOB trends", text)

    def test_facts_hbl_parity_lists_locked_fields(self):
        text = "\n".join(
            format_structured_principle_form_facts(
                self.comparison,
                reward=-50.0,
                strategy_type="lob",
                lob_input_mode="hbl_parity",
            )
        )
        self.assertIn("HBL-parity", text)
        self.assertIn("grace_period", text)
        self.assertIn("pricing_fundamental", text)
        self.assertIn("best_bid_relative", text)
        self.assertIn("NO market_trades", text)
        self.assertNotIn("spread_relative", text)

    def test_reflection_prompt_uses_lob_menus(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(market_history):\n    return 0.1\n",
            batch_summary={
                "zi_benchmark_comparison": self.comparison,
                "per_config_rewards": {"cfg_a": -30.0},
            },
            batch_data={},
            reward=-50.0,
            include_data=False,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
            strategy_type="lob",
        )
        self.assertIn("LOB-only shading strategy", prompt)
        self.assertIn("comma-separated field identifiers", prompt)
        self.assertIn("orders", prompt)
        self.assertNotIn("inventory_remaining_units", prompt)
        self.assertNotIn("ban LOB/bid-ask", prompt)

    def test_validate_lob_reflection(self):
        good = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: ratio\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted, current_pv\n"
            "  parent_form: depth ratio only\n"
            "  why_parent_fails: leftover high; ignores accept/reject in orders\n"
            "  action: replace\n"
            "  diversity_vs_parent: weight shade by reject rate\n"
            "MECHANISM: at t=10 high reject_rate and shade too low; miss fill later.\n"
            "FIX: implement fill_vs_reject with ratio form_class.\n"
        )
        self.assertIsNone(
            validate_structured_principle_form_reflection(good, strategy_type="lob")
        )
        prose = good.replace(
            "features_used: orders, accepted, current_pv",
            "features_used: prior book / trade / current_pv fields from market_history",
        )
        self.assertIsNotNone(
            validate_structured_principle_form_reflection(prose, strategy_type="lob")
        )
        # book_imbalance invalid under raw tape menus, valid under book_orders.
        snap_style = good.replace("fill_vs_reject", "book_imbalance")
        self.assertIsNotNone(
            validate_structured_principle_form_reflection(snap_style, strategy_type="lob")
        )
        self.assertIsNone(
            validate_structured_principle_form_reflection(
                snap_style,
                strategy_type="lob",
                lob_input_mode="book_orders",
            )
        )

    def test_generation_prompt_allows_lob_and_forbids_pv_schedule(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_history):\n    return 0.1\n",
            parent_reflection=(
                "DIAGNOSIS:\n"
                "  principle: side_pressure\n"
                "  form_class: threshold\n"
                "  shade_error_direction: shade_more\n"
                "  features_used: orders, side, current_pv\n"
                "  parent_form: depth ratio\n"
                "  why_parent_fails: ignores bid/ask order flow\n"
                "  diversity_vs_parent: threshold on side_pressure\n"
                "MECHANISM: buy-heavy window with low shade; miss.\n"
                "FIX: threshold form under side_pressure.\n"
            ),
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="lob",
            use_reflection=True,
            parent_performance={
                "reward": -50.0,
                "reward_variance": 1.0,
                "zi_benchmark_comparison": self.comparison,
            },
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
        )
        self.assertIn("agent_strategy(market_history)", prompt)
        self.assertNotIn("units_available", prompt)
        self.assertNotIn("q_max", prompt)
        self.assertIn("market_history→shade", prompt)
        self.assertIn("Use market_history only", prompt)
        self.assertIn("orders", prompt)
        self.assertIn("sanity-check the shade expression", prompt)
        self.assertNotIn("Do NOT use LOB, bid/ask", prompt)
        self.assertNotIn("Use private_values only; no LOB", prompt)
        self.assertNotIn("inventory_weighted", prompt)


class LobSeedTest(unittest.TestCase):
    def test_raw_lob_seed_uses_orders(self):
        tree = RefinementTree(initial_types=["lob"], lob_input_mode="raw")
        node = tree.get(1)
        self.assertIsNotNone(node)
        self.assertIn("orders", node.code)
        self.assertIn("accepted", node.code)
        self.assertNotIn("bids_relative", node.code)


class LobVerifierTest(unittest.TestCase):
    def test_lob_dummy_rejects_missing_inventory_dependency(self):
        code = (
            "def agent_strategy(market_history):\n"
            "    orders = market_history.get('orders') or []\n"
            "    if not orders:\n"
            "        return 0.2\n"
            "    accepted = sum(1 for o in orders if o.get('accepted'))\n"
            "    return min(1.0, accepted / len(orders))\n"
        )
        ok, err, fn = StrategyVerifier().verify(code, required_args=1, input_mode="lob")
        self.assertTrue(ok, err)
        self.assertIsNotNone(fn)


if __name__ == "__main__":
    unittest.main()
