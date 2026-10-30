from typing import Optional
import unittest

from marketsim.LLM.reflection_pipeline.pipeline import (
    CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_SPREAD,
    CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_SPREAD_BOOK,
    _critic_system_prompt,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    combination_generator_instructions,
    format_structured_principle_form_facts,
    principle_form_menus,
    validate_structured_principle_form_reflection,
)
from marketsim.LLM.reflection_pipeline.recipes import get_recipe, list_recipes
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


class SpreadPrincipleRecipeTest(unittest.TestCase):
    def test_recipe_registered(self):
        self.assertIn("simple_spread_zi_advantage_ucb_principle", list_recipes())
        recipe = get_recipe("simple_spread_zi_advantage_ucb_principle")
        self.assertEqual(recipe.critic_mode, "structured_principle_form")
        self.assertEqual(recipe.init_types, ["spread"])
        self.assertEqual(recipe.q_reward_mode, "zi_advantage")
        self.assertEqual(recipe.parent_selection_mode, "ucb")


class SpreadPrincipleMenusTest(unittest.TestCase):
    def test_spread_menus_differ_from_pv_and_lob(self):
        pv_p, pv_f = principle_form_menus("pv")
        lob_p, lob_f = principle_form_menus("lob")
        spread_p, spread_f = principle_form_menus("spread")

        self.assertIn("bid_ask_spread_width", spread_p)
        self.assertIn("inventory_remaining_fraction", spread_p)
        self.assertNotIn("current_pv_in_schedule", spread_p)
        self.assertNotIn("inventory_remaining_units", spread_p)
        self.assertNotIn("prior_spread_or_mid", spread_p)
        self.assertNotIn("fill_vs_reject", spread_p)

        self.assertIn("clamp_linear", spread_f)
        self.assertIn("inventory_weighted", spread_f)
        self.assertNotIn("schedule_index", spread_f)
        self.assertNotIn("ema_over_snapshots", spread_f)
        # Classic/zip_parity may use ema_over_window over the shout tape;
        # LOB uses the same form name over order memory — menus still differ
        # via principles.
        self.assertIn("ema_over_window", spread_f)

        self.assertNotEqual(spread_p, pv_p)
        self.assertNotEqual(spread_p, lob_p)

        # Book mode: hybrid-B Sp-book menus (mid_vs_rhat; not sticky unit-vs-F̃).
        book_p, book_f = principle_form_menus("spread", spread_input_mode="book")
        self.assertNotEqual(book_p, pv_p)
        self.assertNotEqual(book_f, pv_f)
        self.assertIn("mid_vs_rhat", book_p)
        self.assertNotIn("current_pv_vs_fundamental", book_p)
        self.assertIn("pv_vs_touch", book_p)
        self.assertIn("inventory_remaining_fraction", book_p)
        self.assertIn("own_prev_feedback", book_p)
        self.assertIn("quote_liquidity", book_p)
        self.assertNotIn("bid_ask_spread_width", book_p)
        self.assertNotIn("book_imbalance", book_p)
        self.assertNotIn("last_own_accept_reject", book_p)
        self.assertNotIn("unit_schedule_blend", book_f)
        self.assertIn("target_soft_update", book_f)
        self.assertNotIn("ema_over_window", book_f)
        self.assertEqual(len(book_p), 5)
        self.assertEqual(len(book_f), 5)

        # Legacy pvtwin preset: book observation keeps PV 5×5 menus.
        twin_p, twin_f = principle_form_menus(
            "spread",
            spread_input_mode="book",
            preset="spread_book_pvtwin",
        )
        self.assertEqual(twin_p, pv_p)
        self.assertEqual(twin_f, pv_f)
        self.assertIn("current_pv_vs_fundamental", twin_p)

        econ_p, econ_f = principle_form_menus(
            "spread",
            spread_input_mode="book",
            preset="spread_book_econmenus",
        )
        self.assertIn("bid_ask_spread_width", econ_p)
        self.assertIn("ema_over_window", econ_f)

    def test_system_prompt_is_spread(self):
        prompt = _critic_system_prompt(
            "zi_advantage",
            "structured_principle_form",
            strategy_type="spread",
        )
        self.assertEqual(prompt, CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_SPREAD)
        self.assertIn("spread menu pair", prompt)
        self.assertIn("shade_error_direction", prompt)
        self.assertIn("shouts", prompt)
        self.assertNotIn("PV-only: no LOB", prompt)

    def test_system_prompt_book_mode_bans_shouts(self):
        prompt = _critic_system_prompt(
            "zi_advantage",
            "structured_principle_form",
            strategy_type="spread",
            spread_input_mode="book",
        )
        self.assertIn("shade_error_direction", prompt)
        self.assertIn("NO shouts", prompt)
        self.assertIn("prev_event", prompt)
        self.assertIn("private_values", prompt)
        self.assertIn("fundamental_estimate", prompt)
        self.assertIn("current_pv", prompt)
        self.assertIn("market_state", prompt)
        self.assertIn("mid_vs_rhat", prompt)
        self.assertIn("own_prev_feedback", prompt)
        self.assertIn("quote_liquidity", prompt)
        self.assertNotIn("current_pv_vs_fundamental", prompt)
        self.assertTrue(
            prompt.startswith(
                "Return a field-first DIAGNOSIS for a spread-book shading strategy"
            )
        )
        self.assertIn(CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_SPREAD_BOOK, prompt)


class SpreadPrinciplePromptTest(unittest.TestCase):
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

    def test_facts_allow_spread_signals(self):
        text = "\n".join(
            format_structured_principle_form_facts(
                self.comparison, reward=-50.0, strategy_type="spread"
            )
        )
        self.assertIn("market_state", text)
        self.assertIn("shouts", text)
        self.assertNotIn("Banned topics: LOB trends", text)
        self.assertNotIn("in private_values", text)

    def test_reflection_prompt_uses_spread_menus(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(market_state):\n    return 0.1\n",
            batch_summary={
                "zi_benchmark_comparison": self.comparison,
                "per_config_rewards": {"cfg_a": -30.0},
            },
            batch_data={},
            reward=-50.0,
            include_data=False,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
            strategy_type="spread",
        )
        self.assertIn("spread-only shading strategy", prompt)
        self.assertIn("bid_ask_spread_width", prompt)
        self.assertIn("clamp_linear", prompt)
        self.assertNotIn("inventory_remaining_units", prompt)
        self.assertNotIn("prior_spread_or_mid", prompt)
        self.assertNotIn("ban LOB/bid-ask", prompt)

    def test_validate_spread_reflection(self):
        good = (
            "DIAGNOSIS:\n"
            "  principle: pv_vs_mid\n"
            "  form_class: threshold\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: best_bid, best_ask, current_pv\n"
            "  parent_form: constant shade\n"
            "  why_parent_fails: leftover high; ignores spread and inventory\n"
            "  action: replace\n"
            "  diversity_vs_parent: threshold on pv vs mid\n"
            "MECHANISM: at t=10 wide spread and shade too low; miss fill later.\n"
            "FIX: implement pv_vs_mid with threshold form_class.\n"
        )
        self.assertIsNone(
            validate_structured_principle_form_reflection(good, strategy_type="spread")
        )
        pv_style = good.replace("pv_vs_mid", "inventory_remaining_units")
        self.assertIsNotNone(
            validate_structured_principle_form_reflection(pv_style, strategy_type="spread")
        )

    def test_generation_prompt_allows_spread_and_forbids_pv_schedule(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection=(
                "DIAGNOSIS:\n"
                "  principle: bid_ask_spread_width\n"
                "  form_class: ratio\n"
                "  shade_error_direction: shade_more\n"
                "  features_used: best_bid, best_ask\n"
                "  parent_form: constant shade\n"
                "  why_parent_fails: ignores spread\n"
                "  diversity_vs_parent: ratio on spread width\n"
                "MECHANISM: wide spread with low shade; miss.\n"
                "FIX: ratio form under bid_ask_spread_width.\n"
            ),
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            use_reflection=True,
            parent_performance={
                "reward": -50.0,
                "reward_variance": 1.0,
                "zi_benchmark_comparison": self.comparison,
            },
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
        )
        self.assertIn("agent_strategy(market_state)", prompt)
        self.assertIn("market_state→shade", prompt)
        self.assertIn("Use market_state only", prompt)
        self.assertNotIn("Do NOT use LOB, bid/ask", prompt)
        self.assertNotIn("Use private_values only; no LOB", prompt)
        self.assertIn("'shouts'", prompt)
        self.assertIn("fundamental_estimate", prompt)

    def test_generation_prompt_book_mode_bans_shouts(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection=(
                "DIAGNOSIS:\n"
                "  principle: mid_vs_rhat\n"
                "  form_class: ratio\n"
                "  shade_error_direction: shade_more\n"
                "  features_used: best_bid, best_ask, fundamental_estimate\n"
                "  parent_form: constant shade\n"
                "  why_parent_fails: ignores mid vs F̃\n"
                "  diversity_vs_parent: ratio on mid vs r̂\n"
                "MECHANISM: mid below F̃ with low shade; miss.\n"
                "FIX: ratio form under mid_vs_rhat.\n"
            ),
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            spread_input_mode="book",
            use_reflection=True,
            parent_performance={
                "reward": -50.0,
                "reward_variance": 1.0,
                "zi_benchmark_comparison": self.comparison,
            },
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
        )
        self.assertIn("agent_strategy(market_state)", prompt)
        self.assertIn("prev_event", prompt)
        self.assertIn("NO shouts", prompt)
        self.assertNotIn("You MAY use market_state['shouts']", prompt)
        self.assertIn("fundamental_estimate", prompt)
        self.assertIn("private_values", prompt)
        self.assertIn("quote liquidity", prompt.lower())
        self.assertNotIn("Do NOT use LOB, bid/ask", prompt)

    def test_combination_gate_is_spread_specific(self):
        lines = combination_generator_instructions(strategy_type="spread")
        text = "\n".join(lines)
        self.assertIn("prev_event", text)
        self.assertIn("quote liquidity", text)
        self.assertNotIn("pv_mean vs pv_max", text)

    def test_generation_prompt_forbids_unshaded_missing_quotes(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection=(
                "DIAGNOSIS:\n"
                "  principle: bid_ask_spread_width\n"
                "  form_class: ratio\n"
                "  shade_error_direction: shade_more\n"
                "  features_used: best_bid, best_ask\n"
                "  parent_form: constant shade\n"
                "  why_parent_fails: ignores spread\n"
                "  diversity_vs_parent: ratio on spread width\n"
                "MECHANISM: wide spread with low shade; miss.\n"
                "FIX: ratio form under bid_ask_spread_width.\n"
            ),
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            use_reflection=True,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
        )
        self.assertIn("do not treat none as 0.0", prompt.lower())
        self.assertNotIn("fall back to pv", prompt.lower())
        self.assertNotIn("return a high shade (e.g. 1.0)", prompt)
        self.assertIn("shouts", prompt)
        self.assertIn("fundamental_estimate", prompt)

    def test_generation_prompt_book_mode_forbids_unshaded_missing_quotes(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection=(
                "DIAGNOSIS:\n"
                "  principle: own_prev_feedback\n"
                "  form_class: threshold\n"
                "  shade_error_direction: shade_more\n"
                "  features_used: prev_event, best_bid, best_ask\n"
                "  parent_form: constant shade\n"
                "  why_parent_fails: ignores own accept/reject\n"
                "  diversity_vs_parent: threshold on prev_event\n"
                "MECHANISM: rejected last submit; shade too low.\n"
                "FIX: threshold form under own_prev_feedback.\n"
            ),
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            spread_input_mode="book",
            use_reflection=True,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
        )
        self.assertIn("do not treat none as 0.0", prompt.lower())
        self.assertNotIn("fall back to pv", prompt.lower())
        self.assertNotIn("return a high shade (e.g. 1.0)", prompt)
        self.assertIn("prev_event", prompt)
        self.assertIn("bids", prompt.lower())
        self.assertIn("asks", prompt.lower())
        self.assertIn("private_values", prompt)
        self.assertNotIn("You MAY use market_state['shouts']", prompt)


class LastHoldComparisonTest(unittest.TestCase):
    def _diagnosis(
        self,
        *,
        principle: str = "pv_vs_mid",
        form_class: str = "threshold",
        same_failure: Optional[str] = None,
        action: Optional[str] = None,
    ) -> str:
        extra = ""
        if same_failure is not None:
            extra = f"  same_failure_as_last_hold: {same_failure}\n"
        if action is None:
            action = "refine" if same_failure == "yes" else "replace"
        return (
            "DIAGNOSIS:\n"
            f"  principle: {principle}\n"
            f"  form_class: {form_class}\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: best_bid, best_ask, current_pv\n"
            "  parent_form: constant shade\n"
            "  why_parent_fails: leftover high; ignores spread and inventory\n"
            f"  action: {action}\n"
            f"{extra}"
            "  diversity_vs_parent: threshold on pv vs mid\n"
            "MECHANISM: at t=10 wide spread and shade too low; miss fill later.\n"
            "FIX: implement pv_vs_mid with threshold form_class.\n"
        )

    def test_prompt_requires_same_failure_field_when_last_hold_present(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(market_state):\n    return 0.1\n",
            batch_summary={"per_config_rewards": {"market_0001": -30.0}},
            batch_data={
                "prior_hold_card": {
                    "label": "market_0000",
                    "leftover": 0.9,
                    "prior_hold": True,
                }
            },
            reward=-50.0,
            include_data=True,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
            strategy_type="spread",
            run_explored_pairs={("pv_vs_mid", "threshold")},
        )
        self.assertIn("same_failure_as_last_hold", prompt)
        self.assertIn(
            "LAST HOLD: same_failure_as_last_hold=yes requires action=refine",
            prompt,
        )
        self.assertIn("If action=refine: KEEP the parent", prompt)

    def test_validate_requires_same_failure_when_flag_set(self):
        err = validate_structured_principle_form_reflection(
            self._diagnosis(),
            strategy_type="spread",
            require_last_hold_comparison=True,
        )
        self.assertIsNotNone(err)
        self.assertIn("same_failure_as_last_hold", err or "")

    def test_yes_keeps_parent_pair_even_if_explored(self):
        parent = self._diagnosis()
        child = self._diagnosis(same_failure="yes")
        err = validate_structured_principle_form_reflection(
            child,
            strategy_type="spread",
            parent_reflection=parent,
            run_explored_pairs={("pv_vs_mid", "threshold")},
            require_last_hold_comparison=True,
        )
        self.assertIsNone(err)

    def test_yes_rejects_new_pair(self):
        parent = self._diagnosis()
        child = self._diagnosis(
            principle="bid_ask_spread_width",
            form_class="ratio",
            same_failure="yes",
        )
        err = validate_structured_principle_form_reflection(
            child,
            strategy_type="spread",
            parent_reflection=parent,
            require_last_hold_comparison=True,
        )
        self.assertIsNotNone(err)
        self.assertIn("action=refine requires keeping the parent", err or "")
        self.assertIn("re-implementing under that pair", err or "")

    def test_yes_rejects_replace_action(self):
        parent = self._diagnosis()
        child = self._diagnosis(same_failure="yes", action="replace")
        err = validate_structured_principle_form_reflection(
            child,
            strategy_type="spread",
            parent_reflection=parent,
            require_last_hold_comparison=True,
        )
        self.assertIsNotNone(err)
        self.assertIn("same_failure_as_last_hold=yes requires action: refine", err or "")

    def test_no_rejects_refine_action(self):
        parent = self._diagnosis()
        child = self._diagnosis(
            principle="bid_ask_spread_width",
            form_class="ratio",
            same_failure="no",
            action="refine",
        )
        err = validate_structured_principle_form_reflection(
            child,
            strategy_type="spread",
            parent_reflection=parent,
            require_last_hold_comparison=True,
        )
        self.assertIsNotNone(err)
        self.assertIn("same_failure_as_last_hold=no requires action: replace", err or "")

    def test_no_still_requires_new_pair(self):
        parent = self._diagnosis()
        child = self._diagnosis(same_failure="no")
        err = validate_structured_principle_form_reflection(
            child,
            strategy_type="spread",
            parent_reflection=parent,
            require_last_hold_comparison=True,
        )
        self.assertIsNotNone(err)
        self.assertIn("action=replace requires a new", err or "")

    def test_generator_keep_instructions(self):
        from marketsim.LLM.reflection_pipeline.prompts import (
            last_hold_generator_instructions,
            parse_same_failure_as_last_hold,
        )

        self.assertEqual(parse_same_failure_as_last_hold(self._diagnosis()), None)
        self.assertEqual(
            parse_same_failure_as_last_hold(self._diagnosis(same_failure="yes")),
            "yes",
        )
        text = "\n".join(
            last_hold_generator_instructions(
                same_failure="yes",
                parent_principle="pv_vs_mid",
                parent_form="threshold",
            )
        )
        self.assertIn("KEEP", text)
        self.assertIn("pv_vs_mid + threshold", text)


class SpreadMissingQuoteVerifierTest(unittest.TestCase):
    def test_rejects_menu_label_scaffolding_in_code(self):
        from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier

        code = (
            "def agent_strategy(market_state):\n"
            "    principle = 'current_pv_vs_fundamental'\n"
            "    form_class = 'ratio'\n"
            "    if principle == 'current_pv_vs_fundamental':\n"
            "        return 0.01\n"
            "    return 0.01\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="spread"
        )
        self.assertFalse(ok)
        self.assertIn("Menu-label scaffolding", err or "")

    def test_accepts_any_finite_shade_on_missing_quote(self):
        """Missing BBO must not force a prescribed shade; any [0,1] shade is valid."""
        from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier

        code = (
            "def agent_strategy(market_state):\n"
            "    best_bid = market_state.get('best_bid')\n"
            "    best_ask = market_state.get('best_ask')\n"
            "    if best_bid is None or best_ask is None:\n"
            "        return 0.01\n"
            "    return 0.01\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="spread"
        )
        self.assertTrue(ok, err)

    def test_accepts_high_shade_on_missing_quote(self):
        from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
        from marketsim.LLM.reflection_pipeline.tree import RefinementTree

        code = (
            "def agent_strategy(market_state):\n"
            "    best_bid = market_state.get('best_bid')\n"
            "    best_ask = market_state.get('best_ask')\n"
            "    if best_bid is None or best_ask is None:\n"
            "        return 1.0\n"
            "    spread = best_ask - best_bid\n"
            "    if spread <= 0:\n"
            "        return 0.0\n"
            "    return 0.2\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="spread"
        )
        self.assertTrue(ok, err)
        seed = RefinementTree(initial_types=["spread"]).get(1).code
        ok_seed, err_seed, _ = StrategyVerifier().verify(
            seed, required_args=1, input_mode="spread"
        )
        self.assertTrue(ok_seed, err_seed)
        self.assertIn("return 0.01", seed)
        self.assertNotIn(
            "if best_bid is None or best_ask is None:\n        return 1.0",
            seed,
        )


if __name__ == "__main__":
    unittest.main()
