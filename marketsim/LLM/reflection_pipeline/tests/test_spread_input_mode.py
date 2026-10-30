"""Opt-in Spread input modes: classic / zip_parity / book."""

from __future__ import annotations

import unittest

from marketsim.LLM.spread_input import (
    BOOK_SPREAD_KEYS,
    ZIP_PARITY_SPREAD_KEYS,
    build_spread_market_state,
    normalize_spread_input_mode,
    spread_input_schema_lines,
    spread_pricing_fundamental,
)
from marketsim.LLM.reflection_pipeline.pipeline import (
    _fallback_plateau_diagnosis,
    _fallback_principle_form_diagnosis,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    principle_form_menus,
)
from marketsim.LLM.reflection_pipeline.tree import (
    RefinementTree,
    _pv_seed_strategy,
    _spread_seed_strategy,
)
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier


class TestSpreadInputMode(unittest.TestCase):
    def test_normalize_rejects_unknown(self):
        with self.assertRaises(ValueError):
            normalize_spread_input_mode("not_a_mode")

    def test_classic_keeps_estimate_key(self):
        state = build_spread_market_state(
            best_bid=-1.0,
            best_ask=1.0,
            units_available=2,
            q_max=3,
            current_pv=10.0,
            fundamental_estimate=999.0,
            fundamental_true=100_000.0,
            shouts=[],
            spread_input_mode="classic",
        )
        self.assertIn("fundamental_estimate", state)
        self.assertNotIn("fundamental", state)
        self.assertEqual(state["fundamental_estimate"], 999.0)
        self.assertEqual(
            spread_pricing_fundamental(
                fundamental_estimate=999.0,
                fundamental_true=100_000.0,
                spread_input_mode="classic",
            ),
            999.0,
        )

    def test_zip_parity_uses_noisy_fundamental_and_schema_lock(self):
        shouts = [
            {
                "time": 0,
                "price_abs": 100_001.0,
                "price_rel": 1.0,
                "side": 1,
                "accepted": False,
            }
        ]
        state = build_spread_market_state(
            best_bid=-2.0,
            best_ask=3.0,
            units_available=1,
            q_max=3,
            current_pv=50.0,
            fundamental_estimate=999.0,
            fundamental_true=100_000.0,
            shouts=shouts,
            spread_input_mode="zip_parity",
        )
        self.assertEqual(set(state.keys()), set(ZIP_PARITY_SPREAD_KEYS))
        self.assertEqual(state["fundamental"], 999.0)  # estimate, not true F
        self.assertNotIn("fundamental_estimate", state)
        self.assertEqual(
            spread_pricing_fundamental(
                fundamental_estimate=999.0,
                fundamental_true=100_000.0,
                spread_input_mode="zip_parity",
            ),
            999.0,
        )

    def test_book_omits_shouts(self):
        state = build_spread_market_state(
            best_bid=-1.0,
            best_ask=2.0,
            units_available=2,
            q_max=3,
            current_pv=40.0,
            fundamental_estimate=888.0,
            fundamental_true=100_000.0,
            shouts=[{"time": 0, "accepted": True}],
            spread_input_mode="book",
            bids=[{"price_rel": -1.0, "quantity": 2.0}],
            asks=[{"price_rel": 2.0, "quantity": 1.0}],
        )
        self.assertEqual(set(state.keys()), set(BOOK_SPREAD_KEYS))
        self.assertNotIn("shouts", state)
        self.assertNotIn("prev_best_bid", state)
        self.assertEqual(state["fundamental_estimate"], 888.0)
        self.assertIsNone(state["prev_event"])
        self.assertEqual(state["bids"], [{"price_rel": -1.0, "quantity": 2.0}])
        self.assertEqual(state["asks"], [{"price_rel": 2.0, "quantity": 1.0}])

    def test_book_prev_event_and_depth(self):
        state = build_spread_market_state(
            best_bid=-1.0,
            best_ask=2.0,
            units_available=2,
            q_max=3,
            current_pv=40.0,
            fundamental_estimate=888.0,
            fundamental_true=100_000.0,
            shouts=[],
            spread_input_mode="book",
            prev_event={
                "time": 4,
                "price_abs": 1001.0,
                "price_rel": 1.0,
                "side": 1,
                "accepted": False,
            },
            bids=[{"price_rel": -1.5, "quantity": 1.0}],
            asks=[{"price_rel": 2.5, "quantity": 3.0}],
        )
        self.assertNotIn("prev_best_bid", state)
        self.assertEqual(
            state["prev_event"],
            {
                "price_abs": 1001.0,
                "price_rel": 1.0,
                "side": 1,
                "accepted": False,
            },
        )
        self.assertEqual(len(state["bids"]), 1)
        self.assertEqual(len(state["asks"]), 1)

    def test_prompt_schema_switches_with_mode(self):
        classic = "\n".join(spread_input_schema_lines("classic"))
        parity = "\n".join(spread_input_schema_lines("zip_parity"))
        book = "\n".join(spread_input_schema_lines("book"))
        self.assertIn("fundamental_estimate", classic)
        self.assertNotIn("'fundamental':", classic)
        self.assertIn("'fundamental':", parity)
        self.assertIn("noisy", parity.lower())
        self.assertNotIn("TRUE fundamental", parity)
        self.assertIn("do not receive fundamental_estimate", parity.lower())
        self.assertNotIn("'shouts'", book)
        self.assertIn("NO shouts", book)
        self.assertIn("prev_event", book)
        self.assertIn("Sp book", book)
        self.assertIn("private_values", book)
        self.assertIn("'bids'", book)
        self.assertIn("'asks'", book)
        self.assertIn("resting depth", book.lower())
        self.assertNotIn("prev_best_bid", book)
        self.assertIn("L-window orders", book)

        prompt_classic = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            use_reflection=False,
            spread_input_mode="classic",
        )
        prompt_parity = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            use_reflection=False,
            spread_input_mode="zip_parity",
        )
        prompt_book = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.1\n",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="spread",
            use_reflection=False,
            spread_input_mode="book",
        )
        self.assertIn("fundamental_estimate", prompt_classic)
        self.assertIn("'fundamental':", prompt_parity)
        self.assertIn("(fundamental + pv)", prompt_parity)
        self.assertNotIn("(fundamental_estimate + pv)", prompt_parity)
        self.assertIn("NO shouts", prompt_book)
        self.assertIn("prev_event", prompt_book)
        self.assertIn("Sp book", prompt_book)
        self.assertNotIn("'shouts'", prompt_book)

    def test_book_menus_are_sp_book_not_pv(self):
        classic_p, classic_f = principle_form_menus("spread", spread_input_mode="classic")
        book_p, book_f = principle_form_menus("spread", spread_input_mode="book")
        pv_p, pv_f = principle_form_menus("pv")
        self.assertIn("ema_over_window", classic_f)
        self.assertIn("recent_accept_reject", classic_p)
        self.assertIn("shout_competitiveness", classic_p)
        # Book mode: hybrid-B Sp-book menus (distinct from full PV 5×5 and classic).
        self.assertNotEqual(book_p, pv_p)
        self.assertNotEqual(book_f, pv_f)
        self.assertIn("mid_vs_rhat", book_p)
        self.assertNotIn("current_pv_vs_fundamental", book_p)
        self.assertIn("own_prev_feedback", book_p)
        self.assertIn("quote_liquidity", book_p)
        self.assertNotIn("unit_schedule_blend", book_f)
        self.assertNotIn("recent_accept_reject", book_p)
        self.assertNotIn("shout_competitiveness", book_p)
        self.assertNotIn("bid_ask_spread_width", book_p)
        self.assertIn("target_soft_update", book_f)
        self.assertNotIn("ema_over_window", book_f)
        self.assertEqual(len(book_p), 5)
        self.assertEqual(len(book_f), 5)

    def test_verifier_accepts_zip_parity_strategy(self):
        code = (
            "def agent_strategy(market_state):\n"
            "    if market_state.get('best_bid') is None or "
            "market_state.get('best_ask') is None:\n"
            "        return 1.0\n"
            "    _ = market_state['fundamental'] + market_state['current_pv']\n"
            "    return 0.2\n"
        )
        ok, err, fn = StrategyVerifier().verify(
            code, required_args=1, input_mode="spread_zip_parity"
        )
        self.assertTrue(ok, msg=err)
        self.assertIsNotNone(fn)

    def test_verifier_accepts_book_strategy(self):
        code = (
            "def agent_strategy(market_state):\n"
            "    bid = market_state.get('best_bid')\n"
            "    ask = market_state.get('best_ask')\n"
            "    bids = list(market_state.get('bids') or [])\n"
            "    asks = list(market_state.get('asks') or [])\n"
            "    if bid is None or ask is None:\n"
            "        return 1.0\n"
            "    depth = len(bids) + len(asks)\n"
            "    _ = market_state['fundamental_estimate']\n"
            "    prev = market_state.get('prev_event')\n"
            "    if isinstance(prev, dict) and not prev.get('accepted', True):\n"
            "        return 0.25\n"
            "    return 0.15 if depth > 0 else 0.5\n"
        )
        ok, err, fn = StrategyVerifier().verify(
            code, required_args=1, input_mode="spread_book"
        )
        self.assertTrue(ok, msg=err)
        self.assertIsNotNone(fn)
        self.assertNotIn("shouts", code)

    def test_verifier_rejects_book_crash_on_prev_event(self):
        code = (
            "def agent_strategy(market_state):\n"
            "    # Assumes prev_event is always a dict — must fail own-event probe.\n"
            "    return 0.2 if market_state['prev_event']['accepted'] else 0.8\n"
        )
        ok, err, fn = StrategyVerifier().verify(
            code, required_args=1, input_mode="spread_book"
        )
        self.assertFalse(ok)
        self.assertIsNone(fn)
        self.assertIsNotNone(err)

    def test_spread_fallback_samples_live_menu_not_hardcoded(self):
        import random

        from marketsim.LLM.reflection_pipeline.pipeline import _sample_fallback_menu_pair
        from marketsim.LLM.reflection_pipeline.prompts import menu_hypothesis_pairs

        book_pairs = menu_hypothesis_pairs("spread", spread_input_mode="book")
        self.assertEqual(len(book_pairs), 25)
        # Fixed RNG → pair is from the menu, not a hardcoded constant.
        text = _fallback_principle_form_diagnosis(
            strategy_type="spread",
            spread_input_mode="book",
            rng=random.Random(0),
        )
        sampled = _sample_fallback_menu_pair(
            strategy_type="spread",
            spread_input_mode="book",
            rng=random.Random(0),
        )
        self.assertIn(f"principle: {sampled[0]}", text)
        self.assertIn(f"form_class: {sampled[1]}", text)
        self.assertIn(sampled, book_pairs)
        self.assertIn("sampled uniformly", text)
        plateau = _fallback_plateau_diagnosis(
            strategy_type="spread",
            spread_input_mode="book",
            rng=random.Random(1),
        )
        self.assertIn("sampled uniformly", plateau)
        # Different seeds should be able to yield different pairs across the menu.
        seen = {
            _sample_fallback_menu_pair(
                strategy_type="spread",
                spread_input_mode="book",
                rng=random.Random(s),
            )
            for s in range(50)
        }
        self.assertGreater(len(seen), 1)

    def test_spread_seed_binds_bids_asks_and_prev_event(self):
        seed = _spread_seed_strategy(spread_input_mode="book")
        self.assertIn("bids", seed["code"])
        self.assertIn("asks", seed["code"])
        self.assertIn("prev_event", seed["code"])
        self.assertIn("private_values", seed["code"])
        self.assertIn("fundamental_estimate", seed["code"])
        self.assertIn("Sp book foothold", seed["code"])
        # Book quotes bind first; PV fields follow.
        self.assertLess(
            seed["code"].index("best_bid"),
            seed["code"].index("private_values"),
        )

    def test_pv_seed_is_schema_foothold_like_spread(self):
        seed = _pv_seed_strategy()
        self.assertIn("return 0.01", seed["code"])
        self.assertIn("private_values", seed["code"])
        self.assertIn("fundamental_estimate", seed["code"])
        self.assertIn("current_pv", seed["code"])
        self.assertNotIn("unit_pos", seed["code"])
        tree_code = RefinementTree(initial_types=["pv"]).get(1).code
        self.assertEqual(tree_code.strip(), seed["code"].strip())


if __name__ == "__main__":
    unittest.main()
