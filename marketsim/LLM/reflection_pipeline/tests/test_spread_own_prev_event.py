"""Book-mode prev_event must be the tracked seat's own last submit."""

from __future__ import annotations

import unittest

from marketsim.LLM.episode_runner import _append_llm_shout_log
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
from marketsim.LLM.spread_input import build_spread_market_state


class _FakeLLM:
    def __init__(self) -> None:
        self._shout_log = []
        self._own_event_log = []


class SpreadOwnPrevEventTest(unittest.TestCase):
    def test_append_separates_market_tape_and_own_log(self):
        agent = _FakeLLM()
        _append_llm_shout_log(
            agent,
            time=0,
            price_abs=100.0,
            price_rel=1.0,
            side=1,
            accepted=False,
            is_own=False,
        )
        _append_llm_shout_log(
            agent,
            time=1,
            price_abs=101.0,
            price_rel=2.0,
            side=-1,
            accepted=True,
            is_own=True,
        )
        self.assertEqual(len(agent._shout_log), 2)
        self.assertEqual(len(agent._own_event_log), 1)
        self.assertEqual(agent._own_event_log[0]["price_rel"], 2.0)
        self.assertTrue(agent._own_event_log[0]["accepted"])

    def test_book_prev_event_uses_own_not_market_last(self):
        # Mimic take_action: prev_event from own log, shouts from full tape.
        own_last = {
            "time": 1,
            "price_abs": 101.0,
            "price_rel": 2.0,
            "side": -1,
            "accepted": True,
        }
        market_tape = [
            {
                "time": 0,
                "price_abs": 100.0,
                "price_rel": 1.0,
                "side": 1,
                "accepted": False,
            },
            own_last,
            {
                "time": 2,
                "price_abs": 102.0,
                "price_rel": 3.0,
                "side": 1,
                "accepted": False,
            },
        ]
        state = build_spread_market_state(
            best_bid=-1.0,
            best_ask=1.0,
            units_available=2,
            q_max=3,
            current_pv=40.0,
            fundamental_estimate=888.0,
            fundamental_true=100_000.0,
            shouts=market_tape,
            spread_input_mode="book",
            prev_event=own_last,
        )
        self.assertNotIn("shouts", state)
        self.assertEqual(state["prev_event"]["price_rel"], 2.0)
        self.assertTrue(state["prev_event"]["accepted"])
        # Last market shout was price_rel=3; book must not expose that as prev_event.
        self.assertNotEqual(state["prev_event"]["price_rel"], 3.0)

    def test_spread_seed_verifies_under_book_mode(self):
        tree = RefinementTree(initial_types=["spread"], spread_input_mode="book")
        seed = tree._nodes[1]
        self.assertEqual(seed.metadata.get("type"), "spread")
        ok, err, fn = StrategyVerifier().verify(
            seed.code, required_args=1, input_mode="spread_book"
        )
        self.assertTrue(ok, msg=err)
        self.assertIsNotNone(fn)
        self.assertIn("prev_event", seed.code)
        self.assertIn("private_values", seed.code)
        self.assertIn("Sp book foothold", seed.code)


if __name__ == "__main__":
    unittest.main()
