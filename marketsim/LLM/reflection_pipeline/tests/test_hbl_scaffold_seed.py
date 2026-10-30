"""Methodology o_LOB foothold seed (book_orders fields, fixed shade 0.01)."""

from __future__ import annotations

import unittest

from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.hbl_order_memory import strategy_requests_hbl_exact_seed
from marketsim.LLM.lob_features import lob_book_orders_dummy_history
from marketsim.LLM.reflection_pipeline.code_edit import parent_is_complex
from marketsim.LLM.reflection_pipeline.learning_focused import (
    book_orders_basic_seed_code,
    hbl_distilled_lob_order_memory_seed_code,
    hbl_scaffold_lob_order_memory_seed_code,
    lob_order_memory_seed_code,
    normalize_lob_seed_scale,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
from marketsim.fourheap.constants import BUY


def _compile(code: str):
    ns: dict = {}
    exec(code, ns)
    return ns["agent_strategy"]


class TestHblScaffoldSeedWiring(unittest.TestCase):
    def test_normalize_accepts_hbl_scaffold(self):
        self.assertEqual(normalize_lob_seed_scale("hbl_scaffold"), "hbl_scaffold")

    def test_seed_binds_methodology_o_lob_only(self):
        code = hbl_scaffold_lob_order_memory_seed_code()
        self.assertEqual(code, book_orders_basic_seed_code())
        self.assertIn("def agent_strategy", code)
        for token in (
            "best_bid",
            "best_ask",
            "current_pv",
            "fundamental_estimate",
            "bids",
            "asks",
            "orders",
            "time",
            "side",
            "price_rel",
            "quantity",
            "accepted",
            "match_price_rel",
            "agent_id",
            "order_id",
            "return 0.01",
        ):
            self.assertIn(token, code)
        for banned in (
            "pricing_fundamental",
            "grace_period",
            "memory_start_time",
            "transaction_count",
            "best_bid_relative",
            "_ =",
        ):
            self.assertNotIn(banned, code)
        lowered = code.lower()
        self.assertNotIn("def belief", lowered)
        self.assertNotIn("n_grid", code)
        self.assertFalse(parent_is_complex(code), msg=f"lines={len(code.splitlines())}")
        fn = _compile(code)
        self.assertFalse(strategy_requests_hbl_exact_seed(fn))
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_book_orders"
        )
        self.assertTrue(ok, msg=err)

    def test_dispatch_and_tree(self):
        self.assertEqual(
            lob_order_memory_seed_code(
                lob_seed_scale="hbl_scaffold",
                lob_input_mode="book_orders",
            ),
            hbl_scaffold_lob_order_memory_seed_code(),
        )
        self.assertNotEqual(
            hbl_scaffold_lob_order_memory_seed_code(),
            hbl_distilled_lob_order_memory_seed_code(),
        )
        tree = RefinementTree(
            initial_types=["lob"],
            lob_input_mode="book_orders",
            lob_seed_scale="hbl_scaffold",
        )
        lob = tree.get(tree.seed_node_ids()[0])
        self.assertIn("def agent_strategy", lob.code)
        self.assertIn("return 0.01", lob.code)
        self.assertNotIn("n_grid", lob.code)
        self.assertNotIn("pricing_fundamental", lob.code)


class TestHblScaffoldFootholdBehavior(unittest.TestCase):
    def test_foothold_returns_fixed_shade_on_dummy(self):
        fn = _compile(hbl_scaffold_lob_order_memory_seed_code())
        hist = lob_book_orders_dummy_history()
        self.assertEqual(fn(hist), 0.01)
        hist_missing = dict(hist)
        hist_missing["best_bid"] = None
        self.assertEqual(fn(hist_missing), 1.0)

    def test_foothold_runs_episode_without_error(self):
        scaffold = _compile(hbl_scaffold_lob_order_memory_seed_code())
        out = run_episode(
            llm_strategy_func=scaffold,
            special_agent_kind="llm",
            num_zi_buy=5,
            num_zi_sell=5,
            timesteps=80,
            q_max=4,
            shade_range=[0.0, 0.015],
            arrival_rate=0.15,
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
            background_agent_kind="zi",
            strategy_type="lob",
            lob_input_mode="book_orders",
            llm_order_type=BUY,
            random_seed=0,
        )
        self.assertIn("llm_surplus", out)
        self.assertIsInstance(float(out["llm_surplus"]), float)

    def test_simpler_than_distill(self):
        scaffold = hbl_scaffold_lob_order_memory_seed_code()
        distill = hbl_distilled_lob_order_memory_seed_code()
        self.assertLess(len(scaffold.splitlines()), len(distill.splitlines()))
        self.assertNotIn("n_grid", scaffold)
        self.assertIn("n_grid", distill)


if __name__ == "__main__":
    unittest.main()
