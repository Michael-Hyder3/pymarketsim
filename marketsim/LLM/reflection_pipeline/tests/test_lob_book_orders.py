"""LOB book_orders mode: Spread-like state + L-window orders (L not exposed)."""

from __future__ import annotations

import unittest

from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.lob_features import (
    BOOK_ORDERS_LOB_KEYS,
    apply_lob_input_mode,
    is_lob_book_orders_mode,
    lob_book_orders_dummy_history,
    lob_input_schema_lines,
    normalize_lob_input_mode,
)
from marketsim.LLM.reflection_pipeline.learning_focused import (
    book_orders_basic_seed_code,
    lob_order_memory_seed_code,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
from marketsim.fourheap.constants import BUY


class TestLobBookOrders(unittest.TestCase):
    def test_normalize(self):
        self.assertTrue(is_lob_book_orders_mode("book_orders"))
        self.assertEqual(normalize_lob_input_mode("book_orders"), "book_orders")

    def test_schema_lock(self):
        raw = {
            "best_bid_relative": -2.0,
            "best_ask_relative": 3.0,
            "bids": [
                {"price_rel": -2.0, "quantity": 1.0},
                {"price_rel": -8.0, "quantity": 2.0},
            ],
            "asks": [
                {"price_rel": 3.0, "quantity": 1.0},
                {"price_rel": 9.0, "quantity": 4.0},
            ],
            "current_pv": 50.0,
            "fundamental_estimate": 100000.0,
            "orders": [{"time": 1, "side": 1, "price_rel": -1.0, "accepted": True}],
            "L": 6,
            "grace_period": 10.0,
            "lob_snapshots": [{}],
        }
        out = apply_lob_input_mode(raw, lob_input_mode="book_orders")
        self.assertEqual(set(out.keys()), set(BOOK_ORDERS_LOB_KEYS))
        self.assertEqual(out["best_bid"], -2.0)
        self.assertEqual(out["best_ask"], 3.0)
        self.assertEqual(len(out["bids"]), 2)
        self.assertEqual(len(out["asks"]), 2)
        self.assertEqual(out["bids"][1]["quantity"], 2.0)
        self.assertNotIn("L", out)
        self.assertNotIn("grace_period", out)

    def test_schema_lines_omit_l(self):
        text = "\n".join(lob_input_schema_lines("book_orders"))
        self.assertIn("best_bid", text)
        self.assertIn("'bids'", text)
        self.assertIn("'asks'", text)
        self.assertIn("orders", text)
        self.assertIn("NOT provided as a field", text)
        self.assertNotIn("- 'L':", text)

    def test_episode_exposes_full_resting_book(self):
        seen = {"max_bids": 0, "max_asks": 0}

        def strategy(market_history):
            bids = list(market_history.get("bids") or [])
            asks = list(market_history.get("asks") or [])
            seen["max_bids"] = max(seen["max_bids"], len(bids))
            seen["max_asks"] = max(seen["max_asks"], len(asks))
            seen["keys"] = set(market_history.keys())
            if bids:
                seen["bid0"] = bids[0]
            return 0.01

        run_episode(
            llm_strategy_func=strategy,
            llm_order_type=BUY,
            num_zi_buy=6,
            num_zi_sell=6,
            timesteps=120,
            q_max=3,
            shade_range=[0.0, 0.02],
            arrival_rate=0.25,
            obs_noise_var=0.0,
            random_seed=7,
            strategy_type="lob",
            lob_input_mode="book_orders",
            background_agent_kind="zi",
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertEqual(seen.get("keys"), set(BOOK_ORDERS_LOB_KEYS))
        # With several ZI agents, the resting book should exceed a single BBO level.
        self.assertGreaterEqual(seen["max_bids"] + seen["max_asks"], 2)
        if "bid0" in seen:
            self.assertIn("price_rel", seen["bid0"])
            self.assertIn("quantity", seen["bid0"])

    def test_basic_seed_binds_fields(self):
        code = book_orders_basic_seed_code()
        self.assertIn("orders = list(", code)
        self.assertIn("current_pv", code)
        self.assertIn("fundamental_estimate", code)
        self.assertIn("best_bid", code)
        self.assertIn("best_ask", code)
        ns: dict = {}
        exec(code, ns)
        shade = ns["agent_strategy"](lob_book_orders_dummy_history())
        self.assertGreaterEqual(shade, 0.0)
        self.assertLessEqual(shade, 1.0)

    def test_tree_seed_uses_book_orders_code(self):
        tree = RefinementTree(
            initial_types=["lob"],
            lob_input_mode="book_orders",
            lob_seed_scale="basic",
        )
        lob = tree.get(tree.seed_node_ids()[0])
        self.assertEqual(lob.code, lob_order_memory_seed_code(lob_input_mode="book_orders"))
        self.assertIn("best_bid", lob.code)

    def test_verifier_accepts_seed(self):
        ok, err, _ = StrategyVerifier().verify(
            book_orders_basic_seed_code(),
            required_args=1,
            input_mode="lob_book_orders",
        )
        self.assertTrue(ok, err)

    def test_verifier_rejects_l_access(self):
        code = (
            "def agent_strategy(market_history):\n"
            "    L = market_history.get('L')\n"
            "    return 0.01\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_book_orders"
        )
        self.assertFalse(ok)
        self.assertIn("L", err or "")

    def test_verifier_allows_local_spread_relative_name(self):
        """Local vars may be named spread_relative if derived from BBO."""
        code = (
            "def agent_strategy(market_history):\n"
            "    best_bid = market_history.get('best_bid')\n"
            "    best_ask = market_history.get('best_ask')\n"
            "    bids = list(market_history.get('bids') or [])\n"
            "    asks = list(market_history.get('asks') or [])\n"
            "    if best_bid is None or best_ask is None:\n"
            "        return 0.5\n"
            "    spread_relative = best_ask - best_bid\n"
            "    bids_relative = [b.get('price_rel') or 0.0 for b in bids]\n"
            "    asks_relative = [a.get('price_rel') or 0.0 for a in asks]\n"
            "    imb = 0.0\n"
            "    if spread_relative:\n"
            "        bid_p = sum(bids_relative) / max(len(bids_relative), 1)\n"
            "        ask_p = sum(asks_relative) / max(len(asks_relative), 1)\n"
            "        imb = (bid_p - ask_p) / spread_relative\n"
            "    shade = 0.5 + 0.25 * max(-1.0, min(1.0, imb))\n"
            "    return min(max(shade, 0.0), 1.0)\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_book_orders"
        )
        self.assertTrue(ok, err)

    def test_verifier_rejects_market_history_spread_relative_key(self):
        code = (
            "def agent_strategy(market_history):\n"
            "    spread_relative = market_history.get('spread_relative')\n"
            "    return 0.01\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_book_orders"
        )
        self.assertFalse(ok)
        self.assertIn("spread_relative", err or "")

    def test_episode_exposes_book_orders_keys_only(self):
        seen = {}

        def strategy(market_history):
            seen["keys"] = set(market_history.keys())
            seen["has_L"] = "L" in market_history
            seen["fund"] = market_history.get("fundamental_estimate")
            seen["bb"] = market_history.get("best_bid")
            return 0.01

        run_episode(
            llm_strategy_func=strategy,
            llm_order_type=BUY,
            num_zi_buy=3,
            num_zi_sell=3,
            timesteps=60,
            q_max=2,
            shade_range=[0.0, 0.02],
            arrival_rate=0.2,
            obs_noise_var=0.0,
            random_seed=42,
            strategy_type="lob",
            lob_input_mode="book_orders",
            background_agent_kind="zi",
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertEqual(seen.get("keys"), set(BOOK_ORDERS_LOB_KEYS))
        self.assertFalse(seen.get("has_L"))
        self.assertIsNotNone(seen.get("fund"))
        self.assertAlmostEqual(float(seen["fund"]), 100000.0, places=3)


if __name__ == "__main__":
    unittest.main()
