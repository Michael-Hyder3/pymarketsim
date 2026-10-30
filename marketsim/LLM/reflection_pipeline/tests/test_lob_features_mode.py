"""Tests for optional lob_input_mode=features (raw default unchanged)."""

from __future__ import annotations

import unittest

from marketsim.LLM.lob_features import (
    LOB_SCALAR_FEATURE_NAMES,
    apply_lob_input_mode,
    compute_lob_scalar_features,
    lob_features_dummy_history,
    lob_features_seed_strategy_code,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier


class TestLobFeaturesComputation(unittest.TestCase):
    def test_null_book_is_finite(self):
        feats = compute_lob_scalar_features(
            {
                "current_pv": 100.0,
                "lob": {
                    "best_bid_relative": None,
                    "best_ask_relative": None,
                    "bids_relative": [],
                    "asks_relative": [],
                },
                "recent_trade_prices": [],
            }
        )
        self.assertEqual(set(feats), set(LOB_SCALAR_FEATURE_NAMES))
        for name, value in feats.items():
            self.assertTrue(isinstance(value, float), name)
            self.assertEqual(value, value, name)  # not NaN
        self.assertEqual(feats["has_book"], 0.0)
        self.assertEqual(feats["imbalance"], 0.0)

    def test_features_mode_strips_nested_lists(self):
        raw = {
            "timestep": 1,
            "current_pv": 50.0,
            "lob": {
                "best_bid_relative": -5.0,
                "best_ask_relative": 5.0,
                "bids_relative": [-5.0, -10.0],
                "asks_relative": [5.0, 10.0],
            },
            "bids_relative": [-5.0, -10.0],
            "asks_relative": [5.0, 10.0],
            "spread_relative": 10.0,
            "mid_relative": 0.0,
            "recent_trade_prices": [-1.0, 2.0],
            "market_trades": [{"price": -1.0}],
            "lob_snapshots": [{"bids_relative": [-5.0]}],
            "total_trades": 2,
        }
        out = apply_lob_input_mode(raw, lob_input_mode="features")
        self.assertIn("features", out)
        self.assertEqual(out["lob_snapshots"], [])
        self.assertEqual(out["bids_relative"], [])
        self.assertEqual(out["asks_relative"], [])
        self.assertEqual(out["market_trades"], [])
        self.assertEqual(out["features"]["has_book"], 1.0)
        self.assertAlmostEqual(out["features"]["imbalance"], 0.0)
        self.assertAlmostEqual(out["features"]["trade_mom"], 3.0)

    def test_raw_mode_is_identity(self):
        history = {"current_pv": 1.0, "bids_relative": [-1.0]}
        self.assertIs(apply_lob_input_mode(history, lob_input_mode="raw"), history)


class TestLobFeaturesVerifierAndSeed(unittest.TestCase):
    def test_verifier_accepts_features_seed(self):
        code = lob_features_seed_strategy_code()
        ok, err, fn = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_features"
        )
        self.assertTrue(ok, err)
        self.assertIsNotNone(fn)
        shade = fn(lob_features_dummy_history())
        self.assertGreaterEqual(float(shade), 0.0)
        self.assertLessEqual(float(shade), 1.0)

    def test_verifier_rejects_raw_depth_access(self):
        code = "\n".join(
            [
                "def agent_strategy(market_history):",
                "    bids = market_history.get('bids_relative') or []",
                "    return 0.1 if bids else 0.0",
            ]
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_features"
        )
        self.assertFalse(ok)
        self.assertIn("forbids raw nested LOB", err or "")

    def test_tree_seed_uses_features_code(self):
        tree = RefinementTree(initial_types=["lob"], lob_input_mode="features")
        seed_id = tree.seed_node_ids()[0]
        code = tree.get(seed_id).code
        self.assertIn("features", code)
        self.assertNotIn("bids_relative", code)

    def test_tree_raw_seed_uses_order_memory(self):
        tree = RefinementTree(initial_types=["lob"], lob_input_mode="raw")
        seed_id = tree.seed_node_ids()[0]
        code = tree.get(seed_id).code
        self.assertIn("orders", code)
        self.assertIn("accepted", code)
        self.assertNotIn("bids_relative", code)


class TestLobFeaturesPrompts(unittest.TestCase):
    def test_generation_prompt_features_schema(self):
        prompt = build_generation_prompt(
            parent_code=lob_features_seed_strategy_code(),
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="lob",
            lob_input_mode="features",
        )
        self.assertIn("spread_rel", prompt)
        self.assertIn("FIXED scalar feature", prompt)
        self.assertNotIn("'lob_snapshots': list of past", prompt)

    def test_generation_prompt_raw_schema_default(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_history):\n    return 0.0",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="lob",
        )
        self.assertIn("lob_snapshots", prompt)
        self.assertNotIn("FIXED scalar feature", prompt)

    def test_reflection_prompt_features_examples(self):
        prompt = build_reflection_prompt(
            code=lob_features_seed_strategy_code(),
            batch_summary={"zi_benchmark_comparison": None},
            batch_data={},
            reward=0.0,
            critic_mode="structured_principle_form",
            strategy_type="lob",
            lob_input_mode="features",
            menu_exploration_enabled=False,
        )
        self.assertIn("spread_rel", prompt)
        self.assertIn("imbalance", prompt)


if __name__ == "__main__":
    unittest.main()
