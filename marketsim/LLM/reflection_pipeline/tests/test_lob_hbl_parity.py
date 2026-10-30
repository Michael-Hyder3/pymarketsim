"""Opt-in LOB hbl_parity mode + HBL background population."""

from __future__ import annotations

import time
import unittest

from marketsim.LLM.background_agents import (
    is_hbl_background,
    normalize_background_agent_kind,
)
from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.lob_features import (
    HBL_PARITY_LOB_KEYS,
    apply_lob_input_mode,
    is_lob_hbl_parity_mode,
    lob_hbl_parity_dummy_history,
    lob_input_schema_lines,
    normalize_lob_input_mode,
)
from marketsim.LLM.reflection_pipeline.prompts import principle_form_menus
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
from marketsim.fourheap.constants import BUY


class TestLobHblParity(unittest.TestCase):
    def test_normalize_and_helpers(self):
        self.assertEqual(normalize_lob_input_mode(None), "raw")
        self.assertTrue(is_lob_hbl_parity_mode("hbl_parity"))
        self.assertFalse(is_lob_hbl_parity_mode("raw"))
        with self.assertRaises(ValueError):
            normalize_lob_input_mode("not_a_mode")

    def test_schema_lock_strips_extras(self):
        raw = {
            "timestep": 5,
            "L": 6,
            "memory_start_time": 0,
            "transaction_count": 1,
            "orders": [
                {
                    "time": 1,
                    "side": 1,
                    "price_rel": -1.0,
                    "quantity": 1,
                    "accepted": True,
                    "match_price_rel": -0.5,
                    "agent_id": 1,
                    "order_id": 1,
                }
            ],
            "current_pv": 50.0,
            "grace_period": 10.0,
            "side": 1,
            "fundamental_estimate": 100000.0,
            "pricing_fundamental": 100000.0,
            "lob_snapshots": [{"best_bid_relative": -9.0}],
            "market_trades": [{"time": 1, "price": 0.0}],
            "recent_trade_prices": [0.0],
            "last_trade_price": 0.0,
            "total_trades": 1,
            "best_bid_relative": -2.0,
            "best_ask_relative": 3.0,
            "spread_relative": 5.0,
            "mid_relative": 0.5,
            "bids_relative": [-2.0],
            "asks_relative": [3.0],
        }
        out = apply_lob_input_mode(raw, lob_input_mode="hbl_parity")
        self.assertEqual(set(out.keys()), set(HBL_PARITY_LOB_KEYS))
        self.assertEqual(out["best_bid_relative"], -2.0)
        self.assertEqual(out["best_ask_relative"], 3.0)
        self.assertEqual(out["grace_period"], 10.0)
        self.assertEqual(out["side"], 1)
        self.assertEqual(out["fundamental_estimate"], 100000.0)
        self.assertEqual(out["pricing_fundamental"], 100000.0)
        self.assertEqual(len(out["orders"]), 1)
        self.assertEqual(out["orders"][0]["match_price_rel"], -0.5)
        self.assertNotIn("lob_snapshots", out)
        self.assertNotIn("market_trades", out)

    def test_schema_lines_mention_hbl_not_snapshots(self):
        parity = "\n".join(lob_input_schema_lines("hbl_parity"))
        raw = "\n".join(lob_input_schema_lines("raw"))
        self.assertIn("HBL-parity", parity)
        self.assertIn("'orders'", parity)
        self.assertIn("match_price_rel", parity)
        self.assertIn("grace_period", parity)
        # Mentioned only as "you do not receive … lob_snapshots"
        self.assertIn("do not receive", parity)
        self.assertIn("- 'lob_snapshots'", raw)
        self.assertNotIn("- 'lob_snapshots'", parity)

    def test_menus_order_stream_not_snapshot(self):
        p, f = principle_form_menus("lob", lob_input_mode="hbl_parity")
        self.assertIn("fill_vs_reject", p)
        self.assertIn("ema_over_window", f)
        self.assertNotIn("book_imbalance", p)
        self.assertNotIn("ema_over_snapshots", f)

    def test_verifier_accepts_hbl_parity_strategy(self):
        code = (
            "def agent_strategy(market_history):\n"
            "    orders = market_history.get('orders') or []\n"
            "    accepts = sum(1 for o in orders if o.get('accepted'))\n"
            "    n = max(1, len(orders))\n"
            "    return min(max(accepts / n, 0.0), 1.0)\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_hbl_parity"
        )
        self.assertTrue(ok, err)

    def test_verifier_rejects_snapshot_fields(self):
        code = (
            "def agent_strategy(market_history):\n"
            "    snaps = market_history.get('lob_snapshots') or []\n"
            "    return 0.2 if snaps else 0.5\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_hbl_parity"
        )
        self.assertFalse(ok)
        self.assertIn("hbl_parity", err or "")

    def test_dummy_history_keys(self):
        hist = lob_hbl_parity_dummy_history()
        self.assertEqual(set(hist.keys()), set(HBL_PARITY_LOB_KEYS))


class TestHblBackground(unittest.TestCase):
    def test_normalize_accepts_hbl(self):
        self.assertEqual(normalize_background_agent_kind("hbl"), "hbl")
        self.assertTrue(is_hbl_background("hbl"))
        self.assertFalse(is_hbl_background("zi"))

    def test_normalize_accepts_hbl_fast(self):
        self.assertEqual(normalize_background_agent_kind("hbl_fast"), "hbl_fast")
        self.assertTrue(is_hbl_background("hbl_fast"))

    def test_run_holdout_forwards_hbl_bg_parallel_and_lob_parity(self):
        """LOB HBL-bg holdout must forward ecology + episode parallelism."""
        import tempfile
        import textwrap
        from pathlib import Path
        from unittest import mock

        from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig
        from marketsim.LLM.reflection_pipeline.evaluation import BatchEvaluationResult
        from marketsim.LLM.reflection_pipeline.run_holdout import run_holdout

        cfg = ReflectionMarketConfig(
            label="holdout_tiny",
            num_zi_buy=1,
            num_zi_sell=1,
            timesteps=20,
            q_max=1,
            shade_range=[0.0, 0.01],
            arrival_rate=0.2,
        )

        def _fake_batch(**kwargs):
            return BatchEvaluationResult(
                reward=1.0,
                batch_data={},
                per_config_rewards={cfg.label: 1.0},
                baseline_surplus=0.5,
                per_config_reward_variances={cfg.label: 0.0},
                per_config_baseline_surplus={cfg.label: 0.5},
            )

        strategy_src = textwrap.dedent(
            """
            def agent_strategy(market_history):
                return 0.05
            """
        ).strip()
        calls = []

        with tempfile.TemporaryDirectory() as tmp:
            strategy_path = Path(tmp) / "agent_strategy.py"
            strategy_path.write_text(strategy_src + "\n", encoding="utf-8")
            with mock.patch(
                "marketsim.LLM.reflection_pipeline.run_holdout.evaluate_batch",
                side_effect=lambda **kw: calls.append(kw) or _fake_batch(**kw),
            ):
                out = run_holdout(
                    strategy_path,
                    [cfg],
                    episodes_per_config=1,
                    include_bo=False,
                    include_zip=False,
                    background_agent_kind="hbl_fast",
                    lob_input_mode="hbl_parity",
                    strategy_output_mode="shade",
                    parallel_episodes=True,
                    parallel_episode_workers=8,
                )

        self.assertEqual(out["background_agent_kind"], "hbl_fast")
        self.assertEqual(out["lob_input_mode"], "hbl_parity")
        self.assertIsNone(out["bo"])
        self.assertIsNone(out["zip"])
        kinds = [c["special_agent_kind"] for c in calls]
        self.assertEqual(kinds, ["llm", "zi", "hbl"])
        for c in calls:
            self.assertEqual(c["background_agent_kind"], "hbl_fast")
            self.assertEqual(c["lob_input_mode"], "hbl_parity")
            self.assertTrue(c["parallel_episodes"])
            self.assertEqual(c["parallel_workers"], 8)

    def test_hbl_background_episode_runs(self):
        def strategy(market_history):
            orders = market_history.get("orders") or []
            if not orders:
                return 0.5
            accepts = sum(1 for o in orders if o.get("accepted"))
            return min(max(accepts / max(1, len(orders)), 0.0), 1.0)

        result = run_episode(
            llm_strategy_func=strategy,
            llm_order_type=BUY,
            num_zi_buy=2,
            num_zi_sell=2,
            timesteps=40,
            q_max=2,
            shade_range=[0.0, 0.01],
            arrival_rate=0.2,
            random_seed=11,
            strategy_type="lob",
            lob_input_mode="hbl_parity",
            background_agent_kind="hbl",
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertIn("llm_surplus", result)
        self.assertEqual(result["market_config"]["background_agent_kind"], "hbl")
        self.assertTrue(isinstance(result["llm_surplus"], float))

    def test_episode_hbl_parity_exposes_match_price_grace_side(self):
        """Live episode history includes upgraded HBL-parity fields."""
        seen = {}

        def strategy(market_history):
            seen["keys"] = set(market_history.keys())
            seen["grace"] = market_history.get("grace_period")
            seen["side"] = market_history.get("side")
            seen["fund"] = market_history.get("fundamental_estimate")
            seen["pricing_f"] = market_history.get("pricing_fundamental")
            orders = market_history.get("orders") or []
            seen["n_orders"] = len(orders)
            if orders:
                seen["order_keys"] = set(orders[0].keys())
                seen["has_match_field"] = "match_price_rel" in orders[0]
            return 0.3

        run_episode(
            llm_strategy_func=strategy,
            llm_order_type=BUY,
            num_zi_buy=3,
            num_zi_sell=3,
            timesteps=80,
            q_max=2,
            shade_range=[0.0, 0.02],
            arrival_rate=0.2,
            obs_noise_var=0.0,
            random_seed=42,
            strategy_type="lob",
            lob_input_mode="hbl_parity",
            background_agent_kind="zi",
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertTrue(seen.get("n_orders", 0) > 0)
        self.assertEqual(seen.get("grace"), 5.0)  # 1/0.2
        self.assertEqual(seen.get("side"), BUY)
        self.assertIn("fund", seen)
        self.assertIsNotNone(seen.get("pricing_f"))
        self.assertTrue(seen.get("has_match_field"))
        for k in (
            "grace_period",
            "side",
            "fundamental_estimate",
            "pricing_fundamental",
            "orders",
            "best_bid_relative",
            "current_pv",
        ):
            self.assertIn(k, seen["keys"])
        # Never expose latent true F under any key name.
        for banned in (
            "fundamental_true",
            "true_fundamental",
            "fundamental_value",
            "F_true",
        ):
            self.assertNotIn(banned, seen["keys"])
        # Absolute F̃ only (both keys identical; no abs−true relative field).
        self.assertAlmostEqual(float(seen["pricing_f"]), 100000.0, places=3)
        self.assertAlmostEqual(float(seen["fund"]), float(seen["pricing_f"]), places=6)

    def test_special_agent_hbl_l_overrides_tracked_only(self):
        """Ladder seat can freeze a distinct L while background stays default."""
        result = run_episode(
            llm_strategy_func=lambda *_a, **_k: 0.2,
            llm_order_type=BUY,
            num_zi_buy=2,
            num_zi_sell=2,
            timesteps=30,
            q_max=2,
            shade_range=[0.0, 0.01],
            arrival_rate=0.2,
            random_seed=7,
            strategy_type="lob",
            lob_input_mode="hbl_parity",
            background_agent_kind="hbl_fast",
            special_agent_kind="hbl_fast",
            hbl_l_param=6,
            special_agent_hbl_l=2,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertIn("llm_surplus", result)
        self.assertIsInstance(result["llm_surplus"], float)


if __name__ == "__main__":
    unittest.main()
