"""Opt-in ZIP background population + Spread menu sanity."""

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from marketsim.LLM.background_agents import (
    normalize_background_agent_kind,
    is_zip_background,
)
from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.reflection_pipeline.prompts import (
    principle_form_menus,
    menu_hypothesis_pairs,
)
from marketsim.fourheap.constants import BUY


class TestBackgroundAgentKind(unittest.TestCase):
    def test_normalize_default_zi(self):
        self.assertEqual(normalize_background_agent_kind(None), "zi")
        self.assertFalse(is_zip_background("zi"))
        self.assertTrue(is_zip_background("zip"))
        self.assertTrue(is_zip_background("zip_noisy"))
        self.assertTrue(is_zip_background("zip_fixed"))
        self.assertEqual(normalize_background_agent_kind("zip_noisy"), "zip_noisy")
        self.assertEqual(normalize_background_agent_kind("zip_fixed"), "zip_fixed")

    def test_normalize_rejects_unknown(self):
        with self.assertRaises(ValueError):
            normalize_background_agent_kind("bogus")

    def test_normalize_accepts_hbl(self):
        self.assertEqual(normalize_background_agent_kind("hbl"), "hbl")

    def test_zip_background_episode_runs(self):
        def strategy(market_state):
            if market_state.get("best_bid") is None or market_state.get("best_ask") is None:
                return 1.0
            return 0.05

        result = run_episode(
            llm_strategy_func=strategy,
            llm_order_type=BUY,
            num_zi_buy=2,
            num_zi_sell=2,
            timesteps=40,
            q_max=2,
            shade_range=[0.0, 0.01],
            arrival_rate=0.2,
            random_seed=7,
            strategy_type="spread",
            spread_input_mode="zip_parity",
            background_agent_kind="zip",
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertIn("llm_surplus", result)
        # zip is always coerced to zip_noisy (noisy / projected F̂).
        self.assertEqual(result["market_config"]["background_agent_kind"], "zip_noisy")
        # Episode still completes with finite surplus.
        self.assertTrue(isinstance(result["llm_surplus"], float))

    def test_zip_noisy_background_episode_runs(self):
        def strategy(market_state):
            if market_state.get("best_bid") is None or market_state.get("best_ask") is None:
                return 1.0
            return 0.05

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
            strategy_type="spread",
            spread_input_mode="classic",
            background_agent_kind="zip_noisy",
            obs_noise_var=1e6,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertIn("llm_surplus", result)
        self.assertEqual(result["market_config"]["background_agent_kind"], "zip_noisy")
        self.assertTrue(isinstance(result["llm_surplus"], float))

    def test_zip_fixed_background_episode_runs(self):
        def strategy(market_state):
            if market_state.get("best_bid") is None or market_state.get("best_ask") is None:
                return 1.0
            return 0.05

        result = run_episode(
            llm_strategy_func=strategy,
            llm_order_type=BUY,
            num_zi_buy=2,
            num_zi_sell=2,
            timesteps=40,
            q_max=2,
            shade_range=[0.0, 0.01],
            arrival_rate=0.2,
            random_seed=13,
            strategy_type="spread",
            spread_input_mode="book",
            background_agent_kind="zip_fixed",
            obs_noise_var=1e6,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        self.assertIn("llm_surplus", result)
        self.assertEqual(result["market_config"]["background_agent_kind"], "zip_fixed")
        self.assertTrue(isinstance(result["llm_surplus"], float))


class TestHoldoutZipBackgroundWiring(unittest.TestCase):
    def test_run_holdout_forwards_zip_bg_and_skips_bo(self):
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
            def agent_strategy(market_state):
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
                    include_zip=True,
                    background_agent_kind="zip",
                    spread_input_mode="zip_parity",
                    strategy_output_mode="shade",
                )

        self.assertEqual(out["background_agent_kind"], "zip")
        self.assertEqual(out["spread_input_mode"], "zip_parity")
        self.assertIsNone(out["bo"])
        kinds = [c["special_agent_kind"] for c in calls]
        self.assertEqual(kinds, ["llm", "zi", "hbl", "zip"])
        for c in calls:
            self.assertEqual(c["background_agent_kind"], "zip")
            self.assertEqual(c["spread_input_mode"], "zip_parity")


class TestSpreadMenusDistinctFromPv(unittest.TestCase):
    def test_spread_menus_differ_from_pv_and_include_shouts(self):
        pv_p, pv_f = principle_form_menus("pv")
        sp_p, sp_f = principle_form_menus("spread")
        self.assertNotEqual(pv_p, sp_p)
        self.assertTrue(pv_p.isdisjoint(sp_p) or not pv_p.issubset(sp_p))
        # Core PV-only principles must not appear on Spread.
        self.assertNotIn("pv_rank_or_distance_to_best", sp_p)
        self.assertNotIn("nonlinear_abs_pv", sp_p)
        # Spread must expose market_state / shout principles.
        self.assertIn("bid_ask_spread_width", sp_p)
        self.assertIn("recent_accept_reject", sp_p)
        self.assertIn("shout_competitiveness", sp_p)
        self.assertIn("ema_over_window", sp_f)
        # PV form classes that need a PV schedule stay off Spread.
        self.assertNotIn("schedule_index", sp_f)
        self.assertNotIn("rank", sp_f)
        pairs = menu_hypothesis_pairs("spread")
        self.assertGreater(len(pairs), len(menu_hypothesis_pairs("pv")) // 2)


if __name__ == "__main__":
    unittest.main()
