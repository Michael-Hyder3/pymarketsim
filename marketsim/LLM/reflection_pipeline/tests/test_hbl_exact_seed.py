"""Compact HBL-like seed: wiring + surplus near tracked HBL-fast (no exact hook)."""

from __future__ import annotations

import unittest

from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.hbl_order_memory import strategy_requests_hbl_exact_seed
from marketsim.LLM.reflection_pipeline.code_edit import parent_is_complex
from marketsim.LLM.reflection_pipeline.learning_focused import (
    hbl_distilled_lob_order_memory_seed_code,
    lob_order_memory_seed_code,
    normalize_lob_seed_scale,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier
from marketsim.fourheap.constants import BUY, SELL


def _compile(code: str):
    ns: dict = {}
    exec(code, ns)
    return ns["agent_strategy"]


class TestHblDistilledSeedWiring(unittest.TestCase):
    def test_normalize_accepts_hbl(self):
        self.assertEqual(normalize_lob_seed_scale("hbl"), "hbl")

    def test_seed_code_is_editable_hbl_not_exact_hook(self):
        code = hbl_distilled_lob_order_memory_seed_code()
        self.assertIn("def agent_strategy", code)
        self.assertIn("pricing_fundamental", code)
        self.assertIn("belief", code)
        self.assertNotIn("HBL_EXACT_SEED = True", code)
        self.assertNotIn("from __future__", code)
        # Single top-level function → normal full-python refine, not diff mode.
        self.assertFalse(parent_is_complex(code), msg=f"lines={len(code.splitlines())}")
        fn = _compile(code)
        self.assertFalse(strategy_requests_hbl_exact_seed(fn))
        ok, err, _ = StrategyVerifier().verify(
            code, required_args=1, input_mode="lob_hbl_parity"
        )
        self.assertTrue(ok, msg=err)

    def test_dispatch_and_tree(self):
        self.assertEqual(
            lob_order_memory_seed_code(lob_seed_scale="hbl"),
            hbl_distilled_lob_order_memory_seed_code(),
        )
        tree = RefinementTree(
            initial_types=["lob"],
            lob_input_mode="hbl_parity",
            lob_seed_scale="hbl",
        )
        lob = tree.get(tree.seed_node_ids()[0])
        self.assertEqual(lob.metadata.get("seed_kind"), "hbl_distilled")
        self.assertIn("HBL-like", lob.metadata.get("desc", ""))
        self.assertIn("def agent_strategy", lob.code)
        self.assertFalse(parent_is_complex(lob.code))


class TestHblDistilledSeedParity(unittest.TestCase):
    def _common(self, **overrides):
        base = dict(
            num_zi_buy=5,
            num_zi_sell=5,
            timesteps=200,
            q_max=4,
            shade_range=[0.0, 0.015],
            arrival_rate=0.15,
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
            background_agent_kind="zi",
            strategy_type="lob",
            lob_input_mode="hbl_parity",
        )
        base.update(overrides)
        return base

    def test_seed_surplus_near_tracked_hbl_fast_buy(self):
        """Compact seed should trade like HBL (not collapse to ~0)."""
        seed_fn = _compile(hbl_distilled_lob_order_memory_seed_code())

        def _dummy(*_a, **_k):
            return 0.0

        hbl_sum = 0.0
        seed_sum = 0.0
        n = 6
        for seed in range(n):
            common = self._common(llm_order_type=BUY, random_seed=seed)
            hbl = run_episode(
                llm_strategy_func=_dummy,
                special_agent_kind="hbl_fast",
                **common,
            )
            seeded = run_episode(
                llm_strategy_func=seed_fn,
                special_agent_kind="llm",
                **common,
            )
            hbl_sum += float(hbl["llm_surplus"])
            seed_sum += float(seeded["llm_surplus"])
        self.assertGreater(seed_sum / n, 100.0, msg=f"seed_mean={seed_sum/n}")
        ratio = (seed_sum / n) / max(1.0, abs(hbl_sum / n))
        self.assertGreater(ratio, 0.35, msg=f"hbl={hbl_sum/n} seed={seed_sum/n}")
        self.assertLess(ratio, 2.5, msg=f"hbl={hbl_sum/n} seed={seed_sum/n}")

    def test_seed_surplus_near_tracked_hbl_fast_sell(self):
        seed_fn = _compile(hbl_distilled_lob_order_memory_seed_code())

        def _dummy(*_a, **_k):
            return 0.0

        hbl_sum = 0.0
        seed_sum = 0.0
        n = 6
        for seed in range(n):
            common = self._common(llm_order_type=SELL, random_seed=50 + seed)
            hbl = run_episode(
                llm_strategy_func=_dummy,
                special_agent_kind="hbl_fast",
                **common,
            )
            seeded = run_episode(
                llm_strategy_func=seed_fn,
                special_agent_kind="llm",
                **common,
            )
            hbl_sum += float(hbl["llm_surplus"])
            seed_sum += float(seeded["llm_surplus"])
        self.assertGreater(seed_sum / n, 100.0, msg=f"seed_mean={seed_sum/n}")
        ratio = (seed_sum / n) / max(1.0, abs(hbl_sum / n))
        self.assertGreater(ratio, 0.35, msg=f"hbl={hbl_sum/n} seed={seed_sum/n}")
        self.assertLess(ratio, 2.5, msg=f"hbl={hbl_sum/n} seed={seed_sum/n}")

    def test_dense_hbl_bg_seed_not_zero(self):
        seed_fn = _compile(hbl_distilled_lob_order_memory_seed_code())
        surpluses = []
        for seed in (0, 1, 2):
            r = run_episode(
                llm_strategy_func=seed_fn,
                llm_order_type=BUY,
                random_seed=seed,
                num_zi_buy=5,
                num_zi_sell=15,
                timesteps=400,
                q_max=10,
                shade_range=[0.0, 0.025],
                arrival_rate=0.1,
                hbl_l_param=6,
                capture_detail=False,
                use_dummy_fundamental=True,
                background_agent_kind="hbl_fast",
                strategy_type="lob",
                lob_input_mode="hbl_parity",
            )
            surpluses.append(float(r["llm_surplus"]))
        mean = sum(surpluses) / len(surpluses)
        self.assertGreater(mean, 200.0, msg=f"surpluses={surpluses}")


if __name__ == "__main__":
    unittest.main()
