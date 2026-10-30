"""ZIP μ ↔ shade helpers and ladder wiring."""

from __future__ import annotations

import unittest

from marketsim.agent.zip_agent_buy_sell import (
    shade_hi_to_fixed_zip_mu,
    shade_range_to_zip_mu_init,
)
from marketsim.fourheap.constants import BUY, SELL
from marketsim.LLM.reflection_pipeline.config import (
    DEFAULT_ZI_BENCHMARK_SHADE_RANGES,
    DEFAULT_ZIP_BENCHMARK_SHADE_RANGES,
    DEFAULT_ZIP_LADDER_COUNT,
    zip_market_shade_ranges,
)
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
    _tracked_ladder_kind,
    build_ladder_specs,
)


class TestZipMuShadeMapping(unittest.TestCase):
    def test_background_mu_init_from_competitive_shade(self):
        buy = shade_range_to_zip_mu_init([0.0, 0.015], side=BUY)
        sell = shade_range_to_zip_mu_init([0.0, 0.015], side=SELL)
        self.assertLess(buy[0], buy[1])
        self.assertLessEqual(buy[1], 0.0)
        self.assertAlmostEqual(buy[0], -0.015)
        self.assertGreater(sell[0], 0.0)
        self.assertAlmostEqual(sell[1], 0.015)

    def test_ladder_fixed_mu_matches_shade_hi(self):
        self.assertAlmostEqual(shade_hi_to_fixed_zip_mu(0.032, side=BUY), -0.032)
        self.assertAlmostEqual(shade_hi_to_fixed_zip_mu(0.032, side=SELL), 0.032)

    def test_zi_ladder_keeps_twenty_one_rungs(self):
        his = [float(r[-1]) for r in DEFAULT_ZI_BENCHMARK_SHADE_RANGES]
        self.assertEqual(len(his), 21)
        self.assertEqual(len(set(his)), 21)

    def test_zip_ladder_is_eight_fixed_plus_standard(self):
        his = [float(r[-1]) for r in DEFAULT_ZIP_BENCHMARK_SHADE_RANGES]
        self.assertEqual(len(his), 8)
        self.assertEqual(len(set(his)), 8)
        self.assertLessEqual(max(his), 0.012)
        specs = build_ladder_specs(tracked_ladder_kind="zip")
        self.assertEqual(len(specs), DEFAULT_ZIP_LADDER_COUNT)
        self.assertEqual(len(specs), 9)
        self.assertEqual(specs[-1]["ladder_mode"], "standard")
        self.assertEqual(specs[-1]["label"], "standard_zip")
        self.assertTrue(all(s["ladder_mode"] == "fixed_mu" for s in specs[:-1]))
        mus = [shade_hi_to_fixed_zip_mu(s, side=BUY) for s in his]
        self.assertEqual(len(set(mus)), 8)

    def test_market_shades_are_top_subset_of_fixed_ladder(self):
        market = zip_market_shade_ranges(top_k=5)
        fixed = {float(r[-1]) for r in DEFAULT_ZIP_BENCHMARK_SHADE_RANGES}
        his = [float(r[-1]) for r in market]
        self.assertEqual(len(his), 5)
        self.assertTrue(set(his).issubset(fixed))
        self.assertEqual(his, sorted(his))
        # Top-5 by surplus exclude the three weakest fixed rungs.
        self.assertNotIn(0.006, his)
        self.assertNotIn(0.008, his)
        self.assertNotIn(0.012, his)

    def test_tracked_ladder_kind_follows_background(self):
        self.assertEqual(_tracked_ladder_kind("zi"), "zi")
        self.assertEqual(_tracked_ladder_kind("zip"), "zip")
        self.assertEqual(_tracked_ladder_kind("zip_noisy"), "zip_noisy")
        self.assertEqual(_tracked_ladder_kind("hbl"), "hbl_fast")
        self.assertEqual(_tracked_ladder_kind("hbl_fast"), "hbl_fast")


if __name__ == "__main__":
    unittest.main()
