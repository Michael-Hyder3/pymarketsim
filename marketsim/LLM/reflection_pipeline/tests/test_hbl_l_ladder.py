"""HBL L-ladder wiring under HBL / hbl_fast backgrounds."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.config import (
    DEFAULT_HBL_BENCHMARK_L_VALUES,
    DEFAULT_HBL_LADDER_COUNT,
)
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
    _ladder_label,
    _tracked_ladder_kind,
    build_ladder_specs,
)


class TestHblLLadder(unittest.TestCase):
    def test_tracked_ladder_kind_follows_background(self):
        self.assertEqual(_tracked_ladder_kind("zi"), "zi")
        self.assertEqual(_tracked_ladder_kind("zip"), "zip")
        self.assertEqual(_tracked_ladder_kind("zip_noisy"), "zip_noisy")
        self.assertEqual(_tracked_ladder_kind("hbl"), "hbl_fast")
        self.assertEqual(_tracked_ladder_kind("hbl_fast"), "hbl_fast")

    def test_ladder_label(self):
        self.assertEqual(_ladder_label("zi"), "ZI")
        self.assertEqual(_ladder_label("zip"), "ZIP")
        self.assertEqual(_ladder_label("hbl_fast"), "HBL")

    def test_hbl_ladder_is_eight_l_rungs(self):
        self.assertEqual(len(DEFAULT_HBL_BENCHMARK_L_VALUES), 8)
        self.assertEqual(len(set(DEFAULT_HBL_BENCHMARK_L_VALUES)), 8)
        self.assertIn(6, DEFAULT_HBL_BENCHMARK_L_VALUES)
        specs = build_ladder_specs(tracked_ladder_kind="hbl_fast")
        self.assertEqual(len(specs), DEFAULT_HBL_LADDER_COUNT)
        self.assertEqual(len(specs), 8)
        self.assertTrue(all(s["ladder_mode"] == "hbl_L" for s in specs))
        self.assertEqual(
            [int(s["hbl_l"]) for s in specs],
            list(DEFAULT_HBL_BENCHMARK_L_VALUES),
        )
        self.assertEqual(
            [s["label"] for s in specs],
            [f"hbl_L={L}" for L in DEFAULT_HBL_BENCHMARK_L_VALUES],
        )

    def test_hbl_ladder_ignores_shade_ranges_arg_when_building_from_defaults(self):
        # Callers may still pass ZI shade ranges from config; HBL path must not
        # fall through to the ZI 21-cap ladder.
        specs = build_ladder_specs(
            tracked_ladder_kind="hbl",
            shade_ranges=[[0.0, 0.5], [0.0, 1.0]],
        )
        self.assertEqual(len(specs), 8)
        self.assertTrue(all(s.get("hbl_l") is not None for s in specs))


if __name__ == "__main__":
    unittest.main()
