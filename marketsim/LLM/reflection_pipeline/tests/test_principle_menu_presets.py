import unittest

from marketsim.LLM.reflection_pipeline.prompts import (
    list_principle_menu_presets,
    principle_form_menus,
    set_principle_menu_preset,
)


class PrincipleMenuPresetTest(unittest.TestCase):
    def tearDown(self):
        set_principle_menu_preset(None)

    def test_presets_registered(self):
        self.assertIn("lob_e2", list_principle_menu_presets())
        self.assertIn("lob_snapshot", list_principle_menu_presets())
        self.assertIn("spread_e5", list_principle_menu_presets())

    def test_lob_e2_reduces_principles(self):
        set_principle_menu_preset("lob_e2")
        principles, forms = principle_form_menus("lob")
        self.assertEqual(
            principles,
            frozenset({"fill_vs_reject", "side_pressure", "order_flow_imbalance"}),
        )
        self.assertIn("ema_over_window", forms)

    def test_lob_snapshot_preset(self):
        set_principle_menu_preset("lob_snapshot")
        principles, forms = principle_form_menus("lob")
        self.assertIn("book_imbalance", principles)
        self.assertIn("ema_over_snapshots", forms)

    def test_spread_e5_reduces_principles(self):
        set_principle_menu_preset("spread_e5")
        principles, forms = principle_form_menus("spread")
        self.assertEqual(
            principles,
            frozenset(
                {
                    "bid_ask_spread_width",
                    "inventory_remaining_fraction",
                    "pv_vs_mid",
                }
            ),
        )
        self.assertIn("clamp_linear", forms)

    def test_preset_does_not_affect_other_strategy_types(self):
        set_principle_menu_preset("lob_e2")
        pv_p, _ = principle_form_menus("pv")
        self.assertIn("current_pv_vs_fundamental", pv_p)
        self.assertIn("pv_rank_or_distance_to_best", pv_p)


if __name__ == "__main__":
    unittest.main()
