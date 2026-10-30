"""Tests for homeset full-menu sweep warmup (before resample main phase)."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.config import (
    ReflectionConfigSpace,
    build_default_config_pool,
    build_default_config_space,
    resample_market_shade_ranges,
    zip_market_shade_ranges,
)
from marketsim.LLM.reflection_pipeline.pipeline import (
    ELITE_PARENT_BIAS_PROB,
    ReflectionPipeline,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    menu_hypothesis_pairs,
    set_principle_menu_preset,
)


def _tiny_space() -> ReflectionConfigSpace:
    return ReflectionConfigSpace(
        num_zi_buy=[5, 10, 15],
        num_zi_sell=[5, 10, 15],
        timesteps=[100, 200, 300],
        q_max=[8, 10, 12],
        shade_ranges=[[0.0, 0.01], [0.0, 0.02], [0.0, 0.05]],
        arrival_rates=[0.1, 0.15, 0.2],
    )


class HomesetMenuSweepTest(unittest.TestCase):
    def setUp(self):
        set_principle_menu_preset(None)

    def tearDown(self):
        set_principle_menu_preset(None)

    def test_elite_parent_bias_default_is_zero(self):
        self.assertEqual(ELITE_PARENT_BIAS_PROB, 0.0)
        pipe = ReflectionPipeline(
            config_space=_tiny_space(),
            batch_size=1,
            steps=1,
            episodes_per_config=1,
            critic_enabled=False,
            seed=1,
            initial_strategy_types=["lob"],
            run_dir="/tmp/reflection_elite_default_test",
        )
        self.assertEqual(pipe.elite_parent_bias_prob, 0.0)

    def test_lob_full_menu_has_25_pairs(self):
        pairs = sorted(menu_hypothesis_pairs("lob", lob_input_mode="book_orders"))
        self.assertEqual(len(pairs), 25)

    def test_pv_full_menu_has_25_pairs(self):
        pairs = sorted(menu_hypothesis_pairs("pv"))
        self.assertEqual(len(pairs), 25)
        principles = {p for p, _ in pairs}
        forms = {f for _, f in pairs}
        self.assertIn("current_pv_vs_fundamental", principles)
        self.assertIn("unit_schedule_blend", forms)
        self.assertNotIn("inventory_remaining_units", principles)
        self.assertNotIn("schedule_index", forms)

    def test_homeset_sweep_allowed_with_resample(self):
        pipe = ReflectionPipeline(
            config_space=_tiny_space(),
            batch_size=3,
            steps=150,
            episodes_per_config=1,
            critic_enabled=False,
            seed=123,
            home_set_size=3,
            resample_every=25,
            menu_sweep_on_home_set=True,
            lob_input_mode="book_orders",
            initial_strategy_types=["lob"],
            parallel_episodes=True,
            parallel_episode_workers=4,
            run_dir="/tmp/reflection_homeset_menu_sweep_ok",
        )
        self.assertTrue(pipe.menu_sweep_on_home_set)
        self.assertEqual(len(pipe._home_set), 3)
        self.assertEqual(pipe.resample_every, 25)
        self.assertTrue(pipe.parallel_episodes)
        pairs = pipe._menu_sweep_pairs("lob")
        self.assertEqual(len(pairs), 25)

    def test_rejects_homeset_sweep_without_home_set(self):
        with self.assertRaises(ValueError):
            ReflectionPipeline(
                config_space=_tiny_space(),
                steps=1,
                episodes_per_config=1,
                critic_enabled=False,
                seed=1,
                menu_sweep_on_home_set=True,
                home_set_size=0,
                initial_strategy_types=["lob"],
                run_dir="/tmp/reflection_homeset_menu_sweep_bad1",
            )

    def test_rejects_homeset_sweep_with_legacy_menu_sweep_markets(self):
        with self.assertRaises(ValueError):
            ReflectionPipeline(
                config_space=_tiny_space(),
                steps=1,
                episodes_per_config=1,
                critic_enabled=False,
                seed=1,
                home_set_size=3,
                menu_sweep_on_home_set=True,
                menu_sweep_markets=2,
                principle_menu_preset="min2",
                initial_strategy_types=["lob"],
                run_dir="/tmp/reflection_homeset_menu_sweep_bad2",
            )

    def test_seed_resample_slot_from_homeset_after_sweep_helper(self):
        pipe = ReflectionPipeline(
            config_space=_tiny_space(),
            batch_size=3,
            steps=50,
            episodes_per_config=1,
            critic_enabled=False,
            seed=7,
            home_set_size=3,
            resample_every=25,
            menu_sweep_on_home_set=True,
            lob_input_mode="book_orders",
            initial_strategy_types=["lob"],
            run_dir="/tmp/reflection_homeset_menu_sweep_slot",
        )
        # Simulate post-sweep seeding without calling LLM.
        home = list(pipe._home_set)
        pipe._resample_slots = [list(home)]
        pipe._resample_needs_q_realign = False
        slot0 = pipe._ensure_resample_slot(0)
        self.assertEqual([c.label for c in slot0], [c.label for c in home])
        # Next slot should redraw (different markets).
        slot1 = pipe._ensure_resample_slot(25)
        self.assertEqual(len(slot1), 3)
        self.assertNotEqual(
            {c.fingerprint() for c in slot1},
            {c.fingerprint() for c in home},
        )

    def test_homeset_sweep_pins_parent_to_seed(self):
        pipe = ReflectionPipeline(
            config_space=_tiny_space(),
            batch_size=3,
            steps=5,
            episodes_per_config=1,
            critic_enabled=False,
            seed=11,
            home_set_size=3,
            resample_every=25,
            menu_sweep_on_home_set=True,
            lob_input_mode="book_orders",
            initial_strategy_types=["lob"],
            parent_selection_mode="ucb",
            elite_parent_bias_prob=0.0,
            run_dir="/tmp/reflection_homeset_menu_sweep_pin",
        )
        seed = pipe._seed_node_for_strategy_type("lob")
        self.assertIn(seed.node_id, pipe._tree.seed_node_ids())

        # Without override, UCB / elite path is used.
        parent, mode = pipe._select_parent_node()
        self.assertEqual(mode, "ucb")
        self.assertEqual(parent.node_id, seed.node_id)

        # Homeset sweep forces the original base/seed parent.
        pipe._force_parent_id = int(seed.node_id)
        forced_parent, forced_mode = pipe._select_parent_node()
        self.assertEqual(forced_mode, "forced_seed")
        self.assertEqual(forced_parent.node_id, seed.node_id)
        pipe._clear_step_force_overrides()
        self.assertIsNone(pipe._force_parent_id)

        # After clear, selection returns to UCB for the main phase.
        parent2, mode2 = pipe._select_parent_node()
        self.assertEqual(mode2, "ucb")
        self.assertEqual(parent2.node_id, seed.node_id)

    def test_post_homeset_resample_shades_widen_holdout_fps_excluded(self):
        narrow = zip_market_shade_ranges(top_k=5)
        narrow_his = {float(r[-1]) for r in narrow}
        wide_his = {float(r[-1]) for r in resample_market_shade_ranges()}
        self.assertTrue({0.015, 0.02, 0.025}.issubset(wide_his))

        pipe = ReflectionPipeline(
            config_space=build_default_config_space(),
            batch_size=3,
            steps=50,
            episodes_per_config=1,
            critic_enabled=False,
            seed=99,
            home_set_size=3,
            resample_every=25,
            menu_sweep_on_home_set=True,
            lob_input_mode="book_orders",
            initial_strategy_types=["lob"],
            run_dir="/tmp/reflection_homeset_resample_shade_widen",
        )
        home_his = {float(c.shade_range[-1]) for c in pipe._home_set}
        self.assertTrue(home_his.issubset(narrow_his))
        space_his = {float(r[-1]) for r in pipe._config_space.shade_ranges}
        self.assertEqual(space_his, wide_his)

        holdout_cells = {c.ood_cell() for c in build_default_config_pool()}
        pipe._resample_slots = [list(pipe._home_set)]
        redraw = pipe._sample_resample_home_set(1)
        self.assertEqual(len(redraw), 3)
        for cfg in redraw:
            self.assertNotIn(cfg.ood_cell(), holdout_cells)


if __name__ == "__main__":
    unittest.main()
