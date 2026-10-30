"""Support-OOD holdout battery: tiers, A_ood cells, train exclusion."""

from __future__ import annotations

import random
import unittest

from marketsim.LLM.reflection_pipeline.config import (
    HOLDOUT_OOD_TIERS,
    OOD_AXES,
    build_default_config_pool,
    build_default_config_space,
    build_ood_holdout_config_space,
    holdout_configs_by_tier,
    is_outside_train_support,
    outside_train_support_axes,
    sample_ood_holdout_episode_configs,
    sample_stratified_home_set,
    train_support_spec,
)


class TrainSupportSpecTest(unittest.TestCase):
    def test_documents_default_grids_and_ood_axes(self):
        spec = train_support_spec()
        self.assertEqual(spec["axes"], list(OOD_AXES))
        self.assertEqual(spec["num_zi_buy"], [5, 10, 12, 15])
        self.assertEqual(spec["arrival_rates"], [0.1, 0.15])
        self.assertEqual(spec["shock_var"], [1e5])
        self.assertEqual(spec["obs_noise_var"], [1e6])
        self.assertEqual(spec["pv_var"], [1e5])
        self.assertIn(0.025, spec["shade_hi_resample"])
        self.assertNotIn(0.03, spec["shade_hi_union"])
        self.assertNotIn(0.005, spec["shade_hi_union"])


class OodHoldoutPoolTest(unittest.TestCase):
    def test_six_markets_two_per_tier_all_support_ood(self):
        pool = build_default_config_pool()
        self.assertEqual(len(pool), 6)
        by_tier = holdout_configs_by_tier()
        for tier in HOLDOUT_OOD_TIERS:
            self.assertEqual(len(by_tier[tier]), 2, tier)
        cells = [c.ood_cell() for c in pool]
        self.assertEqual(len(set(cells)), 6)
        for cfg in pool:
            self.assertIn(cfg.ood_tier, HOLDOUT_OOD_TIERS)
            self.assertTrue(is_outside_train_support(cfg), cfg.label)
            axes = outside_train_support_axes(cfg)
            self.assertTrue(axes, cfg.label)
            if cfg.ood_tier == "near":
                self.assertTrue(
                    set(axes).isdisjoint({"shock_var", "obs_noise_var"}),
                    cfg.label,
                )
            if cfg.ood_tier == "info":
                self.assertTrue(
                    set(axes) & {"shock_var", "obs_noise_var"},
                    cfg.label,
                )
            if cfg.ood_tier == "compound":
                self.assertTrue(
                    set(axes) & {"shock_var", "obs_noise_var"},
                    cfg.label,
                )
                self.assertTrue(
                    set(axes)
                    & {
                        "num_zi_buy",
                        "num_zi_sell",
                        "timesteps",
                        "q_max",
                        "shade_hi",
                        "arrival_rate",
                    },
                    cfg.label,
                )

    def test_ood_cell_includes_information_axes(self):
        pool = build_default_config_pool()
        info = [c for c in pool if c.ood_tier == "info"]
        self.assertEqual(len(info), 2)
        # Same structural fingerprint possible; ood_cells must still differ via info.
        self.assertEqual(info[0].fingerprint(), info[1].fingerprint())
        self.assertNotEqual(info[0].ood_cell(), info[1].ood_cell())

    def test_homeset_never_hits_holdout_ood_cells(self):
        space = build_default_config_space()
        holdout = build_default_config_pool()
        blocked = {c.ood_cell() for c in holdout}
        for seed in range(16):
            home = sample_stratified_home_set(
                space, random.Random(seed), size=3, exclude=holdout
            )
            for cfg in home:
                self.assertNotIn(cfg.ood_cell(), blocked)
                # Train draws stay inside P_train support.
                self.assertFalse(is_outside_train_support(cfg), cfg.ood_cell())


class SampledOodHoldoutSpaceTest(unittest.TestCase):
    def test_space_levels_disjoint_from_train(self):
        space = build_ood_holdout_config_space()
        spec = train_support_spec()
        self.assertTrue(
            set(space.num_zi_buy).isdisjoint({int(x) for x in spec["num_zi_buy"]})
        )
        self.assertTrue(
            set(round(float(x), 6) for x in space.arrival_rates).isdisjoint(
                {round(float(x), 6) for x in spec["arrival_rates"]}
            )
        )
        self.assertIsNotNone(space.shock_vars)
        self.assertIsNotNone(space.obs_noise_vars)
        self.assertIsNone(space.pv_vars)
        self.assertEqual(float(space.pv_var), 1e5)
        self.assertTrue(
            set(round(float(x), 6) for x in space.shock_vars).isdisjoint(
                {round(float(x), 6) for x in spec["shock_var"]}
            )
        )
        self.assertTrue(
            set(round(float(x), 6) for x in space.obs_noise_vars).isdisjoint(
                {round(float(x), 6) for x in spec["obs_noise_var"]}
            )
        )

    def test_iid_episode_samples_are_all_support_ood(self):
        cfgs = sample_ood_holdout_episode_configs(64, random.Random(0))
        self.assertEqual(len(cfgs), 64)
        labels = [c.label for c in cfgs]
        self.assertEqual(len(set(labels)), 64)
        for cfg in cfgs:
            self.assertEqual(cfg.ood_tier, "sampled")
            self.assertTrue(is_outside_train_support(cfg), cfg.ood_cell())
            # Strong OOD: every A_ood axis leaves train support.
            axes = set(outside_train_support_axes(cfg))
            self.assertEqual(axes, set(OOD_AXES), cfg.ood_cell())

    def test_sample_deterministic_for_seed(self):
        a = sample_ood_holdout_episode_configs(8, random.Random(99))
        b = sample_ood_holdout_episode_configs(8, random.Random(99))
        self.assertEqual([c.ood_cell() for c in a], [c.ood_cell() for c in b])
        c = sample_ood_holdout_episode_configs(8, random.Random(100))
        self.assertNotEqual([x.ood_cell() for x in a], [x.ood_cell() for x in c])


if __name__ == "__main__":
    unittest.main()
