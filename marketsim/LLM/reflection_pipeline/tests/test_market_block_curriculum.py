"""Deterministic schedule tests for market-block + rolling Q suite."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.config import (
    ReflectionConfigSpace,
    build_default_config_space,
)
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline


def _tiny_space() -> ReflectionConfigSpace:
    return ReflectionConfigSpace(
        num_zi_buy=[5, 10],
        num_zi_sell=[5, 10],
        timesteps=[100, 200],
        q_max=[8, 10],
        shade_ranges=[[0.0, 0.01], [0.0, 0.05]],
        arrival_rates=[0.1, 0.2],
    )


class MarketBlockCurriculumTest(unittest.TestCase):
    def test_block_rotation_and_rolling_suite(self):
        pipe = ReflectionPipeline(
            config_space=_tiny_space(),
            batch_size=1,
            steps=1,
            episodes_per_config=1,
            critic_enabled=False,
            seed=42,
            market_block_size=10,
            q_suite_size=2,
            initial_strategy_types=["spread"],
            run_dir="/tmp/reflection_block_curriculum_test",
        )

        # Within first block: same train market; suite size 1.
        train0, suite0 = pipe._train_and_q_configs(0)
        train5, suite5 = pipe._train_and_q_configs(5)
        self.assertEqual(len(train0), 1)
        self.assertEqual(train0[0].label, train5[0].label)
        self.assertEqual([c.label for c in suite0], [train0[0].label])
        self.assertEqual([c.label for c in suite5], [train0[0].label])

        # Block boundary: new train market; suite = last 2.
        train10, suite10 = pipe._train_and_q_configs(10)
        self.assertNotEqual(train10[0].label, train0[0].label)
        self.assertEqual(
            [c.label for c in suite10],
            [train0[0].label, train10[0].label],
        )

        # Hold through rest of block 1.
        train19, suite19 = pipe._train_and_q_configs(19)
        self.assertEqual(train19[0].label, train10[0].label)
        self.assertEqual(
            [c.label for c in suite19],
            [train0[0].label, train10[0].label],
        )

        # Third block drops the oldest from the rolling suite.
        train20, suite20 = pipe._train_and_q_configs(20)
        self.assertNotEqual(train20[0].label, train10[0].label)
        self.assertEqual(
            [c.label for c in suite20],
            [train10[0].label, train20[0].label],
        )
        self.assertEqual(len(pipe._block_markets), 3)

    def test_schedule_deterministic_for_seed(self):
        def schedule(seed: int):
            pipe = ReflectionPipeline(
                config_space=_tiny_space(),
                batch_size=1,
                steps=1,
                episodes_per_config=1,
                critic_enabled=False,
                seed=seed,
                market_block_size=5,
                q_suite_size=3,
                initial_strategy_types=["spread"],
                run_dir="/tmp/reflection_block_curriculum_det",
            )
            fingerprints = []
            suites = []
            for step in range(15):
                train, suite = pipe._train_and_q_configs(step)
                cfg = train[0]
                fingerprints.append(
                    (
                        cfg.label,
                        cfg.num_zi_buy,
                        cfg.num_zi_sell,
                        cfg.timesteps,
                        cfg.q_max,
                        tuple(cfg.shade_range),
                        cfg.arrival_rate,
                    )
                )
                suites.append(
                    tuple(
                        (
                            c.label,
                            c.num_zi_buy,
                            c.timesteps,
                            tuple(c.shade_range),
                        )
                        for c in suite
                    )
                )
            return fingerprints, suites

        a_fp, a_suites = schedule(99)
        b_fp, b_suites = schedule(99)
        c_fp, _ = schedule(100)
        self.assertEqual(a_fp, b_fp)
        self.assertEqual(a_suites, b_suites)
        self.assertNotEqual(a_fp, c_fp)

    def test_rejects_block_with_fixed_pool(self):
        pool = build_default_config_space().sample_batch(
            __import__("random").Random(0), 1
        )
        with self.assertRaises(ValueError):
            ReflectionPipeline(
                config_pool=pool,
                market_block_size=10,
                critic_enabled=False,
                steps=1,
                batch_size=1,
                episodes_per_config=1,
                run_dir="/tmp/reflection_block_curriculum_reject",
            )


if __name__ == "__main__":
    unittest.main()
