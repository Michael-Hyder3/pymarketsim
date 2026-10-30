"""Frozen stratified home set: same K markets every step, opaque to the LLM."""

from __future__ import annotations

import random
import re
import unittest

from marketsim.LLM.reflection_pipeline.config import (
    ReflectionConfigSpace,
    build_default_config_pool,
    build_default_config_space,
    sample_stratified_home_set,
)
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_reflection_prompt,
    format_per_config_breakdown_lines,
)


def _tiny_space() -> ReflectionConfigSpace:
    return ReflectionConfigSpace(
        num_zi_buy=[5, 10, 15],
        num_zi_sell=[5, 10, 15],
        timesteps=[100, 200, 300],
        q_max=[8, 10],
        shade_ranges=[[0.0, 0.005], [0.0, 0.015], [0.0, 0.025]],
        arrival_rates=[0.1, 0.2],
    )


class StratifiedHomeSetSamplerTest(unittest.TestCase):
    def test_three_markets_cover_imbalance_and_shade(self):
        space = _tiny_space()
        home = sample_stratified_home_set(
            space, random.Random(0), size=3, exclude=build_default_config_pool()
        )
        self.assertEqual(len(home), 3)
        self.assertEqual([c.label for c in home], ["market_0000", "market_0001", "market_0002"])
        buyer, seller, balanced = home
        self.assertGreater(buyer.num_zi_buy, buyer.num_zi_sell)
        self.assertLess(seller.num_zi_buy, seller.num_zi_sell)
        self.assertEqual(balanced.num_zi_buy, balanced.num_zi_sell)
        allowed_his = {float(r[-1]) for r in space.shade_ranges}
        for cfg in home:
            self.assertIn(float(cfg.shade_range[-1]), allowed_his)
        # Prefer distinct shade his when the grid has enough options.
        self.assertEqual(len({float(c.shade_range[-1]) for c in home}), 3)

    def test_disjoint_from_holdout_ood_cells(self):
        space = build_default_config_space()
        holdout = build_default_config_pool()
        blocked = {cfg.ood_cell() for cfg in holdout}
        for seed in range(8):
            home = sample_stratified_home_set(
                space, random.Random(seed), size=3, exclude=holdout
            )
            cells = [cfg.ood_cell() for cfg in home]
            self.assertEqual(len(set(cells)), 3)
            for cell in cells:
                self.assertNotIn(cell, blocked)

    def test_deterministic_for_seed(self):
        space = _tiny_space()
        a = sample_stratified_home_set(space, random.Random(7), size=3)
        b = sample_stratified_home_set(space, random.Random(7), size=3)
        self.assertEqual([c.fingerprint() for c in a], [c.fingerprint() for c in b])


class FrozenHomeSetPipelineTest(unittest.TestCase):
    def _pipe(self) -> ReflectionPipeline:
        return ReflectionPipeline(
            config_space=_tiny_space(),
            batch_size=1,
            steps=1,
            episodes_per_config=1,
            critic_enabled=False,
            seed=42,
            home_set_size=3,
            initial_strategy_types=["spread"],
            run_dir="/tmp/reflection_home_set_test",
        )

    def test_frozen_across_steps_and_matches_q(self):
        pipe = self._pipe()
        self.assertEqual(pipe.batch_size, 3)
        self.assertTrue(pipe.critic_per_config_metrics)
        train0, q0 = pipe._train_and_q_configs(0)
        train9, q9 = pipe._train_and_q_configs(9)
        labels0 = [c.label for c in train0]
        self.assertEqual(labels0, ["market_0000", "market_0001", "market_0002"])
        self.assertEqual(labels0, [c.label for c in q0])
        self.assertEqual(labels0, [c.label for c in train9])
        self.assertEqual(
            [c.fingerprint() for c in train0],
            [c.fingerprint() for c in train9],
        )
        self.assertEqual(
            [c.fingerprint() for c in train0],
            [c.fingerprint() for c in q9],
        )

    def test_incompatible_with_blocks(self):
        with self.assertRaises(ValueError):
            ReflectionPipeline(
                config_space=_tiny_space(),
                critic_enabled=False,
                home_set_size=3,
                market_block_size=10,
                initial_strategy_types=["spread"],
                run_dir="/tmp/reflection_home_set_bad",
            )


class OpaquePerConfigPromptTest(unittest.TestCase):
    def test_critic_and_generator_show_opaque_rows_not_hyperparams(self):
        summary = {
            "reward": -10.0,
            "reward_variance": 1.0,
            "per_config_rewards": {
                "market_0000": -12.0,
                "market_0001": -8.0,
                "market_0002": -10.0,
            },
            "per_config_reward_variances": {
                "market_0000": 4.0,
                "market_0001": 1.0,
                "market_0002": 2.0,
            },
            "per_config_metrics": {
                "market_0000": {
                    "llm_surplus_mean": 100.0,
                    "baseline_surplus_mean": 112.0,
                    "avg_shade_mean": 0.002,
                    "config": {"label": "market_0000"},
                },
                "market_0001": {
                    "llm_surplus_mean": 80.0,
                    "baseline_surplus_mean": 88.0,
                    "avg_shade_mean": 0.004,
                    "config": {"label": "market_0001"},
                },
                "market_0002": {
                    "llm_surplus_mean": 90.0,
                    "baseline_surplus_mean": 100.0,
                    "avg_shade_mean": 0.003,
                    "config": {"label": "market_0002"},
                },
            },
        }
        breakdown = "\n".join(format_per_config_breakdown_lines(summary))
        self.assertIn("market_0000", breakdown)
        self.assertIn("market_0001", breakdown)
        self.assertIn("market_0002", breakdown)
        self.assertIn("surplus=100.0000", breakdown)
        self.assertIn("shade=0.0020", breakdown)
        self.assertIsNone(re.search(r"num_zi_buy|shade_range|buy15", breakdown))

        critic = build_reflection_prompt(
            code="def agent_strategy(market_state):\n    return 0.01\n",
            batch_summary=summary,
            batch_data={},
            reward=-10.0,
            reward_variance=1.0,
            include_data=False,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
            strategy_type="spread",
        )
        self.assertIn("market_0000", critic)
        self.assertIn("surplus=80.0000", critic)
        self.assertNotIn("num_zi_buy", critic)

        gen = build_generation_prompt(
            parent_code="def agent_strategy(market_state):\n    return 0.01\n",
            parent_reflection="DIAGNOSIS:\n  principle: inventory_remaining_fraction\n  form_class: ratio\n  shade_error_direction: shade_more\n",
            strategy_output_mode="shade",
            market_description="test market",
            offline_example=None,
            strategy_type="spread",
            parent_performance=summary,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
        )
        self.assertIn("market_0002", gen)
        self.assertIn("advantage=-10.0000", gen)
        self.assertNotIn("num_zi_sell", gen)


if __name__ == "__main__":
    unittest.main()
