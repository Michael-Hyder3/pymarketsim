"""Tests for min2 menu-sweep schedule and forced pair order."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.config import ReflectionConfigSpace
from marketsim.LLM.reflection_pipeline.pipeline import (
    ReflectionPipeline,
    _forced_menu_pair_diagnosis,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    menu_hypothesis_pairs,
    parse_principle_form_fields,
    set_principle_menu_preset,
)


def _tiny_space() -> ReflectionConfigSpace:
    return ReflectionConfigSpace(
        num_zi_buy=[5, 10],
        num_zi_sell=[5, 10],
        timesteps=[100, 200],
        q_max=[8, 10],
        shade_ranges=[[0.0, 0.01], [0.0, 0.05]],
        arrival_rates=[0.1, 0.2],
    )


class MenuSweepCurriculumTest(unittest.TestCase):
    def setUp(self):
        set_principle_menu_preset("min2")

    def tearDown(self):
        set_principle_menu_preset(None)

    def test_min2_spread_has_four_pairs(self):
        pairs = sorted(menu_hypothesis_pairs("spread"))
        self.assertEqual(len(pairs), 4)
        principles = {p for p, _ in pairs}
        forms = {f for _, f in pairs}
        self.assertEqual(
            principles,
            {"inventory_remaining_fraction", "bid_ask_spread_width"},
        )
        self.assertEqual(forms, {"ratio", "inventory_weighted"})

    def test_forced_diagnosis_embeds_pair(self):
        text = _forced_menu_pair_diagnosis(
            principle="bid_ask_spread_width",
            form_class="ratio",
            strategy_type="spread",
        )
        fields = parse_principle_form_fields(text)
        self.assertEqual(fields.get("principle"), "bid_ask_spread_width")
        self.assertEqual(fields.get("form_class"), "ratio")
        self.assertIn("shade_error_direction:", text)

    def test_sweep_schedule_markets_and_pair_order(self):
        pipe = ReflectionPipeline(
            config_space=_tiny_space(),
            batch_size=1,
            steps=1,
            episodes_per_config=1,
            critic_enabled=False,
            seed=7,
            market_block_size=25,
            q_suite_size=3,
            menu_sweep_markets=3,
            principle_menu_preset="min2",
            initial_strategy_types=["spread"],
            run_dir="/tmp/reflection_menu_sweep_test",
        )
        expected_pairs = pipe._menu_sweep_pairs("spread")
        self.assertEqual(len(expected_pairs), 4)

        # Simulate what _run_menu_sweep samples without LLM.
        schedule = []
        for market_i in range(pipe.menu_sweep_markets):
            market = pipe._config_space.sample(
                pipe._rng, batch_index=1000 + market_i
            )
            pipe._sweep_markets.append(market)
            for pair in expected_pairs:
                schedule.append((market_i, market.label, pair))

        self.assertEqual(len(schedule), 12)  # 3 markets × 4 pairs
        # Within each market, all 4 pairs in sorted order.
        for market_i in range(3):
            chunk = schedule[market_i * 4 : (market_i + 1) * 4]
            self.assertEqual([c[2] for c in chunk], expected_pairs)
            labels = {c[1] for c in chunk}
            self.assertEqual(len(labels), 1)

        # Distinct markets across sweep indices.
        labels = [schedule[i * 4][1] for i in range(3)]
        self.assertEqual(len(set(labels)), 3)

        # Q suite history includes sweep markets for later block phase.
        pipe._block_markets.append(
            pipe._config_space.sample(pipe._rng, batch_index=0)
        )
        history = pipe._q_suite_history()
        self.assertEqual(len(history), 4)  # 3 sweep + 1 block
        suite = history[-pipe.q_suite_size :]
        self.assertEqual(len(suite), 3)
        self.assertEqual(suite[-1].label, pipe._block_markets[0].label)

    def test_rejects_sweep_without_min2(self):
        set_principle_menu_preset(None)
        with self.assertRaises(ValueError):
            ReflectionPipeline(
                config_space=_tiny_space(),
                menu_sweep_markets=2,
                principle_menu_preset=None,
                critic_enabled=False,
                steps=1,
                batch_size=1,
                episodes_per_config=1,
                run_dir="/tmp/reflection_menu_sweep_reject",
            )

    def test_schedule_deterministic_for_seed(self):
        def fingerprints(seed: int):
            pipe = ReflectionPipeline(
                config_space=_tiny_space(),
                batch_size=1,
                steps=1,
                episodes_per_config=1,
                critic_enabled=False,
                seed=seed,
                market_block_size=10,
                q_suite_size=2,
                menu_sweep_markets=2,
                principle_menu_preset="min2",
                initial_strategy_types=["spread"],
                run_dir="/tmp/reflection_menu_sweep_det",
            )
            out = []
            for i in range(pipe.menu_sweep_markets):
                m = pipe._config_space.sample(pipe._rng, batch_index=1000 + i)
                out.append(
                    (
                        m.label,
                        m.num_zi_buy,
                        m.num_zi_sell,
                        m.timesteps,
                        m.q_max,
                        tuple(m.shade_range),
                        m.arrival_rate,
                    )
                )
            return out

        self.assertEqual(fingerprints(11), fingerprints(11))
        self.assertNotEqual(fingerprints(11), fingerprints(12))


if __name__ == "__main__":
    unittest.main()
