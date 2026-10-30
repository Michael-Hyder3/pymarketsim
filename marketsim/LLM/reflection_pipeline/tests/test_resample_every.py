"""Hold one sampled market for K steps, then redraw (train=Q)."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.config import (
    ReflectionConfigSpace,
    build_default_config_pool,
    build_default_config_space,
)
from marketsim.LLM.reflection_pipeline.pipeline import (
    ReflectionPipeline,
    _attach_prior_hold_card,
    _compact_case_card_from_payload,
    _compress_batch_data_for_critic,
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


def _pipe(*, seed: int = 42, resample_every: int = 10, **kwargs) -> ReflectionPipeline:
    return ReflectionPipeline(
        config_space=kwargs.pop("config_space", _tiny_space()),
        batch_size=kwargs.pop("batch_size", 3),
        steps=1,
        episodes_per_config=1,
        critic_enabled=False,
        seed=seed,
        resample_every=resample_every,
        initial_strategy_types=["spread"],
        run_dir=kwargs.pop("run_dir", "/tmp/reflection_resample_every_test"),
        **kwargs,
    )


class ResampleEveryTest(unittest.TestCase):
    def test_hold_then_redraw_train_equals_q(self):
        pipe = _pipe()
        self.assertEqual(pipe.batch_size, 1)

        train0, q0 = pipe._train_and_q_configs(0)
        train9, q9 = pipe._train_and_q_configs(9)
        self.assertEqual(len(train0), 1)
        self.assertEqual(train0[0].label, q0[0].label)
        self.assertEqual(train0[0].fingerprint(), q0[0].fingerprint())
        self.assertEqual(train0[0].label, train9[0].label)
        self.assertEqual(train0[0].fingerprint(), train9[0].fingerprint())
        self.assertEqual([c.label for c in q9], [train0[0].label])

        train10, q10 = pipe._train_and_q_configs(10)
        self.assertNotEqual(train10[0].label, train0[0].label)
        self.assertNotEqual(train10[0].fingerprint(), train0[0].fingerprint())
        self.assertEqual(train10[0].fingerprint(), q10[0].fingerprint())
        self.assertTrue(pipe._resample_needs_q_realign)

        train19, q19 = pipe._train_and_q_configs(19)
        self.assertEqual(train19[0].label, train10[0].label)
        self.assertEqual(q19[0].fingerprint(), train10[0].fingerprint())

        train20, _ = pipe._train_and_q_configs(20)
        self.assertNotEqual(train20[0].label, train10[0].label)
        self.assertEqual(len(pipe._resample_markets), 3)

        # Seed eval at step 0 already locked the first hold.
        train0_again, _ = pipe._train_and_q_configs(0)
        self.assertEqual(train0_again[0].fingerprint(), train0[0].fingerprint())

    def test_schedule_deterministic_for_seed(self):
        def schedule(seed: int):
            pipe = _pipe(seed=seed, run_dir=f"/tmp/reflection_resample_every_det_{seed}")
            rows = []
            for step in range(25):
                train, q = pipe._train_and_q_configs(step)
                cfg = train[0]
                rows.append(
                    (
                        cfg.label,
                        cfg.fingerprint(),
                        tuple(c.fingerprint() for c in q),
                    )
                )
            return rows

        self.assertEqual(schedule(99), schedule(99))
        self.assertNotEqual(schedule(99), schedule(100))

    def test_excludes_holdout_ood_cells(self):
        holdout = {cfg.ood_cell() for cfg in build_default_config_pool()}
        pipe = _pipe(
            config_space=build_default_config_space(),
            seed=123,
            run_dir="/tmp/reflection_resample_every_holdout",
        )
        for step in (0, 10, 20, 30, 40):
            train, q = pipe._train_and_q_configs(step)
            self.assertEqual(train[0].ood_cell(), q[0].ood_cell())
            self.assertNotIn(train[0].ood_cell(), holdout)
        cells = [cfg.ood_cell() for cfg in pipe._resample_markets]
        self.assertEqual(len(cells), len(set(cells)))

    def test_rejects_with_home_set_blocks_and_fixed_pool(self):
        # Combined home_set_size + resample_every is allowed (stratified redraw).
        pipe = _pipe(home_set_size=3, resample_every=10)
        self.assertEqual(pipe.batch_size, 3)
        self.assertTrue(pipe.critic_per_config_metrics)
        with self.assertRaises(ValueError):
            _pipe(market_block_size=10)
        with self.assertRaises(ValueError):
            _pipe(menu_sweep_markets=1, principle_menu_preset="min2")
        pool = build_default_config_space().sample_batch(
            __import__("random").Random(0), 1
        )
        with self.assertRaises(ValueError):
            ReflectionPipeline(
                config_pool=pool,
                resample_every=10,
                critic_enabled=False,
                steps=1,
                batch_size=1,
                episodes_per_config=1,
                run_dir="/tmp/reflection_resample_every_reject_pool",
            )

    def test_stratified_home_set_redraw(self):
        pipe = _pipe(
            home_set_size=3,
            resample_every=5,
            run_dir="/tmp/reflection_resample_homeset",
        )
        self.assertEqual(pipe.batch_size, 3)
        train0, q0 = pipe._train_and_q_configs(0)
        self.assertEqual(len(train0), 3)
        self.assertEqual([c.label for c in train0], [c.label for c in q0])
        buyer, seller, balanced = train0
        self.assertGreater(buyer.num_zi_buy, buyer.num_zi_sell)
        self.assertLess(seller.num_zi_buy, seller.num_zi_sell)
        self.assertEqual(balanced.num_zi_buy, balanced.num_zi_sell)
        fps0 = [c.fingerprint() for c in train0]

        train4, _ = pipe._train_and_q_configs(4)
        self.assertEqual([c.fingerprint() for c in train4], fps0)

        train5, q5 = pipe._train_and_q_configs(5)
        self.assertEqual(len(train5), 3)
        self.assertEqual([c.fingerprint() for c in train5], [c.fingerprint() for c in q5])
        self.assertNotEqual([c.fingerprint() for c in train5], fps0)
        self.assertTrue(pipe._resample_needs_q_realign)
        # Opaque labels continue with offset.
        self.assertEqual(
            [c.label for c in train5],
            ["market_0003", "market_0004", "market_0005"],
        )
        buyer2, seller2, balanced2 = train5
        self.assertGreater(buyer2.num_zi_buy, buyer2.num_zi_sell)
        self.assertLess(seller2.num_zi_buy, seller2.num_zi_sell)
        self.assertEqual(balanced2.num_zi_buy, balanced2.num_zi_sell)

    def test_transfer_step_is_first_step_of_new_hold(self):
        pipe = _pipe()
        self.assertFalse(pipe._is_resample_transfer_step(0))
        self.assertFalse(pipe._is_resample_transfer_step(9))
        self.assertTrue(pipe._is_resample_transfer_step(10))
        self.assertFalse(pipe._is_resample_transfer_step(11))
        self.assertTrue(pipe._is_resample_transfer_step(20))
        self.assertFalse(pipe._is_resample_transfer_step(None))

    def test_snapshot_prior_hold_cards_on_redraw(self):
        pipe = _pipe()
        pipe._train_and_q_configs(0)
        node = pipe._tree.get(1)
        node.metadata["hold_case_card"] = {
            "label": "market_0000",
            "advantage": -12.5,
            "leftover": 0.4,
        }
        pipe._train_and_q_configs(10)
        self.assertEqual(
            node.metadata["prior_hold_case_card"]["label"], "market_0000"
        )
        self.assertEqual(pipe._previous_resample_market().label, "market_0000")
        self.assertNotEqual(pipe._resample_markets[-1].label, "market_0000")

    def test_compress_preserves_prior_hold_card(self):
        batch = {
            "configs": [
                {
                    "config": {"label": "market_0001", "q_max": 10},
                    "episodes": [
                        {
                            "reward": -1.0,
                            "llm_surplus": 10.0,
                            "baseline_surplus": 11.0,
                            "episode_stats": {"execution_ratio": 0.5},
                            "diagnostic_json": {"events": []},
                        }
                    ],
                }
            ],
            "prior_hold_card": {
                "label": "market_0000",
                "advantage": -40.0,
                "leftover": 0.9,
                "prior_hold": True,
            },
        }
        compressed = _compress_batch_data_for_critic(batch, max_interactions=0)
        self.assertEqual(compressed["prior_hold_card"]["label"], "market_0000")
        self.assertTrue(compressed["case_cards"])
        self.assertNotIn("prior_hold", compressed["case_cards"][0])
        card = _compact_case_card_from_payload(compressed)
        self.assertEqual(card["label"], "market_0001")
        attached = _attach_prior_hold_card({"configs": []}, {"label": "market_0000"})
        self.assertTrue(attached["prior_hold_card"]["prior_hold"])

    def test_store_context_copies_prior_card_onto_existing_node(self):
        pipe = _pipe()
        node_id = pipe._tree.evaluable_node_ids()[0]
        batch = {
            "configs": [
                {
                    "config": {"label": "market_0001", "q_max": 10},
                    "episodes": [
                        {
                            "reward": -2.0,
                            "llm_surplus": 8.0,
                            "baseline_surplus": 10.0,
                            "episode_stats": {"execution_ratio": 0.4},
                            "diagnostic_json": {"events": []},
                        }
                    ],
                }
            ],
            "prior_hold_card": {
                "label": "market_0000",
                "advantage": -40.0,
                "leftover": 0.9,
                "prior_hold": True,
            },
        }
        pipe._store_node_generator_context(
            node_id,
            reward=-2.0,
            reward_variance=0.0,
            per_config_rewards={"market_0001": -2.0},
            per_config_reward_variances={"market_0001": 0.0},
            batch_data=batch,
        )
        node = pipe._tree.get(node_id)
        self.assertEqual(node.metadata["prior_hold_case_card"]["label"], "market_0000")
        self.assertEqual(node.metadata["hold_case_card"]["label"], "market_0001")
        self.assertNotIn("prior_hold_card", node.metadata["generator_batch_data"])

    def test_critic_prompt_requires_same_failure_when_prior_hold_present(self):
        from marketsim.LLM.reflection_pipeline.prompts import build_reflection_prompt

        prompt = build_reflection_prompt(
            code="def agent_strategy(market_state):\n    return 0.1\n",
            batch_summary={"per_config_rewards": {"market_0001": -10.0}},
            batch_data={
                "prior_hold_card": {
                    "label": "market_0000",
                    "leftover": 0.57,
                    "prior_hold": True,
                }
            },
            reward=-10.0,
            include_data=True,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
            strategy_type="spread",
        )
        self.assertIn("Last hold (same strategy, stats only)", prompt)
        self.assertIn("same_failure_as_last_hold", prompt)


if __name__ == "__main__":
    unittest.main()
