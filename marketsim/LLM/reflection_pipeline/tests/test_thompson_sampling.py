import random
import unittest

from marketsim.LLM.reflection_pipeline.tree import RefinementTree


class ThompsonSamplingTest(unittest.TestCase):
    def test_update_thompson_sets_posterior_mean(self):
        tree = RefinementTree(initial_types=["pv"])
        node_id = tree.seed_node_ids()[0]
        tree.update_thompson_from_zi_comparison(
            node_id,
            {
                "beat_count": 7,
                "total_benchmarks": 10,
                "beat_fraction": 0.7,
            },
            prior_alpha=1.0,
            prior_beta=1.0,
        )
        node = tree.get(node_id)
        self.assertAlmostEqual(float(node.metadata["ts_alpha"]), 8.0)
        self.assertAlmostEqual(float(node.metadata["ts_beta"]), 4.0)
        self.assertAlmostEqual(node.q_value, 8.0 / 12.0)

    def test_update_thompson_accumulates(self):
        tree = RefinementTree(initial_types=["pv"])
        node_id = tree.seed_node_ids()[0]
        comparison = {"beat_count": 4, "total_benchmarks": 10, "beat_fraction": 0.4}
        tree.update_thompson_from_zi_comparison(node_id, comparison)
        tree.update_thompson_from_zi_comparison(node_id, comparison)
        node = tree.get(node_id)
        self.assertAlmostEqual(float(node.metadata["ts_alpha"]), 1.0 + 8.0)
        self.assertAlmostEqual(float(node.metadata["ts_beta"]), 1.0 + 12.0)

    def test_select_thompson_prefers_strong_node(self):
        tree = RefinementTree(initial_types=["pv"])
        good_id = tree.seed_node_ids()[0]
        tree.add_node(parent_id=0, code="weak", reflection="", reward=0.0, metadata={"type": "pv"})
        weak_id = tree._next_id - 1
        tree.update_thompson_from_zi_comparison(
            good_id,
            {"beat_count": 9, "total_benchmarks": 10, "beat_fraction": 0.9},
        )
        tree.update_thompson_from_zi_comparison(
            weak_id,
            {"beat_count": 1, "total_benchmarks": 10, "beat_fraction": 0.1},
        )
        rng = random.Random(0)
        picks = [tree.select_thompson(rng).node_id for _ in range(50)]
        self.assertGreater(picks.count(good_id), picks.count(weak_id))


if __name__ == "__main__":
    unittest.main()
