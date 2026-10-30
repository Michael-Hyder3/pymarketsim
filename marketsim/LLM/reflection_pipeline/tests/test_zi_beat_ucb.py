import unittest

from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


class ZiBeatFractionUcbTest(unittest.TestCase):
    def test_zi_beat_fraction_defaults_to_ucb(self):
        pipeline = ReflectionPipeline(
            steps=1,
            critic_enabled=False,
            q_reward_mode="zi_beat_fraction",
        )
        self.assertEqual(pipeline._resolved_parent_selection_mode(), "ucb")

    def test_zi_beat_fraction_updates_q_from_beat_fraction_not_thompson(self):
        pipeline = ReflectionPipeline(
            steps=1,
            critic_enabled=False,
            q_reward_mode="zi_beat_fraction",
            parent_selection_mode="ucb",
            q_update_mode="replace",
        )
        tree = RefinementTree(initial_types=["pv"])
        node_id = tree.seed_node_ids()[0]
        pipeline._tree = tree
        comparison = {
            "beat_count": 7,
            "total_benchmarks": 21,
            "beat_fraction": 7.0 / 21.0,
        }
        # Seed-style update path: metadata for diagnostics, Q from beat fraction.
        pipeline._apply_zi_node_eval(node_id, comparison=comparison)
        pipeline._tree.update_q_value(node_id, comparison["beat_fraction"], mode="replace")
        node = tree.get(node_id)
        self.assertAlmostEqual(node.q_value, 7.0 / 21.0)
        self.assertEqual(node.metadata["zi_beat_count"], 7)
        self.assertNotIn("ts_alpha", node.metadata)

    def test_ucb_prefers_higher_beat_fraction(self):
        tree = RefinementTree(initial_types=["pv"])
        good_id = tree.seed_node_ids()[0]
        tree.add_node(
            parent_id=0,
            code="weak",
            reflection="",
            reward=0.0,
            metadata={"type": "pv"},
        )
        weak_id = tree._next_id - 1
        tree.update_q_value(good_id, 0.8, mode="replace")
        tree.update_q_value(weak_id, 0.1, mode="replace")
        # Equalize visits so exploit dominates.
        tree.get(good_id).visits = 5
        tree.get(weak_id).visits = 5
        parent = tree.select_ucb(exploration=0.0, total_visits=10)
        self.assertEqual(parent.node_id, good_id)


if __name__ == "__main__":
    unittest.main()
