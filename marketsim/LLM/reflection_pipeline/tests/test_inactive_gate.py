import unittest

from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


class InactiveGateTest(unittest.TestCase):
    def _pipeline(self, **kwargs) -> ReflectionPipeline:
        defaults = dict(
            steps=1,
            critic_enabled=False,
            inactive_gate_enabled=True,
            inactive_gate_min_nodes=10,
            inactive_surplus_gap=300.0,
            inactive_beat_gap=3,
        )
        defaults.update(kwargs)
        return ReflectionPipeline(**defaults)

    def _fill_tree(self, tree: RefinementTree, n: int, *, surplus: float, beats: int) -> None:
        parent = tree.seed_node_ids()[0]
        tree.get(parent).metadata["mean_surplus"] = surplus
        tree.get(parent).metadata["zi_beat_count"] = beats
        for i in range(n - 1):
            node = tree.add_node(
                parent_id=parent,
                code=f"node_{i}",
                reflection="",
                reward=0.0,
                metadata={"type": "pv", "mean_surplus": surplus, "zi_beat_count": beats},
            )
            parent = node.node_id

    def test_surplus_gate_marks_weak_node_inactive(self):
        pipeline = self._pipeline(q_reward_mode="advantage")
        tree = RefinementTree(initial_types=["pv"])
        self._fill_tree(tree, 10, surplus=1000.0, beats=0)
        weak = tree.add_node(
            parent_id=tree.seed_node_ids()[0],
            code="weak",
            reflection="",
            reward=-100.0,
            metadata={"type": "pv", "mean_surplus": 600.0},
        )
        pipeline._tree = tree
        marked = pipeline._maybe_mark_inactive(weak.node_id)
        self.assertTrue(marked)
        self.assertTrue(tree.get(weak.node_id).metadata["inactive"])
        self.assertNotIn(weak.node_id, tree.active_node_ids())

    def test_surplus_gate_keeps_close_node_active(self):
        pipeline = self._pipeline(q_reward_mode="advantage")
        tree = RefinementTree(initial_types=["pv"])
        self._fill_tree(tree, 10, surplus=1000.0, beats=0)
        ok = tree.add_node(
            parent_id=tree.seed_node_ids()[0],
            code="ok",
            reflection="",
            reward=-10.0,
            metadata={"type": "pv", "mean_surplus": 800.0},
        )
        pipeline._tree = tree
        self.assertFalse(pipeline._maybe_mark_inactive(ok.node_id))
        self.assertFalse(tree.get(ok.node_id).metadata.get("inactive", False))

    def test_beat_gate_marks_weak_node_inactive(self):
        pipeline = self._pipeline(q_reward_mode="zi_beat_fraction")
        tree = RefinementTree(initial_types=["pv"])
        self._fill_tree(tree, 10, surplus=500.0, beats=8)
        weak = tree.add_node(
            parent_id=tree.seed_node_ids()[0],
            code="weak",
            reflection="",
            reward=0.1,
            metadata={"type": "pv", "zi_beat_count": 4, "mean_surplus": 400.0},
        )
        pipeline._tree = tree
        self.assertTrue(pipeline._maybe_mark_inactive(weak.node_id))
        self.assertTrue(tree.get(weak.node_id).metadata["inactive"])

    def test_first_ten_nodes_never_gated(self):
        pipeline = self._pipeline(q_reward_mode="advantage")
        tree = RefinementTree(initial_types=["pv"])
        self._fill_tree(tree, 9, surplus=1000.0, beats=0)
        tenth = tree.add_node(
            parent_id=tree.seed_node_ids()[0],
            code="tenth",
            reflection="",
            reward=-100.0,
            metadata={"type": "pv", "mean_surplus": 100.0},
        )
        pipeline._tree = tree
        self.assertEqual(len(tree.evaluable_node_ids()), 10)
        self.assertFalse(pipeline._maybe_mark_inactive(tenth.node_id))

    def test_ucb_skips_inactive_nodes(self):
        tree = RefinementTree(initial_types=["pv"])
        good = tree.seed_node_ids()[0]
        weak = tree.add_node(
            parent_id=0,
            code="weak",
            reflection="",
            reward=0.0,
            metadata={"type": "pv"},
        )
        tree.update_q_value(good, 0.1, mode="replace")
        tree.update_q_value(weak.node_id, 1.0, mode="replace")
        tree.get(good).visits = 5
        tree.get(weak.node_id).visits = 5
        tree.mark_inactive(weak.node_id, reason="test")
        parent = tree.select_ucb(exploration=0.0, total_visits=10)
        self.assertEqual(parent.node_id, good)


if __name__ == "__main__":
    unittest.main()
