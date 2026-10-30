import unittest

from marketsim.LLM.reflection_pipeline.tree import RefinementTree


class QUpdateModeTest(unittest.TestCase):
    def test_replace_overwrites_q_value(self):
        tree = RefinementTree(initial_types=["pv"])
        node_id = tree.seed_node_ids()[0]
        tree.update_q_value(node_id, 5.0, mode="replace")
        tree.update_q_value(node_id, 15.0, mode="replace")
        self.assertAlmostEqual(tree.get(node_id).q_value, 15.0)
        self.assertEqual(tree.get(node_id).metadata.get("q_eval_count"), 2)

    def test_average_keeps_running_mean(self):
        tree = RefinementTree(initial_types=["pv"])
        node_id = tree.seed_node_ids()[0]
        tree.update_q_value(node_id, 10.0, mode="average")
        tree.update_q_value(node_id, 20.0, mode="average")
        tree.update_q_value(node_id, 0.0, mode="average")
        self.assertAlmostEqual(tree.get(node_id).q_value, 10.0)
        self.assertEqual(tree.get(node_id).metadata.get("q_eval_count"), 3)

    def test_apply_q_values_respects_mode(self):
        tree = RefinementTree(initial_types=["pv"])
        node_id = tree.seed_node_ids()[0]
        tree.apply_q_values({node_id: 4.0}, mode="average")
        tree.apply_q_values({node_id: 8.0}, mode="average")
        self.assertAlmostEqual(tree.get(node_id).q_value, 6.0)


if __name__ == "__main__":
    unittest.main()
