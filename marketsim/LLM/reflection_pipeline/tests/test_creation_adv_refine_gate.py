"""Live-Q improve gate, elite selection, and UCB exploit."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.pipeline import (
    HYPOTHESIS_JUMP_PROB,
    PLATEAU_IMPROVE_ADV_EPS,
    PLATEAU_NON_IMPROVE_THRESHOLD,
    ReflectionPipeline,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


PARENT_CODE = """
def agent_strategy(market_history):
    return 0.2
""".strip()

CHILD_CODE = """
def agent_strategy(market_history):
    return 0.3
""".strip()


class LiveQRefineGateTest(unittest.TestCase):
    def _pipeline(self) -> ReflectionPipeline:
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            critic_enabled=False,
            plateau_escape_enabled=True,
            plateau_non_improve_threshold=PLATEAU_NON_IMPROVE_THRESHOLD,
            elite_parent_bias_prob=0.0,
            hypothesis_jump_prob=0.0,
            run_dir="/tmp/reflection_live_q_gate_test",
        )
        pipe._tree = RefinementTree(initial_types=["pv"])
        seed = pipe._tree.get(1)
        seed.code = PARENT_CODE
        seed.q_value = 100.0
        seed.metadata["mean_surplus"] = 1000.0
        return pipe

    def test_defaults(self):
        self.assertEqual(PLATEAU_NON_IMPROVE_THRESHOLD, 8)
        self.assertEqual(PLATEAU_IMPROVE_ADV_EPS, 1.0)
        self.assertEqual(HYPOTHESIS_JUMP_PROB, 0.0)

    def test_store_updates_mean_surplus_only(self):
        pipe = self._pipeline()
        node = pipe._tree.get(1)
        pipe._store_creation_metrics(
            1,
            batch_data=None,
            comparison={"llm_mean_surplus": 50.0},
            reward=999.0,
        )
        self.assertEqual(node.metadata["mean_surplus"], 50.0)
        self.assertIsNone(node.metadata.get("creation_reward"))

    def test_q_improve_resets_streak(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        parent.metadata["non_improve_streak"] = 3
        child = pipe._tree.add_node(
            parent_id=1,
            code=CHILD_CODE,
            reflection="",
            reward=150.0,
            metadata={"type": "pv", "mean_surplus": 900.0},
        )
        child.q_value = 150.0
        pipe._note_parent_refine_outcome(1, child_id=child.node_id, escape_mode=False)
        self.assertEqual(parent.metadata["non_improve_streak"], 0)

    def test_worse_q_increments_streak(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        parent.q_value = 100.0
        parent.metadata["non_improve_streak"] = 0
        child = pipe._tree.add_node(
            parent_id=1,
            code=CHILD_CODE,
            reflection="",
            reward=80.0,
            metadata={"type": "pv", "mean_surplus": 2000.0},
        )
        child.q_value = 80.0
        pipe._note_parent_refine_outcome(1, child_id=child.node_id, escape_mode=False)
        self.assertEqual(parent.metadata["non_improve_streak"], 1)

    def test_elite_prefers_highest_live_q(self):
        pipe = self._pipeline()
        low_q = pipe._tree.add_node(
            parent_id=1,
            code=CHILD_CODE,
            reflection="",
            reward=10.0,
            metadata={"type": "pv", "mean_surplus": 5000.0},
        )
        low_q.q_value = 10.0
        high_q = pipe._tree.add_node(
            parent_id=1,
            code=CHILD_CODE + "\n# b",
            reflection="",
            reward=200.0,
            metadata={"type": "pv", "mean_surplus": 500.0},
        )
        high_q.q_value = 200.0
        elite = pipe._select_elite_parent()
        self.assertIsNotNone(elite)
        self.assertEqual(elite.node_id, high_q.node_id)
        self.assertNotEqual(elite.node_id, low_q.node_id)

    def test_ucb_exploit_uses_live_q_only(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        parent.q_value = 10.0
        parent.visits = 5
        other = pipe._tree.add_node(
            parent_id=1,
            code=CHILD_CODE,
            reflection="",
            reward=40.0,
            metadata={"type": "pv"},
        )
        other.q_value = 40.0
        other.visits = 1
        chosen = pipe._tree.select_ucb(
            exploration=0.0,
            total_visits=10,
            max_children=0,
        )
        self.assertEqual(chosen.node_id, other.node_id)

    def test_ucb_respects_child_cap(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        parent.q_value = 100.0
        parent.visits = 1
        for i in range(20):
            pipe._tree.add_node(
                parent_id=1,
                code=CHILD_CODE + f"\n# c{i}",
                reflection="",
                reward=float(i),
                metadata={"type": "pv"},
            )
        other = pipe._tree.add_node(
            parent_id=1,
            code=CHILD_CODE + "\n# alt",
            reflection="",
            reward=50.0,
            metadata={"type": "pv"},
        )
        other.q_value = 90.0
        other.visits = 1
        self.assertGreaterEqual(len(parent.children), 20)
        chosen = pipe._tree.select_ucb(
            exploration=0.0,
            total_visits=50,
            max_children=20,
        )
        self.assertNotEqual(chosen.node_id, parent.node_id)


if __name__ == "__main__":
    unittest.main()
