import unittest
from types import SimpleNamespace

from marketsim.LLM.reflection_pipeline.pipeline import (
    DUPLICATE_CODE_RECENT_NODE_LIMIT,
    _duplicate_strategy_rejection_message,
    _strategy_code_equivalent,
)
from marketsim.LLM.reflection_pipeline.prompts import build_debug_prompt


PARENT_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    return sum(private_values) / max(private_values)
""".strip()

SIBLING_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    max_pv = max(private_values)
    ratio = sum(private_values) / max_pv
    return min(max(ratio, 0), 1)
""".strip()

RECENT_CODE = """
def agent_strategy(private_values):
    return 0.25
""".strip()

DISTINCT_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    return abs(private_values[0]) / (abs(private_values[0]) + 1.0)
""".strip()


class _FakeTree:
    def __init__(self, nodes):
        self._nodes = {int(n.node_id): n for n in nodes}

    def get(self, node_id: int):
        return self._nodes[int(node_id)]

    def evaluable_node_ids(self):
        return [nid for nid in self._nodes if nid != 0]


class DuplicateStrategyRejectTest(unittest.TestCase):
    def setUp(self):
        self.parent = SimpleNamespace(node_id=17, code=PARENT_CODE, children=[18, 19])
        self.sibling = SimpleNamespace(node_id=18, code=SIBLING_CODE, children=[])
        self.other_sibling = SimpleNamespace(node_id=19, code=PARENT_CODE, children=[])
        self.recent = SimpleNamespace(node_id=30, code=RECENT_CODE, children=[])
        self.tree = _FakeTree(
            [
                SimpleNamespace(node_id=0, code="", children=[17]),
                self.parent,
                self.sibling,
                self.other_sibling,
                self.recent,
            ]
        )

    def test_rejects_parent_duplicate(self):
        msg = _duplicate_strategy_rejection_message(
            PARENT_CODE, parent=self.parent, tree=self.tree
        )
        self.assertIsNotNone(msg)
        self.assertIn("parent", msg)
        self.assertIn("node 17", msg)

    def test_rejects_sibling_duplicate(self):
        msg = _duplicate_strategy_rejection_message(
            SIBLING_CODE, parent=self.parent, tree=self.tree
        )
        self.assertIsNotNone(msg)
        self.assertIn("sibling", msg)
        self.assertIn("node 18", msg)

    def test_rejects_recent_tree_duplicate(self):
        # Parent with no overlapping sibling/parent code; match only via recent node.
        parent = SimpleNamespace(node_id=17, code=PARENT_CODE, children=[])
        tree = _FakeTree(
            [
                parent,
                SimpleNamespace(node_id=30, code=RECENT_CODE, children=[]),
            ]
        )
        msg = _duplicate_strategy_rejection_message(
            RECENT_CODE, parent=parent, tree=tree
        )
        self.assertIsNotNone(msg)
        self.assertIn("recent tree node", msg)
        self.assertIn("node 30", msg)
        self.assertIn(f"last {DUPLICATE_CODE_RECENT_NODE_LIMIT}", msg)

    def test_allows_distinct_code(self):
        msg = _duplicate_strategy_rejection_message(
            DISTINCT_CODE, parent=self.parent, tree=self.tree
        )
        self.assertIsNone(msg)

    def test_comment_only_change_still_equivalent(self):
        commented = SIBLING_CODE + "\n# just a comment\n"
        self.assertTrue(_strategy_code_equivalent(commented, SIBLING_CODE))
        msg = _duplicate_strategy_rejection_message(
            commented, parent=self.parent, tree=self.tree
        )
        self.assertIsNotNone(msg)

    def test_debug_prompt_has_stronger_duplicate_text(self):
        err = (
            "Generated strategy is identical or equivalent to existing sibling "
            "(node 18). Do not return a duplicate of any strategy already in the tree."
        )
        prompt = build_debug_prompt(code=SIBLING_CODE, error=err)
        self.assertIn("parent, a sibling, or another", prompt)
        self.assertIn("argument-order", prompt)
        self.assertIn("Stay under the same principle+form", prompt)
        self.assertIn("probe shades differ", prompt)
        self.assertNotIn("coefficient/threshold/damper tweaks are OK", prompt)
        self.assertIn("form_class", prompt)
        self.assertNotIn("invent a structurally different", prompt.lower())


if __name__ == "__main__":
    unittest.main()
