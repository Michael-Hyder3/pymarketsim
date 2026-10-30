import unittest
from types import SimpleNamespace

from marketsim.LLM.reflection_pipeline.pipeline import (
    PARENT_CODEGEN_FAIL_COOL_THRESHOLD,
    ReflectionPipeline,
    _collect_forbidden_strategy_examples,
)
from marketsim.LLM.reflection_pipeline.prompts import build_debug_prompt, build_generation_prompt
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


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


class _FakeTree:
    def __init__(self, nodes):
        self._nodes = {int(n.node_id): n for n in nodes}

    def get(self, node_id: int):
        return self._nodes[int(node_id)]


class ForbiddenExamplesTest(unittest.TestCase):
    def test_collects_siblings_only_newest_first(self):
        parent = SimpleNamespace(node_id=13, code=PARENT_CODE, children=[20, 21])
        tree = _FakeTree(
            [
                parent,
                SimpleNamespace(node_id=20, code=SIBLING_CODE, children=[]),
                SimpleNamespace(
                    node_id=21,
                    code="def agent_strategy(private_values):\n    return 0.1",
                    children=[],
                ),
            ]
        )
        examples = _collect_forbidden_strategy_examples(parent, tree)
        self.assertEqual([e["role"] for e in examples], ["sibling", "sibling"])
        self.assertEqual([e["node_id"] for e in examples], [21, 20])
        self.assertNotIn(13, [e["node_id"] for e in examples])


class DuplicateRecoveryPromptTest(unittest.TestCase):
    def test_debug_prompt_prefers_reimplement_under_parent(self):
        err = (
            "Generated strategy is identical or equivalent to existing sibling "
            "(node 18)."
        )
        forbidden = [
            {"node_id": 18, "role": "sibling", "code": SIBLING_CODE},
        ]
        prompt = build_debug_prompt(
            code=SIBLING_CODE,
            error=err,
            forbidden_strategies=forbidden,
        )
        self.assertIn("Sibling strategies already tried", prompt)
        self.assertIn("Stay under the same principle+form", prompt)
        self.assertIn("probe shades differ", prompt)
        self.assertNotIn("coefficient/threshold/damper tweaks are OK", prompt)
        self.assertIn("form_class", prompt)
        self.assertNotIn("Invent a NEW", prompt)
        self.assertIn(SIBLING_CODE.splitlines()[0], prompt)

    def test_debug_prompt_plan_guided_duplicate_recovery(self):
        err = (
            "Generated strategy is identical or equivalent to existing sibling "
            "(node 18)."
        )
        reflection = (
            "DIAGNOSIS:\n"
            "  principle: inventory_remaining_units\n"
            "  form_class: inventory_weighted\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: private_values, q_max\n"
            "  parent_form: ratio\n"
            "  why_parent_fails: low advantage\n"
            "  diversity_vs_parent: inventory weighting\n"
            "MECHANISM: x\n"
            "FIX: y\n"
        )
        prompt = build_debug_prompt(
            code=SIBLING_CODE,
            error=err,
            forbidden_strategies=[{"node_id": 18, "role": "sibling", "code": SIBLING_CODE}],
            parent_reflection=reflection,
            plan_guided_recovery=True,
            critic_mode="structured_principle_form",
            strategy_type="pv",
        )
        self.assertIn("Plan-guided fix", prompt)
        self.assertIn("Parent Reflection DIAGNOSIS", prompt)
        self.assertIn("inventory_remaining_units", prompt)
        self.assertIn("inventory_weighted", prompt)
        self.assertNotIn("Prefer a LOCAL edit", prompt)

    def test_generation_prompt_forbidden_prefers_reimplement(self):
        prompt = build_generation_prompt(
            parent_code=PARENT_CODE,
            parent_reflection="",
            strategy_output_mode="x_cap",
            market_description="test market",
            strategy_type="pv",
            use_reflection=False,
            forbidden_strategies=[
                {"node_id": 18, "role": "sibling", "code": SIBLING_CODE},
            ],
        )
        self.assertIn("Sibling strategies already tried", prompt)
        self.assertIn("Stay under the Parent Strategy's principle+form", prompt)
        self.assertIn("probe shades differ", prompt)
        self.assertNotIn("coefficient/threshold/damper tweaks OK", prompt)


class ParentCodegenCoolTest(unittest.TestCase):
    def _pipeline(self) -> ReflectionPipeline:
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            critic_enabled=False,
            parent_codegen_fail_cool_threshold=PARENT_CODEGEN_FAIL_COOL_THRESHOLD,
        )
        pipe._tree = RefinementTree(initial_types=["pv"])
        # Use a non-seed child: seeds are exempt from inactive cool.
        child = pipe._tree.add_node(
            parent_id=1,
            code=PARENT_CODE,
            reflection="",
            reward=0.0,
            metadata={"type": "pv"},
        )
        pipe._cool_parent_id = int(child.node_id)
        return pipe

    def test_cools_parent_after_threshold_fails(self):
        pipe = self._pipeline()
        parent_id = pipe._cool_parent_id
        pipe._note_parent_codegen_outcome(parent_id, success=False)
        node = pipe._tree.get(parent_id)
        self.assertEqual(int(node.metadata.get("codegen_fail_streak", 0)), 1)
        self.assertFalse(bool(node.metadata.get("inactive", False)))

        pipe._note_parent_codegen_outcome(parent_id, success=False)
        node = pipe._tree.get(parent_id)
        self.assertEqual(int(node.metadata.get("codegen_fail_streak", 0)), 2)
        self.assertTrue(bool(node.metadata.get("inactive", False)))
        self.assertIn("codegen_fail_streak", str(node.metadata.get("inactive_reason", "")))

    def test_success_resets_streak(self):
        pipe = self._pipeline()
        parent_id = pipe._cool_parent_id
        pipe._note_parent_codegen_outcome(parent_id, success=False)
        pipe._note_parent_codegen_outcome(parent_id, success=True)
        node = pipe._tree.get(parent_id)
        self.assertEqual(int(node.metadata.get("codegen_fail_streak", -1)), 0)
        self.assertFalse(bool(node.metadata.get("inactive", False)))


if __name__ == "__main__":
    unittest.main()
