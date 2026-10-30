"""Tests for unified-diff refine, helper completeness, and seed cool exemption."""

from __future__ import annotations

import unittest

from marketsim.LLM.code_generator import CodeGenerator
from marketsim.LLM.reflection_pipeline.code_edit import (
    apply_unified_diff,
    complex_parent_edit_instructions,
    extract_code_or_diff_from_response,
    finalize_strategy_code,
    parent_helper_completeness_error,
    parent_is_complex,
    top_level_function_names,
)
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.prompts import build_generation_prompt
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier


_PARENT = """
_HBL_GRID_N = 40

def _clip01(x):
    return min(max(float(x), 0.0), 1.0)

def _belief_buy(p, orders, *, timestep, grace):
    return 0.5

def agent_strategy(market_history):
    orders = list(market_history.get('orders') or [])
    return _clip01(_belief_buy(0.0, orders, timestep=0, grace=10.0))
""".strip() + "\n"


class TestUnifiedDiffRefine(unittest.TestCase):
    def test_apply_simple_hunk(self):
        parent = "def agent_strategy(market_history):\n    return 0.1\n"
        diff = (
            "--- parent.py\n"
            "+++ child.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def agent_strategy(market_history):\n"
            "-    return 0.1\n"
            "+    return 0.2\n"
        )
        out = apply_unified_diff(parent, diff)
        self.assertIn("return 0.2", out)
        self.assertNotIn("return 0.1", out)

    def test_finalize_diff_vs_python(self):
        parent = "def agent_strategy(market_history):\n    return 0.1\n"
        diff = (
            "```diff\n"
            "--- a\n"
            "+++ b\n"
            "@@ -1,2 +1,2 @@\n"
            " def agent_strategy(market_history):\n"
            "-    return 0.1\n"
            "+    return 0.15\n"
            "```\n"
        )
        extracted = extract_code_or_diff_from_response(diff)
        out = finalize_strategy_code(extracted, parent)
        self.assertIn("return 0.15", out)

        py = "```python\ndef agent_strategy(market_history):\n    return 0.3\n```"
        extracted_py = extract_code_or_diff_from_response(py)
        out_py = finalize_strategy_code(extracted_py, parent)
        self.assertIn("return 0.3", out_py)

    def test_code_generator_extracts_diff(self):
        payload = CodeGenerator._extract_strategy_payload(
            "here\n```diff\n@@ -1,1 +1,1 @@\n-a\n+b\n```\n"
        )
        self.assertIn("@@", payload)


class TestHelperCompleteness(unittest.TestCase):
    def test_detects_dropped_helpers(self):
        child = (
            "def agent_strategy(market_history):\n"
            "    return _clip01(_belief_buy(0.0, [], timestep=0, grace=1.0))\n"
        )
        err = parent_helper_completeness_error(_PARENT, child)
        self.assertIsNotNone(err)
        self.assertIn("_clip01", err)
        self.assertIn("_belief_buy", err)

    def test_accepts_complete_child(self):
        err = parent_helper_completeness_error(_PARENT, _PARENT)
        self.assertIsNone(err)

    def test_accepts_intentional_rewrite_without_helpers(self):
        child = "def agent_strategy(market_history):\n    return 0.05\n"
        err = parent_helper_completeness_error(_PARENT, child)
        self.assertIsNone(err)

    def test_verifier_rejects_incomplete(self):
        child = (
            "def agent_strategy(market_history):\n"
            "    return _clip01(0.2)\n"
        )
        ok, err, _ = StrategyVerifier().verify(
            child,
            required_args=1,
            input_mode="lob_hbl_parity",
            parent_code=_PARENT,
        )
        self.assertFalse(ok)
        self.assertIn("Parent-helper completeness", err or "")

    def test_complex_parent_prompt_mentions_diff(self):
        self.assertTrue(parent_is_complex(_PARENT))
        instr = complex_parent_edit_instructions(_PARENT)
        self.assertTrue(any("```diff```" in x for x in instr))
        prompt = build_generation_prompt(
            parent_code=_PARENT,
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="lob",
            use_reflection=False,
            lob_input_mode="hbl_parity",
        )
        self.assertIn("```diff```", prompt)
        self.assertIn("Complex-parent edit format", prompt)


class TestSeedCoolExemption(unittest.TestCase):
    def test_seed_not_marked_inactive(self):
        tree = RefinementTree(
            initial_types=["lob"],
            lob_input_mode="hbl_parity",
            lob_seed_scale="hbl",
        )
        seed_id = tree.seed_node_ids()[0]
        self.assertEqual(tree.get(seed_id).metadata.get("seed_kind"), "hbl_distilled")

        # Minimal pipeline object: only need tree + cool threshold + method.
        pipe = object.__new__(ReflectionPipeline)
        pipe._tree = tree
        pipe.parent_codegen_fail_cool_threshold = 2

        pipe._note_parent_codegen_outcome(seed_id, success=False)
        pipe._note_parent_codegen_outcome(seed_id, success=False)
        pipe._note_parent_codegen_outcome(seed_id, success=False)
        node = tree.get(seed_id)
        self.assertFalse(bool(node.metadata.get("inactive", False)))
        self.assertGreaterEqual(int(node.metadata.get("codegen_fail_streak", 0)), 2)

    def test_non_seed_still_cooled(self):
        tree = RefinementTree(initial_types=["lob"], lob_input_mode="raw")
        seed_id = tree.seed_node_ids()[0]
        child = tree.add_node(
            parent_id=seed_id,
            code="def agent_strategy(market_history):\n    return 0.02\n",
            reflection="child",
            reward=0.0,
            metadata={"type": "lob", "desc": "child"},
        )
        pipe = object.__new__(ReflectionPipeline)
        pipe._tree = tree
        pipe.parent_codegen_fail_cool_threshold = 2
        pipe._note_parent_codegen_outcome(child.node_id, success=False)
        pipe._note_parent_codegen_outcome(child.node_id, success=False)
        self.assertTrue(bool(tree.get(child.node_id).metadata.get("inactive", False)))


class TestTopLevelNames(unittest.TestCase):
    def test_names(self):
        names = top_level_function_names(_PARENT)
        self.assertIn("agent_strategy", names)
        self.assertIn("_belief_buy", names)
        self.assertIn("_clip01", names)


if __name__ == "__main__":
    unittest.main()
