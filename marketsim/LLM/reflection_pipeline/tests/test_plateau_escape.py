import unittest

from marketsim.LLM.reflection_pipeline.pipeline import (
    PLATEAU_NON_IMPROVE_THRESHOLD,
    ReflectionPipeline,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_plateau_revision_prompt,
    parse_principle_form_fields,
    plateau_escape_generator_instructions,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


PARENT_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    return sum(private_values) / max(private_values)
""".strip()

PARENT_REFLECTION = """
DIAGNOSIS:
  principle: inventory_remaining_units
  form_class: ratio
  shade_error_direction: shade_more
  features_used: pv_mean
  parent_form: mean/(mean+max)
  why_parent_fails: local edits stalled
  diversity_vs_parent: none yet
MECHANISM: stalled.
FIX: revise form.
""".strip()


class PlateauPromptTest(unittest.TestCase):
    def test_parse_principle_form_fields(self):
        fields = parse_principle_form_fields(PARENT_REFLECTION)
        self.assertEqual(fields["principle"], "inventory_remaining_units")
        self.assertEqual(fields["form_class"], "ratio")

    def test_plateau_revision_prompt_requires_new_form(self):
        huge_zi = {
            "reward": -400.0,
            "zi_benchmark_comparison": {
                "beat_count": 2,
                "total_benchmarks": 21,
                "beat_fraction": 0.1,
                "llm_mean_surplus": 700.0,
                "per_benchmark": [{"x": i, "blob": "y" * 500} for i in range(50)],
            },
        }
        prompt = build_plateau_revision_prompt(
            parent_code=PARENT_CODE,
            parent_reflection=PARENT_REFLECTION,
            parent_performance=huge_zi,
            non_improve_streak=4,
            critic_mode="structured_principle_form",
            generator_batch_summary=huge_zi,
        )
        self.assertIn("Plateau / stalled hypothesis", prompt)
        self.assertIn("Stalled form_class: ratio", prompt)
        self.assertIn("Change at least one of stalled form_class", prompt)
        self.assertIn("inventory_remaining_units", prompt)
        self.assertNotIn("per_benchmark", prompt)
        self.assertLess(len(prompt), 12000)

    def test_escape_generator_instructions(self):
        instr = plateau_escape_generator_instructions(
            stalled_principle="inventory_remaining_units",
            stalled_form="ratio",
        )
        joined = " ".join(instr)
        self.assertIn("HYPOTHESIS REVISION", joined)
        self.assertIn("ratio", joined)


class PlateauEscapeLogicTest(unittest.TestCase):
    def _pipeline(self) -> ReflectionPipeline:
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            critic_enabled=False,
            plateau_escape_enabled=True,
            plateau_non_improve_threshold=PLATEAU_NON_IMPROVE_THRESHOLD,
            elite_parent_bias_prob=0.0,
        )
        pipe._tree = RefinementTree(initial_types=["pv"])
        seed = pipe._tree.get(1)
        seed.code = PARENT_CODE
        seed.reflection = PARENT_REFLECTION
        seed.metadata["mean_surplus"] = 800.0
        return pipe

    def test_should_escape_after_threshold(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        self.assertFalse(pipe._should_plateau_escape(parent))
        parent.metadata["non_improve_streak"] = PLATEAU_NON_IMPROVE_THRESHOLD
        self.assertTrue(pipe._should_plateau_escape(parent))

    def test_refine_outcome_increments_and_resets(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        parent.q_value = -400.0
        # Add a weak child under parent (worse live Q).
        child = pipe._tree.add_node(
            parent_id=1,
            code="def agent_strategy(private_values):\n    return 0.1\n",
            reflection="",
            reward=-900.0,
            metadata={"type": "pv"},
        )
        child.q_value = -900.0
        child.metadata["mean_surplus"] = 500.0
        pipe._note_parent_refine_outcome(1, child_id=child.node_id, escape_mode=False)
        self.assertEqual(int(parent.metadata["non_improve_streak"]), 1)

        child2 = pipe._tree.add_node(
            parent_id=1,
            code="def agent_strategy(private_values):\n    return 0.2\n",
            reflection="",
            reward=-200.0,
            metadata={"type": "pv"},
        )
        child2.q_value = -200.0
        child2.metadata["mean_surplus"] = 900.0
        pipe._note_parent_refine_outcome(1, child_id=child2.node_id, escape_mode=False)
        self.assertEqual(int(parent.metadata["non_improve_streak"]), 0)

    def test_escape_resets_streak(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        parent.metadata["non_improve_streak"] = 4
        child = pipe._tree.add_node(
            parent_id=1,
            code="def agent_strategy(private_values):\n    return 0.3\n",
            reflection="",
            reward=-500.0,
            metadata={"type": "pv", "escape_mode": True},
        )
        child.metadata["mean_surplus"] = 600.0
        pipe._note_parent_refine_outcome(1, child_id=child.node_id, escape_mode=True)
        self.assertEqual(int(parent.metadata["non_improve_streak"]), 0)

    def test_elite_parent_selection(self):
        pipe = self._pipeline()
        pipe.elite_parent_bias_prob = 1.0
        # Seed starts with visits=1; demote its live Q so elite prefers strong.
        seed = pipe._tree.get(1)
        seed.q_value = -900.0
        weak = pipe._tree.add_node(
            parent_id=1,
            code="def agent_strategy(private_values):\n    return 0.0\n",
            reflection="",
            reward=-800.0,
            metadata={"type": "pv"},
        )
        weak.q_value = -800.0
        weak.metadata["mean_surplus"] = 100.0
        strong = pipe._tree.add_node(
            parent_id=1,
            code="def agent_strategy(private_values):\n    return 0.05\n",
            reflection="",
            reward=-300.0,
            metadata={"type": "pv"},
        )
        strong.q_value = -300.0
        strong.metadata["mean_surplus"] = 950.0
        parent, mode = pipe._select_parent_node()
        self.assertEqual(mode, "elite")
        self.assertEqual(parent.node_id, strong.node_id)


if __name__ == "__main__":
    unittest.main()
