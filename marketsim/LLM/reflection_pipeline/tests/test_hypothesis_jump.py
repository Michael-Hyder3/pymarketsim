import unittest
from unittest.mock import MagicMock, patch

from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.prompts import (
    build_hypothesis_jump_prompt,
    hypothesis_jump_generator_instructions,
    hypothesis_pair_differs_from_parent,
    parse_principle_form_fields,
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

JUMP_REFLECTION = """
DIAGNOSIS:
  principle: pv_rank_or_distance_to_best
  form_class: ratio
  shade_error_direction: shade_more
  features_used: pv_rank
  parent_form: rank-based shade
  why_parent_fails: inventory units too coarse
  diversity_vs_parent: new principle
MECHANISM: rank distance should track competition.
FIX: use rank-based ratio shading.
""".strip()


class HypothesisJumpPromptTest(unittest.TestCase):
    def test_pair_differs_when_principle_changes(self):
        self.assertTrue(
            hypothesis_pair_differs_from_parent(JUMP_REFLECTION, PARENT_REFLECTION)
        )

    def test_pair_same_when_identical(self):
        self.assertFalse(
            hypothesis_pair_differs_from_parent(PARENT_REFLECTION, PARENT_REFLECTION)
        )

    def test_hypothesis_jump_prompt_mentions_parent_and_change(self):
        prompt = build_hypothesis_jump_prompt(
            parent_code=PARENT_CODE,
            parent_reflection=PARENT_REFLECTION,
            parent_performance={"reward": -100.0},
            critic_mode="structured_principle_form",
            strategy_type="pv",
        )
        self.assertIn("Hypothesis exploration jump", prompt)
        self.assertIn("inventory_remaining_units", prompt)
        self.assertIn("Change at least one of principle or form_class", prompt)

    def test_jump_generator_instructions(self):
        instr = hypothesis_jump_generator_instructions(
            parent_principle="inventory_remaining_units",
            parent_form="ratio",
        )
        joined = " ".join(instr)
        self.assertIn("HYPOTHESIS EXPLORATION JUMP", joined)
        self.assertIn("inventory_remaining_units", joined)


class HypothesisJumpLogicTest(unittest.TestCase):
    def _pipeline(self, *, jump_prob: float = 1.0) -> ReflectionPipeline:
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            critic_enabled=False,
            elite_parent_bias_prob=0.0,
            hypothesis_jump_prob=jump_prob,
            plateau_escape_temperature=0.5,
            critic_mode="structured_principle_form",
        )
        pipe._tree = RefinementTree(initial_types=["pv"])
        seed = pipe._tree.get(1)
        seed.code = PARENT_CODE
        seed.reflection = PARENT_REFLECTION
        seed.metadata["mean_surplus"] = 800.0
        pipe._critic = MagicMock()
        pipe._critic.generate_hypothesis_jump.return_value = JUMP_REFLECTION
        return pipe

    def test_hypothesis_jump_revision_changes_pair(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        revision = pipe._generate_hypothesis_jump_revision(parent)
        parent_fields = parse_principle_form_fields(parent.reflection)
        new_fields = parse_principle_form_fields(revision)
        self.assertTrue(
            new_fields.get("principle") != parent_fields.get("principle")
            or new_fields.get("form_class") != parent_fields.get("form_class")
        )

    def test_generate_and_verify_triggers_jump_at_prob_one(self):
        pipe = self._pipeline(jump_prob=1.0)
        parent = pipe._tree.get(1)
        parent.visits = 3
        parent.q_value = 0.5

        with patch.object(pipe, "_save_code", return_value=("code.py", 2)), patch.object(
            pipe, "_verifier"
        ) as mock_verifier, patch.object(pipe, "_generator") as mock_generator, patch(
            "marketsim.LLM.reflection_pipeline.pipeline.validate_code_implements_diagnosis",
            return_value=None,
        ), patch(
            "marketsim.LLM.reflection_pipeline.pipeline._duplicate_strategy_rejection_message",
            return_value=None,
        ):
            mock_generator.generate.return_value = (
                "def agent_strategy(private_values):\n    return 0.2\n"
            )
            mock_verifier.verify.return_value = (True, None, lambda x: 0.2)

            result = pipe.generate_and_verify(step=2)

        self.assertTrue(result.hypothesis_jump)
        self.assertFalse(result.escape_mode)
        pipe._critic.generate_hypothesis_jump.assert_called_once()
        mock_generator.generate.assert_called_once()
        call_kwargs = mock_generator.generate.call_args.kwargs
        self.assertIn(
            "HYPOTHESIS EXPLORATION JUMP",
            " ".join(call_kwargs["extra_instructions"]),
        )


if __name__ == "__main__":
    unittest.main()
