import unittest
from unittest.mock import MagicMock, patch

from marketsim.LLM.reflection_pipeline.pipeline import (
    DEFAULT_CODEGEN_RECOVERY_MODE,
    ReflectionPipeline,
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
  features_used: inventory_weight
MECHANISM: test
FIX: test
""".strip()

CHILD_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    return 0.2
""".strip()


class LegacyCodegenRecoveryTest(unittest.TestCase):
    def test_default_recovery_mode_is_legacy(self):
        pipe = ReflectionPipeline(steps=1, batch_size=1, episodes_per_config=1)
        self.assertEqual(pipe.codegen_recovery_mode, "legacy")
        self.assertEqual(DEFAULT_CODEGEN_RECOVERY_MODE, "legacy")

    def test_legacy_recovery_mode_stored(self):
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            codegen_recovery_mode="legacy",
        )
        self.assertEqual(pipe.codegen_recovery_mode, "legacy")

    def test_legacy_skips_diagnosis_gate(self):
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            critic_enabled=False,
            critic_mode="structured_principle_form",
            codegen_recovery_mode="legacy",
        )
        pipe._tree = RefinementTree(initial_types=["pv"])
        seed = pipe._tree.get(1)
        seed.code = PARENT_CODE
        seed.reflection = PARENT_REFLECTION
        seed.visits = 2

        with patch.object(pipe, "_save_code", return_value=("code.py", 2)), patch.object(
            pipe, "_verifier"
        ) as mock_verifier, patch.object(pipe, "_generator") as mock_generator, patch(
            "marketsim.LLM.reflection_pipeline.pipeline.validate_code_implements_diagnosis",
            return_value="would reject",
        ) as mock_diag:
            mock_generator.generate.return_value = CHILD_CODE
            mock_verifier.verify.return_value = (True, None, lambda x: 0.2)

            result = pipe.generate_and_verify(step=2)

        self.assertTrue(result.success)
        mock_diag.assert_not_called()

    def test_plan_guided_runs_diagnosis_gate(self):
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            critic_enabled=False,
            critic_mode="structured_principle_form",
            codegen_recovery_mode="plan_guided",
        )
        pipe._tree = RefinementTree(initial_types=["pv"])
        seed = pipe._tree.get(1)
        seed.code = PARENT_CODE
        seed.reflection = PARENT_REFLECTION
        seed.visits = 2

        with patch.object(pipe, "_save_code", return_value=("code.py", 2)), patch.object(
            pipe, "_verifier"
        ) as mock_verifier, patch.object(pipe, "_generator") as mock_generator, patch(
            "marketsim.LLM.reflection_pipeline.pipeline.validate_code_implements_diagnosis",
            return_value="missing feature",
        ) as mock_diag, patch(
            "marketsim.LLM.reflection_pipeline.pipeline._duplicate_strategy_rejection_message",
            return_value=None,
        ):
            mock_generator.generate.return_value = CHILD_CODE
            mock_verifier.verify.return_value = (True, None, lambda x: 0.2)
            pipe._generator.debug = MagicMock(return_value=CHILD_CODE)

            pipe.generate_and_verify(step=2)

        mock_diag.assert_called()


if __name__ == "__main__":
    unittest.main()
