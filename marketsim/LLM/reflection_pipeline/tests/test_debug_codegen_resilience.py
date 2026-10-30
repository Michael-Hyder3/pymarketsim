import tempfile
import unittest
from unittest.mock import patch

from marketsim.LLM.code_generator import CodeGenerationError
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline


class DebugCodegenResilienceTest(unittest.TestCase):
    def test_debug_codegen_failure_skips_step_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ReflectionPipeline(
                initial_strategy_types=["lob"],
                batch_size=1,
                steps=1,
                storage_dir=f"{tmp}/llm_calls",
                output_path=f"{tmp}/out.json",
                run_dir=tmp,
                critic_enabled=True,
                critic_mode="structured_principle_form",
                max_debug_retries=2,
            )
            parent = pipeline._tree.get(1)
            parent.code = (
                "def agent_strategy(market_history):\n"
                "    return 0.1\n"
            )
            parent.reflection = (
                "DIAGNOSIS:\n"
                "  principle: book_imbalance\n"
                "  form_class: ratio\n"
                "  shade_error_direction: shade_more\n"
                "  features_used: prior_lob\n"
                "  current_form: constant\n"
                "  why_this_fails: weak\n"
                "  proposed_change: ratio map\n"
                "MECHANISM: flat.\n"
                "FIX: ratio.\n"
            )

            new_code = (
                "def agent_strategy(market_history):\n"
                "    return 0.2\n"
            )
            with patch.object(
                pipeline._generator,
                "generate",
                return_value=new_code,
            ), patch.object(
                pipeline._verifier,
                "verify",
                return_value=(False, "synthetic verify failure", None),
            ), patch.object(
                pipeline._generator,
                "debug",
                side_effect=CodeGenerationError("No Python code block found"),
            ) as debug_mock:
                gen = pipeline.generate_and_verify(step=1)

            self.assertFalse(gen.success)
            self.assertTrue(debug_mock.called)
            self.assertIsNone(gen.strategy_fn)


if __name__ == "__main__":
    unittest.main()
