import tempfile
import unittest
from pathlib import Path

from marketsim.LLM.code_storage import CodeStorage
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


class CodeStorageLayoutTest(unittest.TestCase):
    def test_success_uses_node_id_iteration_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = CodeStorage(tmp)
            path = storage.save_code(
                code="def agent_strategy(x):\n    return 0.0\n",
                function_name="agent_strategy",
                iteration=7,
                task_description="ok",
                success=True,
            )
            self.assertIn("iteration_7", path)
            self.assertFalse(Path(path).parent.parent.name == "attempts")

    def test_failed_attempt_uses_attempts_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = CodeStorage(tmp)
            path = storage.save_failed_attempt(
                code="def agent_strategy(x):\n    return bad\n",
                function_name="agent_strategy",
                step=14,
                attempt=1,
                task_description="fail",
                error_feedback="SyntaxError",
            )
            self.assertIn("attempts/step_0014_try_1", path)
            meta = storage._metadata["agent_strategy_step14_try1"]
            self.assertFalse(meta["success"])
            self.assertEqual(meta["storage_kind"], "failed_attempt")

    def test_next_node_id_matches_saved_iteration(self):
        tree = RefinementTree(initial_types=["pv"])
        self.assertEqual(tree.next_node_id(), 2)
        tree.add_node(
            parent_id=1,
            code="def agent_strategy(x):\n    return 0.1\n",
            reflection="r",
            reward=0.0,
        )
        self.assertEqual(tree.get(2).node_id, 2)

    def test_save_code_paths_match_planned_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp) / "llm_calls"
            pipeline = ReflectionPipeline(
                initial_strategy_types=["pv"],
                batch_size=1,
                steps=1,
                storage_dir=str(storage_dir),
                output_path=str(Path(tmp) / "out.json"),
                run_dir=str(tmp),
                fixed_strategy_code=(
                    "def agent_strategy(private_values):\n"
                    "    return 0.0\n"
                ),
            )
            gen = pipeline.generate_and_verify(step=1)
            self.assertTrue(gen.success)
            self.assertEqual(gen.node_id, 2)
            self.assertIn("iteration_2", gen.code_file or "")


if __name__ == "__main__":
    unittest.main()
