import json
import tempfile
import unittest
from pathlib import Path

from marketsim.LLM.reflection_pipeline.find_duplicate_strategies import (
    collect_node_strategies,
    group_duplicates,
    strategy_key,
)
from marketsim.LLM.reflection_pipeline.find_duplicate_strategies import NodeStrategy


class FindDuplicateStrategiesTest(unittest.TestCase):
    def test_normalized_mode_treats_whitespace_as_duplicate(self):
        a = "def agent_strategy(x):\n    return 1.0\n"
        b = "def agent_strategy(x):\n    return 1.0   \n"
        self.assertEqual(strategy_key(a, "normalized"), strategy_key(b, "normalized"))

    def test_code_mode_splits_fallback_0_vs_else_0_5(self):
        fallback_zero = (
            "def agent_strategy(private_values):\n"
            "    if not private_values:\n"
            "        return 0.0\n"
            "    pv_max = max(private_values)\n"
            "    pv_mean = sum(private_values) / len(private_values)\n"
            "    if pv_mean >= 0:\n"
            "        return (pv_mean / pv_max) ** 2\n"
            "    return 0.0\n"
        )
        else_half = (
            "def agent_strategy(private_values):\n"
            "    if not private_values:\n"
            "        return 0.0\n"
            "    pv_max = max(private_values)\n"
            "    pv_mean = sum(private_values) / len(private_values)\n"
            "    if pv_mean >= 0:\n"
            "        return (pv_mean / pv_max) ** 2\n"
            "    else:\n"
            "        return 0.5\n"
        )
        self.assertNotEqual(strategy_key(fallback_zero, "code"), strategy_key(else_half, "code"))

    def test_code_mode_ignores_comments(self):
        a = (
            "def agent_strategy(private_values):\n"
            "    if not private_values:\n"
            "        return 0.0\n"
            "    return (sum(private_values) / max(private_values)) ** 2  # old comment\n"
        )
        b = (
            "def agent_strategy(private_values):\n"
            "    if not private_values:\n"
            "        return 0.0\n"
            "    return (sum(private_values) / max(private_values)) ** 2  # new comment\n"
        )
        self.assertEqual(strategy_key(a, "code"), strategy_key(b, "code"))

    def test_exact_mode_distinguishes_whitespace(self):
        a = "def agent_strategy(x):\n    return 1.0\n"
        b = "def agent_strategy(x):\n\n    return 1.0\n"
        self.assertNotEqual(strategy_key(a, "exact"), strategy_key(b, "exact"))

    def test_group_duplicates(self):
        strategies = [
            NodeStrategy(1, "def agent_strategy(x):\n    return 1.0\n", "pv", 0, 0.0, 1, "a"),
            NodeStrategy(2, "def agent_strategy(x):\n    return 1.0\n", "pv", 1, 0.0, 1, "b"),
            NodeStrategy(3, "def agent_strategy(x):\n    return 2.0\n", "pv", 1, 0.0, 1, "c"),
        ]
        groups, _ = group_duplicates(strategies, mode="normalized")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].node_ids, [1, 2])

    def test_collect_from_minimal_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            code_dir = run_dir / "llm_calls" / "iteration_1"
            code_dir.mkdir(parents=True)
            code_path = code_dir / "agent_strategy.py"
            code_path.write_text("def agent_strategy(private_values):\n    return 0.5\n", encoding="utf-8")

            summary = {
                "run_dir": str(run_dir),
                "tree": {
                    "nodes": [
                        {"node_id": 0, "parent_id": None, "children": [1], "visits": 1, "q_value": 0.0, "type": None},
                        {"node_id": 1, "parent_id": 0, "children": [2], "visits": 1, "q_value": -1.0, "type": "pv"},
                        {"node_id": 2, "parent_id": 1, "children": [], "visits": 1, "q_value": -2.0, "type": "pv"},
                    ]
                },
                "summary": [
                    {
                        "step": 1,
                        "success": True,
                        "code_file": str(code_path),
                        "q_refresh": {"new_node_id": 2},
                    }
                ],
            }
            summary_path = run_dir / "reflection_pipeline_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            strategies, warnings = collect_node_strategies(summary_path=summary_path, data=summary)
            self.assertEqual([item.node_id for item in strategies], [1, 2])
            self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
