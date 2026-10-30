import unittest
from unittest.mock import MagicMock, patch

from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline


class SeedReflectionStrategyTypeTest(unittest.TestCase):
    @patch("marketsim.LLM.reflection_pipeline.pipeline.ReflectionCritic")
    @patch("marketsim.LLM.reflection_pipeline.pipeline.StrategyGenerator")
    @patch("marketsim.LLM.reflection_pipeline.pipeline.CodeStorage")
    def test_seed_zi_reflection_passes_strategy_type(self, _storage, _gen, critic_cls):
        critic_cls.return_value = MagicMock()

        pipeline = ReflectionPipeline(
            config_pool=[],
            steps=1,
            episodes_per_config=1,
            critic_enabled=True,
            q_reward_mode="zi_advantage",
            q_eval_mode="full_refresh",
            initial_strategy_types=["spread"],
        )
        pipeline._tree.seed_node_ids = MagicMock(return_value=[1])
        node = MagicMock()
        node.code = "def agent_strategy(market_state):\n    return 0.1\n"
        node.metadata = {"type": "spread"}
        pipeline._tree.get = MagicMock(return_value=node)
        pipeline._tree.set_reflection = MagicMock()
        pipeline._tree.update_q_value = MagicMock()
        pipeline._store_node_performance = MagicMock()
        pipeline._store_node_generator_context = MagicMock()
        pipeline._store_creation_metrics = MagicMock()

        eval_result = MagicMock()
        eval_result.reward = -1.0
        eval_result.reward_variance = 0.0
        eval_result.per_config_rewards = {}
        eval_result.per_config_reward_variances = {}
        eval_result.batch_data = {}
        eval_result.zi_benchmark_comparison = None

        with patch.object(
            pipeline, "_generate_reflection", return_value="DIAGNOSIS:\n"
        ) as mock_refl:
            with patch(
                "marketsim.LLM.reflection_pipeline.pipeline.evaluate_batch_with_zi_benchmarks",
                return_value=eval_result,
            ):
                with patch(
                    "marketsim.LLM.reflection_pipeline.pipeline._critic_prompt_inputs",
                    return_value=({}, {}),
                ):
                    pipeline._evaluate_seed_nodes_on_suite_with_zi(MagicMock())

        mock_refl.assert_called_once()
        self.assertEqual(mock_refl.call_args.kwargs.get("strategy_type"), "spread")


if __name__ == "__main__":
    unittest.main()
