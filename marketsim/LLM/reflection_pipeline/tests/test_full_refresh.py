import unittest

from marketsim.LLM.reflection_pipeline.benchmark_suite import build_fixed_eval_suite
from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig
from marketsim.LLM.reflection_pipeline.evaluation import evaluate_batch
from marketsim.LLM.reflection_pipeline.parallel_eval import refresh_nodes_parallel
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier


class FullRefreshTest(unittest.TestCase):
    def test_suite_seeds_match_incremental_mode(self):
        config = ReflectionMarketConfig(
            label="fixed",
            timesteps=40,
            q_max=5,
            num_zi_buy=2,
            num_zi_sell=2,
            pv_var=1e4,
            shade_range=[0.0, 0.5],
            arrival_rate=0.2,
        )
        suite = build_fixed_eval_suite(
            configs=[config],
            episodes_per_config=2,
            base_seed=7,
            step_index=-1,
        )
        for config_index, cfg in enumerate(suite.configs):
            for episode_index in range(2):
                expected = cfg.seed_for(
                    base_seed=7,
                    step=-1,
                    config_index=config_index,
                    episode_index=episode_index,
                )
                self.assertEqual(suite.episode_seeds[config_index][episode_index], expected)

    def test_parallel_refresh_updates_all_nodes(self):
        config = ReflectionMarketConfig(
            label="fixed",
            timesteps=40,
            q_max=5,
            num_zi_buy=2,
            num_zi_sell=2,
            pv_var=1e4,
            shade_range=[0.0, 0.5],
            arrival_rate=0.2,
        )
        suite = build_fixed_eval_suite(
            configs=[config],
            episodes_per_config=1,
            base_seed=11,
            step_index=0,
        )
        node_specs = [
            (
                1,
                "def agent_strategy(private_values):\n    return 0.2\n",
                "pv",
            ),
            (
                2,
                "def agent_strategy(private_values):\n    return 0.8\n",
                "pv",
            ),
        ]
        results = refresh_nodes_parallel(
            node_specs=node_specs,
            suite=suite,
            episodes_per_config=1,
            base_seed=11,
            reward_mode="advantage",
            strategy_output_mode="shade",
            parallel_workers=2,
            include_batch_data=False,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item["success"] for item in results))
        rewards = {item["node_id"]: item["reward"] for item in results}
        self.assertNotEqual(rewards[1], rewards[2])

    def test_full_refresh_pipeline_smoke(self):
        config = ReflectionMarketConfig(
            label="smoke",
            timesteps=40,
            q_max=5,
            num_zi_buy=2,
            num_zi_sell=2,
            pv_var=1e4,
            shade_range=[0.0, 0.5],
            arrival_rate=0.2,
        )
        pipeline = ReflectionPipeline(
            config_pool=[config],
            batch_size=1,
            steps=1,
            episodes_per_config=1,
            exploration_constant=1.0,
            max_debug_retries=1,
            storage_dir="llm_calls_reflection_pipeline_full_refresh_smoke",
            output_path="results/reflection_pipeline_full_refresh_smoke.json",
            seed=42,
            fixed_strategy_code=(
                "def agent_strategy(private_values):\n"
                "    if not private_values:\n"
                "        return 0.0\n"
                "    return 0.4\n"
            ),
            critic_enabled=False,
            initial_strategy_types=["pv"],
            q_eval_mode="full_refresh",
        )
        result = pipeline.run()
        self.assertEqual(result["params"]["q_eval_mode"], "full_refresh")
        self.assertIsNotNone(result["seed_eval_suite"])
        self.assertEqual(len(result["q_refresh_history"]), 2)
        self.assertEqual(result["q_refresh_history"][0].get("phase"), "seed")
        self.assertIn("eval_suite", result["q_refresh_history"][0])
        self.assertTrue(result["summary"][0]["success"])


if __name__ == "__main__":
    unittest.main()
