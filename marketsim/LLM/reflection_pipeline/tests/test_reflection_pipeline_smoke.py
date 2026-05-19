import unittest

from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig
from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline, DEFAULT_STRATEGY_CODE


class ReflectionPipelineSmokeTest(unittest.TestCase):
    def test_smoke_run(self):
        config = ReflectionMarketConfig(
            label="smoke",
            timesteps=50,
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
            storage_dir="llm_calls_reflection_pipeline_smoke",
            output_path="results/reflection_pipeline_smoke.json",
            seed=42,
            fixed_strategy_code=DEFAULT_STRATEGY_CODE,
            critic_enabled=False,
        )
        result = pipeline.run()
        self.assertIn("summary", result)
        self.assertEqual(len(result["summary"]), 1)
        self.assertTrue(result["summary"][0]["success"])


if __name__ == "__main__":
    unittest.main()
