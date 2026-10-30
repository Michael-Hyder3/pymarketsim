import unittest

from marketsim.LLM.reflection_pipeline.pipeline import ReflectionPipeline
from marketsim.LLM.reflection_pipeline.prompts import build_generation_prompt
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


class GeneratorPromptOptionsTest(unittest.TestCase):
    def test_build_generation_prompt_with_critic_data(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(market_history):\n    return 0.5\n",
            parent_reflection="Environment: test",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="lob",
            use_reflection=True,
            parent_performance={
                "reward": 1.5,
                "reward_variance": 0.25,
            },
            generator_batch_summary={
                "reward": 1.5,
                "reward_variance": 0.25,
                "overall": {"reward_mean": 1.5},
            },
            generator_batch_data={
                "median_episode_reward": 1.5,
                "llm_interactions": [
                    {
                        "pre_lob": {"bids_relative": [0.1]},
                        "agent_action": {"x": 0.2},
                        "post_lob": {"asks_relative": [0.2]},
                        "transacted": True,
                    }
                ],
                "interaction_count": 1,
            },
        )
        self.assertIn("## Parent Batch Reward", prompt)
        self.assertIn("reward=1.5000 (variance=0.2500)", prompt)
        self.assertIn("Parent Median-Episode LLM Interactions", prompt)
        self.assertIn("pre_lob", prompt)

    def test_build_generation_prompt_with_critic_on_scalar_reward(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(private_values):\n    return 0.2\n",
            parent_reflection="Environment: test\nPerformance: weak\nFix: tune shade",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="pv",
            use_reflection=True,
            parent_performance={
                "reward": 0.7,
                "reward_variance": 0.21,
            },
            q_reward_mode="zi_beat_fraction",
        )
        self.assertIn("## Parent Q Reward (ZI beat fraction)", prompt)
        self.assertIn("reward=0.7000 (variance=0.2100)", prompt)
        self.assertIn("Parent Reflection", prompt)

    def test_build_generation_prompt_with_sibling(self):
        prompt = build_generation_prompt(
            parent_code="def agent_strategy(private_values):\n    return 0.1\n",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test market",
            strategy_type="pv",
            use_reflection=False,
            sibling_code="def agent_strategy(private_values):\n    return 0.9\n",
            sibling_context={
                "node_id": 7,
                "q_value": -12.5,
                "reward": -10.0,
                "reward_variance": 1.0,
            },
        )
        self.assertIn("Most Recent Sibling Strategy", prompt)
        self.assertIn("Node 7", prompt)
        self.assertIn("return 0.9", prompt)
        self.assertIn("must differ meaningfully from the Most Recent Sibling Strategy", prompt)
        self.assertIn("Do not return code that is identical or near-identical to the sibling", prompt)

    def test_most_recent_sibling_selects_highest_child_id(self):
        tree = RefinementTree(initial_types=["pv"])
        parent_id = tree.seed_node_ids()[0]
        first = tree.add_node(
            parent_id=parent_id,
            code="first",
            reflection="",
            reward=1.0,
        )
        second = tree.add_node(
            parent_id=parent_id,
            code="second",
            reflection="",
            reward=2.0,
        )
        pipeline = ReflectionPipeline(
            steps=1,
            critic_enabled=False,
            generator_include_sibling=True,
        )
        pipeline._tree = tree
        sibling = pipeline._most_recent_sibling(parent_id)
        self.assertIsNotNone(sibling)
        self.assertEqual(sibling.node_id, second.node_id)
        self.assertNotEqual(sibling.node_id, first.node_id)


    def test_zi_mode_auto_enables_generator_critic_data(self):
        pipeline = ReflectionPipeline(
            steps=1,
            q_reward_mode="zi_beat_fraction",
            generator_include_critic_data=False,
            generator_include_sibling=False,
        )
        self.assertTrue(pipeline.generator_include_critic_data)
        # Sibling stays optional (not forced) so frozen recipes can keep DOF low.
        self.assertFalse(pipeline.generator_include_sibling)

    def test_zi_advantage_mode_auto_enables_generator_critic_data(self):
        pipeline = ReflectionPipeline(
            steps=1,
            q_reward_mode="zi_advantage",
            generator_include_critic_data=False,
            generator_include_sibling=False,
        )
        self.assertTrue(pipeline.generator_include_critic_data)
        self.assertFalse(pipeline.generator_include_sibling)


if __name__ == "__main__":
    unittest.main()
