import unittest

from marketsim.LLM.reflection_pipeline.pipeline import (
    CRITIC_MAX_PROMPT_CHARS,
    CRITIC_MAX_OUTPUT_TOKENS,
    _compress_batch_data_for_critic,
    _critic_max_output_tokens,
    _prepare_critic_prompt,
    _slim_llm_interaction_for_critic,
)


class CriticPromptBudgetTest(unittest.TestCase):
    def test_slim_interaction_drops_legacy_fields(self):
        event = {
            "time": 1,
            "transacted": True,
            "agent_action": {"price_relative": 0.1},
            "pre_lob": {"best_bid_relative": 1.0, "bids_relative": [1.0]},
            "post_lob": {"best_ask_relative": 1.1, "asks_relative": [1.1]},
            "lob": {"best_bid_relative": 1.0},
            "agent_bid": True,
            "agent_transacted": True,
            "trade_price_relative": 0.2,
            "surplus_earned": 3.0,
        }
        slim = _slim_llm_interaction_for_critic(event)
        self.assertNotIn("lob", slim)
        self.assertNotIn("agent_bid", slim)
        self.assertEqual(slim["trade_price_relative"], 0.2)

    def test_prepare_critic_prompt_fits_context_budget(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a"},
                    "episodes": [
                        {
                            "reward": 0.0,
                            "diagnostic_json": {
                                "events": [
                                    {
                                        "event_type": "llm_order",
                                        "time": t,
                                        "pre_lob": {
                                            "best_bid_relative": 1.0,
                                            "best_ask_relative": 1.1,
                                            "bids_relative": [1.0] * 4,
                                            "asks_relative": [1.1] * 4,
                                        },
                                        "agent_action": {
                                            "agent_bid": True,
                                            "price_relative": 0.5,
                                            "shade": 0.1,
                                            "position": 0,
                                            "current_pv": 10.0,
                                        },
                                        "post_lob": {
                                            "best_bid_relative": 1.0,
                                            "best_ask_relative": 1.1,
                                            "bids_relative": [1.0] * 4,
                                            "asks_relative": [1.1] * 4,
                                        },
                                        "transacted": False,
                                        "lob": {
                                            "best_bid_relative": 1.0,
                                            "best_ask_relative": 1.1,
                                        },
                                    }
                                    for t in range(25)
                                ]
                            },
                        }
                    ],
                }
            ]
        }
        batch_summary = {
            "reward": -1.0,
            "reward_variance": 2.0,
            "per_config_rewards": {"cfg_a": -1.0},
            "per_config_reward_variances": {"cfg_a": 2.0},
            "overall": {"reward_mean": -1.0, "reward_variance": 2.0},
        }
        code = "def agent_strategy(private_values):\n    return 0.1\n"

        prompt, max_tokens, _system_prompt = _prepare_critic_prompt(
            code=code,
            batch_summary=batch_summary,
            batch_data=batch_data,
            reward=-1.0,
            reward_variance=2.0,
            include_data=True,
        )
        compressed = _compress_batch_data_for_critic(batch_data, max_interactions=8)
        self.assertLessEqual(len(prompt), CRITIC_MAX_PROMPT_CHARS)
        self.assertLessEqual(max_tokens, CRITIC_MAX_OUTPUT_TOKENS)
        self.assertIn("case_cards", compressed)
        self.assertTrue(compressed["case_cards"])
        self.assertNotIn("llm_interactions", compressed)

    def test_max_output_tokens_shrinks_for_large_prompt(self):
        large_prompt = "x" * (CRITIC_MAX_PROMPT_CHARS + 1000)
        max_tokens = _critic_max_output_tokens(
            system_content="system",
            prompt=large_prompt,
        )
        self.assertLess(max_tokens, CRITIC_MAX_OUTPUT_TOKENS)
        self.assertGreaterEqual(max_tokens, 64)


if __name__ == "__main__":
    unittest.main()
