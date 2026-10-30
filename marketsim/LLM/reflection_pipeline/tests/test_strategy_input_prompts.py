import unittest

from marketsim.LLM.reflection_pipeline.prompts import build_generation_prompt


class StrategyInputPromptTest(unittest.TestCase):
    def test_spread_prompt_describes_current_market_state_only(self):
        prompt = build_generation_prompt(
            parent_code="",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="spread",
            use_reflection=False,
        )
        self.assertIn("agent_strategy(market_state)", prompt)
        self.assertIn("CURRENT book at decision time", prompt)
        self.assertIn("current_pv", prompt)
        self.assertIn("CRITICAL scale note", prompt)
        self.assertIn("best_bid or best_ask is None", prompt)
        self.assertIn("private_values", prompt)
        self.assertIn("same full schedule PV strategies receive", prompt)
        self.assertNotIn("do not receive the full private_values schedule", prompt.lower())
        self.assertNotIn("lob_snapshots", prompt)

    def test_lob_prompt_describes_prior_market_history_only(self):
        prompt = build_generation_prompt(
            parent_code="",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="lob",
            use_reflection=False,
        )
        self.assertIn("agent_strategy(market_history)", prompt)
        self.assertIn("PRIOR market information only", prompt)
        self.assertIn("lob_snapshots", prompt)
        self.assertIn("top 4 bid price levels", prompt)
        self.assertIn("current_pv", prompt)
        self.assertIn("CRITICAL scale note", prompt)
        self.assertIn("at most the last 10", prompt)
        self.assertIn("You do not receive market_state", prompt)
        self.assertNotIn("'units_available'", prompt)
        self.assertNotIn("'q_max'", prompt)

    def test_full_prompt_keeps_current_lob_schema(self):
        prompt = build_generation_prompt(
            parent_code="",
            parent_reflection="",
            strategy_output_mode="shade",
            market_description="test",
            strategy_type="full",
            use_reflection=False,
        )
        self.assertIn("CURRENT market context", prompt)
        self.assertIn("top 4 bid price levels", prompt)


if __name__ == "__main__":
    unittest.main()
