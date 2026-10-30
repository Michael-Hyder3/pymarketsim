import unittest

from marketsim.LLM.reflection_pipeline.evaluation import build_case_card_payload
from marketsim.LLM.reflection_pipeline.pipeline import (
    CRITIC_MAX_PROMPT_CHARS,
    _compress_batch_data_for_critic,
    _prepare_critic_prompt,
)
from marketsim.LLM.reflection_pipeline.prompts import format_case_card_sections


def _event(
    *,
    t: int,
    is_buy: bool,
    pv: float,
    bid: float,
    ask: float,
    shade: float,
    filled: bool,
    position: int = 0,
    surplus: float = 0.0,
) -> dict:
    return {
        "event_type": "llm_order",
        "time": t,
        "pre_lob": {
            "best_bid_relative": bid,
            "best_ask_relative": ask,
            "bids_relative": [bid] * 4,
            "asks_relative": [ask] * 4,
        },
        "agent_action": {
            "agent_bid": is_buy,
            "price_relative": bid if is_buy else ask,
            "shade": shade,
            "position": position,
            "current_pv": pv,
        },
        "post_lob": {
            "best_bid_relative": bid,
            "best_ask_relative": ask,
            "bids_relative": [bid] * 4,
            "asks_relative": [ask] * 4,
        },
        "transacted": filled,
        "surplus_earned": surplus if filled else 0.0,
    }


def _episode(
    *,
    reward: float,
    surplus: float,
    zi: float,
    fill: float,
    events: list,
) -> dict:
    return {
        "reward": reward,
        "llm_surplus": surplus,
        "baseline_surplus": surplus - reward,
        "avg_zi_same_surplus": zi,
        "episode_stats": {"execution_ratio": fill, "total_trades": 1},
        "diagnostic_json": {"events": events},
    }


def _batch() -> dict:
    easy_events = [
        _event(t=1, is_buy=True, pv=5.0, bid=-1.0, ask=1.0, shade=0.05, filled=True, surplus=4.0),
        _event(t=2, is_buy=True, pv=4.0, bid=-1.0, ask=1.0, shade=0.04, filled=True, surplus=3.0),
    ]
    hard_events = [
        _event(
            t=1, is_buy=False, pv=8.0, bid=-0.2, ask=0.2, shade=0.9, filled=False, position=0
        ),
        _event(
            t=2, is_buy=False, pv=7.0, bid=-0.2, ask=0.2, shade=0.85, filled=False, position=0
        ),
        _event(
            t=3, is_buy=True, pv=6.0, bid=-0.2, ask=0.2, shade=0.8, filled=False, position=0
        ),
    ]
    return {
        "configs": [
            {
                "config": {
                    "label": "market_0000",
                    "num_zi_buy": 15,
                    "num_zi_sell": 5,
                    "q_max": 10,
                    "shade_range": [0.0, 0.005],
                },
                "episodes": [
                    _episode(
                        reward=-50.0,
                        surplus=200.0,
                        zi=250.0,
                        fill=0.8,
                        events=easy_events,
                    ),
                    _episode(
                        reward=-30.0,
                        surplus=220.0,
                        zi=250.0,
                        fill=0.9,
                        events=easy_events,
                    ),
                ],
            },
            {
                "config": {
                    "label": "market_0001",
                    "num_zi_buy": 5,
                    "num_zi_sell": 15,
                    "q_max": 10,
                    "shade_range": [0.0, 0.025],
                },
                "episodes": [
                    _episode(
                        reward=-400.0,
                        surplus=40.0,
                        zi=300.0,
                        fill=0.1,
                        events=hard_events,
                    ),
                    _episode(
                        reward=-200.0,
                        surplus=80.0,
                        zi=300.0,
                        fill=0.2,
                        events=hard_events[:2],
                    ),
                ],
            },
        ]
    }


class CaseCardPayloadTest(unittest.TestCase):
    def test_worst_market_and_tapes(self):
        payload = build_case_card_payload(_batch(), max_interactions=4)
        self.assertEqual(payload["worst_market"], "market_0001")
        labels = [c["label"] for c in payload["case_cards"]]
        self.assertEqual(labels, ["market_0000", "market_0001"])
        worst_card = [c for c in payload["case_cards"] if c["worst"]][0]
        self.assertEqual(worst_card["label"], "market_0001")
        self.assertGreater(worst_card["leftover"], 0.7)
        self.assertLess(worst_card["fill"], 0.3)
        self.assertIn("worst", payload["tapes"])
        self.assertEqual(payload["tapes"]["worst"]["label"], "market_0001")
        self.assertEqual(payload["tapes"]["worst"]["reward"], -400.0)
        self.assertIn("good", payload["tapes"])
        self.assertEqual(payload["tapes"]["good"]["label"], "market_0000")
        shout = payload["tapes"]["worst"]["shouts"][0]
        self.assertIn("gap", shout)
        self.assertIn("spread", shout)
        self.assertNotIn("bids_relative", shout)
        text = "\n".join(format_case_card_sections(payload))
        self.assertIn("WORST", text)
        self.assertIn("gap=", text)
        self.assertNotIn("num_zi_buy", text)
        self.assertNotIn("shade_range", text)
        self.assertNotIn("bids_relative", text)

    def test_buyer_gap_is_pv_minus_ask(self):
        payload = build_case_card_payload(_batch(), max_interactions=2)
        good = payload["tapes"]["good"]["shouts"][0]
        # pv=5, ask=1 → gap=4
        self.assertAlmostEqual(good["gap"], 4.0)
        self.assertAlmostEqual(good["spread"], 2.0)


class CaseCardPromptBudgetTest(unittest.TestCase):
    def test_critic_prompt_stays_compact_and_opaque(self):
        batch_data = _batch()
        compressed = _compress_batch_data_for_critic(batch_data, max_interactions=8)
        self.assertIn("case_cards", compressed)
        self.assertNotIn("llm_interactions", compressed)
        batch_summary = {
            "reward": -200.0,
            "reward_variance": 10.0,
            "per_config_rewards": {"market_0000": -40.0, "market_0001": -300.0},
            "per_config_reward_variances": {"market_0000": 1.0, "market_0001": 2.0},
        }
        prompt, max_tokens, _ = _prepare_critic_prompt(
            code="def agent_strategy(market_state):\n    return 0.01\n",
            batch_summary=batch_summary,
            batch_data=batch_data,
            reward=-200.0,
            reward_variance=10.0,
            include_data=True,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
            strategy_type="spread",
        )
        self.assertLessEqual(len(prompt), CRITIC_MAX_PROMPT_CHARS)
        self.assertLess(len(prompt), 12000)
        self.assertIn("case cards", prompt.lower())
        self.assertIn("pos_gap_miss", prompt)
        self.assertIn("Do not divide PV by spread", prompt)
        self.assertNotIn("num_zi_buy", prompt)
        self.assertNotIn('"bids_relative"', prompt)
        self.assertNotIn("asks_relative", prompt)

    def test_zero_interactions_drops_tapes_but_keeps_cards(self):
        payload = build_case_card_payload(_batch(), max_interactions=0)
        self.assertTrue(payload["case_cards"])
        self.assertEqual(payload["tapes"]["worst"]["shouts"], [])

    def test_last_hold_card_is_stats_only_and_opaque(self):
        payload = build_case_card_payload(_batch(), max_interactions=4)
        payload["prior_hold_card"] = {
            "label": "market_0000",
            "advantage": -40.0,
            "surplus": 100.0,
            "zi_surplus": 140.0,
            "fill": 0.8,
            "leftover": 0.2,
            "pos_gap_miss": 0.1,
            "buy_surplus": 90.0,
            "sell_surplus": 10.0,
            "shade_fill": 0.01,
            "shade_miss": 0.2,
            "prior_hold": True,
        }
        text = "\n".join(format_case_card_sections(payload))
        self.assertIn("Last hold (same strategy, stats only)", text)
        self.assertIn("LAST HOLD", text)
        self.assertIn("market_0000", text)
        self.assertIn("same_failure_as_last_hold", text)
        self.assertNotIn("num_zi_buy", text)
        # Tapes still only for the current hold.
        self.assertIn("Episode tapes (compact)", text)


if __name__ == "__main__":
    unittest.main()
