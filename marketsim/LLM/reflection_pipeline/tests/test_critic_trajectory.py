import unittest

from marketsim.LLM.reflection_pipeline.evaluation import select_median_episode_interactions


def _episode(reward: float, interaction_times: list[int]) -> dict:
    return {
        "reward": reward,
        "diagnostic_json": {
            "events": [
                {
                    "event_type": "llm_order",
                    "time": t,
                    "pre_lob": {"best_bid_relative": 1.0},
                    "agent_action": {"price_relative": 0.5},
                    "post_lob": {"best_bid_relative": 1.1},
                    "transacted": False,
                }
                for t in interaction_times
            ]
        },
    }


class CriticTrajectoryTest(unittest.TestCase):
    def test_selects_median_episode_and_first_twenty_interactions(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a"},
                    "episodes": [
                        _episode(-10.0, list(range(30))),
                        _episode(0.0, list(range(100, 125))),
                        _episode(10.0, list(range(200, 210))),
                    ],
                }
            ]
        }
        payload = select_median_episode_interactions(batch_data, max_interactions=20)
        self.assertEqual(payload["median_episode_reward"], 0.0)
        self.assertEqual(payload["batch_episode_count"], 3)
        self.assertEqual(payload["interaction_count"], 20)
        self.assertEqual(len(payload["llm_interactions"]), 20)
        self.assertEqual(payload["llm_interactions"][0]["time"], 100)
        self.assertEqual(payload["llm_interactions"][-1]["time"], 119)
        self.assertIn("pre_lob", payload["llm_interactions"][0])
        self.assertIn("post_lob", payload["llm_interactions"][0])

    def test_even_episode_count_uses_upper_median_index(self):
        batch_data = {
            "configs": [
                {
                    "config": {"label": "cfg_a"},
                    "episodes": [
                        _episode(-5.0, [1]),
                        _episode(5.0, [2]),
                    ],
                }
            ]
        }
        payload = select_median_episode_interactions(batch_data, max_interactions=20)
        self.assertEqual(payload["median_episode_reward"], 5.0)


if __name__ == "__main__":
    unittest.main()
