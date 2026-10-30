import unittest

from marketsim.LLM.reflection_pipeline.pipeline_futility_study import (
    SuccessCriterion,
    build_learning_curve,
    futility_decision,
    improvement_slope,
)
from marketsim.LLM.reflection_pipeline.recipes import (
    DEFAULT_SIMPLE_RECIPE,
    get_recipe,
    list_recipes,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree


class ParentSelectionNewestTest(unittest.TestCase):
    def test_select_newest_returns_latest_node(self):
        tree = RefinementTree(initial_types=["pv"])
        seed = tree.seed_node_ids()[0]
        child = tree.add_node(
            parent_id=seed,
            code="child",
            reflection="",
            reward=1.0,
            metadata={"type": "pv"},
        )
        grandchild = tree.add_node(
            parent_id=child.node_id,
            code="grandchild",
            reflection="",
            reward=2.0,
            metadata={"type": "pv"},
        )
        self.assertEqual(tree.select_newest().node_id, grandchild.node_id)


class RecipesTest(unittest.TestCase):
    def test_default_recipe_is_pv_advantage_ucb(self):
        self.assertEqual(DEFAULT_SIMPLE_RECIPE.recipe_id, "simple_pv_advantage_ucb")
        self.assertEqual(DEFAULT_SIMPLE_RECIPE.init_types, ["pv"])
        self.assertEqual(DEFAULT_SIMPLE_RECIPE.q_reward_mode, "advantage")
        self.assertEqual(DEFAULT_SIMPLE_RECIPE.parent_selection_mode, "ucb")
        self.assertEqual(DEFAULT_SIMPLE_RECIPE.episodes_per_config, 50)
        self.assertEqual(DEFAULT_SIMPLE_RECIPE.steps, 125)

    def test_recipe_registry_has_newest_and_zi_arms(self):
        names = list_recipes()
        self.assertIn("simple_pv_advantage_newest", names)
        self.assertIn("simple_pv_zi_beat_ucb", names)
        self.assertEqual(get_recipe("simple_pv_zi_beat_newest").parent_selection_mode, "newest")


class FutilityStudyTest(unittest.TestCase):
    def test_learning_curve_tracks_best_so_far(self):
        steps = [
            {"step": 1, "success": True, "reward": -10.0},
            {"step": 2, "success": True, "reward": -5.0},
            {"step": 3, "success": True, "reward": -8.0, "zi_benchmark_comparison": {"beat_count": 2}},
            {"step": 4, "success": True, "reward": -1.0, "zi_benchmark_comparison": {"beat_count": 4}},
        ]
        curve = build_learning_curve(
            steps,
            checkpoint_every=2,
            success=SuccessCriterion(min_zi_beat_count=3),
        )
        self.assertEqual(curve["n_generator_calls"], 4)
        self.assertEqual(curve["checkpoints"][0]["generator_calls"], 2)
        self.assertEqual(curve["checkpoints"][0]["best_reward_so_far"], -5.0)
        self.assertFalse(curve["checkpoints"][0]["success_so_far"])
        self.assertTrue(curve["checkpoints"][-1]["success_so_far"])
        self.assertEqual(curve["first_success_step"], 4)

    def test_futility_decision_futile_when_flat_and_no_success(self):
        analyses = [
            {
                "learning_curve": {"success_by_end": False},
                "reward_slope_last_window": 0.0,
            },
            {
                "learning_curve": {"success_by_end": False},
                "reward_slope_last_window": 0.0,
            },
        ]
        out = futility_decision(analyses)
        self.assertEqual(out["decision"], "futile")

    def test_futility_decision_need_more_time_when_improving(self):
        analyses = [
            {
                "learning_curve": {"success_by_end": False},
                "reward_slope_last_window": 0.5,
            }
        ]
        out = futility_decision(analyses)
        self.assertEqual(out["decision"], "need_more_time")

    def test_improvement_slope(self):
        checkpoints = [
            {"generator_calls": 25, "best_reward_so_far": -100.0},
            {"generator_calls": 50, "best_reward_so_far": -40.0},
        ]
        slope = improvement_slope(checkpoints, window=20)
        self.assertAlmostEqual(slope, (-40.0 - -100.0) / 25.0)


if __name__ == "__main__":
    unittest.main()
