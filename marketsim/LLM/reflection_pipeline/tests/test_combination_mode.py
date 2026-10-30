import unittest

from marketsim.LLM.reflection_pipeline.pipeline import (
    PLATEAU_NON_IMPROVE_THRESHOLD,
    ReflectionPipeline,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    build_combination_revision_prompt,
    combination_generator_instructions,
    derive_family_id,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier


FAMILY_A_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    return min(1.0, sum(private_values) / (max(private_values) + 100.0))
""".strip()

FAMILY_B_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    mean = sum(private_values) / len(private_values)
    return min(1.0, abs(mean) / (abs(mean) + 100.0))
""".strip()

GATED_CODE = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    if max(private_values) > 80.0:
        return min(1.0, sum(private_values) / (max(private_values) + 100.0))
    spread = max(private_values) - min(private_values)
    return min(1.0, spread / (spread + 50.0))
""".strip()


class CombinationPromptTest(unittest.TestCase):
    def test_derive_family_id_uses_form_and_tags(self):
        reflection = "DIAGNOSIS:\n  principle: inventory_remaining_units\n  form_class: ratio\n  shade_error_direction: shade_more\n"
        fam = derive_family_id(reflection=reflection, code=FAMILY_A_CODE)
        self.assertIn("inventory_remaining_units", fam)
        self.assertIn("ratio", fam)
        self.assertIn("uses_sum", fam)

    def test_combination_revision_prompt_lists_sources(self):
        prompt = build_combination_revision_prompt(
            parent_code=FAMILY_A_CODE,
            parent_reflection="stalled",
            family_sources=[
                {
                    "family_id": "fam_a",
                    "form_class": "ratio",
                    "principle": "inventory_remaining_units",
                    "mean_surplus": 900.0,
                    "code": FAMILY_A_CODE,
                },
                {
                    "family_id": "fam_b",
                    "form_class": "ratio",
                    "principle": "pv_rank_or_distance_to_best",
                    "mean_surplus": 700.0,
                    "code": FAMILY_B_CODE,
                },
            ],
            non_improve_streak=6,
            critic_mode="structured_principle_form",
        )
        self.assertIn("combine strategy families", prompt.lower())
        self.assertIn("fam_a", prompt)
        self.assertIn("fam_b", prompt)
        self.assertIn("gated combination", prompt.lower())

    def test_combination_generator_instructions(self):
        instr = combination_generator_instructions(
            family_a_id="fam_a",
            family_b_id="fam_b",
        )
        joined = " ".join(instr)
        self.assertIn("COMBINATION MODE", joined)
        self.assertIn("fam_a", joined)
        self.assertIn("fam_b", joined)


class CombinationPipelineTest(unittest.TestCase):
    def _pipeline(self) -> ReflectionPipeline:
        pipe = ReflectionPipeline(
            steps=1,
            batch_size=1,
            episodes_per_config=1,
            critic_enabled=False,
            plateau_escape_enabled=True,
            plateau_escape_mode="combination",
            plateau_non_improve_threshold=PLATEAU_NON_IMPROVE_THRESHOLD,
            elite_parent_bias_prob=0.0,
        )
        pipe._tree = RefinementTree(initial_types=["pv"])
        seed = pipe._tree.get(1)
        seed.code = FAMILY_A_CODE
        seed.metadata["mean_surplus"] = 500.0
        seed.reflection = (
            "DIAGNOSIS:\n"
            "  principle: inventory_remaining_units\n"
            "  form_class: ratio\n"
        )
        other = pipe._tree.add_node(
            parent_id=1,
            code=FAMILY_B_CODE,
            reflection=(
                "DIAGNOSIS:\n"
                "  principle: pv_rank_or_distance_to_best\n"
                "  form_class: ratio\n"
            ),
            reward=-100.0,
            metadata={"type": "pv"},
        )
        other.metadata["mean_surplus"] = 900.0
        other.visits = 1
        return pipe

    def test_select_family_sources_picks_distinct_top(self):
        pipe = self._pipeline()
        parent = pipe._tree.get(1)
        parent.visits = 1
        sources = pipe._select_family_sources(parent)
        self.assertEqual(len(sources), 2)
        family_ids = {s["family_id"] for s in sources}
        self.assertEqual(len(family_ids), 2)


class CombinationProbeTest(unittest.TestCase):
    def test_rejects_near_identical_candidate(self):
        verifier = StrategyVerifier()
        err = verifier.combination_behavior_distinct(
            FAMILY_A_CODE,
            [FAMILY_A_CODE, FAMILY_B_CODE],
        )
        self.assertIsNotNone(err)
        self.assertIn("near-identical", err.lower())

    def test_accepts_gated_candidate(self):
        verifier = StrategyVerifier()
        err = verifier.combination_behavior_distinct(
            GATED_CODE,
            [FAMILY_A_CODE, FAMILY_B_CODE],
        )
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
