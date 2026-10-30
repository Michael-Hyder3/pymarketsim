import unittest

from marketsim.LLM.reflection_pipeline.diagnosis_agreement import (
    validate_code_implements_diagnosis,
)
from marketsim.LLM.reflection_pipeline.hypothesis_quality import (
    score_hypothesis_adherence,
)
from marketsim.LLM.reflection_pipeline.prompts import (
    validate_structured_principle_form_reflection,
)


CLAMP_HELPER_CODE = """
def agent_strategy(market_state):
    units_available = market_state.get('units_available', 0)
    q_max = market_state.get('q_max', 1)
    inv = units_available / max(q_max, 1)
    return clamp_linear(inv, 0.0, 1.0)

def clamp_linear(x, lo, hi):
    return min(hi, max(lo, x))
""".strip()

REFLECTION = """
DIAGNOSIS:
  principle: inventory_remaining_fraction
  form_class: clamp_linear
  shade_error_direction: shade_more
  features_used: units_available, q_max
  current_form: constant shade
  why_this_fails: leftover high; shade_fill near 0
  action: replace
  proposed_change: inventory-weighted clamp
MECHANISM: x
FIX: y
""".strip()


class ClampLinearAgreementTest(unittest.TestCase):
    def test_accepts_named_clamp_linear_helper(self):
        err = validate_code_implements_diagnosis(
            CLAMP_HELPER_CODE,
            REFLECTION,
            strategy_type="spread",
        )
        self.assertIsNone(err)


class HypothesisQualityTest(unittest.TestCase):
    def test_scores_high_when_code_matches(self):
        report = score_hypothesis_adherence(
            CLAMP_HELPER_CODE,
            REFLECTION,
            strategy_type="spread",
        )
        self.assertGreaterEqual(report.overall, 0.9)
        self.assertTrue(report.form_class_ok)
        self.assertIsNone(report.violation)

    def test_scores_low_when_features_missing(self):
        code = "def agent_strategy(market_state):\n    return 0.5\n"
        report = score_hypothesis_adherence(code, REFLECTION, strategy_type="spread")
        self.assertLess(report.overall, 0.5)
        self.assertIsNotNone(report.violation)


class HypothesisDiversityValidationTest(unittest.TestCase):
    def test_rejects_duplicate_of_parent_hypothesis(self):
        parent = (
            "DIAGNOSIS:\n"
            "  principle: inventory_remaining_fraction\n"
            "  form_class: clamp_linear\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: units_available\n"
            "  current_form: x\n"
            "  why_this_fails: leftover 0.8\n"
            "  action: replace\n"
            "  proposed_change: z\n"
            "MECHANISM: a\n"
            "FIX: b\n"
        )
        child = parent.replace("current_form: x", "current_form: w")
        err = validate_structured_principle_form_reflection(
            child,
            strategy_type="spread",
            parent_reflection=parent,
        )
        self.assertIsNotNone(err)
        self.assertIn("action=replace", err or "")

    def test_accepts_new_hypothesis_pair(self):
        parent = (
            "DIAGNOSIS:\n"
            "  principle: inventory_remaining_fraction\n"
            "  form_class: clamp_linear\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: units_available\n"
            "  current_form: x\n"
            "  why_this_fails: leftover 0.8\n"
            "  action: replace\n"
            "  proposed_change: z\n"
            "MECHANISM: a\n"
            "FIX: b\n"
        )
        child = (
            "DIAGNOSIS:\n"
            "  principle: bid_ask_spread_width\n"
            "  form_class: ratio\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: best_bid, best_ask\n"
            "  current_form: inventory clamp\n"
            "  why_this_fails: leftover 0.7; shade_fill 0.00\n"
            "  action: replace\n"
            "  proposed_change: spread ratio map\n"
            "MECHANISM: a\n"
            "FIX: b\n"
        )
        err = validate_structured_principle_form_reflection(
            child,
            strategy_type="spread",
            parent_reflection=parent,
        )
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
