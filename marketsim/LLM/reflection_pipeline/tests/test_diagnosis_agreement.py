import unittest

from marketsim.LLM.reflection_pipeline.diagnosis_agreement import (
    validate_code_implements_diagnosis,
    validate_features_used_format,
)


SPREAD_GOOD = """
def agent_strategy(market_state):
  best_bid = market_state.get('best_bid')
  best_ask = market_state.get('best_ask')
  units_available = market_state.get('units_available', 0)
  q_max = market_state.get('q_max', 1)
  current_pv = market_state.get('current_pv', 0.0)
  if best_bid is None or best_ask is None or q_max <= 0:
    return 0.5
  spread = best_ask - best_bid
  inv_frac = units_available / q_max
  return min(max(inv_frac * spread / max(abs(current_pv), 1.0), 0.0), 1.0)
""".strip()

SPREAD_REFLECTION = """
DIAGNOSIS:
  principle: bid_ask_spread_width
  form_class: clamp_linear
  shade_error_direction: shade_more
  features_used: best_bid, best_ask, spread, units_available, q_max, current_pv
  parent_form: constant shade
  why_parent_fails: low batch advantage; ZI ahead in wide-spread bins
  diversity_vs_parent: inventory-weighted spread clamp
MECHANISM: wide spread with high inventory overshades.
FIX: clamp_linear on spread scaled by inventory fraction.
""".strip()

PV_BAD_FEATURES = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    return 0.5
""".strip()

PV_REFLECTION = """
DIAGNOSIS:
  principle: current_pv_in_schedule
  form_class: unit_schedule_blend
  shade_error_direction: shade_more
  features_used: private_values, units_available, q_max
  parent_form: constant
  why_parent_fails: flat shade hurts advantage
  diversity_vs_parent: blend unit with schedule
MECHANISM: early units overshade.
FIX: unit_schedule_blend map.
""".strip()


class DiagnosisAgreementTest(unittest.TestCase):
    def test_accepts_matching_spread_code(self):
        err = validate_code_implements_diagnosis(
            SPREAD_GOOD,
            SPREAD_REFLECTION,
            strategy_type="spread",
        )
        self.assertIsNone(err)

    def test_rejects_missing_features(self):
        err = validate_code_implements_diagnosis(
            PV_BAD_FEATURES,
            PV_REFLECTION,
            strategy_type="pv",
        )
        self.assertIsNotNone(err)
        self.assertIn("features_used", err.lower())

    def test_rejects_wrong_form_class_shape(self):
        code = """
def agent_strategy(private_values):
    if not private_values:
        return 0.0
    current_pv = private_values[0]
    return min(max(current_pv / 1000.0, 0.0), 1.0)
""".strip()
        reflection = """
DIAGNOSIS:
  principle: current_pv_magnitude
  form_class: piecewise_by_sign
  shade_error_direction: shade_more
  features_used: current_pv
  parent_form: constant
  why_parent_fails: low advantage
  diversity_vs_parent: piecewise branches
MECHANISM: x
FIX: y
""".strip()
        err = validate_code_implements_diagnosis(code, reflection, strategy_type="pv")
        self.assertIsNotNone(err)
        self.assertIn("form_class", err.lower())

    def test_skips_unstructured_reflection(self):
        err = validate_code_implements_diagnosis(
            PV_BAD_FEATURES,
            "MECHANISM: plateau prose only\nFIX: try something new",
            strategy_type="pv",
        )
        self.assertIsNone(err)

    def test_lob_accepts_code_when_features_used_is_template_prose(self):
        code = """
def agent_strategy(market_history):
    prior = market_history.get('lob') or {}
    bids = prior.get('bids_relative', []) or []
    asks = prior.get('asks_relative', []) or []
    current_pv = float(market_history.get('current_pv') or 0.0)
    spread = market_history.get('spread_relative')
    if spread is None or current_pv <= 0:
        return 0.5
    if len(bids) > len(asks):
        return min(1.0, abs(spread) / (abs(spread) + current_pv))
    return min(1.0, abs(spread) / (abs(spread) + current_pv + 1.0))
""".strip()
        reflection = """
DIAGNOSIS:
  principle: book_imbalance
  form_class: piecewise_by_sign
  shade_error_direction: shade_more
  features_used: prior book / trade / current_pv fields from market_history
  parent_form: constant
  why_parent_fails: low advantage
  diversity_vs_parent: use book imbalance
MECHANISM: x
FIX: y
""".strip()
        err = validate_code_implements_diagnosis(code, reflection, strategy_type="lob")
        self.assertIsNone(err)

    def test_rejects_prose_features_used_at_critic_validation(self):
        reflection = """
DIAGNOSIS:
  principle: book_imbalance
  form_class: ratio
  shade_error_direction: shade_more
  features_used: prior book / trade / current_pv fields from market_history
  current_form: flat
  why_this_fails: low advantage
  proposed_change: use imbalance ratio
MECHANISM: x
FIX: y
""".strip()
        err = validate_features_used_format(reflection, strategy_type="lob")
        self.assertIsNotNone(err)
        self.assertIn("identifiers", err.lower())


if __name__ == "__main__":
    unittest.main()
