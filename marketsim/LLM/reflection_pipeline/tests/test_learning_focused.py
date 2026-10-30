"""Tests for opt-in learning-focused pipeline options (defaults remain classic)."""

from __future__ import annotations

import unittest

from marketsim.LLM.reflection_pipeline.learning_focused import (
    LessonMemory,
    classic_lob_order_memory_seed_code,
    format_lesson_line,
    hbl_exact_lob_order_memory_seed_code,
    learning_outcome_signals,
    lob_order_memory_seed_code,
    prepend_learning_prompt_sections,
    validate_shade_collapse_reflection,
    validate_zero_trade_reflection,
    zi_scale_lob_order_memory_seed_code,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier


def _zero_trade_summary(*, shade: float = 0.55) -> dict:
    return {
        "zi_benchmark_comparison": {
            "llm_mean_surplus": 0.0,
            "llm_mean_requested_x_cap": shade,
            "llm_mean_shade": shade,
            "beat_fraction": 0.0,
        }
    }


def _trading_summary(*, mean_shade: float = 0.04) -> dict:
    return {
        "zi_benchmark_comparison": {
            "llm_mean_surplus": 400.0,
            "llm_mean_requested_x_cap": 0.04,
            "llm_mean_shade": mean_shade,
            "beat_fraction": 0.3,
        }
    }


def _collapsed_trading_summary() -> dict:
    return _trading_summary(mean_shade=0.002)


_REPLACE_OK = """
DIAGNOSIS:
  current_form: high shade from reject_rate
  why_this_fails: over-shade / no fills; surplus ≈ 0
  principle: fill_vs_reject
  form_class: threshold
  shade_error_direction: shade_more
  features_used: orders, accepted
  mechanism: lower shade floor when reject_rate high
  action: replace
  proposed_change: switch to near-zero shade order-memory rule
MECHANISM: Cut shade so orders can fill.
FIX: Replace with a low-shade fill_vs_reject form.
""".strip()


_REFINE_BAD = """
DIAGNOSIS:
  current_form: high shade from reject_rate
  why_this_fails: leftover advantage on market_0001
  principle: fill_vs_reject
  form_class: threshold
  shade_error_direction: shade_more
  features_used: orders, accepted
  mechanism: tweak constant
  action: refine
  proposed_change: nudge reject weight by 0.01
MECHANISM: Small constant edit.
FIX: Keep form, retune weight.
""".strip()


_COLLAPSE_REPLACE_OK = """
DIAGNOSIS:
  current_form: multiplicative dampers
  why_this_fails: near-zero shade / shade collapse; over-aggressive always-take
  principle: fill_vs_reject
  form_class: threshold
  shade_error_direction: shade_more
  features_used: orders, accepted
  mechanism: restore trade-capable shade band
  action: replace
  proposed_change: piecewise form with shade floor away from zero
MECHANISM: Stop stacking dampers.
FIX: Replace with a fill_vs_reject threshold that keeps shade above collapse.
""".strip()


class TestLobSeedScaleOption(unittest.TestCase):
    def test_classic_seed_unchanged_default(self):
        classic = classic_lob_order_memory_seed_code()
        self.assertEqual(lob_order_memory_seed_code(), classic)
        self.assertIn("return 0.25", classic)
        self.assertNotIn("0.015 + 0.06", classic)

    def test_zi_scale_seed_band(self):
        code = zi_scale_lob_order_memory_seed_code()
        self.assertIn("return 0.02", code)
        self.assertIn("0.015 + 0.06 * raw", code)
        ns: dict = {}
        exec(code, ns)
        fn = ns["agent_strategy"]
        self.assertAlmostEqual(fn({"orders": [], "current_pv": 100.0}), 0.02)
        orders = [
            {"accepted": False, "side": 1, "price_rel": 10.0},
            {"accepted": True, "side": -1, "price_rel": 5.0},
        ]
        shade = fn({"orders": orders, "current_pv": 200.0})
        self.assertGreaterEqual(shade, 0.01)
        self.assertLessEqual(shade, 0.08)

    def test_tree_default_uses_classic(self):
        tree = RefinementTree(initial_types=["lob"], lob_input_mode="raw")
        lob = tree.get(tree.seed_node_ids()[0])
        self.assertIn("return 0.25", lob.code)

    def test_tree_zi_scale_option(self):
        tree = RefinementTree(
            initial_types=["lob"],
            lob_input_mode="raw",
            lob_seed_scale="zi_scale",
        )
        lob = tree.get(tree.seed_node_ids()[0])
        self.assertIn("return 0.02", lob.code)
        self.assertIn("ZI-scale", lob.metadata.get("desc", ""))

    def test_tree_hbl_seed_option(self):
        tree = RefinementTree(
            initial_types=["lob"],
            lob_input_mode="hbl_parity",
            lob_seed_scale="hbl",
        )
        lob = tree.get(tree.seed_node_ids()[0])
        self.assertEqual(lob.code, hbl_exact_lob_order_memory_seed_code())
        self.assertEqual(lob.metadata.get("seed_kind"), "hbl_distilled")
        self.assertIn("HBL-like", lob.metadata.get("desc", ""))
        self.assertNotIn("HBL_EXACT_SEED = True", lob.code)

    def test_tree_hbl_scaffold_seed_option(self):
        from marketsim.LLM.reflection_pipeline.learning_focused import (
            hbl_scaffold_lob_order_memory_seed_code,
        )

        tree = RefinementTree(
            initial_types=["lob"],
            lob_input_mode="hbl_parity",
            lob_seed_scale="hbl_scaffold",
        )
        lob = tree.get(tree.seed_node_ids()[0])
        self.assertEqual(lob.code, hbl_scaffold_lob_order_memory_seed_code())
        self.assertEqual(lob.metadata.get("seed_kind"), "hbl_scaffold")
        self.assertIn("scaffold", lob.metadata.get("desc", "").lower())
        self.assertNotIn("n_grid", lob.code)
        self.assertNotIn("HBL_EXACT_SEED = True", lob.code)


class TestFalsifyZeroTradeOption(unittest.TestCase):
    def test_signals_detect_zero_trade(self):
        signals = learning_outcome_signals(_zero_trade_summary())
        self.assertTrue(signals["zero_trade_regime"])
        self.assertTrue(signals["over_shade_dead"])

    def test_signals_trading_ok(self):
        signals = learning_outcome_signals(_trading_summary())
        self.assertFalse(signals["zero_trade_regime"])
        self.assertFalse(signals["over_shade_dead"])
        self.assertFalse(signals["shade_collapse_regime"])

    def test_validate_rejects_refine_when_zero_trade(self):
        err = validate_zero_trade_reflection(
            _REFINE_BAD,
            batch_summary=_zero_trade_summary(),
        )
        self.assertIsNotNone(err)
        self.assertIn("replace", err.lower())

    def test_validate_accepts_replace_with_over_shade(self):
        err = validate_zero_trade_reflection(
            _REPLACE_OK,
            batch_summary=_zero_trade_summary(),
        )
        self.assertIsNone(err)

    def test_validate_noop_when_trading(self):
        err = validate_zero_trade_reflection(
            _REFINE_BAD,
            batch_summary=_trading_summary(),
        )
        self.assertIsNone(err)

    def test_prompt_inject_only_when_requested(self):
        base = "BASE PROMPT"
        unchanged = prepend_learning_prompt_sections(
            base,
            batch_summary=_zero_trade_summary(),
            falsify_zero_trade=False,
        )
        self.assertEqual(unchanged, base)
        injected = prepend_learning_prompt_sections(
            base,
            batch_summary=_zero_trade_summary(),
            falsify_zero_trade=True,
        )
        self.assertIn("## Outcome signals", injected)
        self.assertIn("zero_trade_regime=True", injected)
        self.assertIn("action MUST be replace", injected)
        self.assertTrue(injected.endswith(base))


class TestFalsifyShadeCollapseOption(unittest.TestCase):
    def test_signals_detect_collapse(self):
        signals = learning_outcome_signals(_collapsed_trading_summary())
        self.assertTrue(signals["shade_collapse_regime"])
        self.assertFalse(signals["zero_trade_regime"])

    def test_validate_rejects_refine_when_collapsed(self):
        err = validate_shade_collapse_reflection(
            _REFINE_BAD,
            batch_summary=_collapsed_trading_summary(),
        )
        self.assertIsNotNone(err)
        self.assertIn("replace", err.lower())

    def test_validate_accepts_collapse_replace(self):
        err = validate_shade_collapse_reflection(
            _COLLAPSE_REPLACE_OK,
            batch_summary=_collapsed_trading_summary(),
        )
        self.assertIsNone(err)

    def test_prompt_inject_collapse_rules(self):
        injected = prepend_learning_prompt_sections(
            "BODY",
            batch_summary=_collapsed_trading_summary(),
            falsify_shade_collapse=True,
        )
        self.assertIn("Shade-collapse falsification", injected)
        self.assertIn("shade_collapse_regime=True", injected)


class TestNoopAndParentDiversity(unittest.TestCase):
    def test_parent_behavior_rejects_identical_shade(self):
        code = "\n".join(
            [
                "def agent_strategy(market_history):",
                "    return 0.05",
            ]
        )
        twin = "\n".join(
            [
                "def agent_strategy(market_history):",
                "    # comment only",
                "    return 0.05",
            ]
        )
        verifier = StrategyVerifier()
        err = verifier.parent_behavior_distinct(
            twin,
            code,
            strategy_type="lob",
            lob_input_mode="raw",
        )
        self.assertIsNotNone(err)
        self.assertIn("near-identical", err.lower())

    def test_probe_rejects_collapsed_constant(self):
        code = "\n".join(
            [
                "def agent_strategy(market_history):",
                "    return 0.001",
            ]
        )
        verifier = StrategyVerifier()
        err = verifier.probe_mean_shade_collapsed(
            code,
            strategy_type="lob",
            lob_input_mode="raw",
            eps=0.01,
        )
        self.assertIsNotNone(err)
        self.assertIn("collapse", err.lower())

    def test_select_ucb_respects_child_cap(self):
        tree = RefinementTree(initial_types=["lob"], lob_seed_scale="zi_scale")
        seed_id = tree.seed_node_ids()[0]
        tree.update_q_value(seed_id, -100.0, mode="replace")
        for i in range(3):
            tree.add_node(
                parent_id=seed_id,
                code=(
                    f"def agent_strategy(market_history):\n"
                    f"    return {0.03 + 0.01 * i}\n"
                ),
                reflection="",
                reward=-200.0 - i,
                metadata={"type": "lob"},
            )
        alt = tree.add_node(
            parent_id=seed_id,
            code="def agent_strategy(market_history):\n    return 0.04\n",
            reflection="",
            reward=-50.0,
            metadata={"type": "lob"},
        )
        tree.update_q_value(alt.node_id, -50.0, mode="replace")
        chosen = tree.select_ucb(
            exploration=0.01,
            total_visits=max(1, sum(n.visits for n in tree.nodes())),
            max_children=3,
        )
        self.assertEqual(chosen.node_id, alt.node_id)


class TestLessonMemoryOption(unittest.TestCase):
    def test_fifo_capacity(self):
        mem = LessonMemory(capacity=2)
        mem.append("a")
        mem.append("b")
        mem.append("c")
        self.assertEqual(mem.as_list(), ["b", "c"])

    def test_prompt_inject_lessons(self):
        line = format_lesson_line(
            step_label="step 1",
            signals=learning_outcome_signals(_zero_trade_summary()),
            action="replace",
            note="try lower shade",
        )
        prompt = prepend_learning_prompt_sections(
            "BODY",
            lessons=[line],
            falsify_zero_trade=False,
        )
        self.assertIn("## Lessons so far", prompt)
        self.assertIn("step 1", prompt)
        self.assertIn("BODY", prompt)


if __name__ == "__main__":
    unittest.main()


class TestOutcomeFirstOption(unittest.TestCase):
    def test_score_line_and_validation(self):
        from marketsim.LLM.reflection_pipeline.learning_focused import (
            format_outcome_score_line,
            validate_outcome_first_reflection,
            enrich_verify_error_hint,
        )

        summary = _trading_summary()
        score = format_outcome_score_line(summary, reward=-10.0)
        self.assertIn("SCORE:", score)
        self.assertIn("Adv=", score)
        good = (
            f"{score}\n"
            "FAILS_BECAUSE: Adv=-10.0 Surp=400 shade shows ZI gaps.\n"
            "CHANGE: Raise shade when reject_rate is high.\n"
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: threshold\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: x\n"
            "  why_this_fails: Adv=-10\n"
            "  action: replace\n"
            "  proposed_change: Raise shade\n"
            "MECHANISM: x\n"
            "FIX: y\n"
        )
        self.assertIsNone(
            validate_outcome_first_reflection(good, batch_summary=summary, reward=-10.0)
        )
        self.assertIn(
            "Missing SCORE",
            validate_outcome_first_reflection("DIAGNOSIS:\n  action: refine\n") or "",
        )
        hint = enrich_verify_error_hint("RuntimeError: KeyError: 'accepted'")
        self.assertIn("o.get('accepted'", hint)

    def test_outcome_first_prompts(self):
        from marketsim.LLM.reflection_pipeline.prompts import (
            build_reflection_prompt,
            build_generation_prompt,
        )

        summary = _trading_summary()
        batch = {
            "tapes": {
                "worst": {
                    "shouts": [
                        {"t": 1, "side": "buy", "shade": 0.01, "gap": 1.0, "filled": 0}
                    ]
                }
            }
        }
        critic = build_reflection_prompt(
            code="def agent_strategy(market_history):\n    return 0.05\n",
            batch_summary=summary,
            batch_data=batch,
            reward=-10.0,
            critic_mode="structured_principle_form",
            strategy_type="lob",
            outcome_first=True,
        )
        self.assertIn("SCORE:", critic)
        self.assertIn("FAILS_BECAUSE", critic)
        self.assertIn("Teaching case", critic)
        gen = build_generation_prompt(
            parent_code="def agent_strategy(market_history):\n    return 0.05\n",
            parent_reflection=(
                "SCORE: Adv=-10 Surp=400 shade=0.04 ZI=0/21\n"
                "FAILS_BECAUSE: Adv=-10 with ZI=0/21.\n"
                "CHANGE: Raise shade band.\n"
                "DIAGNOSIS:\n  principle: fill_vs_reject\n  form_class: threshold\n"
                "  features_used: orders\n  current_form: x\n  why_this_fails: Adv=-10\n"
                "  action: replace\n  proposed_change: Raise shade band\n"
                "MECHANISM: x\nFIX: y\n"
            ),
            strategy_output_mode="shade",
            market_description="CDA",
            strategy_type="lob",
            critic_mode="structured_principle_form",
            parent_performance={
                "reward": -10.0,
                "zi_benchmark_comparison": summary["zi_benchmark_comparison"],
            },
            generator_batch_data=batch,
            outcome_first=True,
        )
        self.assertIn("Learning contract", gen)
        self.assertLess(gen.index("CHANGE:"), gen.index("## Parent Strategy"))
