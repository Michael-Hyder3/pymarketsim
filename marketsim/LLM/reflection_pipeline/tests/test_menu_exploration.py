import unittest

from marketsim.LLM.reflection_pipeline.prompts import (
    build_reflection_prompt,
    collect_hypothesis_pairs_from_reflections,
    compute_menu_coverage_stats,
    format_menu_exploration_section,
    menu_hypothesis_pairs,
    principle_form_menus,
    set_principle_menu_preset,
    validate_structured_principle_form_reflection,
)


class MenuExplorationTest(unittest.TestCase):
    def setUp(self):
        set_principle_menu_preset("lob_e2")

    def tearDown(self):
        set_principle_menu_preset(None)

    def test_lob_e2_menu_size(self):
        pairs = menu_hypothesis_pairs("lob")
        self.assertEqual(len(pairs), 15)

    def test_collect_pairs_from_reflections(self):
        reflections = [
            (
                "DIAGNOSIS:\n"
                "  principle: fill_vs_reject\n"
                "  form_class: piecewise_by_sign\n"
            ),
            (
                "DIAGNOSIS:\n"
                "  principle: side_pressure\n"
                "  form_class: ratio\n"
            ),
        ]
        pairs = collect_hypothesis_pairs_from_reflections(reflections)
        self.assertEqual(
            pairs,
            {
                ("fill_vs_reject", "piecewise_by_sign"),
                ("side_pressure", "ratio"),
            },
        )

    def test_exploration_section_lists_unexplored(self):
        explored = {("fill_vs_reject", "piecewise_by_sign")}
        lines = format_menu_exploration_section(
            strategy_type="lob",
            run_explored_pairs=explored,
        )
        text = "\n".join(lines)
        self.assertIn("If action=refine: KEEP the parent", text)
        self.assertIn("Explored so far in this run: 1/15", text)
        self.assertIn("side_pressure + ratio", text)
        self.assertNotIn("EXPLORATION COMPLETE", text)

    def test_exploration_complete_when_all_tried(self):
        explored = menu_hypothesis_pairs("lob")
        lines = format_menu_exploration_section(
            strategy_type="lob",
            run_explored_pairs=set(explored),
        )
        text = "\n".join(lines)
        self.assertIn("EXPLORATION COMPLETE", text)
        self.assertIn("Explored so far in this run: 15/15", text)

    def test_exploration_section_full_list_when_not_compact(self):
        explored = {("fill_vs_reject", "piecewise_by_sign")}
        lines = format_menu_exploration_section(
            strategy_type="lob",
            run_explored_pairs=explored,
            compact=False,
        )
        text = "\n".join(lines)
        self.assertIn("If action=replace and unexplored pairs remain, pick from this list", text)
        self.assertIn("side_pressure + ratio", text)

    def test_validate_rejects_repeat_during_exploration_phase(self):
        explored = {("fill_vs_reject", "piecewise_by_sign")}
        repeat = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: leftover high; shade_fill near 0\n"
            "  action: replace\n"
            "  proposed_change: try piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        err = validate_structured_principle_form_reflection(
            repeat,
            strategy_type="lob",
            run_explored_pairs=explored,
        )
        self.assertIsNotNone(err)
        self.assertIn("action=replace during menu exploration", err)

    def test_validate_allows_repeat_after_full_coverage(self):
        explored = menu_hypothesis_pairs("lob")
        repeat = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: leftover high; shade_fill near 0\n"
            "  action: replace\n"
            "  proposed_change: retry piecewise with new gate\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        err = validate_structured_principle_form_reflection(
            repeat,
            strategy_type="lob",
            run_explored_pairs=set(explored),
        )
        self.assertIsNone(err)

    def test_refine_keeps_explored_parent_pair(self):
        explored = {("fill_vs_reject", "piecewise_by_sign")}
        parent = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: leftover 0.8; shade_fill 0.00\n"
            "  action: replace\n"
            "  proposed_change: piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        refine = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted, current_pv\n"
            "  current_form: piecewise on fill_vs_reject\n"
            "  why_this_fails: leftover 0.7; pos_gap_miss 0.9\n"
            "  action: refine\n"
            "  proposed_change: gate on leftover inventory\n"
            "MECHANISM: leftover stays high.\n"
            "FIX: keep pair; lower shade when leftover is high.\n"
        )
        err = validate_structured_principle_form_reflection(
            refine,
            strategy_type="lob",
            parent_reflection=parent,
            run_explored_pairs=explored,
        )
        self.assertIsNone(err)

    def test_force_menu_replace_section_inverts_priority(self):
        explored = {("fill_vs_reject", "piecewise_by_sign")}
        classic = "\n".join(
            format_menu_exploration_section(
                strategy_type="lob",
                run_explored_pairs=explored,
            )
        )
        forced = "\n".join(
            format_menu_exploration_section(
                strategy_type="lob",
                run_explored_pairs=explored,
                force_menu_replace=True,
            )
        )
        self.assertIn("If action=refine: KEEP the parent", classic)
        self.assertNotIn("FORCE MENU REPLACE", classic)
        self.assertIn("FORCE MENU REPLACE", forced)
        self.assertIn("action MUST be replace", forced)
        self.assertNotIn("That outranks the unexplored-list rule", forced)

    def test_force_menu_replace_rejects_refine_on_explored_parent(self):
        explored = {("fill_vs_reject", "piecewise_by_sign")}
        parent = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: leftover 0.8; shade_fill 0.00\n"
            "  action: replace\n"
            "  proposed_change: piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        refine = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted, current_pv\n"
            "  current_form: piecewise on fill_vs_reject\n"
            "  why_this_fails: leftover 0.7; pos_gap_miss 0.9\n"
            "  action: refine\n"
            "  proposed_change: gate on leftover inventory\n"
            "MECHANISM: leftover stays high.\n"
            "FIX: keep pair; lower shade when leftover is high.\n"
        )
        # Default off: refine still allowed
        self.assertIsNone(
            validate_structured_principle_form_reflection(
                refine,
                strategy_type="lob",
                parent_reflection=parent,
                run_explored_pairs=explored,
                force_menu_replace=False,
            )
        )
        err = validate_structured_principle_form_reflection(
            refine,
            strategy_type="lob",
            parent_reflection=parent,
            run_explored_pairs=explored,
            force_menu_replace=True,
        )
        self.assertIsNotNone(err)
        self.assertIn("force_menu_replace", err)

    def test_force_menu_replace_accepts_replace_to_unexplored(self):
        explored = {("fill_vs_reject", "piecewise_by_sign")}
        parent = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: leftover 0.8; shade_fill 0.00\n"
            "  action: replace\n"
            "  proposed_change: piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        replace = (
            "DIAGNOSIS:\n"
            "  principle: side_pressure\n"
            "  form_class: ratio\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, side\n"
            "  current_form: piecewise on fill_vs_reject\n"
            "  why_this_fails: leftover 0.7; shade_fill near 0\n"
            "  action: replace\n"
            "  proposed_change: try side_pressure ratio form\n"
            "MECHANISM: parent pair already tried.\n"
            "FIX: unexplored side_pressure ratio.\n"
        )
        err = validate_structured_principle_form_reflection(
            replace,
            strategy_type="lob",
            parent_reflection=parent,
            run_explored_pairs=explored,
            force_menu_replace=True,
        )
        self.assertIsNone(err)

    def test_force_menu_replace_allows_refine_when_menu_complete(self):
        explored = menu_hypothesis_pairs("lob")
        parent = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: leftover 0.8; shade_fill 0.00\n"
            "  action: replace\n"
            "  proposed_change: piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        refine = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted, current_pv\n"
            "  current_form: piecewise on fill_vs_reject\n"
            "  why_this_fails: leftover 0.7; pos_gap_miss 0.9\n"
            "  action: refine\n"
            "  proposed_change: gate on leftover inventory\n"
            "MECHANISM: leftover stays high.\n"
            "FIX: keep pair; lower shade when leftover is high.\n"
        )
        err = validate_structured_principle_form_reflection(
            refine,
            strategy_type="lob",
            parent_reflection=parent,
            run_explored_pairs=set(explored),
            force_menu_replace=True,
        )
        self.assertIsNone(err)

    def test_refine_rejects_new_pair(self):
        parent = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: leftover 0.8; shade_fill 0.00\n"
            "  action: replace\n"
            "  proposed_change: piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        refine_new = (
            "DIAGNOSIS:\n"
            "  principle: side_pressure\n"
            "  form_class: ratio\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, side\n"
            "  current_form: piecewise on fill_vs_reject\n"
            "  why_this_fails: leftover 0.7; pos_gap_miss 0.9\n"
            "  action: refine\n"
            "  proposed_change: switch to spread ratio\n"
            "MECHANISM: leftover stays high.\n"
            "FIX: new pair while claiming refine.\n"
        )
        err = validate_structured_principle_form_reflection(
            refine_new,
            strategy_type="lob",
            parent_reflection=parent,
            run_explored_pairs={("fill_vs_reject", "piecewise_by_sign")},
        )
        self.assertIsNotNone(err)
        self.assertIn("action=refine requires keeping the parent", err or "")

    def test_rejects_canned_why_without_case_card(self):
        canned = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: PRIMARY: low batch advantage / weak fills\n"
            "  action: replace\n"
            "  proposed_change: piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        err = validate_structured_principle_form_reflection(
            canned,
            strategy_type="lob",
        )
        self.assertIsNotNone(err)
        self.assertIn("template sentence", err or "")

    def test_accepts_template_why_when_case_card_cited(self):
        cited = (
            "DIAGNOSIS:\n"
            "  principle: fill_vs_reject\n"
            "  form_class: piecewise_by_sign\n"
            "  shade_error_direction: shade_more\n"
            "  features_used: orders, accepted\n"
            "  current_form: constant shade\n"
            "  why_this_fails: PRIMARY: low batch advantage / weak fills; leftover 0.8\n"
            "  action: replace\n"
            "  proposed_change: piecewise on fill_vs_reject\n"
            "MECHANISM: shade flat.\n"
            "FIX: piecewise map.\n"
        )
        err = validate_structured_principle_form_reflection(
            cited,
            strategy_type="lob",
        )
        self.assertIsNone(err)

    def test_reflection_prompt_includes_menu_exploration(self):
        prompt = build_reflection_prompt(
            code="def agent_strategy(market_history):\n    return 0.1\n",
            batch_summary={"per_config_rewards": {"cfg_a": -30.0}},
            batch_data={},
            reward=-50.0,
            include_data=False,
            q_reward_mode="zi_advantage",
            critic_mode="structured_principle_form",
            strategy_type="lob",
            run_explored_pairs={("fill_vs_reject", "piecewise_by_sign")},
        )
        self.assertIn("## Menu exploration", prompt)
        self.assertIn("Explored so far in this run: 1/15", prompt)
        self.assertIn("If action=refine: KEEP the parent", prompt)
        self.assertIn("side_pressure + ratio", prompt)

    def test_min2_menu_has_four_pairs_per_type(self):
        set_principle_menu_preset("min2")
        try:
            for strategy_type in ("pv", "lob", "spread"):
                pairs = menu_hypothesis_pairs(strategy_type)
                principles, form_classes = principle_form_menus(strategy_type)
                self.assertEqual(len(principles), 2, strategy_type)
                self.assertEqual(len(form_classes), 2, strategy_type)
                self.assertEqual(len(pairs), 4, strategy_type)
        finally:
            set_principle_menu_preset(None)

    def test_compute_menu_coverage_stats(self):
        reflections = [
            (
                "DIAGNOSIS:\n"
                "  principle: fill_vs_reject\n"
                "  form_class: piecewise_by_sign\n"
            ),
            (
                "DIAGNOSIS:\n"
                "  principle: side_pressure\n"
                "  form_class: ratio\n"
            ),
            "not parseable",
        ]
        stats = compute_menu_coverage_stats(strategy_type="lob", reflections=reflections)
        self.assertEqual(stats["menu_size"], 15)
        self.assertEqual(stats["explored_pair_count"], 2)
        self.assertEqual(stats["reflection_count"], 3)
        self.assertEqual(stats["parseable_reflection_count"], 2)
        self.assertAlmostEqual(stats["coverage_fraction"], 2 / 15)
        self.assertFalse(stats["exploration_complete"])
        self.assertEqual(len(stats["unexplored_pairs"]), 13)
        self.assertEqual(stats["principle_counts"]["fill_vs_reject"], 1)
        self.assertGreater(stats["principle_entropy"], 0.0)


if __name__ == "__main__":
    unittest.main()
