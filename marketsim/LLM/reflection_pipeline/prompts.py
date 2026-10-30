from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from marketsim.LLM.lob_features import (
    active_lob_feature_names,
    is_lob_book_orders_mode,
    is_lob_features_mode,
    is_lob_hbl_parity_mode,
    lob_feature_set_name,
    lob_input_schema_lines,
)
from marketsim.LLM.spread_input import (
    DEFAULT_SPREAD_INPUT_MODE,
    get_spread_omit_fundamental,
    is_spread_book_mode,
    is_spread_zip_parity_mode,
    spread_feature_names_for_prompts,
    spread_input_schema_lines,
)
from marketsim.LLM.reflection_pipeline.config import ZI_Q_REWARD_MODES


def format_per_config_breakdown_lines(
    batch_summary: Dict[str, Any],
    *,
    heading: str = "Per-Config Advantage (batch breakdown)",
) -> List[str]:
    per_config = batch_summary.get("per_config_rewards") or {}
    if not per_config:
        return []
    per_config_var = batch_summary.get("per_config_reward_variances") or {}
    per_config_metrics = batch_summary.get("per_config_metrics") or {}
    lines = ["", f"## {heading}"]
    for label in sorted(per_config.keys()):
        adv = float(per_config[label])
        var = float(per_config_var.get(label, 0.0))
        parts = [f"- {label}: advantage={adv:.4f} (variance={var:.4f})"]
        metrics = per_config_metrics.get(label) or {}
        if metrics.get("llm_surplus_mean") is not None:
            parts.append(f"surplus={float(metrics['llm_surplus_mean']):.4f}")
        if metrics.get("baseline_surplus_mean") is not None:
            parts.append(f"baseline={float(metrics['baseline_surplus_mean']):.4f}")
        if metrics.get("avg_shade_mean") is not None:
            parts.append(f"shade={float(metrics['avg_shade_mean']):.4f}")
        lines.append(" ".join(parts))
    return lines


def _fmt_card_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _format_case_card_line(card: Dict[str, Any], *, tag: str = "") -> str:
    return (
        f"- {card.get('label', '?')}{tag}: "
        f"adv={_fmt_card_num(card.get('advantage'), 2)} "
        f"surplus={_fmt_card_num(card.get('surplus'), 1)} "
        f"zi={_fmt_card_num(card.get('zi_surplus'), 1)} "
        f"fill={_fmt_card_num(card.get('fill'), 2)} "
        f"leftover={_fmt_card_num(card.get('leftover'), 2)} "
        f"shade_fill={_fmt_card_num(card.get('shade_fill'), 3)} "
        f"shade_miss={_fmt_card_num(card.get('shade_miss'), 3)} "
        f"buy_sur={_fmt_card_num(card.get('buy_surplus'), 1)} "
        f"sell_sur={_fmt_card_num(card.get('sell_surplus'), 1)} "
        f"pos_gap_miss={_fmt_card_num(card.get('pos_gap_miss'), 2)}"
    )


def format_case_card_sections(
    batch_data: Dict[str, Any],
    *,
    heading_prefix: str = "",
) -> List[str]:
    if not isinstance(batch_data, dict):
        return []
    cards = batch_data.get("case_cards")
    prior = batch_data.get("prior_hold_card")
    if not cards and not prior:
        return []
    worst = batch_data.get("worst_market")
    lines: List[str] = []
    if cards:
        lines += [
            "",
            _evaluation_section_title(heading_prefix, "Per-market case cards"),
            "Opaque labels. Operational stats only (no buy/sell counts or shade caps).",
            "fill = units executed / q_max; leftover = 1-fill; "
            "pos_gap_miss = share of +EV gap shouts that did not fill.",
            f"Worst market: {worst}." if worst else "Worst market: (none).",
        ]
        for card in cards:
            tag = " WORST" if card.get("worst") else ""
            lines.append(_format_case_card_line(card, tag=tag))

    if isinstance(prior, dict) and prior:
        lines += [
            "",
            _evaluation_section_title(
                heading_prefix, "Last hold (same strategy, stats only)"
            ),
            "Same code on the previous market. No tape. Compare leftover, "
            "pos_gap_miss, and buy vs sell surplus to this hold. "
            "You MUST set DIAGNOSIS same_failure_as_last_hold to yes or no. "
            "yes → refine this hypothesis (keep principle+form). "
            "no → new hypothesis (new pair).",
            _format_case_card_line(prior, tag=" LAST HOLD"),
        ]

    tapes = batch_data.get("tapes") or {}
    if not tapes:
        return lines
    lines += [
        "",
        _evaluation_section_title(heading_prefix, "Episode tapes (compact)"),
        "Prices and current_pv are fundamental-relative (~0). "
        "gap = pv-ask (buy) or bid-pv (sell); gap>0 is +EV vs the touch. "
        "Do not divide PV by spread.",
    ]
    for key in ("worst", "good"):
        tape = tapes.get(key)
        if not isinstance(tape, dict):
            continue
        lines.append(
            f"{tape.get('role', key)} on {tape.get('label', '?')}: "
            f"reward={_fmt_card_num(tape.get('reward'), 2)} "
            f"surplus={_fmt_card_num(tape.get('surplus'), 1)} "
            f"fill={_fmt_card_num(tape.get('fill'), 2)} "
            f"leftover={_fmt_card_num(tape.get('leftover'), 2)}"
        )
        shouts = tape.get("shouts") or []
        if not shouts:
            lines.append("  (no shouts logged)")
            continue
        for shout in shouts:
            parts = [
                f"t={shout.get('t')}",
                str(shout.get("side", "?")),
                f"shade={_fmt_card_num(shout.get('shade'), 3)}",
                f"spread={_fmt_card_num(shout.get('spread'), 3)}",
                f"inv={_fmt_card_num(shout.get('inv_frac'), 2)}",
                f"gap={_fmt_card_num(shout.get('gap'), 2)}",
                f"filled={1 if shout.get('filled') else 0}",
            ]
            lines.append("  " + " ".join(parts))
    return lines


def is_zi_q_reward_mode(q_reward_mode: str) -> bool:
    return q_reward_mode in ZI_Q_REWARD_MODES


def is_structured_zi_curriculum(critic_mode: str) -> bool:
    return str(critic_mode) == "structured_zi_curriculum"


def is_hypothesis_loop_critic(critic_mode: str) -> bool:
    return str(critic_mode) == "structured_hypothesis_loop"


def is_structured_principle_form(critic_mode: str) -> bool:
    return str(critic_mode) in (
        "structured_principle_form",
        "structured_hypothesis_loop",
    )


def _lob_features_used_hint(lob_input_mode: str = "raw") -> str:
    if is_lob_features_mode(lob_input_mode):
        names = ", ".join(active_lob_feature_names(lob_input_mode))
        return f"{names}, current_pv"
    if is_lob_book_orders_mode(lob_input_mode):
        return (
            "bids, asks, best_bid, best_ask, orders, accepted, side, price_rel, "
            "current_pv, fundamental_estimate"
        )
    if is_lob_hbl_parity_mode(lob_input_mode):
        return (
            "orders, accepted, side, price_rel, match_price_rel, L, "
            "memory_start_time, transaction_count, grace_period, "
            "best_bid_relative, best_ask_relative, current_pv, "
            "fundamental_estimate, pricing_fundamental"
        )
    return "orders, accepted, side, price_rel, L, current_pv"


LOB_SNAPSHOT_SCHEMA = [
    "Each LOB snapshot is a compact top-of-book view (prices relative to fundamental, ~0):",
    "- 'best_bid_relative': float | null",
    "- 'best_ask_relative': float | null",
    "- 'bids_relative': list[float] (top 4 bid price levels, highest first)",
    "- 'asks_relative': list[float] (top 4 ask price levels, lowest first)",
    "No order quantities are provided — price levels only.",
]


def _spread_input_schema(
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> List[str]:
    return spread_input_schema_lines(
        spread_input_mode,
        omit_fundamental=get_spread_omit_fundamental(),
    )

def _lob_input_schema(lob_input_mode: str = "raw") -> List[str]:
    return lob_input_schema_lines(lob_input_mode)






def _evaluation_section_title(prefix: str, title: str) -> str:
    prefix = prefix.strip()
    if prefix:
        return f"## {prefix} {title}"
    return f"## {title}"




def _format_zi_mean_shade_line(comparison: Optional[Dict[str, Any]], overall: Optional[Dict[str, Any]]) -> str:
    mean_shade = None
    if comparison and comparison.get("llm_mean_shade") is not None:
        mean_shade = float(comparison["llm_mean_shade"])
    elif overall and overall.get("avg_shade_mean") is not None:
        mean_shade = float(overall["avg_shade_mean"])
    if mean_shade is None:
        return ""
    return f"LLM mean effective shade (batch): {mean_shade:.4f}."


def format_zi_benchmark_test_feedback_lines(
    comparison: Dict[str, Any],
    *,
    heading_prefix: str = "",
    failed_only: bool = False,
) -> List[str]:
    tests = comparison.get("benchmark_tests")
    if not tests:
        return []

    total = len(tests)
    passed_count = len(comparison.get("passed_tests") or [])
    failed_tests = comparison.get("failed_tests") or [
        t for t in tests if t.get("status") == "FAIL"
    ]
    passed_tests = comparison.get("passed_tests") or [
        t for t in tests if t.get("status") == "PASS"
    ]

    title = _evaluation_section_title(
        heading_prefix,
        "ZI Benchmark Unit Tests (soft checks — pipeline continues on failure)",
    )
    lines = [
        "",
        title,
        (
            f"Passed {passed_count}/{total}. Each test replays identical episode seeds; "
            "only the tracked agent differs. Assertion: llm_mean_surplus > zi_mean_surplus."
        ),
        "Treat every FAIL like a failing unit test: diagnose from metrics and trajectory.",
    ]

    if failed_tests:
        lines.append("")
        lines.append("FAILED tests (largest surplus shortfall first):")
        for test in failed_tests:
            gap = float(test["surplus_gap"])
            shade_hi = float(test.get("shade_hi", 0.0))
            line = (
                f"  FAIL {test['test_id']} shade_range={test['shade_range']}: "
                f"llm={float(test['llm_mean_surplus']):.1f} zi={float(test['zi_mean_surplus']):.1f} "
                f"gap={gap:.1f} benchmark_shade_hi={shade_hi:g}"
            )
            if test.get("llm_mean_shade") is not None:
                line += f" llm_mean_shade={float(test['llm_mean_shade']):.4f}"
            lines.append(line)

    if not failed_only and passed_tests:
        lines.append("")
        lines.append("PASSED tests:")
        for test in passed_tests:
            gap = float(test["surplus_gap"])
            lines.append(
                f"  PASS {test['test_id']} shade_range={test['shade_range']}: "
                f"gap=+{gap:.1f}"
            )

    return lines








def format_zi_benchmark_comparison_section(
    comparison: Dict[str, Any],
    *,
    heading_prefix: str = "",
    q_reward_mode: str = "zi_beat_fraction",
) -> List[str]:
    if not comparison:
        return []
    beaten = comparison.get("beaten_shade_ranges") or []
    not_beaten = comparison.get("not_beaten_shade_ranges") or []
    shade_line = _format_zi_mean_shade_line(comparison, None)
    beat_fraction = float(comparison.get("beat_fraction", 0.0))
    if q_reward_mode == "zi_advantage":
        beat_reward_text = (
            f"Beat {int(comparison.get('beat_count', 0))}/"
            f"{int(comparison.get('total_benchmarks', 0))} ZI shade benchmarks "
            f"(beat fraction {beat_fraction:.4f} diagnostic only; Q reward is batch advantage)."
        )
    else:
        beat_reward_text = (
            f"Beat {int(comparison.get('beat_count', 0))}/"
            f"{int(comparison.get('total_benchmarks', 0))} ZI shade benchmarks "
            f"(Q reward = beat fraction {beat_fraction:.4f})."
        )
    lines = [
        "",
        _evaluation_section_title(
            heading_prefix,
            "ZI Benchmark Comparison (identical episode seeds; only tracked agent differs)",
        ),
        (
            f"LLM mean surplus: {float(comparison.get('llm_mean_surplus', 0.0)):.4f}. "
            f"{beat_reward_text}"
        ),
    ]
    if shade_line:
        lines.append(shade_line)
    test_lines = format_zi_benchmark_test_feedback_lines(
        comparison,
        heading_prefix=heading_prefix,
    )
    if test_lines:
        lines += test_lines
    else:
        lines += [
            "Beat these ZI shade ranges:",
            json.dumps(beaten, indent=2),
            "Did not beat these ZI shade ranges:",
            json.dumps(not_beaten, indent=2),
            "Per-benchmark details:",
            json.dumps(comparison.get("benchmarks", []), indent=2),
        ]
    return lines


def format_evaluation_trajectory_sections(
    *,
    batch_summary: Dict[str, Any],
    batch_data: Dict[str, Any],
    reward: Optional[float] = None,
    reward_variance: Optional[float] = None,
    heading_prefix: str = "Parent",
    q_reward_mode: str = "advantage",
    omit_zi_tables: bool = False,
) -> List[str]:
    sections: List[str] = []
    comparison = batch_summary.get("zi_benchmark_comparison") or batch_data.get(
        "zi_benchmark_comparison"
    )
    is_zi_mode = is_zi_q_reward_mode(q_reward_mode) or comparison is not None

    if reward is not None and not omit_zi_tables:
        if is_zi_mode and q_reward_mode == "zi_beat_fraction":
            reward_label = "ZI Beat Fraction (Q reward)"
        elif is_zi_mode:
            reward_label = "Batch Reward (advantage; ZI benchmarks diagnostic)"
        else:
            reward_label = "Batch Reward"
        sections += [
            "",
            (
                f"{_evaluation_section_title(heading_prefix, reward_label)}: "
                f"{reward:.4f} (variance: {float(reward_variance or 0.0):.4f})"
            ),
        ]
    if comparison and not omit_zi_tables:
        sections += format_zi_benchmark_comparison_section(
            comparison,
            heading_prefix=heading_prefix,
            q_reward_mode=q_reward_mode,
        )
    elif is_zi_mode and batch_summary.get("overall") and not omit_zi_tables:
        shade_line = _format_zi_mean_shade_line(None, batch_summary["overall"])
        if shade_line:
            sections += [
                "",
                _evaluation_section_title(heading_prefix, "Batch Shade"),
                shade_line,
            ]
    if batch_data:
        if batch_data.get("case_cards") is not None or batch_data.get("prior_hold_card"):
            sections += format_case_card_sections(
                batch_data,
                heading_prefix=heading_prefix,
            )
        elif batch_data.get("llm_interactions") or batch_data.get("error"):
            data_text = json.dumps(batch_data, indent=2, default=str)
            interaction_count = batch_data.get(
                "interaction_count",
                len(batch_data.get("llm_interactions", [])),
            )
            sections += [
                "",
                (
                    f"{_evaluation_section_title(heading_prefix, 'Median-Episode LLM Interactions')} "
                    f"(first {interaction_count} full state-action-next-state steps)"
                ),
                "Each interaction has pre_lob (state before the LLM acts), agent_action, "
                "post_lob (state after order submission and matching), and transacted.",
                data_text,
            ]
    return sections














def _shade_hi_from_range(shade_range: Any) -> Optional[float]:
    if not isinstance(shade_range, (list, tuple)) or not shade_range:
        return None
    if len(shade_range) > 1:
        return float(shade_range[1])
    return float(shade_range[0])


def _benchmark_pass_fail_lists(
    comparison: Optional[Dict[str, Any]],
) -> Tuple[List[float], List[float]]:
    beaten_his: List[float] = []
    failed_his: List[float] = []
    if not comparison:
        return beaten_his, failed_his
    tests = comparison.get("benchmark_tests") or comparison.get("benchmarks") or []
    for item in tests:
        shade_hi = item.get("shade_hi")
        if shade_hi is None:
            shade_hi = _shade_hi_from_range(item.get("shade_range"))
        if shade_hi is None:
            continue
        beat = bool(item.get("beat")) if "beat" in item else item.get("status") == "PASS"
        if beat:
            beaten_his.append(float(shade_hi))
        else:
            failed_his.append(float(shade_hi))
    return beaten_his, failed_his


def _shade_level_from_surplus_and_realized(
    *,
    has_failures: bool,
    all_beaten: bool,
    trade_shade: Optional[float],
    x_cap: Optional[float],
) -> Tuple[str, str, str, Optional[float]]:
    trade = float(trade_shade) if trade_shade is not None else None
    cap = float(x_cap) if x_cap is not None else None
    collapsed = (
        trade is not None
        and cap is not None
        and cap >= 0.05
        and trade <= 0.02
        and trade < 0.25 * cap
    )
    low_realized = trade is not None and trade <= 0.05

    if has_failures:
        if collapsed or low_realized or trade is None:
            direction = "shade_more"
            error_type = "level_too_low"
            fix_class = "add_offset"
            if trade is not None:
                target: Optional[float] = min(
                    1.0, max(trade * 4.0, trade + 0.05, 0.05)
                )
            elif cap is not None:
                target = min(1.0, max(cap * 2.0, cap + 0.05))
            else:
                target = 0.1
        else:
            direction = "reshape_pv_map"
            error_type = "pv_mapping_wrong"
            fix_class = "change_pv_feature"
            target = cap if cap is not None else trade
        return direction, error_type, fix_class, target

    if all_beaten:
        if trade is not None and trade >= 0.15:
            return (
                "shade_less",
                "level_too_high",
                "scale_base",
                max(0.0, trade * 0.5),
            )
        return ("reshape_pv_map", "pv_mapping_wrong", "change_pv_feature", trade)

    return ("reshape_pv_map", "pv_mapping_wrong", "change_pv_feature", None)


def infer_curriculum_shade_diagnosis(
    comparison: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    beaten_his, failed_his = _benchmark_pass_fail_lists(comparison)
    llm_x_cap = None
    llm_trade_shade = None
    if comparison:
        llm_x_cap = comparison.get("llm_mean_requested_x_cap")
        if llm_x_cap is None and isinstance(comparison.get("overall"), dict):
            llm_x_cap = comparison["overall"].get("avg_requested_x_cap_mean")
        llm_trade_shade = comparison.get("llm_mean_shade")
        if llm_trade_shade is None and isinstance(comparison.get("overall"), dict):
            llm_trade_shade = comparison["overall"].get("avg_shade_mean")

    has_failures = bool(failed_his)
    all_beaten = bool(beaten_his) and not failed_his
    direction, error_type, fix_class, suggested_target = (
        _shade_level_from_surplus_and_realized(
            has_failures=has_failures,
            all_beaten=all_beaten,
            trade_shade=float(llm_trade_shade) if llm_trade_shade is not None else None,
            x_cap=float(llm_x_cap) if llm_x_cap is not None else None,
        )
    )

    return {
        "requested_x_cap": float(llm_x_cap) if llm_x_cap is not None else None,
        "llm_trade_shade": float(llm_trade_shade) if llm_trade_shade is not None else None,
        "highest_beaten_shade_hi": max(beaten_his) if beaten_his else None,
        "lowest_failed_shade_hi": min(failed_his) if failed_his else None,
        "shade_error_direction": direction,
        "error_type": error_type,
        "fix_class": fix_class,
        "suggested_target": suggested_target,
    }








_CURRICULUM_DIRECTIONS = frozenset({"shade_more", "shade_less", "reshape_pv_map"})
_CURRICULUM_ERROR_TYPES = frozenset(
    {"level_too_low", "level_too_high", "pv_mapping_wrong"}
)
_CURRICULUM_FIX_CLASSES = frozenset(
    {"scale_base", "add_offset", "change_pv_feature", "change_formula"}
)




_PRINCIPLE_FORM_PRINCIPLES = frozenset(
    {
        "pv_rank_or_distance_to_best",
        "pv_vs_fundamental",
        "fill_surplus_tradeoff",
        "current_pv_vs_fundamental",
        "current_pv_in_schedule",
    }
)
_PRINCIPLE_FORM_CLASSES = frozenset(
    {
        "ratio",
        "rank",
        "signed_power",
        "piecewise_by_sign",
        "unit_schedule_blend",
    }
)

_PRINCIPLE_FORM_PRINCIPLES_LOB = frozenset(
    {
        "fill_vs_reject",
        "side_pressure",
        "order_flow_imbalance",
        "recent_aggressiveness",
    }
)
_PRINCIPLE_FORM_CLASSES_LOB = frozenset(
    {
        "ratio",
        "threshold",
        "piecewise_by_sign",
        "ema_over_window",
        "signed_power",
    }
)

_PRINCIPLE_FORM_PRINCIPLES_LOB_BOOK_ORDERS = frozenset(
    {
        "fill_vs_reject",
        "order_flow_imbalance",
        "recent_aggressiveness",
        "book_imbalance",
        "depth_asymmetry",
    }
)
_PRINCIPLE_FORM_CLASSES_LOB_BOOK_ORDERS = _PRINCIPLE_FORM_CLASSES_LOB

_LOB_SHADE_SANITY_CHECK = (
    "Before returning code, sanity-check the shade expression: if you compute a "
    "market signal (imbalance, pressure, fill rate, etc.), confirm it can take "
    "nontrivial values under normal books — not algebraically constant 0/1; "
    "confirm the returned shade depends on that signal so changing bids, asks, "
    "or orders would change x (not only a fixed constant or +ε); prefer a simple "
    "correct formula over a long pipeline whose intermediates cancel or never "
    "enter the return."
)

_SPREAD_SHADE_SANITY_CHECK = (
    "Before returning code, sanity-check the shade expression: if you compute a "
    "market signal from the provided inputs, confirm it can take nontrivial "
    "values under normal books — not algebraically constant 0/1; confirm the "
    "returned shade depends on that signal so changing those inputs would "
    "change x (not only a fixed constant or +ε); prefer a simple correct "
    "formula over a long pipeline whose intermediates cancel or never enter "
    "the return. "
    "If best_bid or best_ask is None, that quote is absent — do not treat None "
    "as 0.0 in arithmetic."
)

_PRINCIPLE_FORM_PRINCIPLES_LOB_SNAPSHOT = frozenset(
    {
        "prior_spread_or_mid",
        "book_imbalance",
        "depth_asymmetry",
        "trade_momentum",
        "current_pv_vs_mid",
    }
)
_PRINCIPLE_FORM_CLASSES_LOB_SNAPSHOT = frozenset(
    {
        "ratio",
        "threshold",
        "piecewise_by_sign",
        "ema_over_snapshots",
        "signed_power",
    }
)

_PRINCIPLE_FORM_PRINCIPLES_SPREAD = frozenset(
    {
        "bid_ask_spread_width",
        "inventory_remaining_fraction",
        "pv_vs_mid",
        "pv_vs_touch",
        "current_pv_magnitude",
        "recent_accept_reject",
        "shout_competitiveness",
    }
)
_PRINCIPLE_FORM_PRINCIPLES_SPREAD_BOOK = frozenset(
    {
        "mid_vs_rhat",
        "pv_vs_touch",
        "inventory_remaining_fraction",
        "own_prev_feedback",
        "quote_liquidity",
    }
)
_PRINCIPLE_FORM_PRINCIPLES_SPREAD_BOOK_ECONMENUS = frozenset(
    {
        "bid_ask_spread_width",
        "inventory_remaining_fraction",
        "book_imbalance",
        "pv_vs_touch",
        "last_own_accept_reject",
    }
)
_PRINCIPLE_FORM_CLASSES_SPREAD = frozenset(
    {
        "ratio",
        "threshold",
        "piecewise_by_sign",
        "clamp_linear",
        "inventory_weighted",
        "ema_over_window",
    }
)
_PRINCIPLE_FORM_CLASSES_SPREAD_BOOK = frozenset(
    {
        "ratio",
        "threshold",
        "piecewise_by_sign",
        "inventory_weighted",
        "target_soft_update",
    }
)
_PRINCIPLE_FORM_CLASSES_SPREAD_BOOK_ECONMENUS = frozenset(
    {
        "ratio",
        "threshold",
        "piecewise_by_sign",
        "inventory_weighted",
        "ema_over_window",
    }
)

_PRINCIPLE_MENU_PRESETS: Dict[str, Dict[str, Any]] = {
    "lob_e2": {
        "lob": frozenset(
            {"fill_vs_reject", "side_pressure", "order_flow_imbalance"}
        ),
    },
    "lob_snapshot": {
        "lob": {
            "principles": _PRINCIPLE_FORM_PRINCIPLES_LOB_SNAPSHOT,
            "form_classes": _PRINCIPLE_FORM_CLASSES_LOB_SNAPSHOT,
        },
    },
    "spread_e5": {
        "spread": frozenset(
            {"bid_ask_spread_width", "inventory_remaining_fraction", "pv_vs_mid"}
        ),
    },
    "spread_book_pvtwin": {
        "spread": {
            "principles": _PRINCIPLE_FORM_PRINCIPLES,
            "form_classes": _PRINCIPLE_FORM_CLASSES,
        },
    },
    "spread_book_econmenus": {
        "spread": {
            "principles": _PRINCIPLE_FORM_PRINCIPLES_SPREAD_BOOK_ECONMENUS,
            "form_classes": _PRINCIPLE_FORM_CLASSES_SPREAD_BOOK_ECONMENUS,
        },
    },
    "min2": {
        "pv": {
            "principles": frozenset(
                {"current_pv_vs_fundamental", "pv_rank_or_distance_to_best"}
            ),
            "form_classes": frozenset({"ratio", "unit_schedule_blend"}),
        },
        "lob": {
            "principles": frozenset({"fill_vs_reject", "side_pressure"}),
            "form_classes": frozenset({"ratio", "piecewise_by_sign"}),
        },
        "spread": {
            "principles": frozenset(
                {"inventory_remaining_fraction", "bid_ask_spread_width"}
            ),
            "form_classes": frozenset({"ratio", "inventory_weighted"}),
        },
    },
}

_ACTIVE_PRINCIPLE_MENU_PRESET: Optional[str] = None


def list_principle_menu_presets() -> List[str]:
    return sorted(_PRINCIPLE_MENU_PRESETS.keys())


def set_principle_menu_preset(preset: Optional[str]) -> None:
    global _ACTIVE_PRINCIPLE_MENU_PRESET
    if preset is not None and preset not in _PRINCIPLE_MENU_PRESETS:
        known = ", ".join(list_principle_menu_presets())
        raise ValueError(f"Unknown principle_menu_preset '{preset}'. Known: {known}")
    _ACTIVE_PRINCIPLE_MENU_PRESET = preset




def _apply_menu_preset_override(
    override: Any,
    principles: frozenset,
    form_classes: frozenset,
) -> Tuple[frozenset, frozenset]:
    if isinstance(override, frozenset):
        return override, form_classes
    if isinstance(override, dict):
        return (
            override.get("principles", principles),
            override.get("form_classes", form_classes),
        )
    return principles, form_classes


def principle_form_menus(
    strategy_type: str = "pv",
    *,
    preset: Optional[str] = None,
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> Tuple[frozenset, frozenset]:
    if strategy_type == "lob":
        if is_lob_features_mode(lob_input_mode):
            principles, form_classes = (
                _PRINCIPLE_FORM_PRINCIPLES_LOB_SNAPSHOT,
                _PRINCIPLE_FORM_CLASSES_LOB_SNAPSHOT,
            )
        elif is_lob_book_orders_mode(lob_input_mode):
            principles, form_classes = (
                _PRINCIPLE_FORM_PRINCIPLES_LOB_BOOK_ORDERS,
                _PRINCIPLE_FORM_CLASSES_LOB_BOOK_ORDERS,
            )
        else:
            principles, form_classes = (
                _PRINCIPLE_FORM_PRINCIPLES_LOB,
                _PRINCIPLE_FORM_CLASSES_LOB,
            )
    elif strategy_type == "spread":
        if is_spread_book_mode(spread_input_mode):
            principles = _PRINCIPLE_FORM_PRINCIPLES_SPREAD_BOOK
            form_classes = _PRINCIPLE_FORM_CLASSES_SPREAD_BOOK
        else:
            principles = _PRINCIPLE_FORM_PRINCIPLES_SPREAD
            form_classes = _PRINCIPLE_FORM_CLASSES_SPREAD
    else:
        principles, form_classes = (
            _PRINCIPLE_FORM_PRINCIPLES,
            _PRINCIPLE_FORM_CLASSES,
        )

    active_preset = preset if preset is not None else _ACTIVE_PRINCIPLE_MENU_PRESET
    if active_preset:
        override = _PRINCIPLE_MENU_PRESETS.get(active_preset, {}).get(strategy_type)
        if override is not None:
            principles, form_classes = _apply_menu_preset_override(
                override, principles, form_classes
            )
    return principles, form_classes


def _reflection_search_span_for_next_hypothesis(text: str) -> str:
    raw = text or ""
    next_m = re.search(
        r"(?is)\bNEXT_HYPOTHESIS\b\s*:?(.*?)(?=\n\s*(?:MECHANISM|FIX)\b|\Z)",
        raw,
    )
    if next_m:
        return next_m.group(1)
    diag_m = re.search(
        r"(?is)\bDIAGNOSIS\b\s*:?(.*?)(?=\n\s*(?:MECHANISM|FIX)\b|\Z)",
        raw,
    )
    if diag_m:
        return diag_m.group(1)
    return raw


def parse_principle_form_fields(text: str) -> Dict[str, Optional[str]]:
    search = _reflection_search_span_for_next_hypothesis(text or "")
    principle_match = re.search(r"principle:\s*(\S+)", search, flags=re.IGNORECASE)
    form_match = re.search(r"form_class:\s*(\S+)", search, flags=re.IGNORECASE)
    direction_match = re.search(
        r"shade_error_direction:\s*(\S+)", search, flags=re.IGNORECASE
    )
    return {
        "principle": principle_match.group(1).strip() if principle_match else None,
        "form_class": form_match.group(1).strip() if form_match else None,
        "shade_error_direction": (
            direction_match.group(1).strip() if direction_match else None
        ),
    }


def parse_same_failure_as_last_hold(text: str) -> Optional[str]:
    match = re.search(
        r"same_failure_as_last_hold:\s*(yes|no)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).strip().lower()


def parse_diagnosis_action(text: str) -> Optional[str]:
    match = re.search(
        r"(?m)^\s*action:\s*(refine|replace)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).strip().lower()


def diagnosis_action_generator_instructions(
    *,
    action: str,
    parent_principle: Optional[str] = None,
    parent_form: Optional[str] = None,
) -> List[str]:
    pair = ""
    if parent_principle and parent_form:
        pair = f" ({parent_principle} + {parent_form})"
    if action == "refine":
        return [
            "Parent DIAGNOSIS action=refine: KEEP that principle and form_class"
            f"{pair}. Change the shade map under that pair so probe shades differ "
            "from the parent (not comments/renames/whitespace-only). Do not switch "
            "to a new menu pair.",
        ]
    if action == "replace":
        return [
            "Parent DIAGNOSIS action=replace: implement a NEW (principle, form_class), "
            f"not{pair or ' the parent pair'}. Change how shade is computed vs the "
            "parent (not comment-only).",
        ]
    return []




_TEMPLATE_WHY_RE = re.compile(
    r"primary:\s*low batch advantage\s*/\s*weak fills",
    flags=re.IGNORECASE,
)
_CASE_CARD_WHY_TOKENS = (
    "leftover",
    "pos_gap_miss",
    "shade_fill",
    "shade_miss",
    "buy_sur",
    "sell_sur",
    "buy surplus",
    "sell surplus",
)


def _why_this_fails_body(text: str) -> str:
    match = re.search(
        r"(?:why_this_fails|why_parent_fails):\s*(.+?)(?=\n\s*\w+:|\n[A-Z]|\Z)",
        text or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _why_cites_case_card(why: str) -> bool:
    lowered = (why or "").lower()
    return any(token in lowered for token in _CASE_CARD_WHY_TOKENS)


def _diagnosis_field_present(text: str, *labels: str) -> bool:
    lowered = (text or "").lower()
    return any(f"{label}:" in lowered for label in labels)


def _hypothesis_pair_from_reflection(reflection: str) -> Optional[Tuple[str, str]]:
    fields = parse_principle_form_fields(reflection or "")
    principle = fields.get("principle")
    form_class = fields.get("form_class")
    if not principle or not form_class:
        return None
    return (principle.strip().lower(), form_class.strip().lower())


def menu_hypothesis_pairs(
    strategy_type: str = "pv",
    *,
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> frozenset[Tuple[str, str]]:
    principles, form_classes = principle_form_menus(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )
    return frozenset(
        (principle, form_class)
        for principle in sorted(principles)
        for form_class in sorted(form_classes)
    )


def collect_hypothesis_pairs_from_reflections(
    reflections: Iterable[str],
) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for reflection in reflections:
        pair = _hypothesis_pair_from_reflection(reflection or "")
        if pair is not None:
            pairs.add(pair)
    return pairs


def compute_menu_coverage_stats(
    *,
    strategy_type: str = "pv",
    reflections: Iterable[str],
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> Dict[str, Any]:
    reflection_list = [str(r) for r in reflections if (r or "").strip()]
    menu_pairs = menu_hypothesis_pairs(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )
    principles, form_classes = principle_form_menus(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )

    pair_counts: Counter[Tuple[str, str]] = Counter()
    principle_counts: Counter[str] = Counter()
    for reflection in reflection_list:
        pair = _hypothesis_pair_from_reflection(reflection)
        if pair is None:
            continue
        pair_counts[pair] += 1
        principle_counts[pair[0]] += 1

    explored_pairs = set(pair_counts.keys())
    unexplored = sorted(menu_pairs - explored_pairs)
    menu_size = len(menu_pairs)
    explored_count = len(explored_pairs)
    parseable = sum(pair_counts.values())

    entropy = 0.0
    if parseable > 0:
        for count in principle_counts.values():
            prob = count / parseable
            entropy -= prob * math.log(prob)
    max_entropy = math.log(len(principles)) if len(principles) > 1 else 0.0

    return {
        "strategy_type": strategy_type,
        "menu_size": menu_size,
        "principle_count": len(principles),
        "form_class_count": len(form_classes),
        "reflection_count": len(reflection_list),
        "parseable_reflection_count": parseable,
        "explored_pair_count": explored_count,
        "coverage_fraction": explored_count / menu_size if menu_size else 0.0,
        "exploration_complete": explored_count >= menu_size if menu_size else False,
        "unexplored_pairs": [
            {"principle": principle, "form_class": form_class}
            for principle, form_class in unexplored
        ],
        "explored_pairs": [
            {
                "principle": principle,
                "form_class": form_class,
                "count": int(pair_counts[(principle, form_class)]),
            }
            for principle, form_class in sorted(explored_pairs)
        ],
        "principle_counts": dict(sorted(principle_counts.items())),
        "principle_entropy": entropy,
        "principle_entropy_normalized": (
            entropy / max_entropy if max_entropy > 0 else 0.0
        ),
        "missing_principles": sorted(set(principles) - set(principle_counts.keys())),
    }


def format_menu_exploration_section(
    *,
    strategy_type: str = "pv",
    run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    require_last_hold_comparison: bool = False,
    compact: bool = False,
    force_menu_replace: bool = False,
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> List[str]:
    menu_pairs = menu_hypothesis_pairs(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )
    explored = set(run_explored_pairs or ())
    unexplored = sorted(menu_pairs - explored)
    principles, _ = principle_form_menus(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )
    explored_principles = sorted({principle for principle, _ in explored})
    missing_principles = sorted(set(principles) - set(explored_principles))

    lines = [
        "## Menu exploration",
        f"Menu size: {len(menu_pairs)} (principle, form_class) pairs.",
        f"Explored so far in this run: {len(explored)}/{len(menu_pairs)}.",
    ]
    if explored:
        preview = sorted(explored)[:10]
        extra = len(explored) - len(preview)
        lines.append(
            "Already tried: "
            + ", ".join(f"{principle}+{form_class}" for principle, form_class in preview)
            + (f" (+{extra} more)" if extra > 0 else "")
        )
    if missing_principles:
        lines.append(
            "Principles not yet used as primary hypothesis: "
            + ", ".join(missing_principles)
        )

    if unexplored:
        lines.append("")
        if force_menu_replace:
            lines += [
                "FORCE MENU REPLACE (incomplete menu): if the parent "
                "(principle, form_class) was already tried, action MUST be replace "
                "to an unexplored pair — refine is forbidden until every pair has "
                "been tried (exception: same_failure_as_last_hold=yes).",
                "Unexplored pairs (pick one; do not repeat parent, sibling, or a "
                "pair already tried):",
            ]
        else:
            lines += [
                "If action=refine: KEEP the parent (principle, form_class) and "
                "prefer a LOCAL within-family edit of the shade map "
                "(thresholds/coefficients/dampers OK) when staying on the same "
                "hypothesis is clearly justified.",
                "If the parent pair is already explored and surplus is not improving, "
                "prefer action=replace to an unexplored pair (keep searching the menu).",
                "If action=replace and unexplored pairs remain, pick from this list "
                "(do not repeat parent, sibling, or a pair already tried):",
            ]
        for principle, form_class in unexplored:
            lines.append(f"- {principle} + {form_class}")
    else:
        lines += [
            "",
            "EXPLORATION COMPLETE — every menu pair has been tried at least once. "
            "Keep searching: if refining the parent pair is not clearly improving "
            "surplus, prefer action=replace to a different principle (any form_class) "
            "rather than refining one pair forever. action=refine still keeps the "
            "parent pair; action=replace must differ from the parent.",
        ]
    if require_last_hold_comparison:
        lines += [
            "",
            "LAST HOLD: same_failure_as_last_hold=yes requires action=refine; "
            "no requires action=replace.",
        ]
    return lines

def _slim_plateau_performance(performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    perf = performance or {}
    slim: Dict[str, Any] = {}
    for key in ("reward", "reward_variance", "per_config_rewards"):
        if key in perf:
            slim[key] = perf[key]
    overall = perf.get("overall")
    if isinstance(overall, dict):
        slim["overall"] = {
            k: overall.get(k)
            for k in (
                "llm_surplus_mean",
                "llm_mean_x_cap",
                "llm_trade_shade",
            )
            if k in overall
        }
    comparison = perf.get("zi_benchmark_comparison")
    if isinstance(comparison, dict):
        slim["zi"] = {
            k: comparison.get(k)
            for k in (
                "beat_count",
                "total_benchmarks",
                "beat_fraction",
                "llm_mean_surplus",
                "llm_mean_x_cap",
                "llm_trade_shade",
            )
            if k in comparison
        }
    return slim


def _slim_plateau_batch_summary(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    slim: Dict[str, Any] = {}
    for key in ("reward", "reward_variance", "per_config_rewards"):
        if key in summary:
            slim[key] = summary[key]
    comparison = summary.get("zi_benchmark_comparison")
    if isinstance(comparison, dict):
        slim["zi"] = {
            k: comparison.get(k)
            for k in (
                "beat_count",
                "total_benchmarks",
                "beat_fraction",
                "llm_mean_surplus",
                "llm_mean_x_cap",
                "llm_trade_shade",
            )
            if k in comparison
        }
    return slim


def build_plateau_revision_prompt(
    *,
    parent_code: str,
    parent_reflection: str,
    parent_performance: Optional[Dict[str, Any]],
    non_improve_streak: int,
    critic_mode: str = "default",
    generator_batch_summary: Optional[Dict[str, Any]] = None,
    strategy_type: str = "pv",
    run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    max_code_chars: int = 2500,
    max_reflection_chars: Optional[int] = None,
) -> str:
    fields = parse_principle_form_fields(parent_reflection or "")
    stalled_principle = fields.get("principle") or "unknown"
    stalled_form = fields.get("form_class") or "unknown"
    code = (parent_code or "").strip()
    if len(code) > max_code_chars:
        code = code[: max_code_chars - 20].rstrip() + "\n# ... truncated ..."
    reflection = (parent_reflection or "").strip() or "(none)"
    if max_reflection_chars is not None and len(reflection) > max_reflection_chars:
        reflection = reflection[: max_reflection_chars - 20].rstrip() + "\n... truncated ..."
    sections: List[str] = [
        "## Plateau / stalled hypothesis",
        (
            f"Local refinements of this parent failed to improve mean surplus for "
            f"{int(non_improve_streak)} consecutive children."
        ),
        "Treat the current principle+form as a hypothesis that has exhausted useful "
        "local edits. Revise the scientific hypothesis — do not retune coefficients.",
        "",
        f"Stalled principle: {stalled_principle}",
        f"Stalled form_class: {stalled_form}",
        "",
        "## Parent Strategy",
        "```python",
        code,
        "```",
        "",
        "## Prior Parent Reflection",
        reflection,
        "",
        "## Parent Performance (compact)",
        json.dumps(_slim_plateau_performance(parent_performance), indent=2),
    ]
    slim_summary = _slim_plateau_batch_summary(generator_batch_summary)
    if slim_summary:
        sections += [
            "",
            "## Parent batch summary (compact)",
            json.dumps(slim_summary, indent=2),
        ]

    if is_structured_principle_form(critic_mode):
        sections += [
            "",
            *format_menu_exploration_section(
                strategy_type=strategy_type,
                run_explored_pairs=run_explored_pairs,
                compact=False,
                lob_input_mode=lob_input_mode,
                spread_input_mode=spread_input_mode,
            ),
            "",
            *format_structured_principle_form_diagnosis_template(strategy_type=strategy_type),
            "",
            "## Plateau revision constraints",
            f"- Change at least one of stalled form_class ({stalled_form}) or "
            f"principle ({stalled_principle}); you may keep the other.",
            "- Prefer an unused menu pair from the coverage list above when any remain.",
            "- why_parent_fails must explain why local edits under the stalled pair failed.",
            "- diversity_vs_parent must name how the revised pair (or kept-side change) "
            "differs under probe shades.",
            "- FIX must implement the revised principle + form_class (no full code).",
            "- Set action: replace (plateau escape is not a refine).",
        ]
    return "\n".join(sections)






def hypothesis_jump_generator_instructions(
    *,
    parent_principle: Optional[str] = None,
    parent_form: Optional[str] = None,
) -> List[str]:
    principle = parent_principle or "unknown"
    form = parent_form or "unknown"
    return [
        "HYPOTHESIS EXPLORATION JUMP: implement a new scientific hypothesis from this parent.",
        f"Parent hypothesis was principle={principle}, form_class={form}.",
        "Change at least one of principle or form_class per the Parent Reflection DIAGNOSIS.",
        "Follow the Parent Reflection DIAGNOSIS — new structure/features/control flow, "
        "not only constants.",
    ]


def plateau_escape_generator_instructions(
    *,
    stalled_principle: Optional[str] = None,
    stalled_form: Optional[str] = None,
) -> List[str]:
    principle = stalled_principle or "unknown"
    form = stalled_form or "unknown"
    return [
        "HYPOTHESIS REVISION (plateau escape): local edits of the parent stalled.",
        f"Change at least one of stalled principle={principle} or "
        f"form_class={form} (you may keep the other).",
        "Follow the Parent Reflection DIAGNOSIS/Fix from the plateau revision.",
        "Change how shade is computed enough that probe shades differ from the "
        "parent — new driving fields and/or control flow are preferred.",
        "Forbidden: multiplying the parent shade by a constant, renaming variables, "
        "or comment-only edits.",
        "The child must look like a different strategy under probe shades, not a "
        "near-clone of the parent.",
    ]


def _code_shape_tags(code: str) -> List[str]:
    text = (code or "").lower()
    tags: List[str] = []
    if "sum(private_values)" in text or "sum(pv" in text:
        tags.append("uses_sum")
    if "len(private_values)" in text or "len(pv" in text:
        tags.append("uses_len")
    if "spread_relative" in text or "mid_relative" in text:
        tags.append("uses_spread_mid")
    if "lob_snapshots" in text:
        tags.append("uses_snapshots")
    if "bids_relative" in text or "asks_relative" in text:
        tags.append("uses_depth")
    if "recent_trade_prices" in text or "last_trade_price" in text:
        tags.append("uses_trades")
    if "**" in text:
        tags.append("uses_pow")
    if "math.sqrt" in text or "**0.5" in text:
        tags.append("uses_sqrt")
    if text.count("if ") >= 2:
        tags.append("piecewise")
    if "max(" in text:
        tags.append("uses_max")
    if "min(" in text:
        tags.append("uses_min")
    if "abs(" in text:
        tags.append("uses_abs")
    return sorted(tags)


def derive_family_id(
    *,
    reflection: str = "",
    code: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    meta = metadata or {}
    if meta.get("family_id"):
        return str(meta["family_id"])
    fields = parse_principle_form_fields(reflection or "")
    form = str(meta.get("form_class") or fields.get("form_class") or "unknown")
    principle = str(meta.get("principle") or fields.get("principle") or "unknown")
    tags = _code_shape_tags(code)
    tag_suffix = "+".join(tags) if tags else "plain"
    return f"{principle}:{form}:{tag_suffix}"






def validate_structured_principle_form_reflection(
    text: str,
    *,
    strategy_type: str = "pv",
    parent_reflection: Optional[str] = None,
    sibling_reflections: Optional[List[str]] = None,
    run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    require_last_hold_comparison: bool = False,
    force_menu_replace: bool = False,
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> Optional[str]:
    if not text or not text.strip():
        return "Empty reflection"
    lowered = text.lower()
    required_markers = [
        "diagnosis:",
        "principle:",
        "form_class:",
        "shade_error_direction:",
        "features_used:",
        "mechanism:",
        "fix:",
        "action:",
    ]
    missing = [marker for marker in required_markers if marker not in lowered]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"

    if not _diagnosis_field_present(text, "current_form", "parent_form"):
        return "Missing current_form (describe this strategy's functional form)"
    if not _diagnosis_field_present(text, "why_this_fails", "why_parent_fails"):
        return "Missing why_this_fails (tie to batch advantage first)"
    if not _diagnosis_field_present(text, "proposed_change", "diversity_vs_parent"):
        return "Missing proposed_change (how the new hypothesis differs)"

    direction_match = re.search(
        r"shade_error_direction:\s*(\S+)", text, flags=re.IGNORECASE
    )
    if not direction_match or direction_match.group(1) not in _CURRICULUM_DIRECTIONS:
        return "shade_error_direction must be shade_more, shade_less, or reshape_pv_map"

    action = parse_diagnosis_action(text)
    if action is None:
        return "Missing action: refine or replace"

    why_body = _why_this_fails_body(text)
    if _TEMPLATE_WHY_RE.search(why_body) and not _why_cites_case_card(why_body):
        return (
            "why_this_fails is the template sentence. Cite leftover, shade_fill, "
            "or pos_gap_miss from the case card — do not paste PRIMARY/SECONDARY."
        )

    same_fail = parse_same_failure_as_last_hold(text)
    if require_last_hold_comparison and same_fail is None:
        return (
            "Missing same_failure_as_last_hold: yes or no "
            "(required when a LAST HOLD card is present)"
        )
    if require_last_hold_comparison and same_fail == "yes" and action != "refine":
        return "same_failure_as_last_hold=yes requires action: refine"
    if require_last_hold_comparison and same_fail == "no" and action != "replace":
        return "same_failure_as_last_hold=no requires action: replace"

    principles, form_classes = principle_form_menus(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )
    principle_match = re.search(r"principle:\s*(\S+)", text, flags=re.IGNORECASE)
    if not principle_match or principle_match.group(1) not in principles:
        return "principle must be one of: " + ", ".join(sorted(principles))

    form_match = re.search(r"form_class:\s*(\S+)", text, flags=re.IGNORECASE)
    if not form_match or form_match.group(1) not in form_classes:
        return "form_class must be one of: " + ", ".join(sorted(form_classes))

    proposed = _hypothesis_pair_from_reflection(text)
    if proposed is not None:
        parent_pair = _hypothesis_pair_from_reflection(parent_reflection or "")
        menu_pairs = menu_hypothesis_pairs(
            strategy_type,
            lob_input_mode=lob_input_mode,
            spread_input_mode=spread_input_mode,
        )
        explored = set(run_explored_pairs or ())
        unexplored = menu_pairs - explored

        if (
            force_menu_replace
            and action == "refine"
            and same_fail != "yes"
            and parent_pair is not None
            and unexplored
            and parent_pair in explored
        ):
            return (
                "force_menu_replace: parent (principle, form_class) was already "
                "explored and unexplored pairs remain — use action: replace and "
                "pick an unexplored pair"
            )

        refine_parent = action == "refine" and parent_pair is not None
        if refine_parent:
            if proposed != parent_pair:
                return (
                    "action=refine requires keeping the parent "
                    "(principle, form_class) and re-implementing under that pair — "
                    "do not pick a new pair."
                )
        else:
            if unexplored and proposed in explored:
                return (
                    "action=replace during menu exploration: pick an unexplored "
                    "(principle, form_class). "
                    f"Already explored {len(explored)}/{len(menu_pairs)} pairs — "
                    f"({proposed[0]}, {proposed[1]}) was already tried in this run."
                )
            if parent_pair is not None and proposed == parent_pair:
                return (
                    "action=replace requires a new (principle, form_class) vs the parent "
                    "— use action: refine to keep this hypothesis."
                )
            for idx, sibling_reflection in enumerate(sibling_reflections or []):
                sibling_pair = _hypothesis_pair_from_reflection(sibling_reflection or "")
                if sibling_pair is not None and proposed == sibling_pair:
                    return (
                        "Proposed (principle, form_class) duplicates sibling hypothesis "
                        f"#{idx + 1} — pick a different principle or form_class."
                    )

    if "environment:" in lowered:
        return "Do not include an Environment section; use the DIAGNOSIS block only"
    if re.search(r"(?m)^\s*performance\s*:", text, flags=re.IGNORECASE):
        return "Do not include a Performance section; use the DIAGNOSIS block only"

    if strategy_type == "lob":
        if re.search(r"private_values\s*=", text, flags=re.IGNORECASE):
            return "Do not invent a full private_values schedule in LOB-only mode"
        from marketsim.LLM.reflection_pipeline.diagnosis_agreement import (
            validate_features_used_format,
        )

        features_err = validate_features_used_format(text, strategy_type=strategy_type)
        if features_err:
            return features_err
    elif strategy_type == "spread":
        if re.search(r"lob_snapshots|market_history\s*[\[.(]", text, flags=re.IGNORECASE):
            return "Do not invent LOB depth or market_history fields in spread-only mode"
    else:
        if re.search(r"pv_mean\s*>=\s*0\.[0-9]", text, flags=re.IGNORECASE):
            return "Do not propose pv_mean >= 0.x thresholds (private_values are not [0,1])"

    return None


def format_zi_surplus_gap_facts(
    comparison: Optional[Dict[str, Any]],
    *,
    top_n: int = 5,
) -> List[str]:
    if not comparison:
        return []
    llm = float(comparison.get("llm_mean_surplus", 0.0))
    results = comparison.get("benchmark_results") or comparison.get("benchmarks") or []
    if not results:
        return []

    gaps: List[Tuple[float, Dict[str, Any]]] = []
    for bench in results:
        zi_surplus = float(bench.get("mean_surplus", 0.0))
        gap = zi_surplus - llm
        if gap > 0:
            gaps.append((gap, bench))
    gaps.sort(key=lambda item: item[0], reverse=True)

    lines = [
        "## ZI Surplus Gaps (secondary — explain advantage failures)",
        "Primary objective is batch advantage. Use these gaps in why_this_fails "
        "and to justify shade_error_direction — do NOT hardcode shade_hi floats.",
        f"LLM mean surplus: {llm:.2f}.",
    ]
    if not gaps:
        lines.append(
            "No positive ZI surplus gaps — LLM meets or beats all benchmark surplus."
        )
        return lines
    for gap, bench in gaps[:top_n]:
        shade = bench.get("shade_range", "?")
        zi_surplus = float(bench.get("mean_surplus", 0.0))
        lines.append(
            f"- shade {shade}: ZI surplus {zi_surplus:.2f}, gap +{gap:.2f} (ZI ahead)."
        )
    return lines


def format_shade_level_hypothesis_facts(
    comparison: Optional[Dict[str, Any]],
) -> List[str]:
    lines = [
        "## Shade-level hypothesis (from ladder / menu probe params)",
        "Your DIAGNOSIS must include shade_error_direction as a level hypothesis "
        "(shade_more / shade_less / reshape_pv_map). Pick principle + form_class "
        "from the menus as the PARAMETERS that implement that hypothesis.",
    ]
    if not comparison:
        lines.append(
            "No ZI comparison — default shade_error_direction to reshape_pv_map "
            "unless case cards clearly show a global level error."
        )
        return lines

    diag = infer_curriculum_shade_diagnosis(comparison)
    beaten_his, failed_his = _benchmark_pass_fail_lists(comparison)
    llm_x_cap = diag["requested_x_cap"]
    suggested_target = diag["suggested_target"]
    trade = diag["llm_trade_shade"]
    lines += [
        f"Suggested shade_error_direction: {diag['shade_error_direction']}.",
        (
            f"LLM mean realized trade shade: {float(trade):.4f}."
            if trade is not None
            else "LLM mean realized trade shade: (not reported)."
        ),
        (
            f"LLM mean requested x_cap (strategy output; not compared to ladder "
            f"shade_hi): {float(llm_x_cap):.4f}."
            if llm_x_cap is not None
            else "LLM mean requested x_cap: (not reported)."
        ),
        (
            f"Classical seats ahead on surplus: {len(failed_his)} failed / "
            f"{len(beaten_his)} beaten (ladder shade_hi is a label only — do not "
            f"match it)."
        ),
        (
            f"Suggested target realized shade (HINT only — do NOT hardcode): "
            f"{float(suggested_target):.4f}."
            if suggested_target is not None
            else "Suggested target realized shade: (unknown)."
        ),
        "Direction rule: classical surplus gaps + realized trade shade drive "
        "shade_more / shade_less — never compare x_cap to shade_hi.",
        "shade_more → raise typical returned shade under the chosen menu pair; "
        "shade_less → lower it; reshape_pv_map → change which menu features drive shade.",
        "Do NOT paste shade_hi / suggested_target floats into agent_strategy.",
    ]
    return lines


def format_structured_principle_form_facts(
    comparison: Optional[Dict[str, Any]],
    *,
    reward: Optional[float] = None,
    strategy_type: str = "pv",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    lob_input_mode: str = "raw",
) -> List[str]:
    lines = [
        "## Evaluation Facts (PRIMARY: batch advantage; SECONDARY: ZI surplus gaps)",
        (
            f"Batch advantage (Q reward) — PRIMARY objective: {float(reward):.4f}."
            if reward is not None
            else "Batch advantage (Q reward) — PRIMARY objective: (not reported)."
        ),
    ]
    if comparison:
        llm_x_cap = comparison.get("llm_mean_requested_x_cap")
        llm_trade_shade = comparison.get("llm_mean_shade")
        lines += [
            (
                f"LLM mean requested x_cap (strategy output): {float(llm_x_cap):.4f}."
                if llm_x_cap is not None
                else "LLM mean requested x_cap: (not reported)."
            ),
            (
                f"LLM mean realized trade shade (ex-post fill metric; NOT an objective "
                f"and NOT comparable to x_cap): {float(llm_trade_shade):.4f}."
                if llm_trade_shade is not None
                else "LLM mean realized trade shade: (not reported)."
            ),
            (
                f"ZI beat count (secondary diagnostic): "
                f"{int(comparison.get('beat_count', 0))}/"
                f"{int(comparison.get('total_benchmarks', 0))} "
                f"(fraction={float(comparison.get('beat_fraction', 0.0)):.4f})."
            ),
            f"LLM mean surplus: {float(comparison.get('llm_mean_surplus', 0.0)):.2f}.",
        ]
        lines += ["", *format_zi_surplus_gap_facts(comparison)]
        lines += ["", *format_shade_level_hypothesis_facts(comparison)]
    else:
        lines.append("No ZI comparison was provided.")
        lines += ["", *format_shade_level_hypothesis_facts(None)]

    lines += [
        "PRIMARY: propose a menu (principle, form_class) that improves batch advantage "
        "AND state the shade-level hypothesis it realizes (shade_more / shade_less / "
        "reshape_pv_map).",
        "SECONDARY: cite ZI surplus gaps / ladder shade_hi to justify shade_error_direction.",
        "Do NOT ask to make requested x_cap equal realized trade shade.",
    ]
    if strategy_type == "lob":
        if is_lob_hbl_parity_mode(lob_input_mode):
            lob_obs = (
                "LOB HBL-parity: use only the HBL information set — orders (L-window "
                "submit tape with accepted/side/price_rel/match_price_rel), L, "
                "memory_start_time, transaction_count, grace_period, side, "
                "best_bid_relative, best_ask_relative, current_pv, "
                "fundamental_estimate, pricing_fundamental. Primary signal is orders "
                "(same memory HBL uses). BBO fields are live peeks, not a full book. "
                "NO market_trades, lob_snapshots, or depth lists."
            )
        elif is_lob_book_orders_mode(lob_input_mode):
            lob_obs = (
                "LOB book_orders: use live bids/asks/best_bid/best_ask plus the L-window "
                "orders tape, current_pv, and fundamental_estimate. "
                "CRITICAL: bids/asks = resting book; orders = historical tape."
            )
        else:
            lob_obs = (
                "LOB-only: primary signal is market_history['orders'] — every bid/ask in "
                "the last-L-transaction window with accepted True/False. Prefer orders / "
                "accepted / side / price_rel / L over snapshot depth alone."
            )
        lines += [
            "CRITICAL scale note: LOB price fields (price_rel and book prices) are relative "
            "to fundamental (~0 scale). current_pv is a large monetary unit (often hundreds+), "
            "NOT a fraction in [0,1]. Shade/x_cap outputs MUST be in [0,1].",
            "Shade semantics: higher x_cap = more shade = less aggressive. "
            "Lower x_cap = less shade = more aggressive.",
            lob_obs,
            "Banned: inventing a full private_values schedule, inventory fields, "
            "unavailable market_state fields, or time-of-day stories. Ignoring order "
            "memory when only using book depth is a failure mode.",
        ]
    elif strategy_type == "spread" and is_spread_book_mode(spread_input_mode):
        _feats = spread_feature_names_for_prompts(spread_input_mode)
        _omit = get_spread_omit_fundamental()
        lines += [
            "CRITICAL scale note: best_bid/best_ask/price_rel and resting book prices "
            "are ~0-relative; current_pv and price_abs are large monetary units "
            "(often hundreds+), NOT fractions in [0,1]. "
            + (
                ""
                if _omit
                else
                "F̃ (fundamental_estimate) is absolute (~1e5); do not subtract/divide "
                "F̃ with relative quotes as if same units. "
            )
            + "Shade/x_cap outputs MUST be in [0,1].",
            "Shade semantics: higher x_cap = more shade = less aggressive. "
            "Lower x_cap = less shade = more aggressive.",
            "Banned topics: time-of-day stories, inventing a full shout tape or "
            "L-window orders history, and signals not in market_state.",
            f"Sp book packaging: use market_state fields ({_feats}). Prefer hybrid-B "
            "principles: mid vs r̂ (F̃), PV vs touch, inventory remaining, own "
            "prev_event feedback (accept/reject and/or prev price vs touch), and "
            "quote liquidity (spread width / touch depth / imbalance as ONE cue). "
            "Do NOT use current_pv_vs_fundamental as the named principle. "
            "Relative quotes are ~0-scale; do NOT threshold them like fractions "
            "(e.g. spread < 0.01). NO shouts / L-window orders tape."
            + (
                " Banned: fundamental_estimate (omitted this run)."
                if _omit
                else ""
            ),
        ]
    elif strategy_type == "spread":
        _omit = get_spread_omit_fundamental()
        spread_obs = (
            "Spread-only: use market_state fields (best_bid, best_ask, units_available, "
            "q_max, current_pv, private_values, shouts, plus fundamental under zip_parity or "
            "fundamental_estimate under classic). private_values is the remaining-unit "
            "schedule (same vector as PV). Handle None bids/asks safely."
        )
        spread_ban = (
            "Banned: inventing LOB depth lists or time-of-day stories. "
            "private_values is provided — use it, do not invent a second schedule."
        )
        _scale = (
            "CRITICAL scale note: best_bid/best_ask/price_rel are ~0-relative; current_pv "
            "and price_abs are large monetary units (often hundreds+), NOT fractions in "
            "[0,1]. Shade/x_cap outputs MUST be in [0,1]."
            if _omit
            else
            "CRITICAL scale note: best_bid/best_ask/price_rel are ~0-relative; current_pv, "
            "fundamental / fundamental_estimate, and price_abs are large monetary units "
            "(often hundreds+), NOT fractions in [0,1]. Shade/x_cap outputs MUST be in [0,1]."
        )
        lines += [
            _scale,
            "Shade semantics: higher x_cap = more shade = less aggressive. "
            "Lower x_cap = less shade = more aggressive.",
            spread_obs,
            spread_ban,
        ]
    else:
        lines += [
            "CRITICAL scale note: private_values are large monetary units (often hundreds+), "
            "NOT fractions in [0,1]. Comparing pv_mean to 0.5/0.9/etc. usually never fires. "
            "Shade/x_cap outputs MUST be in [0,1].",
            "Shade semantics: higher x_cap = more shade = less aggressive. "
            "Lower x_cap = less shade = more aggressive.",
            "Banned topics: LOB trends, best-bid chasing, time-of-day, and any signal not "
            "in private_values.",
        ]
    lines.append(
        "Anti-patterns (do not recommend): hardcoding shade floats or test_ids; "
        "comment-only or rename-only edits vs the parent."
    )
    return lines





def format_structured_principle_form_diagnosis_template(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    require_last_hold_comparison: bool = False,
) -> List[str]:
    principles, form_classes = principle_form_menus(
        strategy_type,
        lob_input_mode=lob_input_mode,
        spread_input_mode=spread_input_mode,
    )
    principle_menu = ", ".join(sorted(principles))
    form_menu = ", ".join(sorted(form_classes))
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        features_hint = (
            "comma-separated identifiers only from this run's feature set: "
            f"{_lob_features_used_hint(lob_input_mode)}"
        )
        mechanism_hint = (
            "cite ≥2 trajectory moments: time, current_pv, shade, feature values "
            "from the allowed set, fill/no-fill"
        )
    elif strategy_type == "lob" and is_lob_hbl_parity_mode(lob_input_mode):
        features_hint = (
            "comma-separated identifiers only from the HBL-parity set: "
            f"{_lob_features_used_hint(lob_input_mode)}"
        )
        mechanism_hint = (
            "cite ≥2 trajectory moments: time, current_pv, shade, order-window "
            "accept/reject or side pressure, BBO peek when used, fill/no-fill"
        )
    elif strategy_type == "lob":
        features_hint = (
            "comma-separated identifiers only, e.g. "
            f"{_lob_features_used_hint(lob_input_mode)}"
        )
        mechanism_hint = (
            "cite ≥2 trajectory moments: time, current_pv, shade, order-window "
            "accept/reject or side pressure, fill/no-fill"
        )
    elif strategy_type == "spread":
        if is_spread_book_mode(spread_input_mode):
            features_hint = (
                "comma-separated from best_bid/best_ask, bids/asks, units_available/"
                "q_max, prev_event, current_pv, private_values "
                "(and fundamental_estimate when present); NO shouts"
            )
            mechanism_hint = (
                "cite ≥2 trajectory moments: time, shade, fill/no-fill, and the "
                "book/inventory/touch signal the principle uses"
            )
        else:
            features_hint = (
                "which market_state fields (best_bid, best_ask, units_available, q_max, "
                "current_pv, private_values, shouts, fundamental or fundamental_estimate); "
                "private_values is the remaining-unit schedule; handle None bids/asks safely"
            )
            mechanism_hint = (
                "cite case-card stats (leftover, fill, pos_gap_miss) and ≥2 tape shouts: "
                "t, side, shade, spread, gap_to_touch, filled"
            )
    else:
        features_hint = (
            "comma-separated from private_values, fundamental_estimate, current_pv "
            "(and derived stats); note which of Θ / F̃ / unit the hypothesis uses"
        )
        mechanism_hint = (
            "cite ≥2 trajectory moments: time, current_pv, shade, fill/no-fill "
            "(and schedule/F̃ context when relevant)"
        )
    diagnosis_fields = [
        "DIAGNOSIS:",
        "  principle: <one from menu — parameters of the hypothesis>",
        "  form_class: <one from menu — parameters of the hypothesis>",
        "  shade_error_direction: <shade_more | shade_less | reshape_pv_map — "
        "level hypothesis realized VIA the menu pair above>",
        f"  features_used: <{features_hint}>",
        "  current_form: <one line describing THIS strategy's functional form>",
        "  why_this_fails: <cite leftover / shade_fill / pos_gap_miss from the case card>",
        "  action: <refine = keep this principle+form and re-implement shade map; "
        "replace = new pair>",
    ]
    if require_last_hold_comparison:
        diagnosis_fields.append(
            "  same_failure_as_last_hold: <yes or no — leftover / pos_gap_miss / "
            "buy vs sell vs LAST HOLD card; yes→action=refine, no→action=replace>"
        )
    diagnosis_fields.append(
        "  proposed_change: <refine: how the new implementation differs "
        "(features/normalization/branches) to realize shade_error_direction; "
        "replace: how the new pair differs from current_form and realizes the "
        "level hypothesis>"
    )
    pair_rules = [
        "You MUST set action to refine or replace.",
        "action=refine: KEEP this strategy's (principle, form_class) and improve "
        "the shade map under that pair — forbid comment-only or AST-identical "
        "clones; the new implementation must compute shade differently enough "
        "that probe shades differ.",
        "action=replace: pick a NEW pair (unexplored if any remain; must differ "
        "from parent/siblings). Prefer replace when the parent pair is already "
        "explored and surplus is not improving.",
    ]
    if require_last_hold_comparison:
        pair_rules.append(
            "LAST HOLD: yes → action=refine; no → action=replace."
        )
    menu_lines = [
        "## Allowed menus (pick exactly one principle and one form_class)",
        f"principle ∈ {{{principle_menu}}}",
        f"form_class ∈ {{{form_menu}}}",
    ]
    if strategy_type == "pv":
        menu_lines.extend(
            [
                "Prefer hypotheses that can use Θ (private_values), F̃ "
                "(fundamental_estimate), and the unit being priced (current_pv) when "
                "useful — unused args are allowed, but schedule-only rules often "
                "underuse current_pv.",
                "current_pv_vs_fundamental: shade from the unit value vs F̃.",
                "current_pv_in_schedule: shade from where the current unit sits in Θ.",
                "pv_vs_fundamental: schedule aggregates and/or current_pv vs F̃.",
                "fill_surplus_tradeoff: aggressiveness vs surplus left on the table "
                "(often current_pv with schedule context).",
                "pv_rank_or_distance_to_best: rank/distance of units in the schedule "
                "(can incorporate current_pv).",
                "unit_schedule_blend: combine a unit-level term with a schedule-level "
                "term (natural place for all three inputs).",
            ]
        )
    elif strategy_type == "spread" and is_spread_book_mode(spread_input_mode):
        menu_lines.extend(
            [
                "Prefer hypotheses that use mid vs r̂, PV vs touch, inventory, "
                "own prev_event feedback, and/or quote liquidity — match the named "
                "principle. Relative quotes ~0; never treat them as [0,1] fractions.",
                "Principles (what economic signal drives shade):",
                "mid_vs_rhat: midquote ((best_bid+best_ask)/2) or touch vs "
                "fundamental_estimate (ZIP/Preist fair-price target).",
                "pv_vs_touch: current_pv vs best quote on the relevant side "
                "(ZIP-style competitiveness).",
                "inventory_remaining_fraction: remaining demand / q_max "
                "(AA inventory urgency).",
                "own_prev_feedback: own prev_event accepted flag and/or prev "
                "price vs live touch (ZIP/GD margin update).",
                "quote_liquidity: ONE cue from quoted spread width, touch depth, "
                "and/or book imbalance (do not invent a second liquidity principle).",
                "Forms (how that signal becomes a shade map):",
                "ratio: continuous normalize/fraction map from the principle signal.",
                "threshold: step / gated map once the signal crosses a cut.",
                "piecewise_by_sign: different shade branches by sign or regime.",
                "inventory_weighted: modulate the principle signal by remaining "
                "inventory fraction.",
                "target_soft_update: move shade toward a computed target "
                "(ZIP/Preist soft update), not a static ratio of the raw signal.",
            ]
        )
    return [
        *menu_lines,
        "",
        "## Required DIAGNOSIS Output (hypothesis for the NEXT codegen step)",
        "You are evaluating THIS strategy's batch results and proposing the NEXT "
        "hypothesis for the generator: a shade-level claim "
        "(shade_error_direction) realized by menu parameters "
        "(principle + form_class).",
        "Use the exact labels. Fill every field. Do not invent extra top-level sections.",
        "Copy Suggested shade_error_direction from Shade-level hypothesis Facts "
        "unless MECHANISM justifies a different choice.",
        "",
        *diagnosis_fields,
        "",
        f"MECHANISM: (max 3 sentences; {mechanism_hint})",
        "FIX: (max 2 sentences; state shade_error_direction and how the chosen "
        "principle+form_class realizes it; no full code; no magic shade floats)",
        "",
        "Do NOT write Environment, Performance, or Principle essay sections.",
        *pair_rules,
        "shade_error_direction must be shade_more, shade_less, or reshape_pv_map.",
        (
            "Example FIX: 'Hypothesis shade_more under fill_vs_reject + "
            "piecewise_by_sign: raise shade after rejects, cut after fills.'"
            if strategy_type == "lob"
            else
            "Example FIX: 'Hypothesis shade_more under mid_vs_rhat + "
            "piecewise_by_sign: raise shade when midquote is below F̃.'"
            if strategy_type == "spread" and is_spread_book_mode(spread_input_mode)
            else
            "Example FIX: 'Hypothesis shade_more under inventory_remaining_fraction + "
            "ratio: raise shade when remaining inventory fraction is high.'"
            if strategy_type == "spread"
            else
            "Example FIX: 'Hypothesis shade_more under current_pv_vs_fundamental + "
            "piecewise_by_sign: raise shade when current_pv is below F̃.'"
        ),
        "Do NOT paste shade_hi / suggested_target floats into FIX as code constants.",
        (
            "features_used must be comma-separated identifiers "
            "(e.g. spread_rel, imbalance, current_pv), never template prose."
            if strategy_type == "lob"
            and is_lob_features_mode(lob_input_mode)
            else
            "features_used must be comma-separated identifiers "
            "(e.g. bids_relative, current_pv), never template prose."
        ),
    ]
















def format_structured_principle_form_parent_eval(
    parent_performance: Dict[str, Any],
    *,
    q_reward_mode: str = "zi_advantage",
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> List[str]:
    comparison = parent_performance.get("zi_benchmark_comparison")
    reward = parent_performance.get("reward")
    reward_var = float(parent_performance.get("reward_variance", 0.0))
    lines = [
        "",
        "## Parent Evaluation (batch metrics — implement Parent Reflection DIAGNOSIS)",
    ]
    lines += format_structured_principle_form_facts(
        comparison,
        reward=float(reward) if reward is not None else None,
        strategy_type=strategy_type,
        spread_input_mode=spread_input_mode,
        lob_input_mode=lob_input_mode,
    )
    lines.append(f"Batch reward variance: {reward_var:.4f}")

    per_config = parent_performance.get("per_config_rewards") or {}
    per_config_var = parent_performance.get("per_config_reward_variances") or {}
    if per_config:
        lines += ["", "## Parent Per-Config Advantage (batch breakdown)"]
        for label in sorted(per_config.keys()):
            adv = float(per_config[label])
            var = float(per_config_var.get(label, 0.0))
            lines.append(f"- {label}: advantage={adv:.4f} (variance={var:.4f})")

    if comparison and is_zi_q_reward_mode(q_reward_mode):
        lines += [
            "",
            "## Parent ZI Surplus Gaps (soft diagnostic — not Fix targets)",
            (
                f"Beat {int(comparison.get('beat_count', 0))}/"
                f"{int(comparison.get('total_benchmarks', 0))} benchmarks "
                f"(fraction={float(comparison.get('beat_fraction', 0.0)):.4f})."
            ),
            "Use gaps only as context for why_parent_fails; do not hardcode shade_hi "
            "or test_ids.",
        ]
    return lines


def _structured_principle_form_critic_instructions(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> List[str]:
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        names = ", ".join(active_lob_feature_names(lob_input_mode))
        return [
            "You are reviewing a LOB features-mode shading strategy in a continuous "
            "double auction.",
            "PRIMARY objective: batch advantage (Q reward = LLM surplus minus baseline; "
            "higher is better).",
            "SECONDARY diagnostic: ZI surplus gaps — use to explain weak regimes in "
            "why_parent_fails, not as the optimization target.",
            "Your job is to propose a shade-level hypothesis (shade_error_direction) "
            "realized by a market-principled menu pair (principle + form_class) — not a "
            "ZI shade_hi paste.",
            "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template).",
            "Then write MECHANISM (max 3 sentences) and FIX (max 2 sentences).",
            "Do NOT write Environment, Performance, or Principle essay sections.",
            "Pick principle and form_class from the allowed menus; set shade_error_direction "
            "from Shade-level hypothesis Facts.",
            f"features_used must be comma-separated identifiers from this run's feature "
            f"set only: {names}, current_pv — never template prose.",
            "proposed_change must say how the menu pair realizes shade_error_direction "
            "(form/feature change vs current_form, not only a coefficient tweak).",
            "Anti-patterns: never recommend hardcoding shade floats/test_ids, or "
            "comment-only / rename-only edits.",
            "Higher x_cap means MORE shade (less aggressive). Do not use raw lob_snapshots "
            "/ depth lists or invent a full private_values schedule.",
        ]
    if strategy_type == "lob":
        if is_lob_book_orders_mode(lob_input_mode):
            return [
                "You are reviewing a LOB book_orders shading strategy in a continuous "
                "double auction.",
                "PRIMARY objective: batch advantage (Q reward = LLM surplus minus baseline; "
                "higher is better).",
                "SECONDARY diagnostic: ZI surplus gaps — use to explain weak regimes in "
                "why_parent_fails, not as the optimization target.",
                "Your job is to propose a shade-level hypothesis (shade_error_direction) "
                "realized by a market-principled menu pair (principle + form_class) — not a "
                "ZI shade_hi paste.",
                "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template).",
                "Then write MECHANISM (max 3 sentences) and FIX (max 2 sentences).",
                "Do NOT write Environment, Performance, or Principle essay sections.",
                "Pick principle and form_class from the allowed LOB book_orders menus "
                "(fill_vs_reject, order_flow_imbalance, recent_aggressiveness, "
                "book_imbalance, depth_asymmetry); set shade_error_direction from Facts.",
                "features_used must be comma-separated field identifiers from market_history "
                "(prefer bids, asks, best_bid, best_ask, orders, accepted, side, price_rel, "
                "current_pv, fundamental_estimate) — not prose.",
                "CRITICAL: 'bids'/'asks' are the FULL live resting book; 'orders' is the "
                "historical L-window submit tape. book_imbalance / depth_asymmetry must use "
                "bids/asks depth; tape principles must use orders.",
                "proposed_change must say how the menu pair realizes shade_error_direction "
                "(form/feature change vs current_form, not only a coefficient tweak).",
                "Anti-patterns: never recommend hardcoding shade floats/test_ids, or "
                "comment-only / rename-only edits.",
                "Higher x_cap means MORE shade (less aggressive). LOB book_orders: ban "
                "inventing a full private_values schedule, inventory fields, L, or "
                "unavailable market_state fields.",
            ]
        if is_lob_hbl_parity_mode(lob_input_mode):
            return [
                "You are reviewing a LOB HBL-parity shading strategy in a continuous "
                "double auction.",
                "PRIMARY objective: batch advantage (Q reward = LLM surplus minus baseline; "
                "higher is better).",
                "SECONDARY diagnostic: ZI surplus gaps — use to explain weak regimes in "
                "why_parent_fails, not as the optimization target.",
                "Your job is to propose a shade-level hypothesis (shade_error_direction) "
                "realized by a market-principled menu pair (principle + form_class) — not a "
                "ZI shade_hi paste.",
                "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template).",
                "Then write MECHANISM (max 3 sentences) and FIX (max 2 sentences).",
                "Do NOT write Environment, Performance, or Principle essay sections.",
                "Pick principle and form_class from the allowed LOB order-memory menus "
                "(fill_vs_reject, side_pressure, order_flow_imbalance, "
                "recent_aggressiveness); set shade_error_direction from Facts.",
                "features_used must be comma-separated field identifiers from the HBL-parity "
                "set only: orders, accepted, side, price_rel, match_price_rel, L, "
                "memory_start_time, transaction_count, grace_period, best_bid_relative, "
                "best_ask_relative, current_pv, fundamental_estimate, pricing_fundamental "
                "— not prose.",
                "Primary signal is orders (same L-window memory HBL uses). BBO "
                "(best_bid_relative / best_ask_relative) is a live peek, not a full book. "
                "If the parent ignores orders/accepted, say so in why_parent_fails and prefer "
                "action=replace toward an order-memory principle.",
                "proposed_change must say how the menu pair realizes shade_error_direction "
                "(form/feature change vs current_form, not only a coefficient tweak).",
                "Anti-patterns: never recommend hardcoding shade floats/test_ids, or "
                "comment-only / rename-only edits.",
                "Higher x_cap means MORE shade (less aggressive). Ban inventing a full "
                "private_values schedule, inventory fields, market_trades, lob_snapshots, "
                "or unavailable market_state fields.",
            ]
        return [
            "You are reviewing a LOB-only shading strategy in a continuous double auction.",
            "PRIMARY objective: batch advantage (Q reward = LLM surplus minus baseline; "
            "higher is better).",
            "SECONDARY diagnostic: ZI surplus gaps — use to explain weak regimes in "
            "why_parent_fails, not as the optimization target.",
            "Your job is to propose a shade-level hypothesis (shade_error_direction) "
            "realized by a market-principled menu pair (principle + form_class) — not a "
            "ZI shade_hi paste.",
            "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template).",
            "Then write MECHANISM (max 3 sentences) and FIX (max 2 sentences).",
            "Do NOT write Environment, Performance, or Principle essay sections.",
            "Pick principle and form_class from the allowed LOB order-memory menus "
            "(fill_vs_reject, side_pressure, order_flow_imbalance, "
            "recent_aggressiveness); set shade_error_direction from Facts.",
            "features_used must be comma-separated field identifiers from market_history "
            "(prefer orders, accepted, side, price_rel, L, memory_start_time, current_pv) "
            "— not prose like 'prior book / trade fields'.",
            "If the parent ignores orders/accepted and only uses snapshot depth, say so in "
            "why_parent_fails and prefer action=replace toward an order-memory principle.",
            "proposed_change must say how the menu pair realizes shade_error_direction "
            "(form/feature change vs current_form, not only a coefficient tweak).",
            "Anti-patterns: never recommend hardcoding shade floats/test_ids, or "
            "comment-only / rename-only edits.",
            "Higher x_cap means MORE shade (less aggressive). LOB-only: ban inventing a "
            "full private_values schedule, inventory fields, or unavailable market_state "
            "fields.",
        ]
    if strategy_type == "spread":
        if is_spread_book_mode(spread_input_mode):
            omit = get_spread_omit_fundamental()
            feats = spread_feature_names_for_prompts(spread_input_mode)
            return [
                "You are reviewing a spread-book shading strategy in a continuous "
                "double auction.",
                "PRIMARY objective: batch advantage (Q reward = LLM surplus minus baseline; "
                "higher is better).",
                "SECONDARY diagnostic: ZI surplus gaps — use to explain weak regimes in "
                "why_this_fails, not as the optimization target.",
                "Your job is to propose a shade-level hypothesis (shade_error_direction) "
                "realized by a market-principled menu pair (principle + form_class) — not a "
                "ZI shade_hi paste.",
                "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template).",
                "Then write MECHANISM (max 3 sentences) and FIX (max 2 sentences).",
                "Do NOT write Environment, Performance, or Principle essay sections.",
                "Pick principle and form_class from the Sp-book hybrid-B menus "
                "(mid_vs_rhat / pv_vs_touch / inventory_remaining_fraction / "
                "own_prev_feedback / quote_liquidity); "
                "set shade_error_direction from Shade-level hypothesis Facts.",
                "features_used must be scale-safe (relative quotes vs large current_pv "
                "/ F̃). Do not threshold relative spreads like fractions.",
                "proposed_change must say how the menu pair realizes shade_error_direction "
                "(form/feature change vs current_form, not only a coefficient tweak).",
                "Anti-patterns: never recommend hardcoding shade floats/test_ids, or "
                "comment-only / rename-only edits.",
                "Higher x_cap means MORE shade (less aggressive). Prefer mid-vs-r̂, "
                "PV-vs-touch, inventory, own-prev feedback, and quote-liquidity "
                "hypotheses matching the menus."
                + (
                    " Do not cite fundamental_estimate (omitted this run)."
                    if omit
                    else " You MAY use fundamental_estimate when the principle needs it."
                ),
                f"market_state fields: {feats}. NO shouts / full tape. "
                "action=refine keeps principle+form; action=replace picks a new pair.",
            ]
        return [
            "You are reviewing a spread-only shading strategy in a continuous double auction.",
            "PRIMARY objective: batch advantage (Q reward = LLM surplus minus baseline; "
            "higher is better).",
            "SECONDARY diagnostic: ZI surplus gaps — use to explain weak regimes in "
            "why_this_fails, not as the optimization target.",
            "Your job is to propose a shade-level hypothesis (shade_error_direction) "
            "realized by a market-principled menu pair (principle + form_class) — not a "
            "ZI shade_hi paste.",
            "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template).",
            "Then write MECHANISM (max 3 sentences) and FIX (max 2 sentences).",
            "Do NOT write Environment, Performance, or Principle essay sections.",
            "Pick principle and form_class from the allowed SPREAD menus (not PV menus): "
            "principles cover bid/ask geometry, inventory fraction, PV vs touch/mid, "
            "and shout-tape signals (recent_accept_reject, shout_competitiveness); "
            "set shade_error_direction from Facts.",
            "features_used must come from market_state only: best_bid, best_ask, "
            "units_available, q_max, current_pv, private_values, shouts, and fundamental "
            "(zip_parity) or fundamental_estimate (classic). private_values is the "
            "remaining-unit schedule. Handle None bids/asks safely.",
            "action=refine keeps this principle+form and re-implements the shade "
            "map; action=replace picks a new pair.",
            "proposed_change must match the chosen action and say how the menu pair "
            "realizes shade_error_direction.",
            "Anti-patterns: never recommend hardcoding shade floats/test_ids, or "
            "comment-only / rename-only edits.",
            "Higher x_cap means MORE shade (less aggressive). Spread-only: use market_state "
            "including shouts and private_values; do not invent LOB depth.",
        ]
    return [
        "You are reviewing a PV-only shading strategy in a continuous double auction.",
        "PRIMARY objective: batch advantage (Q reward = LLM surplus minus baseline; "
        "higher is better).",
        "SECONDARY diagnostic: ZI surplus gaps — use to explain weak regimes in "
        "why_parent_fails, not as the optimization target.",
        "Your job is to propose a shade-level hypothesis (shade_error_direction) "
        "realized by a market-principled menu pair (principle + form_class) — not a "
        "ZI shade_hi paste.",
        "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template).",
        "Then write MECHANISM (max 3 sentences) and FIX (max 2 sentences).",
        "Do NOT write Environment, Performance, or Principle essay sections.",
        "Pick principle and form_class from the allowed menus; set shade_error_direction "
        "from Shade-level hypothesis Facts.",
        "features_used must be scale-safe for large private_values (not [0,1] fractions).",
        "proposed_change must say how the menu pair realizes shade_error_direction "
        "(form/feature change vs current_form, not only a coefficient tweak).",
        "Anti-patterns: never recommend hardcoding shade floats/test_ids, or "
        "comment-only / rename-only edits.",
        "Higher x_cap means MORE shade (less aggressive). PV-only: ban LOB/bid-ask/trend "
        "stories. You MAY use fundamental_estimate (absolute F̃) with the schedule.",
    ]


_SHADE_LEVEL_HYPOTHESIS_GENERATOR_LINES = [
    "Parent DIAGNOSIS is a shade-level hypothesis (shade_error_direction) realized "
    "by menu parameters (principle + form_class).",
    "Priority: (1) shade_error_direction under the named principle+form_class, "
    "(2) features_used / proposed_change / action, (3) MECHANISM/FIX prose.",
    "shade_more: raise typical returned x_cap in [0,1] under that menu pair.",
    "shade_less: lower typical returned x_cap in [0,1] under that menu pair.",
    "reshape_pv_map: change which menu features drive shade (not a global level nudge).",
    "Do NOT hardcode shade floats, shade_hi, suggested_target, test_id, or "
    "benchmark lists. Do NOT equate x_cap with realized trade shade.",
    "DIAGNOSIS principle / form_class / shade_error_direction are labels for you — "
    "do NOT write them into agent_strategy as string variables, comments that only "
    "name the menu pair, or if principle == ... / if form_class == ... dispatch. "
    "Implement the math those labels describe using market inputs only.",
]


def _structured_principle_form_generator_task(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> List[str]:
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        return [
            "Read Parent Hypothesis DIAGNOSIS first (principle, form_class, "
            "shade_error_direction, features_used, proposed_change).",
            "Implement that hypothesis using Parent Evaluation facts and trajectory when "
            "provided.",
            *_SHADE_LEVEL_HYPOTHESIS_GENERATOR_LINES,
            "Invent a NEW general features→shade rule of the named form_class using only "
            "the fixed scalar feature vector — not a market-specific patch and not a "
            "pasted critic float.",
            "Change functional form / features vs the Parent Strategy (not only "
            "coefficients).",
            "Do NOT copy the parent code. Do NOT use raw lob_snapshots / depth lists or "
            "invent a full private_values schedule.",
            "Your agent_strategy MUST differ meaningfully from the Parent Strategy "
            "(logic / formulas / control flow — not comment-only or rename-only).",
            _LOB_SHADE_SANITY_CHECK,
        ]
    if strategy_type == "lob" and is_lob_book_orders_mode(lob_input_mode):
        return [
            "Read Parent Hypothesis DIAGNOSIS first (principle, form_class, "
            "shade_error_direction, features_used, proposed_change).",
            "Implement that hypothesis using Parent Evaluation facts and trajectory when "
            "provided.",
            *_SHADE_LEVEL_HYPOTHESIS_GENERATOR_LINES,
            "Invent a NEW general market_history→shade rule of the named form_class — not a "
            "market-specific patch and not a pasted critic float.",
            "Use live bids/asks/best_bid/best_ask (resting book) plus the L-window "
            "orders tape, current_pv, and fundamental_estimate. "
            "CRITICAL: bids/asks = resting book; orders = historical tape. "
            "Do NOT read L, lob_snapshots, or spread_relative from market_history.",
            "Change functional form / features vs the Parent Strategy (not only "
            "coefficients).",
            "Do NOT copy the parent code. Do NOT invent a full private_values schedule.",
            "Your agent_strategy MUST differ meaningfully from the Parent Strategy "
            "(logic / formulas / control flow — not comment-only or rename-only).",
            _LOB_SHADE_SANITY_CHECK,
        ]
    if strategy_type == "lob":
        return [
            "Read Parent Hypothesis DIAGNOSIS first (principle, form_class, "
            "shade_error_direction, features_used, proposed_change).",
            "Implement that hypothesis using Parent Evaluation facts and trajectory when "
            "provided.",
            *_SHADE_LEVEL_HYPOTHESIS_GENERATOR_LINES,
            "Invent a NEW general market_history→shade rule of the named form_class — not a "
            "market-specific patch and not a pasted critic float.",
            "Prefer market_history['orders'] (accepted / side / price_rel / L) as the primary "
            "signal — same L-window memory HBL uses. Snapshot depth alone is weaker.",
            "Change functional form / features vs the Parent Strategy (not only "
            "coefficients).",
            "Do NOT copy the parent code. Do NOT invent a full private_values schedule.",
            "Your agent_strategy MUST differ meaningfully from the Parent Strategy "
            "(logic / formulas / control flow — not comment-only or rename-only).",
            _LOB_SHADE_SANITY_CHECK,
        ]
    if strategy_type == "spread" and is_spread_book_mode(spread_input_mode):
        omit = get_spread_omit_fundamental()
        return [
            "Read Parent Hypothesis DIAGNOSIS first (principle, form_class, "
            "shade_error_direction, features_used, action, proposed_change).",
            "Implement that hypothesis using Parent Evaluation facts and trajectory when "
            "provided.",
            *_SHADE_LEVEL_HYPOTHESIS_GENERATOR_LINES,
            "If action=refine: KEEP principle+form; re-implement the shade map under "
            "that pair so it realizes shade_error_direction (different features / "
            "normalization / branches). Coefficient-only or +ε damper edits are forbidden.",
            "If action=replace: write a NEW market_state→shade rule of the named "
            "form_class that realizes shade_error_direction. It must differ from the "
            "parent in structure, not only comments.",
            "Invent a NEW general market_state→shade rule under the named principle "
            "(mid vs r̂, PV vs touch, inventory fraction, own prev_event "
            "feedback, or quote liquidity) — not a market-specific patch and not a "
            "pasted critic float.",
            f"Signature must be agent_strategy(market_state): use fields from "
            f"({spread_feature_names_for_prompts(spread_input_mode)}). Prefer the "
            "signals named by principle/features_used. NO shouts / L-window orders "
            "tape."
            + (
                " Do NOT read or invent fundamental_estimate."
                if omit
                else
                " F̃ is absolute; relative quotes/PV are not — never subtract/divide "
                "as same units; never divide PV by relative spread."
            ),
            "If best_bid or best_ask is None, that quote is absent — do not treat "
            "None as 0.0 in arithmetic.",
            "Your agent_strategy MUST differ meaningfully from the Parent Strategy "
            "(logic / formulas / control flow — not comment-only or rename-only).",
            _SPREAD_SHADE_SANITY_CHECK,
        ]
    if strategy_type == "spread":
        return [
            "Read Parent Hypothesis DIAGNOSIS first (principle, form_class, "
            "shade_error_direction, features_used, action, proposed_change).",
            "Implement that hypothesis using Parent Evaluation and case cards when provided.",
            *_SHADE_LEVEL_HYPOTHESIS_GENERATOR_LINES,
            "If action=refine: KEEP principle+form; re-implement the shade map under "
            "that pair so it realizes shade_error_direction (different features / "
            "normalization / branches). Coefficient-only or +ε damper edits are forbidden.",
            "If action=replace: write a NEW market_state→shade rule of the named "
            "form_class that realizes shade_error_direction. It must differ from the "
            "parent in structure, not only comments.",
            "You MAY use market_state['shouts'] (accepted and rejected) and "
            "market_state['private_values'] (remaining-unit schedule, same as PV). "
            "Do NOT invent LOB depth lists.",
            _SPREAD_SHADE_SANITY_CHECK,
        ]
    return [
        "Read Parent Hypothesis DIAGNOSIS first (principle, form_class, "
        "shade_error_direction, features_used, proposed_change).",
        "Implement that hypothesis using Parent Evaluation facts and trajectory when "
        "provided.",
        *_SHADE_LEVEL_HYPOTHESIS_GENERATOR_LINES,
        "Invent a NEW general "
        "(private_values, fundamental_estimate, current_pv)→shade rule of the "
        "named form_class — not a market-specific patch and not a pasted critic float.",
        "Change functional form / features vs the Parent Strategy (not only "
        "coefficients).",
        "Do NOT copy the parent code. Do NOT use LOB, bid/ask, or market-trend signals.",
        "Signature must be "
        "agent_strategy(private_values, fundamental_estimate, current_pv): "
        "full schedule Θ, absolute F̃, and the unit being priced now. "
        "Prefer using all three when the hypothesis benefits; unused args are "
        "allowed but schedule-only rules often underuse current_pv.",
        "Your agent_strategy MUST differ meaningfully from the Parent Strategy "
        "(logic / formulas / control flow — not comment-only or rename-only).",
    ]










def format_forbidden_strategies_section(
    forbidden_strategies: Optional[List[Dict[str, Any]]],
) -> List[str]:
    if not forbidden_strategies:
        return []
    lines: List[str] = [
        "",
        "## Sibling strategies already tried (do NOT return AST-equivalent variants)",
        "These are prior children of the same parent. Do not copy them.",
        "Stay under the Parent Strategy's principle+form with a shade expression "
        "whose probe shades differ from these siblings — do not invent a new "
        "form_class unless DIAGNOSIS action=replace, and do not return "
        "AST-identical or comment-only clones.",
    ]
    for item in forbidden_strategies:
        role = str(item.get("role") or "strategy")
        node_id = item.get("node_id", "?")
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        lines += [
            "",
            f"### Forbidden {role} (node {node_id})",
            "```python",
            code,
            "```",
        ]
    return lines


def build_generation_prompt(
    *,
    parent_code: str,
    parent_reflection: str,
    strategy_output_mode: str,
    market_description: str,
    offline_example: Optional[Dict[str, Any]] = None,
    strategy_type: str = "full",
    use_reflection: bool = True,
    parent_performance: Optional[Dict[str, Any]] = None,
    generator_batch_summary: Optional[Dict[str, Any]] = None,
    generator_batch_data: Optional[Dict[str, Any]] = None,
    sibling_code: Optional[str] = None,
    sibling_context: Optional[Dict[str, Any]] = None,
    q_reward_mode: str = "advantage",
    critic_mode: str = "default",
    forbid_numpy_random: bool = False,
    forbidden_strategies: Optional[List[Dict[str, Any]]] = None,
    extra_instructions: Optional[List[str]] = None,
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    outcome_first: bool = False,
) -> str:
    is_pv_only = strategy_type == "pv"
    is_spread_only = strategy_type == "spread"
    is_lob_only = strategy_type == "lob"
    is_lob_features = is_lob_only and is_lob_features_mode(lob_input_mode)
    zip_parity = is_spread_only and is_spread_zip_parity_mode(spread_input_mode)
    fund_key = "fundamental" if zip_parity else "fundamental_estimate"
    requirements = [
        "Return exactly one Python function named agent_strategy.",
        (
            "Signature: agent_strategy(private_values, fundamental_estimate, current_pv)."
            if is_pv_only
            else "Signature: agent_strategy(market_state)."
            if is_spread_only
            else "Signature: agent_strategy(market_history)."
            if is_lob_only
            else "Signature: agent_strategy(private_values, market_history, market_state)."
        ),
        "Return a single float x in [0,1] representing the shading level.",
        f"Buyers: bid = ({fund_key} + pv) - x * ({fund_key} + pv).",
        f"Sellers: ask = ({fund_key} + pv) + x * ({fund_key} + pv).",
        "pv is the marginal unit private value (buyer) or cost (seller, ascending schedule).",
        "x=0 is unshaded (buyer bids full value; seller asks reservation); x=1 is maximum shade.",
        "Assume the market only allows single-unit trades.",
        "Only use provided inputs; do not import external libraries.",
    ]
    if strategy_output_mode == "x_cap":
        requirements.append("Interpret x as x_cap (max shade); output should remain in [0,1].")
    else:
        requirements.append("Interpret x as shading fraction; output should remain in [0,1].")

    sections = [
        "You are creating a trading strategy for a continuous double auction.",
        "",
        "## Market Description",
        market_description.strip(),
        "",
        "## Input Schema",
    ]
    if is_pv_only:
        sections.append(
            "private_values is the full relative private-value schedule for all units "
            "(length q_max; entries are NOT removed after trades)."
        )
        sections.append(
            "current_pv is the relative private value/cost of the unit being priced "
            "on this decision (same role as LOB/spread current_pv)."
        )
        sections.append(
            "fundamental_estimate is the absolute noisy projected fundamental F̃ "
            "(same quantity ShadeToLimit uses for pricing). Never invent true F_t."
        )
        sections.append(
            "Prefer private_values, fundamental_estimate, and current_pv when deciding "
            "the shading level (use all three when the hypothesis benefits); "
            "do not use bid-ask or LOB fields."
        )
    elif is_spread_only:
        sections += _spread_input_schema(spread_input_mode)
    elif is_lob_only:
        sections += _lob_input_schema(lob_input_mode)
    if offline_example:
        example_text = json.dumps(offline_example, indent=2)
        sections += [
            "",
            "## Offline ZI Example (single agent perspective)",
            example_text,
            "Use this example to understand the data format and typical values.",
        ]

    if parent_code.strip():
        sections += [
            "",
            "## Parent Strategy",
            "```python",
            parent_code.strip(),
            "```",
        ]
        curriculum_mode = is_structured_zi_curriculum(critic_mode)
        hypothesis_loop = is_hypothesis_loop_critic(critic_mode)
        principle_form_mode = is_structured_principle_form(critic_mode)
        if parent_performance:
            if principle_form_mode:
                sections += format_structured_principle_form_parent_eval(
                    parent_performance,
                    q_reward_mode=q_reward_mode,
                    strategy_type=strategy_type,
                    lob_input_mode=lob_input_mode,
                    spread_input_mode=spread_input_mode,
                )
        if use_reflection:
            sections += [
                "",
                    "## Parent Hypothesis (DIAGNOSIS — implement this experiment)",
                parent_reflection.strip() or "(none)",
            ]
            sections += [
                "",
                "## Refinement Task",
            ]
            if principle_form_mode:
                sections += _structured_principle_form_generator_task(
                    strategy_type=strategy_type,
                    lob_input_mode=lob_input_mode,
                    spread_input_mode=spread_input_mode,
                )
                diagnosis_action = (parse_diagnosis_action(parent_reflection or "") or "").strip().lower()
                requirements += [
                    "Implement Parent Hypothesis DIAGNOSIS (principle, form_class, action) first.",
                    "Do not hardcode critic floats or test_ids.",
                ]
                if outcome_first:
                    requirements.insert(
                        -1,
                        "Outcome-first: implement the CHANGE line from the learning "
                        "contract before principle/form labels.",
                    )
                if diagnosis_action == "refine":
                    requirements.append(
                        "action=refine: keep the parent principle+form; change the "
                        "shade map under that pair so probe shades differ from the "
                        "parent (not comment-only)."
                    )
                else:
                    requirements.append(
                        "action=replace (or unspecified): change how shade is "
                        "computed vs the Parent Strategy (not comment-only)."
                    )
                if is_lob_features:
                    requirements.append(
                        "Use only market_history['features'] scalars (and current_pv); "
                        "do not use raw lob_snapshots / depth lists or invent a full "
                        "private_values schedule."
                    )
                elif is_lob_only:
                    requirements.append(
                        "Use market_history only; do not invent a full private_values schedule."
                    )
                    requirements.append(_LOB_SHADE_SANITY_CHECK)
                elif is_spread_only:
                    if is_spread_book_mode(spread_input_mode):
                        requirements.append(
                            (
                                "Sp-book menus: use quotes, resting bids/asks, inventory, "
                                "own prev_event, and/or current_pv as named by the "
                                "principle. Do not invent shouts, fundamental_estimate, "
                                "or an L-window orders tape."
                                if get_spread_omit_fundamental()
                                else
                                "Sp-book menus: use quotes, resting bids/asks, inventory, "
                                "own prev_event, PV-vs-touch, and related market_state "
                                "fields as named by the principle. Do not invent shouts "
                                "or an L-window orders tape."
                            )
                        )
                        requirements.append(_SPREAD_SHADE_SANITY_CHECK)
                    else:
                        requirements.append(
                            "Use market_state only (including shouts and private_values); "
                            "do not invent LOB depth lists."
                        )
                        requirements.append(_SPREAD_SHADE_SANITY_CHECK)
                else:
                    requirements.append(
                        "Use private_values, fundamental_estimate, and current_pv only; "
                        "no LOB or market-trend inputs. Signature must be "
                        "agent_strategy(private_values, fundamental_estimate, current_pv)."
                    )

    if generator_batch_summary is not None and generator_batch_data is not None:
        sections += format_evaluation_trajectory_sections(
            batch_summary=generator_batch_summary,
            batch_data=generator_batch_data,
            heading_prefix="Parent",
            q_reward_mode=q_reward_mode,
        )

    if sibling_code and sibling_code.strip():
        sibling = sibling_context or {}
        sections += [
            "",
            "## Most Recent Sibling Strategy",
            (
                f"Node {sibling.get('node_id', '?')} "
                f"(Q={float(sibling.get('q_value', 0.0)):.4f}, "
                f"reward={float(sibling.get('reward', 0.0)):.4f}, "
                f"variance={float(sibling.get('reward_variance', 0.0)):.4f})"
            ),
            "```python",
            sibling_code.strip(),
            "```",
            "This sibling is the most recent prior child of the same parent (a prior revision attempt).",
            "Compare your proposed strategy to this sibling: learn what was already tried, but do NOT "
            "copy or return the same code.",
        ]
        requirements += [
            "Your agent_strategy must differ meaningfully from the Most Recent Sibling Strategy.",
            "Do not return code that is identical or near-identical to the sibling; propose a distinct revision.",
            "Use the sibling only as a contrast — show a different approach than that prior attempt.",
        ]

    sections += format_forbidden_strategies_section(forbidden_strategies)
    if forbidden_strategies:
        requirements += [
            "Do not return any listed sibling strategy or an AST-equivalent variant.",
            "Stay under the Parent Strategy's principle+form with probe-distinct "
            "shades unless DIAGNOSIS action=replace requires a new menu pair.",
        ]

    if extra_instructions:
        sections += ["", "## Extra instructions", *[f"- {x}" for x in extra_instructions if str(x).strip()]]
        requirements += [str(x).strip() for x in extra_instructions if str(x).strip()]

    from marketsim.LLM.reflection_pipeline.code_edit import (
        complex_parent_edit_instructions,
        parent_is_complex,
    )

    complex_edit = complex_parent_edit_instructions(parent_code)
    if complex_edit:
        sections += ["", "## Complex-parent edit format", *[f"- {x}" for x in complex_edit]]
        requirements += list(complex_edit)

    response_format = (
        "Respond with exactly one ```diff``` unified diff (preferred for this parent) "
        "OR one COMPLETE ```python``` block that keeps every helper you still call."
        if parent_is_complex(parent_code)
        else "Respond with exactly one ```python``` block."
    )
    sections += [
        "",
        "## Requirements",
        *[f"- {req}" for req in requirements],
        "",
        response_format,
    ]
    return "\n".join(sections)


def build_debug_prompt(
    *,
    code: str,
    error: str,
    forbidden_strategies: Optional[List[Dict[str, Any]]] = None,
    force_new_form: bool = False,
    parent_reflection: Optional[str] = None,
    plan_guided_recovery: bool = False,
    critic_mode: str = "default",
    strategy_type: str = "pv",
) -> str:
    error_text = error.strip()
    extra: List[str] = []
    lowered = error_text.lower()
    is_duplicate = "identical or equivalent" in lowered
    is_diagnosis_mismatch = (
        "does not implement parent reflection" in lowered
        or "does not implement principle" in lowered
        or "does not match form_class" in lowered
    )
    plan_guided = bool(
        plan_guided_recovery
        and parent_reflection
        and is_structured_principle_form(critic_mode)
    )
    if plan_guided and (is_duplicate or is_diagnosis_mismatch or forbidden_strategies):
        fields = parse_principle_form_fields(parent_reflection or "")
        extra = [
            "",
            "## Plan-guided fix (implement Parent Reflection DIAGNOSIS)",
            "- Keep the strategy fully deterministic pure Python.",
            "- Do NOT import numpy or random; do NOT add noise/sampling.",
            "- Implement the Parent Reflection DIAGNOSIS below — principle, form_class, "
            "and features_used must appear in real formulas (not comments only).",
            (
                f"- Required principle: {fields.get('principle') or '(from reflection)'}; "
                f"form_class: {fields.get('form_class') or '(from reflection)'}."
            ),
            "- Produce code structurally distinct from sibling strategies listed below.",
            "- Do NOT fall back to a LOCAL coefficient tweak of the parent when the "
            "diagnosis calls for a new form_class or feature set.",
            "- Do NOT return AST-identical code to a sibling or recent tree node.",
        ]
        if force_new_form:
            extra += [
                "- Plateau escape: you may combine features from the diagnosis with a "
                "clearly new control flow vs the parent.",
            ]
    elif is_duplicate or forbidden_strategies:
        extra = [
            "",
            "## Fix constraints",
            "- Keep the strategy fully deterministic pure Python.",
            "- Do NOT import numpy or random; do NOT add noise/sampling.",
            "- Do not return AST-identical code to the parent, a sibling, or another "
            "recent tree node — not comments, renames, whitespace, or argument-order tweaks.",
            "- Stay under the same principle+form as the parent: change the shade "
            "expression so probe shades differ; comments, renames, whitespace, or "
            "AST-identical clones are not enough.",
            "- Do NOT invent a brand-new form_class unless the diagnosis action is "
            "replace or the error clearly requires a new menu pair.",
            "- Avoid sibling AST clones; keep the parent principle+form while "
            "making probe shades differ.",
        ]
        if force_new_form:
            extra += [
                "- If a previous attempt was still a near-clone, change the shade "
                "computation so probe shades differ while keeping the parent "
                "principle+form.",
            ]
    elif is_diagnosis_mismatch and plan_guided:
        fields = parse_principle_form_fields(parent_reflection or "")
        extra = [
            "",
            "## Plan-guided fix (implement Parent Reflection DIAGNOSIS)",
            "- Implement the Parent Reflection DIAGNOSIS — use declared features in code.",
            (
                f"- Required principle: {fields.get('principle') or '(from reflection)'}; "
                f"form_class: {fields.get('form_class') or '(from reflection)'}."
            ),
            "- Keep deterministic pure Python; no numpy/random.",
        ]
    elif (
        "forbidden import" in lowered
        or "numpy" in lowered
        or "random" in lowered
        or "no attribute 'random'" in lowered
    ):
        extra = [
            "",
            "## Fix constraints",
            "- Do NOT import or use numpy or random.",
            "- Do NOT call math.random or any RNG — those are invalid here.",
            "- Keep the strategy fully deterministic pure Python.",
            "- To differ from a prior version, change constants/formulas/thresholds, not noise.",
        ]
    forbidden_section = format_forbidden_strategies_section(forbidden_strategies)
    parent_section: List[str] = []
    if plan_guided and parent_reflection:
        parent_section = [
            "",
            "## Parent Reflection DIAGNOSIS (implement this plan)",
            parent_reflection.strip(),
        ]
    intro = (
        "The following strategy code failed verification or does not implement the "
        "Parent Reflection plan."
        if plan_guided
        else "The following strategy code failed verification."
    )
    fix_line = (
        "Fix by implementing the Parent Reflection DIAGNOSIS and return corrected code."
        if plan_guided
        else "Fix the errors with a local edit and return a corrected version."
    )
    is_helper_completeness = "parent-helper completeness" in lowered
    response_fmt = (
        "Respond with exactly one ```diff``` unified diff against the parent "
        "(preferred) OR one COMPLETE ```python``` module that includes every "
        "helper you still call."
        if is_helper_completeness
        else "Respond with exactly one ```python``` block "
        "(or ```diff``` if fixing a complex multi-helper parent)."
    )
    return "\n".join(
        [
            intro,
            fix_line,
            "",
            "## Error",
            error_text,
            *extra,
            *parent_section,
            *forbidden_section,
            "",
            "## Code (failed / rejected — do not return an AST-equivalent)",
            "```python",
            code.strip(),
            "```",
            "",
            response_fmt,
        ]
    )


def build_reflection_prompt(
    *,
    code: str,
    batch_summary: Dict[str, Any],
    batch_data: Any,
    reward: float,
    reward_variance: float = 0.0,
    include_data: bool = True,
    q_reward_mode: str = "advantage",
    critic_mode: str = "default",
    strategy_type: str = "pv",
    run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    menu_exploration_enabled: bool = True,
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    require_last_hold_comparison: Optional[bool] = None,
    force_menu_replace: bool = False,
    outcome_first: bool = False,
) -> str:
    comparison = batch_summary.get("zi_benchmark_comparison")
    is_zi_mode = is_zi_q_reward_mode(q_reward_mode) or comparison is not None
    structured_curriculum = is_structured_zi_curriculum(critic_mode)
    hypothesis_loop = is_hypothesis_loop_critic(critic_mode)
    structured_principle = is_structured_principle_form(critic_mode)
    if require_last_hold_comparison is None:
        require_last_hold_comparison = bool(
            isinstance(batch_data, dict) and batch_data.get("prior_hold_card")
        )

    if structured_principle:
        sections = list(
            _structured_principle_form_critic_instructions(
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
                spread_input_mode=spread_input_mode,
            )
        )
        sections += [
            "",
            *format_structured_principle_form_facts(
                comparison,
                reward=reward,
                strategy_type=strategy_type,
                spread_input_mode=spread_input_mode,
                lob_input_mode=lob_input_mode,
            ),
        ]
        if menu_exploration_enabled:
            sections += [
                "",
                *format_menu_exploration_section(
                    strategy_type=strategy_type,
                    run_explored_pairs=run_explored_pairs,
                    require_last_hold_comparison=require_last_hold_comparison,
                    compact=True,
                    force_menu_replace=force_menu_replace,
                    lob_input_mode=lob_input_mode,
                    spread_input_mode=spread_input_mode,
                ),
            ]
        sections += [
            "",
            *format_structured_principle_form_diagnosis_template(
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
                spread_input_mode=spread_input_mode,
                require_last_hold_comparison=require_last_hold_comparison,
            ),
        ]

    sections += [
        "",
        "## Strategy",
        "```python",
        code.strip(),
        "```",
        "",
    ]
    if q_reward_mode == "zi_beat_fraction":
        sections.append(f"## Batch Reward (ZI beat fraction): {reward:.4f}")
        sections.append(
            "Beat fraction = (# ZI shade benchmarks beaten) / (total benchmarks) "
            "on identical episode seeds."
        )
    elif is_zi_mode or structured_curriculum or structured_principle:
        if not structured_principle:
            sections.append(
                f"## Batch Reward (advantage): {reward:.4f} (variance: {reward_variance:.4f})"
            )
        if comparison and not structured_principle:
            sections.append(
                f"ZI beat fraction (diagnostic): "
                f"{float(comparison.get('beat_fraction', 0.0)):.4f} "
                f"({int(comparison.get('beat_count', 0))}/"
                f"{int(comparison.get('total_benchmarks', 0))} benchmarks)."
            )
    else:
        sections.append(
            f"## Batch Reward: {reward:.4f} (variance: {reward_variance:.4f})"
        )

    if structured_curriculum or structured_principle:
        if structured_curriculum and comparison:
            sections += format_zi_benchmark_comparison_section(
                comparison,
                q_reward_mode=q_reward_mode,
            )
        sections += format_per_config_breakdown_lines(batch_summary)
        if include_data:
            sections += format_evaluation_trajectory_sections(
                batch_summary=batch_summary,
                batch_data=batch_data if isinstance(batch_data, dict) else {},
                heading_prefix="",
                q_reward_mode=q_reward_mode,
            )
        elif not comparison and not (batch_summary.get("per_config_rewards") or {}):
            fact_name = (
                "Curriculum Facts" if structured_curriculum else "Evaluation Facts"
            )
            sections += [
                "",
                "## Evaluation Data",
                f"No episode trajectories were provided. Use {fact_name} and "
                "strategy code above.",
            ]
        return "\n".join(sections)
