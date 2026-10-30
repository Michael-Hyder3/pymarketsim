from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from marketsim.LLM.lob_features import (
    active_lob_feature_names,
    is_lob_features_mode,
    lob_feature_set_name,
)
from marketsim.LLM.spread_input import (
    DEFAULT_SPREAD_INPUT_MODE,
    is_spread_zip_parity_mode,
    spread_input_schema_lines,
)
from marketsim.LLM.reflection_pipeline.config import ZI_Q_REWARD_MODES


def format_per_config_breakdown_lines(
    batch_summary: Dict[str, Any],
    *,
    heading: str = "Per-Config Advantage (batch breakdown)",
) -> List[str]:
    """Opaque per-market rows: advantage plus surplus/shade when present."""
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
    """Compact operational stats + two episode tapes. No hyperparams, no LOB dumps."""
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


def is_structured_principle_form(critic_mode: str) -> bool:
    return str(critic_mode) == "structured_principle_form"


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
    return spread_input_schema_lines(spread_input_mode)


def _lob_input_schema() -> List[str]:
    return [
        "market_history contains PRIOR market information only (not the current pre-action book):",
        "- 'timestep': int — current decision timestep",
        "- 'lob_snapshots': list of past post-step LOB snapshots (oldest to newest), each with:",
        "  {'best_bid_relative','best_ask_relative','bids_relative','asks_relative'}",
        *LOB_SNAPSHOT_SCHEMA,
        "- 'lob': most recent prior snapshot (same schema as one lob_snapshots entry)",
        "- 'best_bid_relative', 'best_ask_relative', 'bids_relative', 'asks_relative':",
        "  copied from the most recent prior snapshot for convenience",
        "- 'spread_relative': float | null — ask minus bid from most recent prior snapshot",
        "- 'mid_relative': float | null — midpoint from most recent prior snapshot",
        "- 'market_trades': list of executed trades before this decision (up to 10),",
        "  each {'time','price','buyer_id','seller_id','agent_involved'}",
        "- 'total_trades': int — trades seen so far before this decision",
        "- 'recent_trade_prices': list[float] (last up to 10 trade prices)",
        "- 'last_trade_price': float | None",
        "- 'current_pv': float — relative private value for the unit being priced now (large monetary scale)",
        "lob_snapshots contains at most the last 10 prior post-step snapshots (oldest to newest).",
        "On early timesteps the prior book may be empty (null bids/asks, empty depth lists).",
        "total_trades counts all trades seen so far; market_trades lists only the last up to 10.",
        "CRITICAL scale note: LOB price fields are relative to fundamental (~0 scale). "
        "current_pv is a large monetary unit (often hundreds+), NOT a fraction in [0,1]. "
        "Shade outputs MUST be in [0,1]. Handle None bids/asks safely.",
        "Use ONLY market_history. You do not receive market_state or the full private_values schedule.",
    ]


def _lob_features_input_schema(lob_input_mode: str = "features_full") -> List[str]:
    names = active_lob_feature_names(lob_input_mode)
    set_name = lob_feature_set_name(lob_input_mode)
    name_list = ", ".join(names)
    return [
        f"LOB features mode ({set_name}): market_history exposes a FIXED scalar feature vector only.",
        "Signature remains agent_strategy(market_history), but you MUST use only:",
        "- 'timestep': int",
        "- 'current_pv': float (large monetary scale; also summarized as log_pv when present)",
        "- 'features': dict of precomputed finite scalars (also mirrored as top-level keys)",
        f"  features keys (always present, never None): {name_list}",
        "  Semantics:",
        "  - spread_rel: prior ask−bid (0 if missing)",
        "  - mid_rel: prior midpoint (0 if missing)",
        "  - imbalance: (depth_bid − depth_ask) / (depth_bid + depth_ask), else 0",
        "  - depth_bid, depth_ask: prior book side lengths",
        "  - trade_mom: recent trade-price change (0 if <2 trades)",
        "  - log_pv: log1p(|current_pv|)",
        "  - pv_minus_mid: current_pv − mid_rel when mid is defined, else 0",
        "  - has_book: 1.0 if prior book non-empty else 0.0",
        "Use ONLY the listed feature keys — other feature names are not provided in this run.",
        "Nested raw LOB structure is intentionally empty/stripped — do NOT use",
        "bids_relative, asks_relative, lob_snapshots, market_trades, or best_*_relative lists.",
        "Map features → shade in [0,1]. Prefer market_history['features'][name] or top-level scalars.",
        "You do not receive market_state or the full private_values schedule.",
    ]


def _full_input_schema() -> List[str]:
    return [
        "private_values is a list of relative private values for each unit.",
        "market_history is the CURRENT market context at decision time:",
        "- 'timestep': int",
        "- 'market_trades': list of dicts with keys {'time', 'price', 'buyer_id', 'seller_id', 'agent_involved'} (may be empty)",
        "- 'total_trades': int",
        "- 'recent_trade_prices': list[float] (last up to 10 trades)",
        "- 'last_trade_price': float | None",
        "- 'lob': current LOB snapshot with",
        "  {'best_bid_relative','best_ask_relative','bids_relative','asks_relative'}",
        *LOB_SNAPSHOT_SCHEMA,
        "- 'best_bid_relative', 'best_ask_relative', 'bids_relative', 'asks_relative',",
        "  'spread_relative', 'mid_relative': derived from the current LOB snapshot",
        "market_state is the CURRENT point-in-time state:",
        "- 'best_bid', 'best_ask': float | null — relative to fundamental",
        "- 'units_available', 'q_max', 'fundamental_estimate', 'current_pv'",
        "Use market_history['market_trades'] to compute trade-price stats;",
        "if empty, fall back to market_state['fundamental_estimate'] or 0.",
        "If best_bid or best_ask is None, avoid using them directly;",
        "use a safe fallback (e.g., fundamental_estimate) and skip division.",
        "If best_ask <= best_bid, treat spread as zero and avoid division.",
    ]


def _evaluation_section_title(prefix: str, title: str) -> str:
    prefix = prefix.strip()
    if prefix:
        return f"## {prefix} {title}"
    return f"## {title}"


def _zi_benchmark_context_note(
    q_reward_mode: str = "zi_beat_fraction",
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
) -> List[str]:
    reward_line = (
        "Each Q reward is the fraction of fixed ZI shade benchmarks beaten on identical "
        "episode seeds (only the tracked agent differs): reward = beat_count / 10 "
        "(e.g. beat 1 → 0.1, beat 3 → 0.3; not all-or-nothing)."
        if q_reward_mode == "zi_beat_fraction"
        else
        "Q reward is batch advantage (mean LLM surplus minus baseline on the evaluation "
        "batch). ZI shade benchmarks and unit tests are diagnostic: beat_count / "
        "n_benchmarks is reported but is NOT the Q reward."
    )
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        names = ", ".join(active_lob_feature_names(lob_input_mode))
        input_line = (
            f"Use market_history features only (LOB features mode {lob_feature_set_name(lob_input_mode)}): "
            f"map the fixed scalar feature vector ({names}) to shade; do not use raw "
            "lob_snapshots / depth lists or invent a full private_values schedule."
        )
    elif strategy_type == "lob":
        input_line = (
            "Use market_history only (LOB-only mode): map prior book / trade / "
            "current_pv signals to shade; do not invent a full "
            "private_values schedule, inventory fields, or market_state fields "
            "that are not provided."
        )
    elif strategy_type == "spread":
        input_line = (
            "Use market_state only (spread-only mode): map bid/ask, inventory, "
            "current_pv, and shouts to shade; do not invent a full private_values "
            "schedule or LOB depth lists."
        )
    else:
        input_line = (
            "Use private_values, fundamental_estimate, and current_pv (PV-only mode): "
            "map the unit schedule plus the current unit PV and absolute F̃ to shade; "
            "do not use bid-ask spread or LOB fields."
        )
    return [
        "## ZI Benchmark Evaluation (read before revising)",
        reward_line,
        "Each benchmark ZI uses shade_hi as a fixed x_cap; each order draws "
        "shade ~ Uniform(0, shade_hi).",
        "Your strategy outputs shade x in [0,1]: higher x shades more (buyers bid lower; "
        "sellers ask higher).",
        input_line,
        "Each fixed ZI shade range is a soft unit test (PASS/FAIL). The pipeline continues "
        "on failure; use test metrics and trajectory data to diagnose FAILs.",
    ]


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
    """Soft unit-test style report for each fixed ZI benchmark."""
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


def format_advantage_config_test_feedback_lines(
    batch_summary: Dict[str, Any],
    *,
    heading_prefix: str = "",
    failed_only: bool = False,
) -> List[str]:
    """Soft unit-test style report for each market config (advantage mode)."""
    tests = batch_summary.get("config_unit_tests")
    if not tests:
        return []

    total = int(batch_summary.get("config_tests_total", len(tests)))
    passed_count = int(batch_summary.get("config_tests_pass_count", 0))
    failed_tests = batch_summary.get("failed_config_tests") or [
        t for t in tests if t.get("status") == "FAIL"
    ]
    passed_tests = batch_summary.get("passed_config_tests") or [
        t for t in tests if t.get("status") == "PASS"
    ]

    title = _evaluation_section_title(
        heading_prefix,
        "Per-Config Unit Tests (soft checks — pipeline continues on failure)",
    )
    lines = [
        "",
        title,
        (
            f"Passed {passed_count}/{total}. Each test is one market config in the "
            "evaluation batch. Assertion: advantage_mean (LLM surplus minus baseline) > 0."
        ),
        "Treat every FAIL like a failing unit test: diagnose from metrics and trajectory.",
    ]

    if failed_tests:
        lines.append("")
        lines.append("FAILED tests (lowest advantage first):")
        for test in failed_tests:
            gap = float(test["advantage_gap"])
            line = (
                f"  FAIL {test['test_id']} config={test['config_label']!r}: "
                f"advantage={float(test['advantage_mean']):.1f} gap={gap:.1f}"
            )
            if test.get("llm_surplus_mean") is not None:
                line += f" llm={float(test['llm_surplus_mean']):.1f}"
            if test.get("baseline_surplus_mean") is not None:
                line += f" baseline={float(test['baseline_surplus_mean']):.1f}"
            if test.get("avg_shade_mean") is not None:
                line += f" avg_shade={float(test['avg_shade_mean']):.4f}"
            lines.append(line)

    if not failed_only and passed_tests:
        lines.append("")
        lines.append("PASSED tests:")
        for test in passed_tests:
            gap = float(test["advantage_gap"])
            lines.append(
                f"  PASS {test['test_id']} config={test['config_label']!r}: "
                f"advantage=+{gap:.1f}"
            )

    return lines


def format_advantage_config_tests_section(
    batch_summary: Dict[str, Any],
    *,
    heading_prefix: str = "",
) -> List[str]:
    if not batch_summary.get("config_unit_tests"):
        return []
    overall = batch_summary.get("overall") or {}
    shade_line = _format_zi_mean_shade_line(None, overall)
    lines = [
        "",
        _evaluation_section_title(
            heading_prefix,
            "Per-Config Advantage Summary",
        ),
        (
            f"Batch reward (mean advantage): {float(batch_summary.get('reward', 0.0)):.4f} "
            f"(variance: {float(batch_summary.get('reward_variance', 0.0)):.4f}). "
            f"Passed {int(batch_summary.get('config_tests_pass_count', 0))}/"
            f"{int(batch_summary.get('config_tests_total', 0))} config unit tests."
        ),
    ]
    if shade_line:
        lines.append(shade_line)
    lines += format_advantage_config_test_feedback_lines(
        batch_summary,
        heading_prefix=heading_prefix,
    )
    return lines


def format_generator_advantage_config_section(
    batch_summary: Dict[str, Any],
) -> List[str]:
    if not batch_summary.get("config_unit_tests"):
        return []
    overall = batch_summary.get("overall") or {}
    shade_line = _format_zi_mean_shade_line(None, overall)
    lines = [
        "",
        "## Parent Per-Config Results",
        (
            f"Passed {int(batch_summary.get('config_tests_pass_count', 0))}/"
            f"{int(batch_summary.get('config_tests_total', 0))} config unit tests. "
            f"Batch reward = {float(batch_summary.get('reward', 0.0)):.4f}."
        ),
    ]
    if shade_line:
        lines.append(shade_line)
    lines += format_advantage_config_test_feedback_lines(batch_summary, failed_only=False)
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
    if batch_summary.get("per_config_metrics") and not is_zi_mode:
        test_lines = format_advantage_config_test_feedback_lines(
            batch_summary,
            heading_prefix=heading_prefix,
        )
        if test_lines:
            sections += [
                "",
                _evaluation_section_title(
                    heading_prefix,
                    "Per-Config Performance (mean and variance per market config)",
                ),
                json.dumps(batch_summary["per_config_metrics"], indent=2),
                "",
                _evaluation_section_title(
                    heading_prefix,
                    "Overall Batch Performance (pooled across all configs)",
                ),
                json.dumps(batch_summary.get("overall", {}), indent=2),
            ]
            sections += test_lines
        else:
            sections += [
                "",
                _evaluation_section_title(
                    heading_prefix,
                    "Per-Config Performance (mean and variance per market config)",
                ),
                json.dumps(batch_summary["per_config_metrics"], indent=2),
                "",
                _evaluation_section_title(
                    heading_prefix,
                    "Overall Batch Performance (pooled across all configs)",
                ),
                json.dumps(batch_summary.get("overall", {}), indent=2),
            ]
    elif batch_summary and not is_zi_mode:
        sections += format_advantage_config_tests_section(
            batch_summary,
            heading_prefix=heading_prefix,
        )
        if not batch_summary.get("config_unit_tests"):
            sections += [
                "",
                _evaluation_section_title(
                    heading_prefix,
                    "Batch Summary (Aggregated metrics; means and variances)",
                ),
                json.dumps(batch_summary, indent=2),
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
            # Legacy median-episode dump (tests / old stored nodes).
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


def parent_reward_section_title(q_reward_mode: str) -> str:
    if q_reward_mode == "zi_beat_fraction":
        return "Parent Q Reward (ZI beat fraction)"
    if q_reward_mode == "zi_advantage":
        return "Parent Batch Reward (advantage)"
    return "Parent Batch Reward"


def format_parent_reward_section(
    parent_performance: Dict[str, Any],
    *,
    q_reward_mode: str = "advantage",
) -> List[str]:
    reward = float(parent_performance.get("reward", 0.0))
    reward_var = float(parent_performance.get("reward_variance", 0.0))
    sections = [
        "",
        f"## {parent_reward_section_title(q_reward_mode)}",
        f"reward={reward:.4f} (variance={reward_var:.4f})",
    ]
    if q_reward_mode == "zi_beat_fraction":
        sections.append(
            "Higher is better. Reward is the fraction of fixed ZI shade benchmarks "
            "the parent beat on identical episode seeds."
        )
        comparison = parent_performance.get("zi_benchmark_comparison")
        overall = parent_performance.get("overall")
        shade_line = _format_zi_mean_shade_line(comparison, overall)
        if shade_line:
            sections.append(shade_line)
    elif q_reward_mode == "zi_advantage":
        sections.append(
            "Higher is better. Q reward is batch advantage (LLM surplus minus baseline). "
            "ZI benchmark beat fraction is reported separately as diagnostic context."
        )
        comparison = parent_performance.get("zi_benchmark_comparison")
        if comparison:
            sections.append(
                f"ZI beat fraction (diagnostic): "
                f"{float(comparison.get('beat_fraction', 0.0)):.4f} "
                f"({int(comparison.get('beat_count', 0))}/"
                f"{int(comparison.get('total_benchmarks', 0))} benchmarks)."
            )
    return sections


def format_generator_zi_benchmark_section(
    comparison: Dict[str, Any],
    *,
    q_reward_mode: str = "zi_beat_fraction",
) -> List[str]:
    if not comparison:
        return []
    beaten = comparison.get("beaten_shade_ranges") or []
    not_beaten = comparison.get("not_beaten_shade_ranges") or []
    shade_line = _format_zi_mean_shade_line(comparison, None)
    beat_fraction = float(comparison.get("beat_fraction", 0.0))
    if q_reward_mode == "zi_advantage":
        summary = (
            f"Beat {int(comparison.get('beat_count', 0))}/"
            f"{int(comparison.get('total_benchmarks', 0))} benchmarks "
            f"(beat fraction {beat_fraction:.4f} diagnostic). "
            f"LLM mean surplus = {float(comparison.get('llm_mean_surplus', 0.0)):.4f}."
        )
    else:
        summary = (
            f"Beat {int(comparison.get('beat_count', 0))}/"
            f"{int(comparison.get('total_benchmarks', 0))} benchmarks "
            f"(Q = {beat_fraction:.4f}). "
            f"LLM mean surplus = {float(comparison.get('llm_mean_surplus', 0.0)):.4f}."
        )
    lines = [
        "",
        "## Parent ZI Benchmark Results (same seeds; only tracked agent differed)",
        summary,
    ]
    if shade_line:
        lines.append(shade_line)
    test_lines = format_zi_benchmark_test_feedback_lines(comparison, failed_only=False)
    if test_lines:
        lines += test_lines
    else:
        lines += [
            "Shade ranges the parent BEAT:",
            json.dumps(beaten, indent=2) if beaten else "[]",
            "Shade ranges the parent DID NOT beat (priority targets for revision):",
            json.dumps(not_beaten, indent=2) if not_beaten else "[]",
            "Per-benchmark mean surplus (LLM vs each ZI shade cohort):",
            json.dumps(comparison.get("benchmarks", []), indent=2),
        ]
    return lines


def _advantage_critic_instructions() -> List[str]:
    return [
        "You are reviewing a strategy evaluated in advantage mode.",
        "Batch reward is mean LLM surplus minus baseline surplus across episodes "
        "(higher is better). Each market config is also a soft unit test "
        "(PASS if per-config advantage_mean > 0).",
        "Provide a 5-part note with the exact headings:",
        "Environment:",
        "Performance:",
        "Mechanism:",
        "Principle:",
        "Fix:",
        "",
        "Use these rules (keep each section brief):",
        "- Environment: evaluation setting and which market configs mattered (cite config labels).",
        "- Performance: batch reward, per-config advantage, mean shade if shown, and cite "
        "which unit tests FAILED vs PASSED (use test_id and config_label).",
        "- Mechanism: cite at least TWO moments from the median-episode trajectory "
        "(time, shade, current_pv, best_bid/ask, transacted). Explain why those moments "
        "are consistent with specific FAILED config tests (cite test_id and metrics).",
        "- Principle: one reusable rule linking inputs to shade or bidding that would flip "
        "the most important FAILED tests to PASS.",
        "- Fix: implement the Principle in agent_strategy; name which FAILED test_ids "
        "this change targets.",
        "Do not give vague advice like 'tune shading'; ground the Mechanism in trajectory data.",
    ]


def _zi_critic_instructions(q_reward_mode: str = "zi_beat_fraction") -> List[str]:
    if q_reward_mode == "zi_advantage":
        reward_lines = [
            "You are reviewing a strategy evaluated with ZI benchmark unit tests.",
            "Q reward is batch advantage (mean LLM surplus minus baseline; higher is better).",
            "ZI beat fraction and per-benchmark unit tests are diagnostic (not the Q reward).",
        ]
        performance_line = (
            "- Performance: batch advantage reward, beat fraction, LLM mean surplus, "
            "mean shade if shown, and cite which ZI unit tests FAILED vs PASSED."
        )
    else:
        reward_lines = [
            "You are reviewing a strategy evaluated in ZI beat-fraction mode.",
            "The batch reward is beat_count / total_benchmarks on identical episode seeds "
            "(0 = beat none, 1 = beat every benchmark; partial credit for each one beaten).",
        ]
        performance_line = (
            "- Performance: beat fraction, LLM mean surplus, mean shade if shown, and cite "
            "which unit tests FAILED vs PASSED (use test_id and shade_range)."
        )
    return [
        *reward_lines,
        "Each benchmark opponent is a ZI agent with fixed x_cap = shade_hi; "
        "each order draws shade ~ Uniform(0, shade_hi).",
        "Each benchmark is also a soft unit test (PASS if LLM surplus beats that ZI cohort).",
        "Provide a 5-part note with the exact headings:",
        "Environment:",
        "Performance:",
        "Mechanism:",
        "Principle:",
        "Fix:",
        "",
        "Use these rules (keep each section brief):",
        "- Environment: evaluation setting and which ZI shade benchmarks mattered.",
        performance_line,
        "- Mechanism: cite at least TWO moments from the median-episode trajectory "
        "(time, shade, current_pv, best_bid/ask, transacted). Explain why those moments "
        "are consistent with specific FAILED benchmark tests.",
        "- Principle: one reusable rule linking private_values to shade that would improve "
        "general performance against varied opponents.",
        "- Fix: describe one concrete revision direction for the next generator step; "
        "do not paste full replacement code.",
        "Do not give vague advice like 'tune shading'; ground the Mechanism in trajectory data.",
    ]


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


def infer_curriculum_shade_diagnosis(
    comparison: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministic shade-level diagnosis from ZI comparison (batch-level)."""
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

    shade_error_direction = "reshape_pv_map"
    error_type = "pv_mapping_wrong"
    fix_class = "change_pv_feature"
    suggested_target: Optional[float] = None

    if llm_x_cap is not None:
        x_cap = float(llm_x_cap)
        if failed_his and x_cap < min(failed_his):
            shade_error_direction = "shade_more"
            error_type = "level_too_low"
            fix_class = "add_offset"
            suggested_target = min(1.0, max(min(failed_his) * 2.0, x_cap + 0.01))
        elif beaten_his and x_cap > max(beaten_his) and not failed_his:
            shade_error_direction = "shade_less"
            error_type = "level_too_high"
            fix_class = "scale_base"
            suggested_target = max(0.0, max(beaten_his) * 0.5)
        elif failed_his:
            center = sum(failed_his) / len(failed_his)
            if x_cap <= center:
                shade_error_direction = "shade_more"
                error_type = "level_too_low"
                fix_class = "add_offset"
            else:
                shade_error_direction = "shade_less"
                error_type = "level_too_high"
                fix_class = "scale_base"
            suggested_target = center

    return {
        "requested_x_cap": float(llm_x_cap) if llm_x_cap is not None else None,
        "llm_trade_shade": float(llm_trade_shade) if llm_trade_shade is not None else None,
        "highest_beaten_shade_hi": max(beaten_his) if beaten_his else None,
        "lowest_failed_shade_hi": min(failed_his) if failed_his else None,
        "shade_error_direction": shade_error_direction,
        "error_type": error_type,
        "fix_class": fix_class,
        "suggested_target": suggested_target,
    }


def infer_per_config_shade_directions(
    comparison: Optional[Dict[str, Any]],
    batch_data: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Per-market shade direction from each config's mean requested x_cap."""
    if not comparison or not batch_data:
        return {}
    beaten_his, failed_his = _benchmark_pass_fail_lists(comparison)
    from marketsim.LLM.reflection_pipeline.metrics import summarize_per_config_performance

    per_config = summarize_per_config_performance(batch_data)
    directions: Dict[str, str] = {}
    for label, metrics in per_config.items():
        x_cap_raw = metrics.get("avg_requested_x_cap_mean")
        if x_cap_raw is None:
            continue
        x_cap = float(x_cap_raw)
        if failed_his and x_cap < min(failed_his):
            directions[label] = "shade_more"
        elif beaten_his and x_cap > max(beaten_his) and not failed_his:
            directions[label] = "shade_less"
        elif failed_his:
            center = sum(failed_his) / len(failed_his)
            directions[label] = "shade_more" if x_cap <= center else "shade_less"
        else:
            directions[label] = "reshape_pv_map"
    return directions


def infer_curriculum_consensus(per_config_directions: Dict[str, str]) -> str:
    if not per_config_directions:
        return "n/a"
    unique = set(per_config_directions.values())
    if len(unique) == 1:
        return "aligned"
    return "mixed"


def format_curriculum_diagnosis_template(
    comparison: Optional[Dict[str, Any]],
    *,
    batch_data: Optional[Dict[str, Any]] = None,
    multi_config: bool = False,
) -> List[str]:
    """Pre-filled DIAGNOSIS block the critic must echo and refine."""
    diag = infer_curriculum_shade_diagnosis(comparison)
    per_config_directions = (
        infer_per_config_shade_directions(comparison, batch_data) if multi_config else {}
    )
    consensus = (
        infer_curriculum_consensus(per_config_directions)
        if multi_config
        else "n/a"
    )
    if multi_config and consensus == "mixed":
        diag = dict(diag)
        diag["shade_error_direction"] = "reshape_pv_map"
        diag["error_type"] = "pv_mapping_wrong"
        diag["fix_class"] = "change_pv_feature"

    def _fmt_optional(value: Optional[float]) -> str:
        return f"{value:g}" if value is not None else "none"

    lines = [
        "## Required DIAGNOSIS Output (start your reply with this block)",
        "Copy the field values below from computed facts unless you justify a change "
        "in MECHANISM. Use the exact labels.",
        "",
        "DIAGNOSIS:",
        f"  requested_x_cap: {_fmt_optional(diag['requested_x_cap'])}",
        f"  highest_beaten_shade_hi: {_fmt_optional(diag['highest_beaten_shade_hi'])}",
        f"  lowest_failed_shade_hi: {_fmt_optional(diag['lowest_failed_shade_hi'])}",
        f"  shade_error_direction: {diag['shade_error_direction']}",
        f"  error_type: {diag['error_type']}",
        f"  fix_class: {diag['fix_class']}",
        f"  consensus: {consensus}",
    ]
    if multi_config and per_config_directions:
        lines.append("  per_config_directions:")
        for label in sorted(per_config_directions):
            lines.append(f"    {label}: {per_config_directions[label]}")
    if diag["suggested_target"] is not None:
        lines.append(
            f"  suggested_target_x_cap: {diag['suggested_target']:.4f} "
            "(DIAGNOSTIC HINT — do not paste literally into code)"
        )
    lines += [
        "",
        "MECHANISM: (max 2 sentences; cite requested_x_cap and one failed test_id)",
        "FIX: (max 2 sentences; name fix_class; no full replacement code)",
        "",
        "Do NOT write an Environment section. Do NOT invent estimated_mean_shade; "
        "use requested_x_cap only.",
    ]
    return lines


_CURRICULUM_DIRECTIONS = frozenset({"shade_more", "shade_less", "reshape_pv_map"})
_CURRICULUM_ERROR_TYPES = frozenset(
    {"level_too_low", "level_too_high", "pv_mapping_wrong"}
)
_CURRICULUM_FIX_CLASSES = frozenset(
    {"scale_base", "add_offset", "change_pv_feature", "change_formula"}
)


def validate_structured_curriculum_reflection(
    text: str,
    *,
    multi_config: bool = False,
) -> Optional[str]:
    """Return an error message if the critic output is missing required structure."""
    if not text or not text.strip():
        return "Empty reflection"
    lowered = text.lower()
    required_markers = [
        "diagnosis:",
        "requested_x_cap:",
        "shade_error_direction:",
        "error_type:",
        "fix_class:",
        "consensus:",
        "mechanism:",
        "fix:",
    ]
    missing = [marker for marker in required_markers if marker not in lowered]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"

    direction_match = re.search(
        r"shade_error_direction:\s*(\S+)", text, flags=re.IGNORECASE
    )
    if not direction_match or direction_match.group(1) not in _CURRICULUM_DIRECTIONS:
        return "shade_error_direction must be shade_more, shade_less, or reshape_pv_map"

    error_match = re.search(r"error_type:\s*(\S+)", text, flags=re.IGNORECASE)
    if not error_match or error_match.group(1) not in _CURRICULUM_ERROR_TYPES:
        return "error_type must be level_too_low, level_too_high, or pv_mapping_wrong"

    fix_match = re.search(r"fix_class:\s*(\S+)", text, flags=re.IGNORECASE)
    if not fix_match or fix_match.group(1) not in _CURRICULUM_FIX_CLASSES:
        return "fix_class must be scale_base, add_offset, change_pv_feature, or change_formula"

    if multi_config and "per_config_directions:" not in lowered:
        return "Missing per_config_directions for multi-config batch"

    if re.search(r"pv_mean\s*>=\s*0\.[0-9]", text, flags=re.IGNORECASE):
        return "Do not propose pv_mean >= 0.x thresholds (private_values are not [0,1])"

    if "environment:" in lowered:
        return "Do not include an Environment section; use the DIAGNOSIS block only"

    return None


# ---------------------------------------------------------------------------
# structured_principle_form critic mode (principle + form_class menus)
# ---------------------------------------------------------------------------

_PRINCIPLE_FORM_PRINCIPLES = frozenset(
    {
        "inventory_remaining_units",
        "pv_rank_or_distance_to_best",
        "signed_asymmetry",
        "nonlinear_abs_pv",
        "fill_surplus_tradeoff",
    }
)
_PRINCIPLE_FORM_CLASSES = frozenset(
    {
        "ratio",
        "rank",
        "schedule_index",
        "signed_power",
        "piecewise_by_sign",
        "inventory_weighted",
    }
)

# LOB-only menus: only use fields present in market_history (prior book, trades,
# current_pv). Parallel structure to PV menus, without inventory fields.
_PRINCIPLE_FORM_PRINCIPLES_LOB = frozenset(
    {
        "prior_spread_or_mid",
        "book_imbalance",
        "depth_asymmetry",
        "trade_momentum",
        "current_pv_vs_mid",
    }
)
_PRINCIPLE_FORM_CLASSES_LOB = frozenset(
    {
        "ratio",
        "threshold",
        "piecewise_by_sign",
        "ema_over_snapshots",
        "signed_power",
    }
)

# Spread-only menus: market_state fields (bid/ask, inventory, PV, shouts tape).
# Distinct from PV menus (private_values schedule only) and LOB menus (order memory).
_PRINCIPLE_FORM_PRINCIPLES_SPREAD = frozenset(
    {
        "bid_ask_spread_width",
        "inventory_remaining_fraction",
        "pv_vs_mid",
        "pv_vs_touch",
        "current_pv_magnitude",
        # Shout-tape / ZIP observation stream (market_state['shouts'])
        "recent_accept_reject",
        "shout_competitiveness",
    }
)
_PRINCIPLE_FORM_CLASSES_SPREAD = frozenset(
    {
        "ratio",
        "threshold",
        "piecewise_by_sign",
        "clamp_linear",
        "inventory_weighted",
        "ema_over_window",  # aggregate over shouts (accept/reject history)
    }
)

# Experiment menu presets. Per strategy_type, value is either:
# - frozenset: override principles only (legacy lob_e2 / spread_e5)
# - dict with optional "principles" and/or "form_classes" frozensets
_PRINCIPLE_MENU_PRESETS: Dict[str, Dict[str, Any]] = {
    "lob_e2": {
        "lob": frozenset(
            {"prior_spread_or_mid", "book_imbalance", "current_pv_vs_mid"}
        ),
    },
    "spread_e5": {
        "spread": frozenset(
            {"bid_ask_spread_width", "inventory_remaining_fraction", "pv_vs_mid"}
        ),
    },
    "min2": {
        "pv": {
            "principles": frozenset(
                {"inventory_remaining_units", "pv_rank_or_distance_to_best"}
            ),
            "form_classes": frozenset({"ratio", "rank"}),
        },
        "lob": {
            "principles": frozenset({"book_imbalance", "current_pv_vs_mid"}),
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
    """Set the active principle-menu preset for this process (None = full menus)."""
    global _ACTIVE_PRINCIPLE_MENU_PRESET
    if preset is not None and preset not in _PRINCIPLE_MENU_PRESETS:
        known = ", ".join(list_principle_menu_presets())
        raise ValueError(f"Unknown principle_menu_preset '{preset}'. Known: {known}")
    _ACTIVE_PRINCIPLE_MENU_PRESET = preset


def get_principle_menu_preset() -> Optional[str]:
    return _ACTIVE_PRINCIPLE_MENU_PRESET


def _apply_menu_preset_override(
    override: Any,
    principles: frozenset,
    form_classes: frozenset,
) -> Tuple[frozenset, frozenset]:
    """Apply a preset override (principles-only frozenset or full dict)."""
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
) -> Tuple[frozenset, frozenset]:
    """Return (principles, form_classes) menus for the given strategy type."""
    if strategy_type == "lob":
        principles, form_classes = (
            _PRINCIPLE_FORM_PRINCIPLES_LOB,
            _PRINCIPLE_FORM_CLASSES_LOB,
        )
    elif strategy_type == "spread":
        principles, form_classes = (
            _PRINCIPLE_FORM_PRINCIPLES_SPREAD,
            _PRINCIPLE_FORM_CLASSES_SPREAD,
        )
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


def parse_principle_form_fields(text: str) -> Dict[str, Optional[str]]:
    """Extract principle / form_class from a DIAGNOSIS reflection (best-effort)."""
    principle_match = re.search(r"principle:\s*(\S+)", text or "", flags=re.IGNORECASE)
    form_match = re.search(r"form_class:\s*(\S+)", text or "", flags=re.IGNORECASE)
    return {
        "principle": principle_match.group(1).strip() if principle_match else None,
        "form_class": form_match.group(1).strip() if form_match else None,
    }


def parse_same_failure_as_last_hold(text: str) -> Optional[str]:
    """Return 'yes' or 'no' from DIAGNOSIS, else None."""
    match = re.search(
        r"same_failure_as_last_hold:\s*(yes|no)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return match.group(1).strip().lower()


def parse_diagnosis_action(text: str) -> Optional[str]:
    """Return 'refine' or 'replace' from DIAGNOSIS, else None."""
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


def last_hold_generator_instructions(
    *,
    same_failure: str,
    parent_principle: Optional[str] = None,
    parent_form: Optional[str] = None,
) -> List[str]:
    if same_failure == "yes":
        return diagnosis_action_generator_instructions(
            action="refine",
            parent_principle=parent_principle,
            parent_form=parent_form,
        )
    if same_failure == "no":
        return diagnosis_action_generator_instructions(
            action="replace",
            parent_principle=parent_principle,
            parent_form=parent_form,
        )
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


def menu_hypothesis_pairs(strategy_type: str = "pv") -> frozenset[Tuple[str, str]]:
    """All valid (principle, form_class) pairs from the active menu."""
    principles, form_classes = principle_form_menus(strategy_type)
    return frozenset(
        (principle, form_class)
        for principle in sorted(principles)
        for form_class in sorted(form_classes)
    )


def collect_hypothesis_pairs_from_reflections(
    reflections: Iterable[str],
) -> Set[Tuple[str, str]]:
    """Extract explored (principle, form_class) pairs from prior reflections."""
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
) -> Dict[str, Any]:
    """Summarize how fully the critic explored the principle menu in a run."""
    reflection_list = [str(r) for r in reflections if (r or "").strip()]
    menu_pairs = menu_hypothesis_pairs(strategy_type)
    principles, form_classes = principle_form_menus(strategy_type)

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
    compact: bool = True,
) -> List[str]:
    """Prompt section nudging the critic to cover the full principle menu.

    compact=True (default): one-line coverage stats — enough for refine-first loops.
    compact=False: full unexplored pair list (use on plateau replace escapes).
    """
    menu_pairs = menu_hypothesis_pairs(strategy_type)
    explored = set(run_explored_pairs or ())
    unexplored = sorted(menu_pairs - explored)
    principles, _ = principle_form_menus(strategy_type)
    explored_principles = sorted({principle for principle, _ in explored})
    missing_principles = sorted(set(principles) - set(explored_principles))

    lines = [
        "## Menu exploration",
        f"Covered {len(explored)}/{len(menu_pairs)} (principle, form_class) pairs so far.",
        "Prefer action=refine when the case cards show the same failure mode. "
        "Use action=replace only when the hypothesis itself is wrong.",
    ]
    if missing_principles and not compact:
        lines.append(
            "Principles not yet used: " + ", ".join(missing_principles)
        )

    if compact:
        if unexplored:
            lines.append(
                f"{len(unexplored)} pairs still unexplored — available on replace/plateau."
            )
        else:
            lines.append("Menu fully explored; refine or replace any pair.")
    elif unexplored:
        lines += [
            "",
            "If action=replace, pick from this unexplored list:",
        ]
        for principle, form_class in unexplored:
            lines.append(f"- {principle} + {form_class}")
    else:
        lines += [
            "",
            "Menu fully explored; action=replace may repeat a pair but must still "
            "differ from the parent unless refining.",
        ]
    if require_last_hold_comparison:
        lines += [
            "",
            "LAST HOLD: same_failure_as_last_hold=yes → action=refine; "
            "no → action=replace.",
        ]
    return lines


def _slim_plateau_performance(performance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only compact scalars for plateau prompts (avoid huge ZI tables)."""
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
    max_code_chars: int = 2500,
    max_reflection_chars: Optional[int] = None,
) -> str:
    """
    Critic prompt for hypothesis revision after local refine stalls.

    Asks for a new principle/form (learning-flavored escape), not a coefficient tweak.
    Kept deliberately compact for 8k-context models.
    """
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
    else:
        sections += [
            "",
            "## Required output",
            "Write a short reflection with headings Mechanism, Principle, Fix.",
            "Principle must state a revised shading hypothesis different from the parent.",
            "Fix must describe a structural change (new features / control flow / form), "
            "not coefficient retuning.",
        ]
    return "\n".join(sections)


def hypothesis_pair_differs_from_parent(
    reflection: str,
    parent_reflection: str,
) -> bool:
    """True when at least one of principle or form_class differs from the parent."""
    parent_fields = parse_principle_form_fields(parent_reflection or "")
    new_fields = parse_principle_form_fields(reflection or "")
    parent_principle = (parent_fields.get("principle") or "").strip().lower()
    parent_form = (parent_fields.get("form_class") or "").strip().lower()
    new_principle = (new_fields.get("principle") or "").strip().lower()
    new_form = (new_fields.get("form_class") or "").strip().lower()
    if not new_principle or not new_form:
        return False
    return new_principle != parent_principle or new_form != parent_form


def build_hypothesis_jump_prompt(
    *,
    parent_code: str,
    parent_reflection: str,
    parent_performance: Optional[Dict[str, Any]],
    critic_mode: str = "default",
    generator_batch_summary: Optional[Dict[str, Any]] = None,
    strategy_type: str = "pv",
    run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    max_code_chars: int = 2500,
    max_reflection_chars: Optional[int] = None,
) -> str:
    """Critic prompt for a probabilistic UCB exploration jump to a new hypothesis."""
    fields = parse_principle_form_fields(parent_reflection or "")
    parent_principle = fields.get("principle") or "unknown"
    parent_form = fields.get("form_class") or "unknown"
    code = (parent_code or "").strip()
    if len(code) > max_code_chars:
        code = code[: max_code_chars - 20].rstrip() + "\n# ... truncated ..."
    reflection = (parent_reflection or "").strip() or "(none)"
    if max_reflection_chars is not None and len(reflection) > max_reflection_chars:
        reflection = reflection[: max_reflection_chars - 20].rstrip() + "\n... truncated ..."
    sections: List[str] = [
        "## Hypothesis exploration jump",
        (
            "UCB selected this parent for tree search. Explore a NEW scientific "
            "hypothesis from this lineage — not a coefficient tweak."
        ),
        "Change at least one of principle or form_class vs the parent hypothesis.",
        "",
        f"Parent principle: {parent_principle}",
        f"Parent form_class: {parent_form}",
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
            ),
            "",
            *format_structured_principle_form_diagnosis_template(strategy_type=strategy_type),
            "",
            "## Hypothesis jump constraints",
            "- Change at least one of principle or form_class vs the parent hypothesis.",
            "- Prefer a structurally different map, not coefficient retuning.",
            "- why_this_fails should explain limits of the parent's current hypothesis.",
            "- proposed_change must describe the new principle/form experiment.",
            "- FIX must implement the new principle + form_class (no full code).",
            "- Set action: replace (a hypothesis jump is not a refine).",
        ]
    else:
        sections += [
            "",
            "## Required output",
            "Write a short reflection with headings Mechanism, Principle, Fix.",
            "Principle must state a revised shading hypothesis different from the parent.",
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
    """Shallow code-shape tags for lightweight family grouping."""
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
    """Lightweight family key from form_class + shallow code-shape tags."""
    meta = metadata or {}
    if meta.get("family_id"):
        return str(meta["family_id"])
    fields = parse_principle_form_fields(reflection or "")
    form = str(meta.get("form_class") or fields.get("form_class") or "unknown")
    principle = str(meta.get("principle") or fields.get("principle") or "unknown")
    tags = _code_shape_tags(code)
    tag_suffix = "+".join(tags) if tags else "plain"
    return f"{principle}:{form}:{tag_suffix}"


def build_combination_revision_prompt(
    *,
    parent_code: str,
    parent_reflection: str,
    family_sources: List[Dict[str, Any]],
    non_improve_streak: int,
    critic_mode: str = "default",
    strategy_type: str = "pv",
    max_code_chars: int = 1800,
    max_reflection_chars: Optional[int] = None,
) -> str:
    """
    Critic prompt for gated combination of two strategy families on plateau escape.
    """
    parent = (parent_code or "").strip()
    if len(parent) > max_code_chars:
        parent = parent[: max_code_chars - 20].rstrip() + "\n# ... truncated ..."
    reflection = (parent_reflection or "").strip() or "(none)"
    if max_reflection_chars is not None and len(reflection) > max_reflection_chars:
        reflection = reflection[: max_reflection_chars - 20].rstrip() + "\n... truncated ..."
    sections: List[str] = [
        "## Plateau escape — combine strategy families",
        (
            f"Local refinements of the selected parent failed to improve mean surplus for "
            f"{int(non_improve_streak)} consecutive children."
        ),
        "Instead of another single-family revision, propose a **gated combination** of two "
        "existing families from the tree.",
        "Combine by conditional branching (if gate then family A else family B), "
        "**not** by averaging coefficients or blending into one formula.",
        "",
        "## Parent Strategy (stalled lineage)",
        "```python",
        parent,
        "```",
        "",
        "## Parent reflection",
        reflection,
    ]
    for idx, source in enumerate(family_sources[:2], start=1):
        fam_id = source.get("family_id", "unknown")
        form = source.get("form_class", "unknown")
        principle = source.get("principle", "unknown")
        surplus = source.get("mean_surplus")
        code = str(source.get("code") or "").strip()
        if len(code) > max_code_chars:
            code = code[: max_code_chars - 20].rstrip() + "\n# ... truncated ..."
        sections += [
            "",
            f"## Source family {idx} ({fam_id})",
            f"principle: {principle}",
            f"form_class: {form}",
            f"mean_surplus: {surplus}",
            "```python",
            code,
            "```",
        ]
    if is_structured_principle_form(critic_mode):
        sections += [
            "",
            *format_structured_principle_form_diagnosis_template(
                strategy_type=strategy_type
            ),
            "",
            "## Combination constraints",
            "- form_class should be piecewise/gated (e.g. piecewise_by_sign, gated_ratio).",
            "- parent_form must name the gate condition and both branch formulas.",
            "- why_parent_fails: explain why single-family local edits stalled.",
            "- diversity_vs_parent: describe the gate + which family each branch uses.",
            "- FIX: implement gated composition (no full code).",
            "- Do NOT return a pure weighted average of the two families.",
        ]
    else:
        sections += [
            "",
            "## Required output",
            "Write Mechanism, Principle, Fix.",
            "Principle: gated combination of the two source families.",
            "Fix: gate condition + branch formulas (structural, not coefficient blend).",
        ]
    return "\n".join(sections)


def combination_generator_instructions(
    *,
    family_a_id: Optional[str] = None,
    family_b_id: Optional[str] = None,
    strategy_type: str = "pv",
) -> List[str]:
    fam_a = family_a_id or "family_A"
    fam_b = family_b_id or "family_B"
    if strategy_type == "lob":
        gate_line = (
            "Use a simple LOB gate (spread vs mid, book imbalance sign, depth ratio, "
            "or current_pv vs mid)."
        )
    elif strategy_type == "spread":
        gate_line = (
            "Use a simple spread gate (bid_ask spread width, units_available/q_max, "
            "current_pv vs mid of best_bid/best_ask, or current_pv vs touch)."
        )
    else:
        gate_line = (
            "Use a simple PV gate (pv_mean vs pv_max, sign, dispersion, inventory proxy)."
        )
    return [
        "COMBINATION MODE (plateau escape): combine two strategy families with gating.",
        f"Source families: {fam_a} and {fam_b}.",
        gate_line,
        "if <gate>: use family A shading formula; else: use family B shading formula.",
        "Do NOT blend with a single weighted average — branches must differ structurally.",
        "Keep deterministic pure Python; clamp final shade to [0, 1].",
        "Follow Parent Reflection DIAGNOSIS gate + FIX.",
    ]


def validate_structured_principle_form_reflection(
    text: str,
    *,
    strategy_type: str = "pv",
    parent_reflection: Optional[str] = None,
    sibling_reflections: Optional[List[str]] = None,
    run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    require_last_hold_comparison: bool = False,
) -> Optional[str]:
    """Return an error message if the principle/form critic output is invalid."""
    if not text or not text.strip():
        return "Empty reflection"
    lowered = text.lower()
    required_markers = [
        "diagnosis:",
        "principle:",
        "form_class:",
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

    principles, form_classes = principle_form_menus(strategy_type)
    principle_match = re.search(r"principle:\s*(\S+)", text, flags=re.IGNORECASE)
    if not principle_match or principle_match.group(1) not in principles:
        return "principle must be one of: " + ", ".join(sorted(principles))

    form_match = re.search(r"form_class:\s*(\S+)", text, flags=re.IGNORECASE)
    if not form_match or form_match.group(1) not in form_classes:
        return "form_class must be one of: " + ", ".join(sorted(form_classes))

    proposed = _hypothesis_pair_from_reflection(text)
    if proposed is not None:
        parent_pair = _hypothesis_pair_from_reflection(parent_reflection or "")
        refine_parent = action == "refine" and parent_pair is not None
        if refine_parent:
            if proposed != parent_pair:
                return (
                    "action=refine requires keeping the parent "
                    "(principle, form_class) and re-implementing under that pair — "
                    "do not pick a new pair."
                )
        else:
            menu_pairs = menu_hypothesis_pairs(strategy_type)
            explored = set(run_explored_pairs or ())
            unexplored = menu_pairs - explored
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
        if re.search(r"private_values\s*=", text, flags=re.IGNORECASE):
            return "Do not invent a full private_values schedule in spread-only mode"
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
    """Secondary ZI surplus gaps where benchmarks beat the LLM (positive gap)."""
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
        "Primary objective is batch advantage. Use these gaps in why_parent_fails "
        "when advantage is weak — do NOT hardcode shade_hi or chase shade levels.",
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


def format_structured_principle_form_facts(
    comparison: Optional[Dict[str, Any]],
    *,
    reward: Optional[float] = None,
    strategy_type: str = "pv",
    include_zi_gaps: bool = False,
) -> List[str]:
    """Advantage-first facts for principle/form critic (no shade-level teaching).

    By default omits ZI surplus-gap tables — those pull the model toward matching
    shade_hi. Keep a single surplus / beat-count line when comparison is present.
    """
    lines = [
        "## Evaluation Facts",
        (
            f"Batch advantage (PRIMARY objective): {float(reward):.4f}."
            if reward is not None
            else "Batch advantage (PRIMARY objective): (not reported)."
        ),
    ]
    if comparison:
        llm_shade = comparison.get("llm_mean_requested_x_cap")
        if llm_shade is None:
            llm_shade = comparison.get("llm_mean_shade")
        lines += [
            (
                f"LLM mean shade output: {float(llm_shade):.4f}."
                if llm_shade is not None
                else "LLM mean shade output: (not reported)."
            ),
            f"LLM mean surplus: {float(comparison.get('llm_mean_surplus', 0.0)):.2f}.",
            (
                f"ZI beat count (diagnostic only): "
                f"{int(comparison.get('beat_count', 0))}/"
                f"{int(comparison.get('total_benchmarks', 0))}."
            ),
        ]
        if include_zi_gaps:
            lines += ["", *format_zi_surplus_gap_facts(comparison)]
    else:
        lines.append("No ZI comparison was provided.")

    lines += [
        "Propose a functional-form change that can improve batch advantage.",
        "Do NOT propose shade_more / shade_less or matching ZI shade floats.",
    ]
    if strategy_type == "lob":
        lines.append(
            "LOB-only: use market_history only. Shade output must be in [0,1]. "
            "Do not invent a private_values schedule."
        )
    elif strategy_type == "spread":
        lines.append(
            "Use market_state only (including shouts). Shade output must be in [0,1]. "
            "None bids/asks are absent — do not treat None as 0.0 in arithmetic. "
            "Do not invent a PV schedule or LOB depth."
        )
    else:
        lines.append(
            "Use private_values, fundamental_estimate, and current_pv only. "
            "Shade output must be in [0,1]. "
            "Private values are large monetary units, not fractions."
        )
    return lines


def format_structured_principle_form_diagnosis_template(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
    require_last_hold_comparison: bool = False,
) -> List[str]:
    """Empty DIAGNOSIS scaffold the critic must fill (no precomputed shade direction)."""
    principles, form_classes = principle_form_menus(strategy_type)
    principle_menu = ", ".join(sorted(principles))
    form_menu = ", ".join(sorted(form_classes))
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        names = ", ".join(active_lob_feature_names(lob_input_mode))
        features_hint = (
            f"comma-separated identifiers only from this run's feature set: {names}, "
            "current_pv"
        )
        mechanism_hint = (
            "cite ≥2 trajectory moments: time, current_pv, shade, feature values "
            "from the allowed set, fill/no-fill"
        )
    elif strategy_type == "lob":
        features_hint = (
            "comma-separated identifiers only, e.g. bids_relative, asks_relative, "
            "spread_relative, mid_relative, current_pv, market_trades, prior_lob"
        )
        mechanism_hint = (
            "cite ≥2 trajectory moments: time, current_pv, shade, prior spread/mid "
            "or best bid/ask, fill/no-fill"
        )
    elif strategy_type == "spread":
        features_hint = (
            "which market_state fields (bid/ask, spread, inventory fraction, current_pv, "
            "fundamental_estimate, shouts); handle None bids/asks safely"
        )
        mechanism_hint = (
            "cite case-card stats (leftover, fill, pos_gap_miss) and ≥2 tape shouts: "
            "t, side, shade, spread, gap_to_touch, filled"
        )
    else:
        features_hint = "which PV stats / schedule index / sign / …; scale-safe"
        mechanism_hint = (
            "cite ≥2 trajectory moments: time, current_pv, shade, fill/no-fill"
        )
    diagnosis_fields = [
        "DIAGNOSIS:",
        "  principle: <one from menu — proposed hypothesis>",
        "  form_class: <one from menu — proposed hypothesis>",
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
        "(features/normalization/branches); replace: how the new pair differs "
        "from current_form>"
    )
    pair_rules = [
        "You MUST set action to refine or replace.",
        "action=refine: KEEP this strategy's (principle, form_class) and improve "
        "the shade map under that pair — forbid comment-only or AST-identical "
        "clones; the new implementation must compute shade differently enough "
        "that probe shades differ. "
        "This outranks menu-exploration and the differ-from-parent rule.",
        "action=replace: pick a NEW pair (unexplored if any remain; must differ "
        "from parent/siblings).",
    ]
    if require_last_hold_comparison:
        pair_rules.append(
            "LAST HOLD: yes → action=refine; no → action=replace."
        )
    return [
        "## Allowed menus (pick exactly one principle and one form_class)",
        f"principle ∈ {{{principle_menu}}}",
        f"form_class ∈ {{{form_menu}}}",
        "",
        "## Required DIAGNOSIS Output (hypothesis for the NEXT codegen step)",
        "You are evaluating THIS strategy's batch results and proposing the NEXT "
        "experiment (principle + form_class) for the generator to implement.",
        "Use the exact labels. Fill every field. Do not invent extra top-level sections.",
        "",
        *diagnosis_fields,
        "",
        f"MECHANISM: (max 3 sentences; {mechanism_hint})",
        "FIX: (max 2 sentences; state the hypothesis for codegen; no full code; "
        "no magic shade floats)",
        "",
        "Do NOT write Environment, Performance, or Principle essay sections.",
        *pair_rules,
        "Do NOT use shade_more / shade_less / match shade_hi as the Fix.",
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
) -> List[str]:
    """Parent eval facts for the generator under principle/form critic mode."""
    comparison = parent_performance.get("zi_benchmark_comparison")
    reward = parent_performance.get("reward")
    reward_var = float(parent_performance.get("reward_variance", 0.0))
    lines = [
        "",
        "## Parent Evaluation (batch metrics — implement Parent Hypothesis DIAGNOSIS)",
    ]
    lines += format_structured_principle_form_facts(
        comparison,
        reward=float(reward) if reward is not None else None,
        strategy_type=strategy_type,
    )
    lines.append(f"Batch reward variance: {reward_var:.4f}")
    lines += format_per_config_breakdown_lines(
        parent_performance,
        heading="Parent Per-Config Advantage (batch breakdown)",
    )
    return lines


def _structured_principle_form_critic_instructions(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
) -> List[str]:
    if strategy_type == "spread":
        return [
            "You are reviewing a spread-only shading strategy.",
            "Job: read the data, state a hypothesis (principle + form_class), and choose "
            "refine (re-implement under same pair) or replace (new pair).",
            "PRIMARY objective: batch advantage (higher is better).",
            "Reply MUST begin with the DIAGNOSIS block, then MECHANISM (≤3 sentences) "
            "and FIX (≤2 sentences). No essay sections.",
            "features_used ⊆ market_state "
            "(best_bid, best_ask, units_available, q_max, fundamental_estimate, "
            "current_pv, shouts). Handle None bids/asks.",
            "Do not hardcode shade floats. Higher shade = less aggressive.",
        ]
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        names = ", ".join(active_lob_feature_names(lob_input_mode))
        return [
            "You are reviewing a LOB features-mode shading strategy.",
            "Job: read the data, state a hypothesis (principle + form_class), and choose "
            "refine or replace.",
            "PRIMARY objective: batch advantage (higher is better).",
            "Reply MUST begin with the DIAGNOSIS block, then MECHANISM (≤3 sentences) "
            "and FIX (≤2 sentences).",
            f"features_used ⊆ fixed feature set ({names}, current_pv).",
            "Do not hardcode shade floats. Higher shade = less aggressive.",
        ]
    if strategy_type == "lob":
        return [
            "You are reviewing a LOB-only shading strategy.",
            "Job: read the data, state a hypothesis (principle + form_class), and choose "
            "refine or replace.",
            "PRIMARY objective: batch advantage (higher is better).",
            "Reply MUST begin with the DIAGNOSIS block, then MECHANISM (≤3 sentences) "
            "and FIX (≤2 sentences).",
            "features_used ⊆ market_history fields. Do not invent a PV schedule.",
            "Do not hardcode shade floats. Higher shade = less aggressive.",
        ]
    return [
        "You are reviewing a PV-only shading strategy.",
        "Job: read the data, state a hypothesis (principle + form_class), and choose "
        "refine or replace.",
        "PRIMARY objective: batch advantage (higher is better).",
        "Reply MUST begin with the DIAGNOSIS block, then MECHANISM (≤3 sentences) "
        "and FIX (≤2 sentences).",
        "features_used must be scale-safe for large private_values.",
        "Do not hardcode shade floats. Higher shade = less aggressive. No LOB stories.",
    ]


def _structured_principle_form_generator_task(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = "raw",
) -> List[str]:
    shared_head = [
        "Read Parent Hypothesis DIAGNOSIS first (principle, form_class, features_used, "
        "action, proposed_change).",
        "Implement that hypothesis using Parent Evaluation facts when provided.",
        "Priority: (1) principle + form_class + action, (2) features_used / "
        "proposed_change, (3) MECHANISM/FIX prose.",
        "If action=refine: KEEP principle+form; re-implement the shade map under "
        "that pair (different features / normalization / branches). "
        "Coefficient-only or +ε damper edits are forbidden.",
    ]
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        return shared_head + [
            "If action=replace: write a NEW features→shade rule of the named form_class "
            "using only the fixed scalar feature vector. Differ in structure, not only comments.",
            "Do NOT use raw lob_snapshots / depth lists or invent a private_values schedule.",
            "Do NOT hardcode critic shade floats.",
        ]
    if strategy_type == "lob":
        return shared_head + [
            "If action=replace: write a NEW market_history→shade rule of the named "
            "form_class. Differ in structure, not only comments.",
            "Do NOT invent a private_values schedule.",
            "Do NOT hardcode critic shade floats.",
        ]
    if strategy_type == "spread":
        return shared_head + [
            "If action=replace: write a NEW market_state→shade rule of the named "
            "form_class. Differ in structure, not only comments.",
            "You MAY use market_state['shouts'] (accepted and rejected). "
            "Do NOT invent a private_values schedule or LOB depth lists.",
            "If best_bid or best_ask is None, that quote is absent — do not treat "
            "None as 0.0 in arithmetic.",
            "Do NOT hardcode critic shade floats.",
        ]
    return shared_head + [
        "If action=replace: write a NEW private_values→shade rule of the named "
        "form_class. Differ in structure, not only comments.",
        "Use private_values, fundamental_estimate, and current_pv only; "
        "no LOB or market-trend signals.",
        "Do NOT hardcode critic shade floats.",
    ]


def format_structured_zi_curriculum_facts(
    comparison: Optional[Dict[str, Any]],
    *,
    reward: Optional[float] = None,
) -> List[str]:
    """Factual shade anchors for the structured curriculum critic (no storytelling)."""
    if not comparison:
        return [
            "## Curriculum Facts (from evaluation)",
            "No ZI comparison was provided.",
        ]
    diag = infer_curriculum_shade_diagnosis(comparison)
    beaten_his, failed_his = _benchmark_pass_fail_lists(comparison)
    suggested_direction = diag["shade_error_direction"]
    suggested_target = diag["suggested_target"]
    llm_x_cap = diag["requested_x_cap"]
    llm_trade_shade = diag["llm_trade_shade"]

    lines = [
        "## Curriculum Facts (from evaluation — use these; do not invent market trends)",
        (
            f"Batch advantage (Q reward): {float(reward):.4f}."
            if reward is not None
            else "Batch advantage (Q reward): (not reported)."
        ),
        (
            f"LLM mean requested x_cap (strategy output; COMPARE THIS to ZI shade_hi): "
            f"{float(llm_x_cap):.4f}."
            if llm_x_cap is not None
            else "LLM mean requested x_cap: (not reported)."
        ),
        (
            f"LLM mean realized trade shade (NOT an x_cap; do not compare to shade_hi): "
            f"{float(llm_trade_shade):.4f}."
            if llm_trade_shade is not None
            else "LLM mean realized trade shade: (not reported)."
        ),
        (
            f"ZI beat (diagnostic only): {int(comparison.get('beat_count', 0))}/"
            f"{int(comparison.get('total_benchmarks', 0))} "
            f"(fraction={float(comparison.get('beat_fraction', 0.0)):.4f})."
        ),
        f"LLM mean surplus: {float(comparison.get('llm_mean_surplus', 0.0)):.2f}.",
        (
            f"Highest BEATEN shade_hi: {max(beaten_his):g}."
            if beaten_his
            else "Highest BEATEN shade_hi: (none beaten)."
        ),
        (
            f"Lowest FAILED shade_hi: {min(failed_his):g}."
            if failed_his
            else "Lowest FAILED shade_hi: (none failed)."
        ),
        f"Suggested shade_error_direction (from requested x_cap vs ZI caps): {suggested_direction}.",
        f"Suggested error_type: {diag['error_type']}.",
        f"Suggested fix_class: {diag['fix_class']}.",
        (
            f"Suggested target_shade_level (DIAGNOSTIC HINT only — do NOT hardcode this "
            f"float into agent_strategy): {suggested_target:.4f}."
            if suggested_target is not None
            else "Suggested target_shade_level: (unknown)."
        ),
        "Write a GENERAL shading rule from private_values → x_cap. Curriculum caps are "
        "probes to diagnose level error, not constants to paste into code.",
        "Metric note: ZI shade_hi is an x_cap. Your agent also returns an x_cap; runtime "
        "samples shade ~ Uniform(0, x_cap). Compare requested x_cap to shade_hi only.",
        "CRITICAL scale note: private_values are large monetary units (often hundreds+), "
        "NOT fractions in [0,1]. Comparing pv_mean to 0.5/0.9/etc. usually never fires "
        "and does not change shade. Shade/x_cap outputs MUST be in [0,1].",
        "Shade semantics (PV-only): higher x = more shade = buyers bid lower / sellers ask higher "
        "(less aggressive). Lower x = less shade = more aggressive.",
        "Banned topics: LOB trends, best-bid chasing, time-of-day, and any signal not in private_values.",
    ]
    return lines


def format_structured_zi_curriculum_parent_eval(
    parent_performance: Dict[str, Any],
    *,
    q_reward_mode: str = "zi_advantage",
    reference_only: bool = False,
) -> List[str]:
    """Parent eval facts for the generator — mirrors structured curriculum critic inputs."""
    comparison = parent_performance.get("zi_benchmark_comparison")
    reward = parent_performance.get("reward")
    reward_var = float(parent_performance.get("reward_variance", 0.0))
    if reference_only:
        header = (
            "## Parent Evaluation (reference only — critic already interpreted these facts; "
            "do NOT re-diagnose; implement Parent Reflection instead)"
        )
    else:
        header = "## Parent Evaluation (batch metrics and curriculum facts)"
    lines = [
        "",
        header,
    ]
    lines += format_structured_zi_curriculum_facts(
        comparison,
        reward=float(reward) if reward is not None else None,
    )
    lines.append(f"Batch reward variance: {reward_var:.4f}")
    lines += format_per_config_breakdown_lines(
        parent_performance,
        heading="Parent Per-Config Advantage (batch breakdown)",
    )

    if comparison:
        lines += format_zi_benchmark_comparison_section(
            comparison,
            heading_prefix="Parent",
            q_reward_mode=q_reward_mode,
        )
    return lines


def _structured_zi_curriculum_critic_instructions() -> List[str]:
    return [
        "You are reviewing a PV-only shading strategy in a continuous double auction.",
        "Q reward is batch advantage (LLM surplus minus baseline; higher is better).",
        "ZI beat fraction and curriculum shade caps are diagnostic probes — not constants "
        "to hardcode into code.",
        "Your reply MUST begin with the DIAGNOSIS block (exact labels from the template below).",
        "Copy computed field values from Curriculum Facts unless you justify a change in MECHANISM.",
        "Then write MECHANISM (max 2 sentences) and FIX (max 2 sentences).",
        "Do NOT write Environment, Performance, or Principle sections.",
        "shade_error_direction must be one of: shade_more, shade_less, reshape_pv_map.",
        "error_type must be one of: level_too_low, level_too_high, pv_mapping_wrong.",
        "fix_class must be one of: scale_base, add_offset, change_pv_feature, change_formula.",
        "When consensus is mixed (multi-config), set shade_error_direction to reshape_pv_map "
        "and error_type to pv_mapping_wrong.",
        "Use LLM mean requested x_cap (not realized trade shade) when comparing to ZI shade_hi.",
        "private_values are large currency units — do NOT propose pv_mean >= 0.5 thresholds.",
        "Higher x_cap means MORE shade (less aggressive). PV-only: ban LOB/bid-ask/trend stories.",
    ]


def _structured_zi_curriculum_generator_task() -> List[str]:
    return [
        "Read Parent Reflection DIAGNOSIS first (shade_error_direction, error_type, fix_class).",
        "Implement that diagnosis using Parent Evaluation facts and trajectory when provided.",
        "Priority order: (1) shade_error_direction, (2) fix_class, (3) MECHANISM/FIX prose.",
        "When consensus is mixed, change the PV→x_cap mapping (reshape_pv_map) — do not chase "
        "one global level target.",
        "Invent a NEW general rule (not a market-specific patch or pasted critic float):",
        "- shade_more / level_too_low + add_offset: raise typical returned x_cap in [0,1].",
        "- shade_less / level_too_high + scale_base: lower typical returned x_cap in [0,1].",
        "- reshape_pv_map / change_pv_feature: change which PV features drive x_cap.",
        "Do NOT hardcode target_shade_level, shade_hi, test_id, or benchmark lists.",
        "Do NOT copy the parent code. Do NOT use LOB, bid/ask, or market-trend signals.",
        "DETERMINISTIC ONLY: no import numpy, no import random, no RNG of any kind.",
        "Your agent_strategy MUST differ meaningfully from the Parent Strategy "
        "(logic / formulas / control flow — not dead PV thresholds or pasted critic floats).",
    ]


def format_forbidden_strategies_section(
    forbidden_strategies: Optional[List[Dict[str, Any]]],
) -> List[str]:
    """Prompt section listing sibling strategies that must not be returned (or equivalents)."""
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
    if is_zi_q_reward_mode(q_reward_mode) and not is_structured_principle_form(critic_mode):
        sections += _zi_benchmark_context_note(
            q_reward_mode,
            strategy_type=strategy_type,
            lob_input_mode=lob_input_mode,
        )
        sections.append("")
    if is_pv_only:
        sections.append("private_values is a list of relative private values for each unit.")
        sections.append("Use only private_values to decide the shading level.")
    elif is_spread_only:
        sections += _spread_input_schema(spread_input_mode)
    elif is_lob_features:
        sections += _lob_features_input_schema(lob_input_mode)
    elif is_lob_only:
        sections += _lob_input_schema()
    else:
        sections += _full_input_schema()
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
        principle_form_mode = is_structured_principle_form(critic_mode)
        if parent_performance:
            if curriculum_mode:
                sections += format_structured_zi_curriculum_parent_eval(
                    parent_performance,
                    q_reward_mode=q_reward_mode,
                    reference_only=False,
                )
            elif principle_form_mode:
                sections += format_structured_principle_form_parent_eval(
                    parent_performance,
                    q_reward_mode=q_reward_mode,
                    strategy_type=strategy_type,
                )
            else:
                sections += format_parent_reward_section(
                    parent_performance,
                    q_reward_mode=q_reward_mode,
                )
                if is_zi_q_reward_mode(q_reward_mode):
                    comparison = parent_performance.get("zi_benchmark_comparison")
                    if comparison:
                        sections += format_generator_zi_benchmark_section(
                            comparison,
                            q_reward_mode=q_reward_mode,
                        )
                elif parent_performance.get("config_unit_tests"):
                    sections += format_generator_advantage_config_section(parent_performance)
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
            if curriculum_mode:
                sections += _structured_zi_curriculum_generator_task()
                requirements += [
                    "Use only deterministic pure Python (stdlib math is ok for "
                    "deterministic functions like sqrt/exp; never use RNG).",
                    "Never import numpy or random; never sample or add noise.",
                ]
                requirements += [
                    "Implement Parent Reflection DIAGNOSIS (shade_error_direction, fix_class) first.",
                    "Re-check Parent Evaluation facts and trajectory when provided.",
                    "Do not hardcode critic floats, shade_hi, or test_ids.",
                    "Your agent_strategy must differ meaningfully from the Parent Strategy.",
                ]
                if is_lob_only:
                    requirements.append(
                        "Use market_history only; do not invent a full private_values schedule."
                    )
                else:
                    requirements.append(
                        "Use private_values, fundamental_estimate, and current_pv only; "
                        "no LOB or market-trend inputs. Signature must be "
                        "agent_strategy(private_values, fundamental_estimate, current_pv)."
                    )
            elif principle_form_mode:
                sections += _structured_principle_form_generator_task(
                    strategy_type=strategy_type,
                    lob_input_mode=lob_input_mode,
                )
                diagnosis_action = (parse_diagnosis_action(parent_reflection or "") or "").strip().lower()
                requirements += [
                    "Implement Parent Hypothesis DIAGNOSIS (principle, form_class, action) first.",
                    "Do not hardcode critic floats or test_ids.",
                ]
                if diagnosis_action == "refine":
                    requirements.append(
                        "action=refine: keep the parent principle+form; change the "
                        "shade map under that pair so probe shades differ from the "
                        "parent (not comment-only)."
                    )
                else:
                    requirements.append(
                        "action=replace (or unspecified): change functional form / features "
                        "vs the Parent Strategy (not comment-only)."
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
                elif is_spread_only:
                    requirements.append(
                        "Use market_state only (including shouts); do not invent a full "
                        "private_values schedule or LOB depth lists."
                    )
                else:
                    requirements.append(
                        "Use private_values, fundamental_estimate, and current_pv only; "
                        "no LOB or market-trend inputs. Signature must be "
                        "agent_strategy(private_values, fundamental_estimate, current_pv)."
                    )
            elif is_zi_q_reward_mode(q_reward_mode):
                sections += [
                    "Read the Parent Reflection (Mechanism, Principle, and Fix) for diagnosis only.",
                    "Do NOT copy the Parent Strategy or paraphrase the reflection's Fix section into code.",
                    "Propose a new general shading rule informed by Mechanism/Principle, with different "
                    "logic than the parent (new formulas, thresholds, PV aggregation, or control flow).",
                    "Use FAILED ZI Benchmark Unit Tests only as diagnostics (surplus gaps, "
                    "shade_hi, trajectory) — not as a checklist of special cases to encode.",
                    "Write one general shading rule that should perform well across this "
                    "market against many opponents. Do NOT hardcode, branch on, or otherwise "
                    "refer to individual benchmark shade ranges / test_ids in the code.",
                    "Your agent_strategy MUST differ meaningfully from the Parent Strategy "
                    "(change logic, formulas, thresholds, or control flow). Do not return "
                    "identical, near-identical, comment-only, or rename-only revisions.",
                ]
                requirements += [
                    "Use Parent Reflection for diagnosis only; do not copy its Fix verbatim.",
                    "Ground changes in trajectory evidence when batch interactions are provided.",
                    "Your agent_strategy must differ meaningfully from the Parent Strategy.",
                    "Optimize for general market performance; never encode individual ZI "
                    "benchmarks (shade ranges, test_ids, or per-opponent special cases) in code.",
                ]
                if forbid_numpy_random:
                    sections += [
                        "DETERMINISTIC ONLY: no import numpy, no import random, no RNG of any kind.",
                    ]
                    requirements += [
                        "Use only deterministic pure Python (stdlib math is ok for "
                        "deterministic functions like sqrt/exp; never use RNG).",
                        "Never import numpy or random; never sample or add noise.",
                    ]
            else:
                sections += [
                    "Read the Parent Reflection (Mechanism, Principle, and Fix) and the "
                    "Parent evaluation data below.",
                    "Implement the Fix so it follows the Principle and addresses Mechanism "
                    "findings from the trajectory.",
                    "Address each FAILED entry in Per-Config Unit Tests (diagnose from "
                    "advantage gaps, surplus metrics, shade, and trajectory; no prescriptive hints).",
                    "You MUST make at least one meaningful code change; do not return identical code.",
                ]
                requirements += [
                    "Follow the Parent Reflection Principle and Fix sections.",
                    "Ground changes in trajectory evidence when batch interactions are provided.",
                    "Fix every FAILED config test you can; cite test_id and config_label in comments.",
                ]
        else:
            performance = parent_performance or {}
            sections += [
                "",
                "## Parent Performance (batch evaluation metrics)",
                json.dumps(performance, indent=2),
                "",
                "## Refinement Task",
                "No reflection was provided. Use the Parent Strategy and Parent Performance",
                "metrics to propose a revised strategy that may improve batch reward.",
                "You MUST make at least one meaningful change to the Parent Strategy code",
                "(logic, formula, thresholds, or control flow).",
                "Do not return identical or near-identical code.",
            ]
            requirements += [
                "Your agent_strategy must differ meaningfully from the Parent Strategy.",
                "Use the parent performance metrics when deciding how to revise the strategy.",
            ]

    # Include parent trajectory/metrics for generator (all critic modes when data available).
    if generator_batch_summary is not None and generator_batch_data is not None:
        sections += format_evaluation_trajectory_sections(
            batch_summary=generator_batch_summary,
            batch_data=generator_batch_data,
            heading_prefix="Parent",
            q_reward_mode=q_reward_mode,
            omit_zi_tables=is_structured_principle_form(critic_mode),
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

    sections += [
        "",
        "## Requirements",
        *[f"- {req}" for req in requirements],
        "",
        "Respond with exactly one ```python``` block.",
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
            # Kept for API compatibility; still require probe-distinct in-family edit.
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
            "Respond with exactly one ```python``` block.",
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
    require_last_hold_comparison: Optional[bool] = None,
) -> str:
    comparison = batch_summary.get("zi_benchmark_comparison")
    is_zi_mode = is_zi_q_reward_mode(q_reward_mode) or comparison is not None
    structured_curriculum = is_structured_zi_curriculum(critic_mode)
    structured_principle = is_structured_principle_form(critic_mode)
    if require_last_hold_comparison is None:
        require_last_hold_comparison = bool(
            isinstance(batch_data, dict) and batch_data.get("prior_hold_card")
        )

    if structured_curriculum:
        per_config = batch_summary.get("per_config_rewards") or {}
        multi_config = len(per_config) > 1
        sections = list(_structured_zi_curriculum_critic_instructions())
        sections += [
            "",
            *_zi_benchmark_context_note(
                q_reward_mode,
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
            ),
        ]
        sections += ["", *format_structured_zi_curriculum_facts(comparison, reward=reward)]
        sections += [
            "",
            *format_curriculum_diagnosis_template(
                comparison,
                batch_data=batch_data if isinstance(batch_data, dict) else None,
                multi_config=multi_config,
            ),
        ]
    elif structured_principle:
        sections = list(
            _structured_principle_form_critic_instructions(
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
            )
        )
        sections += [
            "",
            *format_structured_principle_form_facts(
                comparison,
                reward=reward,
                strategy_type=strategy_type,
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
                ),
            ]
        sections += [
            "",
            *format_structured_principle_form_diagnosis_template(
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
                require_last_hold_comparison=require_last_hold_comparison,
            ),
        ]
    elif is_zi_mode:
        sections = list(_zi_critic_instructions(q_reward_mode))
        sections += [
            "",
            *_zi_benchmark_context_note(
                q_reward_mode,
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
            ),
        ]
    else:
        sections = list(_advantage_critic_instructions())

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
                omit_zi_tables=structured_principle,
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

    if not include_data:
        sections += [
            "",
            "## Evaluation Data",
            "No batch metrics or episode trajectories were provided. "
            "Use only the strategy code and scalar batch reward above.",
        ]
        if comparison:
            sections += format_zi_benchmark_comparison_section(
                comparison,
                q_reward_mode=q_reward_mode,
            )
        elif batch_summary.get("config_unit_tests"):
            sections += format_advantage_config_tests_section(batch_summary)
        return "\n".join(sections)

    sections += format_evaluation_trajectory_sections(
        batch_summary=batch_summary,
        batch_data=batch_data if isinstance(batch_data, dict) else {},
        heading_prefix="",
        q_reward_mode=q_reward_mode,
    )
    return "\n".join(sections)