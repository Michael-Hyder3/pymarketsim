from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Set

from marketsim.LLM.reflection_pipeline.prompts import parse_principle_form_fields

_NON_INPUT_FEATURES = frozenset(
    {
        "shade",
        "fill",
        "no-fill",
        "time",
        "trajectory",
        "fills",
        "surplus",
        "advantage",
    }
)

_FEATURE_CODE_ALIASES: Dict[str, List[str]] = {
    "best_bid": ["best_bid"],
    "best_ask": ["best_ask"],
    "spread": ["spread", "best_ask", "best_bid", "spread_relative"],
    "spread_relative": ["spread_relative", "best_ask", "best_bid"],
    "mid": ["mid_relative", "best_bid", "best_ask"],
    "mid_relative": ["mid_relative", "best_bid", "best_ask"],
    "current_pv": ["current_pv"],
    "units_available": ["units_available"],
    "q_max": ["q_max"],
    "fundamental_estimate": ["fundamental_estimate"],
    "fundamental": ["fundamental", "fundamental_estimate"],
    "shouts": ["shouts"],
    "inventory": ["units_available", "q_max"],
    "bids_relative": ["bids_relative"],
    "asks_relative": ["asks_relative"],
    "lob": ["lob", "lob_snapshots", "market_history"],
    "lob_snapshots": ["lob_snapshots"],
    "market_trades": ["market_trades", "recent_trade_prices"],
    "recent_trade_prices": ["recent_trade_prices", "last_trade_price"],
    "private_values": ["private_values"],
    "pv_mean": ["private_values", "pv_mean", "sum("],
    "pv_max": ["private_values", "pv_max", "max("],
    "prior_lob": ["lob", "lob_snapshots", "market_history"],
    "prior_book": ["lob", "lob_snapshots", "market_history", "bids_relative", "asks_relative"],
    "trade": ["market_trades", "recent_trade_prices", "last_trade_price"],
    "trades": ["market_trades", "recent_trade_prices", "last_trade_price"],
    "orders": ["orders"],
    "accepted": ["accepted"],
    "side": ["side", "order_type"],
    "price_rel": ["price_rel"],
    "memory_start_time": ["memory_start_time"],
    "l": ["'l'", '"l"', "memory_start", "orders"],
    "transaction_count": ["transaction_count"],
    "features": ["features"],
    "spread_rel": ["spread_rel", "features"],
    "mid_rel": ["mid_rel", "features"],
    "imbalance": ["imbalance", "features"],
    "depth_bid": ["depth_bid", "features"],
    "depth_ask": ["depth_ask", "features"],
    "trade_mom": ["trade_mom", "features"],
    "log_pv": ["log_pv", "features"],
    "pv_minus_mid": ["pv_minus_mid", "features"],
    "has_book": ["has_book", "features"],
}

_LOB_PROSE_FEATURE_MARKERS = (
    "prior book",
    "prior lob",
    "fields from market_history",
    "from market_history",
    "which prior lob",
)

_PRINCIPLE_CODE_HINTS: Dict[str, List[str]] = {
    "bid_ask_spread_width": ["best_bid", "best_ask", "spread"],
    "inventory_remaining_fraction": ["units_available", "q_max"],
    "inventory_remaining_units": ["units_available", "q_max", "private_values"],
    "pv_vs_mid": ["current_pv", "mid_relative", "best_bid", "best_ask"],
    "pv_vs_touch": ["current_pv", "best_bid", "best_ask"],
    "current_pv_magnitude": ["current_pv"],
    "recent_accept_reject": ["shouts", "accepted"],
    "last_own_accept_reject": ["prev_event", "accepted"],
    "shout_competitiveness": ["shouts", "price_rel", "price_abs", "best_bid", "best_ask"],
    "current_pv_vs_mid": ["current_pv", "mid_relative", "mid"],
    "prior_spread_or_mid": ["spread_relative", "mid_relative", "spread", "mid"],
    "book_imbalance": ["bids", "asks", "bids_relative", "asks_relative", "imbalance"],
    "depth_asymmetry": ["bids", "asks", "bids_relative", "asks_relative"],
    "trade_momentum": ["recent_trade_prices", "market_trades", "last_trade_price"],
    "fill_vs_reject": ["orders", "accepted"],
    "side_pressure": ["orders", "side"],
    "order_flow_imbalance": ["orders", "side"],
    "recent_aggressiveness": ["orders", "price_rel", "accepted"],
    "current_pv_vs_flow": ["current_pv", "orders"],
    "pv_rank_or_distance_to_best": ["private_values", "max(", "min(", "current_pv"],
    "signed_asymmetry": ["private_values"],
    "nonlinear_abs_pv": ["abs(", "private_values"],
    "fill_surplus_tradeoff": ["private_values", "current_pv"],
    "pv_vs_fundamental": [
        "private_values",
        "fundamental_estimate",
        "current_pv",
    ],
    "current_pv_vs_fundamental": ["current_pv", "fundamental_estimate"],
    "current_pv_in_schedule": ["current_pv", "private_values"],
}


def _code_lower(code: str) -> str:
    return (code or "").lower()


def _parse_features_used(reflection: str) -> List[str]:
    match = re.search(r"features_used:\s*([^\n]+)", reflection or "", flags=re.IGNORECASE)
    if not match:
        return []
    raw = match.group(1).strip().strip("*")
    parts = re.split(r"[,;]", raw)
    return [p.strip().lower() for p in parts if p.strip()]


def _expand_features_used_tokens(
    tokens: List[str],
    *,
    strategy_type: str = "pv",
) -> List[str]:
    expanded: List[str] = []
    for token in tokens:
        lower = token.strip().lower()
        if not lower:
            continue
        if strategy_type == "lob" and (
            any(marker in lower for marker in _LOB_PROSE_FEATURE_MARKERS)
            or (
                "/" in lower
                and "," not in lower
                and not re.match(r"^[a-z][a-z0-9_]*$", lower)
            )
        ):
            if "book" in lower or "lob" in lower or "market_history" in lower:
                expanded.append("prior_lob")
            if "trade" in lower:
                expanded.append("market_trades")
            if "current_pv" in lower or re.search(r"\bpv\b", lower):
                expanded.append("current_pv")
            if not any(
                x in expanded for x in ("prior_lob", "market_trades", "current_pv")
            ):
                expanded.extend(["prior_lob", "current_pv"])
            continue
        expanded.append(lower)
    return expanded


def validate_features_used_format(
    reflection: str,
    *,
    strategy_type: str = "pv",
) -> Optional[str]:
    match = re.search(r"features_used:\s*([^\n]+)", reflection or "", flags=re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip().strip("*")
    lower = raw.lower()
    if any(marker in lower for marker in _LOB_PROSE_FEATURE_MARKERS):
        return (
            "features_used must list comma-separated field identifiers "
            "(e.g. orders, accepted, side, price_rel, L, current_pv), "
            "not template prose"
        )
    if strategy_type == "lob":
        examples = (
            "orders, accepted, side, price_rel, L, memory_start_time, current_pv"
        )
        for token in [t.strip() for t in re.split(r"[,;]", raw) if t.strip()]:
            if not re.match(r"^[a-z][a-z0-9_]*$", token.strip("*"), flags=re.IGNORECASE):
                return (
                    "features_used must be comma-separated identifiers like "
                    f"{examples}; got {token!r}"
                )
    return None


def _feature_present(code: str, feature: str) -> bool:
    token = feature.strip().lower()
    if token in _NON_INPUT_FEATURES:
        return True
    aliases = _FEATURE_CODE_ALIASES.get(token)
    if aliases is None:
        normalized = token.replace("-", "_").replace(" ", "_")
        aliases = _FEATURE_CODE_ALIASES.get(normalized, [token.replace(" ", "_")])
    lower = _code_lower(code)
    return any(alias in lower for alias in aliases)


def _form_class_shape_ok(form_class: str, code: str) -> bool:
    form = (form_class or "").strip().lower()
    lower = _code_lower(code)
    if form == "clamp_linear":
        if "clamp_linear" in lower:
            return True
        if "min(" in lower and "max(" in lower:
            return True
        if "clip(" in lower:
            return True
        return False
    checks: Dict[str, Callable[[], bool]] = {
        "ratio": lambda: "/" in code,
        "threshold": lambda: "if " in lower and bool(re.search(r"[<>=]", code)),
        "piecewise_by_sign": lambda: lower.count("if ") >= 2,
        "inventory_weighted": lambda: (
            ("*" in code or "/" in code)
            and (
                "units_available" in lower
                or "q_max" in lower
                or "private_values" in lower
            )
        ),
        "rank": lambda: "sorted" in lower or "rank" in lower,
        "signed_power": lambda: "**" in code or "pow(" in lower,
        "ema_over_snapshots": lambda: "lob_snapshots" in lower,
        "ema_over_window": lambda: (
            ("orders" in lower or "shouts" in lower)
            and ("for " in lower or "sum(" in lower or "ema" in lower)
        ),
        "schedule_index": lambda: "len(" in lower,
        "unit_schedule_blend": lambda: (
            "current_pv" in lower
            and "private_values" in lower
            and ("/" in code or "*" in code or "+" in code)
        ),
    }
    checker = checks.get(form)
    if checker is None:
        return True
    return checker()


def _principle_hint_ok(principle: str, code: str) -> bool:
    hints = _PRINCIPLE_CODE_HINTS.get((principle or "").strip().lower())
    if not hints:
        return True
    lower = _code_lower(code)
    return any(hint in lower for hint in hints)


def validate_code_implements_diagnosis(
    code: str,
    reflection: str,
    *,
    strategy_type: str = "pv",
) -> Optional[str]:
    fields = parse_principle_form_fields(reflection or "")
    principle = fields.get("principle")
    form_class = fields.get("form_class")
    if not principle or not form_class:
        return None

    features = [
        f
        for f in _expand_features_used_tokens(
            _parse_features_used(reflection or ""),
            strategy_type=strategy_type,
        )
        if f and f not in _NON_INPUT_FEATURES
    ]
    substantive = [f for f in features if f not in _NON_INPUT_FEATURES]
    if substantive:
        matched = sum(1 for feat in substantive if _feature_present(code, feat))
        required = max(1, min(len(substantive), 2))
        if matched < required:
            missing = [f for f in substantive if not _feature_present(code, f)]
            return (
                "Generated code does not implement Parent Reflection features_used: "
                f"missing evidence for {', '.join(missing[:4])}. "
                "Use the named inputs in formulas (not only in comments)."
            )

    if not _principle_hint_ok(principle, code):
        return (
            f"Generated code does not implement principle '{principle}' — "
            "use the corresponding input signals in the shading rule."
        )

    if not _form_class_shape_ok(form_class, code):
        return (
            f"Generated code does not match form_class '{form_class}' structure "
            "(e.g. ratio needs division, clamp_linear needs min/max, "
            "piecewise_by_sign needs multiple branches)."
        )

    return None


