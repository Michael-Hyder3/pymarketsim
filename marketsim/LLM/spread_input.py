from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

SPREAD_INPUT_MODES = ("classic", "zip_parity", "book")
DEFAULT_SPREAD_INPUT_MODE = "classic"

_ACTIVE_SPREAD_OMIT_FUNDAMENTAL: bool = False


def set_spread_omit_fundamental(omit: bool) -> None:
    global _ACTIVE_SPREAD_OMIT_FUNDAMENTAL
    _ACTIVE_SPREAD_OMIT_FUNDAMENTAL = bool(omit)


def get_spread_omit_fundamental() -> bool:
    return bool(_ACTIVE_SPREAD_OMIT_FUNDAMENTAL)


def resolve_spread_omit_fundamental(omit: Optional[bool] = None) -> bool:
    if omit is None:
        return get_spread_omit_fundamental()
    return bool(omit)


ZIP_PARITY_SPREAD_KEYS = (
    "current_pv",
    "fundamental",
    "best_bid",
    "best_ask",
    "units_available",
    "q_max",
    "shouts",
    "private_values",
)

BOOK_CURRENT_KEYS = (
    "best_bid",
    "best_ask",
    "bids",
    "asks",
    "units_available",
    "q_max",
    "fundamental_estimate",
    "current_pv",
)

BOOK_SPREAD_KEYS = BOOK_CURRENT_KEYS + ("prev_event", "private_values")

SPREAD_FUNDAMENTAL_OBS_KEYS = ("fundamental_estimate", "fundamental")


def normalize_spread_input_mode(mode: Optional[str]) -> str:
    resolved = str(mode or DEFAULT_SPREAD_INPUT_MODE).strip().lower()
    if resolved not in SPREAD_INPUT_MODES:
        known = ", ".join(SPREAD_INPUT_MODES)
        raise ValueError(f"Unsupported spread_input_mode {mode!r}. Known: {known}")
    return resolved


def is_spread_zip_parity_mode(mode: Optional[str]) -> bool:
    try:
        return normalize_spread_input_mode(mode) == "zip_parity"
    except ValueError:
        return False


def is_spread_book_mode(mode: Optional[str]) -> bool:
    try:
        return normalize_spread_input_mode(mode) == "book"
    except ValueError:
        return False


def normalize_prev_event(event: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(event, dict) or not event:
        return None
    return {
        "price_abs": float(event["price_abs"]),
        "price_rel": float(event["price_rel"]),
        "side": int(event["side"]),
        "accepted": bool(event["accepted"]),
    }


def _drop_fundamental_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    for key in SPREAD_FUNDAMENTAL_OBS_KEYS:
        state.pop(key, None)
    return state






def _normalize_private_values(values: Optional[List[Any]]) -> List[float]:
    out: List[float] = []
    for value in values or []:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _normalize_resting_levels(
    levels: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in levels or []:
        if not isinstance(entry, dict):
            continue
        try:
            price = float(entry["price_rel"])
            qty = float(entry["quantity"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"price_rel": price, "quantity": qty})
    return out


def build_spread_market_state(
    *,
    best_bid: Optional[float],
    best_ask: Optional[float],
    units_available: int,
    q_max: int,
    current_pv: float,
    fundamental_estimate: float,
    fundamental_true: float,
    shouts: Optional[List[Dict[str, Any]]] = None,
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    prev_event: Optional[Dict[str, Any]] = None,
    bids: Optional[List[Dict[str, Any]]] = None,
    asks: Optional[List[Dict[str, Any]]] = None,
    omit_fundamental: Optional[bool] = None,
    private_values: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    del fundamental_true
    mode = normalize_spread_input_mode(spread_input_mode)
    omit = resolve_spread_omit_fundamental(omit_fundamental)
    shout_list = list(shouts or [])
    schedule = _normalize_private_values(private_values)
    if mode == "zip_parity":
        state: Dict[str, Any] = {
            "current_pv": float(current_pv),
            "fundamental": float(fundamental_estimate),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "units_available": int(units_available),
            "q_max": int(q_max),
            "shouts": shout_list,
            "private_values": schedule,
        }
        return _drop_fundamental_keys(state) if omit else state
    if mode == "book":
        state = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bids": _normalize_resting_levels(bids),
            "asks": _normalize_resting_levels(asks),
            "units_available": int(units_available),
            "q_max": int(q_max),
            "fundamental_estimate": float(fundamental_estimate),
            "current_pv": float(current_pv),
            "prev_event": normalize_prev_event(prev_event),
            "private_values": schedule,
        }
        return _drop_fundamental_keys(state) if omit else state
    state = {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "units_available": int(units_available),
        "q_max": int(q_max),
        "fundamental_estimate": float(fundamental_estimate),
        "current_pv": float(current_pv),
        "shouts": shout_list,
        "private_values": schedule,
    }
    return _drop_fundamental_keys(state) if omit else state


def spread_pricing_fundamental(
    *,
    fundamental_estimate: float,
    fundamental_true: float,
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
) -> float:
    del fundamental_true, spread_input_mode
    return float(fundamental_estimate)


def spread_feature_names_for_prompts(
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    *,
    omit_fundamental: Optional[bool] = None,
) -> str:
    mode = normalize_spread_input_mode(spread_input_mode)
    omit = resolve_spread_omit_fundamental(omit_fundamental)
    if mode == "book":
        names = ["private_values", "current_pv"]
        if not omit:
            names.insert(1, "fundamental_estimate")
        names += [
            "best_bid",
            "best_ask",
            "bids",
            "asks",
            "units_available",
            "q_max",
            "prev_event",
        ]
        return ", ".join(names)
    if mode == "zip_parity":
        names = [
            "best_bid",
            "best_ask",
            "units_available",
            "q_max",
            "current_pv",
            "shouts",
        ]
        if not omit:
            names.insert(5, "fundamental")
        names.append("private_values")
        return ", ".join(names)
    names = [
        "best_bid",
        "best_ask",
        "units_available",
        "q_max",
        "current_pv",
        "shouts",
    ]
    if not omit:
        names.insert(5, "fundamental_estimate")
    names.append("private_values")
    return ", ".join(names)


def spread_input_schema_lines(
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    *,
    omit_fundamental: Optional[bool] = None,
) -> List[str]:
    mode = normalize_spread_input_mode(spread_input_mode)
    omit = resolve_spread_omit_fundamental(omit_fundamental)
    no_f_note = (
        "Banned observation: fundamental_estimate / fundamental (F̂) are NOT "
        "provided this run — do not read or invent them. Normalize using quotes, "
        "inventory, current_pv magnitude, or own prev_event only."
    )
    if mode == "zip_parity":
        lines = [
            "market_state is the CURRENT book at decision time (ZIP-parity schema; "
            "you receive only this dict):",
            "- 'current_pv': float — relative private value for the unit being priced now "
            "(large monetary scale; ZIP λ_rel)",
        ]
        if not omit:
            lines.append(
                "- 'fundamental': float — agent's noisy / projected fundamental estimate F̂ "
                "(large monetary scale; same quantity ZIP-noisy uses for "
                "λ = current_pv + F̂). NOT the latent true F_t."
            )
        lines += [
            "- 'best_bid': float | null — best bid, relative to fundamental (~0 scale)",
            "- 'best_ask': float | null — best ask, relative to fundamental (~0 scale)",
            "- 'units_available': int — remaining units the agent can still trade",
            "- 'q_max': int — maximum inventory capacity",
            "- 'private_values': list[float] — remaining-unit relative private values "
            "(same full schedule PV strategies receive; large monetary units, not [0,1])",
            "- 'shouts': list[dict] — every prior market shout this episode "
            "(accepted and rejected), identical to the ZIP observation stream, each:",
            "  {'time': int, 'price_abs': float, 'price_rel': float, "
            "'side': int (BUY=1, SELL=-1), 'accepted': bool}",
        ]
        if omit:
            lines.append(
                "CRITICAL scale note: best_bid/best_ask/price_rel are ~0-relative; "
                "current_pv and price_abs are large monetary units (often hundreds+), "
                "NOT fractions in [0,1]. Shade outputs MUST be in [0,1]."
            )
            lines.append(no_f_note)
        else:
            lines.append(
                "CRITICAL scale note: best_bid/best_ask/price_rel are ~0-relative; "
                "current_pv, fundamental, and price_abs are large monetary units "
                "(often hundreds+), NOT fractions in [0,1]. Shade outputs MUST be in [0,1]."
            )
            lines.append(
                "ZIP parity: λ_abs ≈ current_pv + fundamental. Higher shade = less aggressive."
            )
        lines += [
            "If best_bid or best_ask is None, that quote is absent — do not treat None "
            "as 0.0 in arithmetic. If both quotes exist and best_ask <= best_bid, "
            "treat spread as zero.",
            "You may compute spread as (best_ask - best_bid) when both are non-null.",
            "Use ONLY market_state. 'private_values' is the remaining-unit relative "
            "schedule (same vector PV strategies receive; large monetary units). "
            "You do not receive fundamental_estimate, LOB depth lists, or "
            "accepted-trades-only history — shouts already include rejects "
            "(ZIP observation parity).",
        ]
        return lines
    if mode == "book":
        lines = [
            "Sp book observation as market_state (signature "
            "agent_strategy(market_state)). Prefer quote/depth/inventory/own-feedback "
            "signals matching the Sp-book principle menu:",
            "- 'best_bid' / 'best_ask': float | null — relative to fundamental (~0 scale)",
            "- 'bids' / 'asks': list[dict] — current resting depth only, best-first, each "
            "{'price_rel': float, 'quantity': float} (NOT an L-window orders tape)",
            "- 'units_available' / 'q_max': int — remaining demand and capacity",
            "- 'prev_event': null | dict — this seat's last own submit "
            "{'price_abs','price_rel','side','accepted'}, or None",
            "- 'current_pv': float — relative private value/cost of the unit being "
            "priced on this decision (large monetary scale)",
            "- 'private_values': list[float] — full relative private-value schedule for "
            "all units (length q_max; entries are NOT removed after trades; large "
            "monetary units, NOT fractions in [0,1])",
        ]
        if not omit:
            lines.append(
                "- 'fundamental_estimate': float — absolute noisy projected fundamental "
                "F̃ (same quantity ShadeToLimit uses for pricing). Never invent true F_t. "
                "F̃ is absolute (~1e5 scale); private_values / current_pv are relative "
                "(often hundreds+) — do not subtract or divide them as if same units."
            )
        else:
            lines.append(no_f_note)
        lines += [
            "CRITICAL scale note: private_values / current_pv are large relative monetary "
            "units (often hundreds+), NOT [0,1]. F̃ is absolute (~1e5); "
            "best_bid/best_ask/price_rel are ~0-relative. Shade outputs MUST be in [0,1].",
            "NO shouts / full accept-reject tape / L-window orders. "
            "If best_bid or best_ask is None, that quote is absent — do not treat None "
            "as 0.0 in arithmetic.",
        ]
        return lines
    lines = [
        "market_state is the CURRENT book at decision time (you receive only this dict):",
        "- 'best_bid': float | null — best bid price, relative to fundamental (~0 scale)",
        "- 'best_ask': float | null — best ask price, relative to fundamental (~0 scale)",
        "- 'units_available': int — remaining units the agent can still trade",
        "- 'q_max': int — maximum inventory capacity",
    ]
    if not omit:
        lines.append(
            "- 'fundamental_estimate': float — agent's noisy estimate of fundamental "
            "(large monetary scale)"
        )
    lines += [
        "- 'current_pv': float — relative private value for the unit being priced now "
        "(large monetary scale)",
        "- 'private_values': list[float] — remaining-unit relative private values "
        "(same full schedule PV strategies receive; large monetary units, not [0,1])",
        "- 'shouts': list[dict] — every prior market shout this episode "
        "(accepted and rejected), each:",
        "  {'time': int, 'price_abs': float, 'price_rel': float, "
        "'side': int (BUY=1, SELL=-1), 'accepted': bool}",
    ]
    if omit:
        lines.append(
            "CRITICAL scale note: best_bid/best_ask/price_rel are ~0-relative; current_pv "
            "and price_abs are large monetary units (often hundreds+), NOT fractions in "
            "[0,1]. Shade outputs MUST be in [0,1]."
        )
        lines.append(no_f_note)
    else:
        lines.append(
            "CRITICAL scale note: best_bid/best_ask/price_rel are ~0-relative; current_pv, "
            "fundamental_estimate, and price_abs are large monetary units (often hundreds+), NOT "
            "fractions in [0,1]. Shade outputs MUST be in [0,1]."
        )
    lines += [
        "If best_bid or best_ask is None, that quote is absent — do not treat None "
        "as 0.0 in arithmetic. If both quotes exist and best_ask <= best_bid, "
        "treat spread as zero.",
        "You may compute spread as (best_ask - best_bid) when both are non-null.",
        "Use ONLY market_state. private_values is the remaining-unit schedule "
        "(same as PV). You do not receive LOB depth lists or accepted-trades-only "
        "history — shouts already include rejects.",
    ]
    return lines




def spread_book_dummy_state(
    *,
    omit_fundamental: Optional[bool] = None,
) -> Dict[str, Any]:
    state = {
        "best_bid": -1.0,
        "best_ask": 1.0,
        "bids": [{"price_rel": -1.0, "quantity": 1.0}],
        "asks": [{"price_rel": 1.0, "quantity": 1.0}],
        "units_available": 3,
        "q_max": 3,
        "fundamental_estimate": 100_000.0,
        "current_pv": 100.0,
        "prev_event": None,
        "private_values": [100.0, 50.0, 25.0],
    }
    if resolve_spread_omit_fundamental(omit_fundamental):
        return _drop_fundamental_keys(state)
    return state
