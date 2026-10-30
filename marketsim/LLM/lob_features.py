from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

LOB_INPUT_MODES = (
    "raw",
    "hbl_parity",
    "book_orders",
    "features",
    "features_full",
    "features_f3",
    "features_f1",
)
DEFAULT_LOB_INPUT_MODE = "raw"

HBL_PARITY_LOB_KEYS = (
    "timestep",
    "L",
    "memory_start_time",
    "transaction_count",
    "orders",
    "current_pv",
    "best_bid_relative",
    "best_ask_relative",
    "grace_period",
    "side",
    "fundamental_estimate",
    "pricing_fundamental",
)

BOOK_ORDERS_LOB_KEYS = (
    "best_bid",
    "best_ask",
    "bids",
    "asks",
    "current_pv",
    "fundamental_estimate",
    "orders",
)

LOB_SCALAR_FEATURE_NAMES = (
    "spread_rel",
    "mid_rel",
    "imbalance",
    "depth_bid",
    "depth_ask",
    "trade_mom",
    "log_pv",
    "pv_minus_mid",
    "has_book",
)

LOB_FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "full": LOB_SCALAR_FEATURE_NAMES,
    "f3": ("spread_rel", "log_pv", "imbalance"),
    "f1": ("log_pv",),
}


def normalize_lob_input_mode(mode: Optional[str]) -> str:
    resolved = str(mode or DEFAULT_LOB_INPUT_MODE).strip().lower()
    if resolved == "features":
        resolved = "features_full"
    if resolved not in {
        "raw",
        "hbl_parity",
        "book_orders",
        "features_full",
        "features_f3",
        "features_f1",
    }:
        known = ", ".join(LOB_INPUT_MODES)
        raise ValueError(f"Unsupported lob_input_mode {mode!r}. Known: {known}")
    return resolved


def is_lob_features_mode(mode: Optional[str]) -> bool:
    try:
        resolved = normalize_lob_input_mode(mode)
    except ValueError:
        return False
    return resolved.startswith("features")


def is_lob_hbl_parity_mode(mode: Optional[str]) -> bool:
    try:
        return normalize_lob_input_mode(mode) == "hbl_parity"
    except ValueError:
        return False


def is_lob_book_orders_mode(mode: Optional[str]) -> bool:
    try:
        return normalize_lob_input_mode(mode) == "book_orders"
    except ValueError:
        return False


def lob_feature_set_name(mode: Optional[str]) -> str:
    resolved = normalize_lob_input_mode(mode)
    if resolved == "features_f3":
        return "f3"
    if resolved == "features_f1":
        return "f1"
    return "full"


def active_lob_feature_names(mode: Optional[str] = "features_full") -> Tuple[str, ...]:
    return LOB_FEATURE_SETS[lob_feature_set_name(mode)]


def compute_lob_scalar_features(history: Mapping[str, Any]) -> Dict[str, float]:
    lob = history.get("lob") if isinstance(history.get("lob"), dict) else {}
    bids = list(lob.get("bids_relative") or history.get("bids_relative") or [])
    asks = list(lob.get("asks_relative") or history.get("asks_relative") or [])
    depth_bid = float(len(bids))
    depth_ask = float(len(asks))
    depth = depth_bid + depth_ask
    has_book = 1.0 if depth > 0 else 0.0
    imbalance = ((depth_bid - depth_ask) / depth) if depth > 0 else 0.0

    spread = history.get("spread_relative")
    if spread is None:
        best_bid = history.get("best_bid_relative", lob.get("best_bid_relative"))
        best_ask = history.get("best_ask_relative", lob.get("best_ask_relative"))
        if best_bid is not None and best_ask is not None:
            spread = float(best_ask) - float(best_bid)
    spread_rel = float(spread) if spread is not None else 0.0

    mid = history.get("mid_relative")
    if mid is None:
        best_bid = history.get("best_bid_relative", lob.get("best_bid_relative"))
        best_ask = history.get("best_ask_relative", lob.get("best_ask_relative"))
        if best_bid is not None and best_ask is not None:
            mid = 0.5 * (float(best_bid) + float(best_ask))
    mid_rel = float(mid) if mid is not None else 0.0

    prices = [
        float(p)
        for p in (history.get("recent_trade_prices") or [])
        if p is not None
    ]
    if len(prices) >= 2:
        trade_mom = prices[-1] - prices[0]
    else:
        trade_mom = 0.0

    current_pv = float(history.get("current_pv") or 0.0)
    log_pv = math.log1p(abs(current_pv))
    pv_minus_mid = current_pv - mid_rel if mid is not None else 0.0

    return {
        "spread_rel": spread_rel,
        "mid_rel": mid_rel,
        "imbalance": imbalance,
        "depth_bid": depth_bid,
        "depth_ask": depth_ask,
        "trade_mom": float(trade_mom),
        "log_pv": float(log_pv),
        "pv_minus_mid": float(pv_minus_mid),
        "has_book": has_book,
    }


def _select_features(
    all_features: Mapping[str, float],
    *,
    lob_input_mode: str,
) -> Dict[str, float]:
    names = active_lob_feature_names(lob_input_mode)
    return {name: float(all_features[name]) for name in names}


def apply_lob_input_mode(
    history: Dict[str, Any],
    *,
    lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
) -> Dict[str, Any]:
    mode = normalize_lob_input_mode(lob_input_mode)
    if mode == "raw":
        return history
    if mode == "hbl_parity":
        lob = history.get("lob") if isinstance(history.get("lob"), dict) else {}
        best_bid = history.get("best_bid_relative", lob.get("best_bid_relative"))
        best_ask = history.get("best_ask_relative", lob.get("best_ask_relative"))
        grace = history.get("grace_period")
        side = history.get("side")
        fund_est = history.get("fundamental_estimate")
        pricing_f = history.get("pricing_fundamental")
        return {
            "timestep": history.get("timestep"),
            "L": history.get("L"),
            "memory_start_time": history.get("memory_start_time"),
            "transaction_count": history.get("transaction_count"),
            "orders": list(history.get("orders") or []),
            "current_pv": float(history.get("current_pv") or 0.0),
            "best_bid_relative": (
                None if best_bid is None else float(best_bid)
            ),
            "best_ask_relative": (
                None if best_ask is None else float(best_ask)
            ),
            "grace_period": (
                None if grace is None else float(grace)
            ),
            "side": (None if side is None else int(side)),
            "fundamental_estimate": (
                None if fund_est is None else float(fund_est)
            ),
            "pricing_fundamental": (
                None if pricing_f is None else float(pricing_f)
            ),
        }
    if mode == "book_orders":
        lob = history.get("lob") if isinstance(history.get("lob"), dict) else {}
        best_bid = history.get("best_bid")
        if best_bid is None:
            best_bid = history.get("best_bid_relative", lob.get("best_bid_relative"))
        best_ask = history.get("best_ask")
        if best_ask is None:
            best_ask = history.get("best_ask_relative", lob.get("best_ask_relative"))
        fund_est = history.get("fundamental_estimate")
        if fund_est is None:
            fund_est = history.get("pricing_fundamental")
        bids = history.get("bids")
        if bids is None:
            bids = lob.get("bids")
        if bids is None:
            bids = [
                {"price_rel": float(p), "quantity": 1.0}
                for p in (history.get("bids_relative") or lob.get("bids_relative") or [])
            ]
        asks = history.get("asks")
        if asks is None:
            asks = lob.get("asks")
        if asks is None:
            asks = [
                {"price_rel": float(p), "quantity": 1.0}
                for p in (history.get("asks_relative") or lob.get("asks_relative") or [])
            ]
        return {
            "best_bid": (None if best_bid is None else float(best_bid)),
            "best_ask": (None if best_ask is None else float(best_ask)),
            "bids": [dict(e) for e in list(bids or [])],
            "asks": [dict(e) for e in list(asks or [])],
            "current_pv": float(history.get("current_pv") or 0.0),
            "fundamental_estimate": (
                None if fund_est is None else float(fund_est)
            ),
            "orders": list(history.get("orders") or []),
        }

    all_features = compute_lob_scalar_features(history)
    features = _select_features(all_features, lob_input_mode=mode)
    out: Dict[str, Any] = {
        "timestep": history.get("timestep"),
        "current_pv": float(history.get("current_pv") or 0.0),
        "features": dict(features),
        **features,
        "lob_snapshots": [],
        "market_trades": [],
        "recent_trade_prices": [],
        "last_trade_price": None,
        "total_trades": int(history.get("total_trades") or 0),
        "lob": {
            "best_bid_relative": None,
            "best_ask_relative": None,
            "bids_relative": [],
            "asks_relative": [],
        },
        "best_bid_relative": None,
        "best_ask_relative": None,
        "bids_relative": [],
        "asks_relative": [],
        "orders": [],
        "L": history.get("L"),
        "memory_start_time": history.get("memory_start_time"),
        "transaction_count": history.get("transaction_count"),
        "spread_relative": features.get("spread_rel", 0.0),
        "mid_relative": (
            all_features.get("mid_rel", 0.0) if "mid_rel" in features else 0.0
        ),
    }
    return out








def lob_book_orders_dummy_history() -> Dict[str, Any]:
    rawish = {
        "orders": [
            {
                "time": 3,
                "side": 1,
                "price_rel": -5.0,
                "quantity": 1,
                "accepted": True,
                "match_price_rel": -4.5,
                "agent_id": 1,
                "order_id": 101,
            },
            {
                "time": 4,
                "side": -1,
                "price_rel": 6.0,
                "quantity": 1,
                "accepted": False,
                "match_price_rel": None,
                "agent_id": 2,
                "order_id": 102,
            },
        ],
        "current_pv": 100.0,
        "fundamental_estimate": 100000.0,
        "best_bid_relative": -5.0,
        "best_ask_relative": 6.0,
        "bids": [
            {"price_rel": -5.0, "quantity": 1.0},
            {"price_rel": -12.0, "quantity": 2.0},
        ],
        "asks": [
            {"price_rel": 6.0, "quantity": 1.0},
            {"price_rel": 15.0, "quantity": 3.0},
        ],
        "lob": {
            "best_bid_relative": -5.0,
            "best_ask_relative": 6.0,
            "bids": [
                {"price_rel": -5.0, "quantity": 1.0},
                {"price_rel": -12.0, "quantity": 2.0},
            ],
            "asks": [
                {"price_rel": 6.0, "quantity": 1.0},
                {"price_rel": 15.0, "quantity": 3.0},
            ],
        },
        "L": 6,
        "grace_period": 10.0,
        "side": 1,
        "lob_snapshots": [{"best_bid_relative": -1.0}],
        "market_trades": [{"time": 1, "price": 0.0}],
    }
    return apply_lob_input_mode(rawish, lob_input_mode="book_orders")


def lob_input_schema_lines(
    lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
) -> List[str]:
    mode = normalize_lob_input_mode(lob_input_mode)
    if mode == "book_orders":
        return [
            "market_history is the CURRENT full limit-order-book state plus the recent "
            "order tape (you receive only this dict):",
            "- 'best_bid': float | null — live best bid (relative to fundamental, ~0)",
            "- 'best_ask': float | null — live best ask (relative to fundamental, ~0)",
            "- 'bids': list of ALL unmatched resting BUY orders currently in the book, "
            "best-first. Each entry: {'price_rel': float, 'quantity': float}. "
            "This is the full live bid side, not only the best bid.",
            "- 'asks': list of ALL unmatched resting SELL orders currently in the book, "
            "best-first. Each entry: {'price_rel': float, 'quantity': float}. "
            "This is the full live ask side, not only the best ask.",
            "- 'current_pv': float — relative private value for the unit being priced "
            "now (large monetary scale)",
            "- 'fundamental_estimate': float — agent's noisy / projected fundamental "
            "estimate F̂ (absolute / large scale). NOT the latent true F_t.",
            "- 'orders': list of ALL bids and asks entered during the calendar time "
            "spanning the last L completed transactions (idle timesteps included; "
            "L is fixed by the simulator and is NOT provided as a field). Each entry:",
            "  {'time': int, 'side': 1=BUY / -1=SELL, 'price_rel': float "
            "(submit limit), 'quantity': int, "
            "'accepted': bool (True iff the order later matched), "
            "'match_price_rel': float | null, "
            "'agent_id': int, 'order_id': int}",
            "CRITICAL: 'bids'/'asks' = current resting book; 'orders' = historical "
            "submit tape in the L-window. Do not confuse them.",
            "CRITICAL keys: each bids/asks/orders entry uses 'price_rel' and "
            "'quantity' — never 'price'. Spread is not a field: if needed, set "
            "spread = best_ask - best_bid after None-checks (local variable OK).",
            "Do NOT call market_history.get('spread_relative'), "
            "market_history.get('bids_relative'), market_history.get('asks_relative'), "
            "or market_history.get('mid_relative') — those keys are absent.",
            "CRITICAL scale note: best_bid/best_ask and price_rel are relative (~0). "
            "fundamental_estimate and current_pv are large monetary units. Shade "
            "outputs MUST be in [0,1]. Handle None bids/asks / match_price_rel safely. "
            "When orders/bids/asks lists are empty, do NOT call np.mean/np.std on "
            "empty lists — return a small default shade instead.",
            "Use ONLY market_history. You do not receive market_state, L, grace_period, "
            "memory_start_time, lob_snapshots, or the full private_values schedule.",
        ]
    if mode == "hbl_parity":
        return [
            "market_history contains PRIOR market information only "
            "(HBL-parity schema; you receive only this dict):",
            "- 'timestep': int — current decision timestep",
            "- 'L': int — HBL memory length (number of recent transactions "
            "defining the order window)",
            "- 'memory_start_time': int — earliest timestep included in the "
            "HBL order memory",
            "- 'transaction_count': int — total completed transactions so far",
            "- 'grace_period': float — HBL unmatched-order grace window "
            "(= 1/arrival_rate); used to weight rejected quotes in beliefs",
            "- 'side': int — focal seat side (BUY=+1, SELL=-1)",
            "- 'fundamental_estimate': float — agent's noisy / projected "
            "fundamental estimate in *absolute* form (same value as "
            "pricing_fundamental). NOT latent true F_t.",
            "- 'pricing_fundamental': float — the *same* absolute estimate "
            "(used by the shade→price channel). NOT latent true F_t.",
            "- 'orders': list of ALL bids and asks entered into the market from "
            "memory_start_time through the current timestep "
            "(identical to HBL strategy memory). Each entry:",
            "  {'time': int, 'side': 1=BUY / -1=SELL, 'price_rel': float "
            "(submit limit), 'quantity': int, "
            "'accepted': bool (True iff the order later matched), "
            "'match_price_rel': float | null (trade price on the matched leg; "
            "HBL TBL/TAL uses this, not price_rel), "
            "'agent_id': int, 'order_id': int}",
            "  This is the primary LOB signal — same stream HBL's belief uses.",
            "- 'best_bid_relative': float | null — live best bid at decision "
            "time (relative to fundamental, ~0 scale; HBL peeks the book)",
            "- 'best_ask_relative': float | null — live best ask at decision time",
            "- 'current_pv': float — relative private value for the unit being "
            "priced now (large monetary scale)",
            "orders covers every bid/ask submitted during the time it takes to "
            "complete the last L transactions (or from t=0 until L transactions "
            "exist). Same window HBL uses.",
            "CRITICAL scale note: price_rel, match_price_rel, best_*_relative "
            "are relative to fundamental (~0 scale). "
            "fundamental_estimate and pricing_fundamental are absolute (large, "
            "identical). current_pv is a large "
            "monetary unit (often hundreds+), NOT a fraction in [0,1]. Shade "
            "outputs MUST be in [0,1]. Handle None bids/asks / match_price_rel "
            "safely.",
            "Use ONLY market_history. You do not receive market_state, "
            "lob_snapshots, market_trades, depth lists, or the full "
            "private_values schedule — HBL observation parity.",
        ]
    return [
        "market_history contains PRIOR market information only (not the current "
        "pre-action book):",
        "- 'timestep': int — current decision timestep",
        "- 'L': int — HBL memory length (number of recent transactions defining "
        "the order window)",
        "- 'memory_start_time': int — earliest timestep included in the HBL "
        "order memory",
        "- 'transaction_count': int — total completed transactions so far",
        "- 'orders': list of ALL bids and asks entered into the market from "
        "memory_start_time through the current timestep "
        "(identical to HBL strategy memory). Each entry:",
        "  {'time': int, 'side': 1=BUY / -1=SELL, 'price_rel': float, "
        "'quantity': int, 'accepted': bool (True iff the order later matched), "
        "'agent_id': int, 'order_id': int}",
        "  This is the primary LOB signal — use it to reconstruct book "
        "pressure, rejects, and trades.",
        "- 'lob_snapshots': list of past post-step LOB snapshots "
        "(oldest to newest), each with:",
        "  {'best_bid_relative','best_ask_relative','bids_relative',"
        "'asks_relative'}",
        "- 'lob': most recent prior snapshot (same schema as one lob_snapshots "
        "entry)",
        "- 'best_bid_relative', 'best_ask_relative', 'bids_relative', "
        "'asks_relative': copied from the most recent prior snapshot for "
        "convenience",
        "- 'spread_relative': float | null — ask minus bid from most recent "
        "prior snapshot",
        "- 'mid_relative': float | null — midpoint from most recent prior "
        "snapshot",
        "- 'market_trades': list of executed trades before this decision "
        "(up to 10), each {'time','price','buyer_id','seller_id',"
        "'agent_involved'}",
        "- 'total_trades': int — trades seen so far before this decision",
        "- 'recent_trade_prices': list[float] (last up to 10 trade prices)",
        "- 'last_trade_price': float | None",
        "- 'current_pv': float — relative private value for the unit being "
        "priced now (large monetary scale)",
        "orders covers every bid/ask submitted during the time it takes to "
        "complete the last L transactions (or from t=0 until L transactions "
        "exist). Same window HBL uses.",
        "lob_snapshots contains at most the last 10 prior post-step snapshots "
        "(oldest to newest).",
        "On early timesteps the prior book may be empty (null bids/asks, empty "
        "depth lists).",
        "total_trades counts all trades seen so far; market_trades lists only "
        "the last up to 10.",
        "CRITICAL scale note: LOB price fields (price_rel and book prices) are "
        "relative to fundamental (~0 scale). current_pv is a large monetary "
        "unit (often hundreds+), NOT a fraction in [0,1]. Shade outputs MUST "
        "be in [0,1]. Handle None bids/asks safely.",
        "Use ONLY market_history. You do not receive market_state or the full "
        "private_values schedule.",
    ]
