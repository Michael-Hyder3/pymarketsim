from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from marketsim.LLM.hbl_order_memory import (
    HBL_EXACT_SEED_FLAG,
    strategy_requests_hbl_exact_seed,
)

__all_seed_helpers__ = (
    "HBL_EXACT_SEED_FLAG",
    "strategy_requests_hbl_exact_seed",
)

LOB_SEED_SCALE_CLASSIC = "classic"
LOB_SEED_SCALE_ZI = "zi_scale"
LOB_SEED_SCALE_HBL = "hbl"
LOB_SEED_SCALE_HBL_SCAFFOLD = "hbl_scaffold"
LOB_SEED_SCALE_BASIC = "basic"
LOB_SEED_SCALES = (
    LOB_SEED_SCALE_CLASSIC,
    LOB_SEED_SCALE_ZI,
    LOB_SEED_SCALE_HBL,
    LOB_SEED_SCALE_HBL_SCAFFOLD,
    LOB_SEED_SCALE_BASIC,
)

DEFAULT_LOB_SEED_SCALE = LOB_SEED_SCALE_CLASSIC
DEFAULT_LEARNING_LESSON_CAPACITY = 8
ZERO_TRADE_SURPLUS_EPS = 1e-6
OVER_SHADE_X_CAP_THRESHOLD = 0.1
SHADE_COLLAPSE_MEAN_EPS = 0.01
DEFAULT_LEARNING_PARENT_MAX_CHILDREN = 20

_OVER_SHADE_WHY_RE = re.compile(
    r"(over[\s-]?shade|no\s+fills?|zero\s+surplus|surplus\s*(≈|=|~)?\s*0|"
    r"did\s+not\s+trade|never\s+trades?|no\s+trades?)",
    re.IGNORECASE,
)
_SHADE_COLLAPSE_WHY_RE = re.compile(
    r"(shade\s*collaps|near[\s-]?zero\s+shade|mean\s+shade|always[\s-]?take|"
    r"over[\s-]?aggress|too\s+aggress|shade\s*(≈|=|~)?\s*0|under[\s-]?shade)",
    re.IGNORECASE,
)


def normalize_lob_seed_scale(value: Optional[str]) -> str:
    raw = str(value or DEFAULT_LOB_SEED_SCALE).strip().lower()
    return raw


def classic_lob_order_memory_seed_code() -> str:
    return "\n".join(
        [
            "def agent_strategy(market_history):",
            "    # HBL-lite seed: shade from L-window order memory (same stream HBL sees).",
            "    orders = list(market_history.get('orders') or [])",
            "    current_pv = abs(float(market_history.get('current_pv') or 0.0))",
            "    if not orders:",
            "        return 0.25",
            "    n = float(len(orders))",
            "    accepted = sum(1 for o in orders if o.get('accepted'))",
            "    reject_rate = 1.0 - (accepted / n)",
            "    buys = sum(1 for o in orders if int(o.get('side', 0)) == 1)",
            "    sells = sum(1 for o in orders if int(o.get('side', 0)) == -1)",
            "    side_pressure = abs(buys - sells) / n",
            "    # Accepted aggressiveness: mean |price_rel| among fills (0 if none).",
            "    filled_prices = [",
            "        abs(float(o.get('price_rel', 0.0)))",
            "        for o in orders if o.get('accepted')",
            "    ]",
            "    if filled_prices:",
            "        aggr = sum(filled_prices) / len(filled_prices)",
            "        aggr_term = aggr / (aggr + 50.0)",
            "    else:",
            "        aggr_term = 0.0",
            "    pv_term = current_pv / (current_pv + 500.0) if current_pv > 0 else 0.0",
            "    shade = (",
            "        0.35 * reject_rate",
            "        + 0.30 * side_pressure",
            "        + 0.20 * aggr_term",
            "        + 0.15 * pv_term",
            "    )",
            "    return min(max(shade, 0.0), 1.0)",
        ]
    )






def book_orders_basic_seed_code() -> str:
    return _agent_strategy_source_from_module("hbl_scaffold_agent_strategy.py")


def _agent_strategy_source_from_module(filename: str) -> str:
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / filename
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text.rstrip() + "\n"
    keep = [
        node
        for node in tree.body
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign),
        )
    ]
    parts = [ast.get_source_segment(text, node) or "" for node in keep]
    return ("\n\n".join(p for p in parts if p.strip())).rstrip() + "\n"








def lob_order_memory_seed_code(
    *,
    lob_seed_scale: str = DEFAULT_LOB_SEED_SCALE,
    lob_input_mode: str = "raw",
) -> str:
    from marketsim.LLM.lob_features import is_lob_book_orders_mode, normalize_lob_input_mode

    mode = normalize_lob_input_mode(lob_input_mode)
    if is_lob_book_orders_mode(mode):
        return book_orders_basic_seed_code()
    scale = normalize_lob_seed_scale(lob_seed_scale)
    return classic_lob_order_memory_seed_code()


def lob_order_memory_seed_description(
    *,
    lob_seed_scale: str = DEFAULT_LOB_SEED_SCALE,
    lob_input_mode: str = "raw",
) -> str:
    from marketsim.LLM.lob_features import is_lob_book_orders_mode, normalize_lob_input_mode

    if is_lob_book_orders_mode(normalize_lob_input_mode(lob_input_mode)):
        return (
            "Schema foothold: bind o_LOB (BBO, PV, F̃, resting bids/asks, "
            "orders tape); fixed shade 0.01"
        )
    scale = normalize_lob_seed_scale(lob_seed_scale)
    return "HBL-lite order-memory shading heuristic"














class LessonMemory:
    def __init__(self, *, capacity: int = DEFAULT_LEARNING_LESSON_CAPACITY) -> None:
        self.capacity = max(1, int(capacity))
        self._lessons: List[str] = []
























_OUTCOME_SCORE_TOKEN_RE = re.compile(
    r"\b(adv|surp|surplus|shade|zi)\b|[+-]?\d+(?:\.\d+)?",
    flags=re.IGNORECASE,
)




def enrich_verify_error_hint(error: Optional[str]) -> str:
    message = str(error)
    lowered = message.lower()
    hints: List[str] = []
    if (
        "near-identical" in lowered
        or "identical or equivalent" in lowered
        or "outputs matched" in lowered
    ):
        if "outputs matched parent on probes" not in lowered:
            hints.append(
                "Hint: shade outputs matched an existing strategy on probes — "
                "change a constant, feature, or branch so probe shades differ."
            )
    if not hints:
        return message
    return message + " " + " ".join(hints)
