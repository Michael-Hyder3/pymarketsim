from __future__ import annotations

import ast
import inspect
import re
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from marketsim.LLM.lob_features import (
    apply_lob_input_mode,
    lob_book_orders_dummy_history,
)
from marketsim.LLM.spread_input import (
    spread_book_dummy_state,
)
from marketsim.LLM.utils import parse_strategy_output, safe_strategy_call


_MIN_SHADE_ON_MISSING_QUOTE = 0.0


def _spread_dummy_state(
    *,
    omit_fundamental: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "current_pv": 100.0,
        "best_bid": None,
        "best_ask": None,
        "units_available": 3,
        "q_max": 3,
        "shouts": [],
        "private_values": [100.0, 50.0, 25.0],
    }
    if not omit_fundamental:
        state["fundamental_estimate"] = 1000.0
    state.update(overrides)
    if omit_fundamental:
        state.pop("fundamental_estimate", None)
        state.pop("fundamental", None)
    return state


def _spread_zip_parity_dummy(
    *,
    omit_fundamental: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    state = spread_zip_parity_dummy_state(omit_fundamental=omit_fundamental)
    state.update(overrides)
    if omit_fundamental:
        state.pop("fundamental_estimate", None)
        state.pop("fundamental", None)
    return state


def _spread_book_dummy(
    *,
    omit_fundamental: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    state = spread_book_dummy_state(omit_fundamental=omit_fundamental)
    state.update(overrides)
    if omit_fundamental:
        state.pop("fundamental_estimate", None)
        state.pop("fundamental", None)
    return state


_FORBIDDEN_NUMERIC_MODULES: FrozenSet[str] = frozenset({"numpy", "random"})

_FORBIDDEN_RAW_LOB_TOKENS_IN_FEATURES_MODE: FrozenSet[str] = frozenset(
    {
        "bids_relative",
        "asks_relative",
        "lob_snapshots",
        "orders",
        "market_trades",
        "recent_trade_prices",
        "best_bid_relative",
        "best_ask_relative",
    }
)

_FORBIDDEN_RAW_LOB_TOKENS_IN_HBL_PARITY: FrozenSet[str] = frozenset(
    {
        "lob_snapshots",
        "market_trades",
        "recent_trade_prices",
        "last_trade_price",
        "bids_relative",
        "asks_relative",
        "spread_relative",
        "mid_relative",
        "total_trades",
    }
)

_FORBIDDEN_RAW_LOB_TOKENS_IN_BOOK_ORDERS: FrozenSet[str] = frozenset(
    {
        "lob_snapshots",
        "market_trades",
        "recent_trade_prices",
        "last_trade_price",
        "bids_relative",
        "asks_relative",
        "spread_relative",
        "mid_relative",
        "total_trades",
        "best_bid_relative",
        "best_ask_relative",
        "memory_start_time",
        "transaction_count",
        "grace_period",
        "pricing_fundamental",
    }
)

_COMBINATION_PROBE_PV_GRIDS: Tuple[List[float], ...] = (
    [100.0, 50.0, 25.0],
    [10.0, 20.0, 30.0],
    [-10.0, 5.0, 15.0],
    [100.0, 100.0, 100.0],
    [100.0, 10.0, 1.0],
    [5.0, -3.0, 8.0],
    [200.0, 50.0, 10.0, 5.0],
)

_COMBINATION_PROBE_LOB_HISTORIES: Tuple[Dict[str, Any], ...] = (
    {
        "timestep": 1,
        "L": 6,
        "memory_start_time": 0,
        "transaction_count": 0,
        "orders": [
            {"time": 0, "side": 1, "price_rel": -5.0, "quantity": 1, "accepted": False, "agent_id": 1, "order_id": 1},
            {"time": 0, "side": -1, "price_rel": 5.0, "quantity": 1, "accepted": False, "agent_id": 2, "order_id": 2},
        ],
        "lob": {
            "best_bid_relative": -5.0,
            "best_ask_relative": 5.0,
            "bids_relative": [-5.0, -10.0],
            "asks_relative": [5.0, 10.0],
        },
        "best_bid_relative": -5.0,
        "best_ask_relative": 5.0,
        "bids_relative": [-5.0, -10.0],
        "asks_relative": [5.0, 10.0],
        "spread_relative": 10.0,
        "mid_relative": 0.0,
        "lob_snapshots": [],
        "market_trades": [],
        "total_trades": 0,
        "recent_trade_prices": [],
        "last_trade_price": None,
        "current_pv": 100.0,
    },
    {
        "timestep": 2,
        "L": 6,
        "memory_start_time": 0,
        "transaction_count": 1,
        "orders": [
            {"time": 0, "side": 1, "price_rel": -20.0, "quantity": 1, "accepted": True, "agent_id": 0, "order_id": 3},
            {"time": 1, "side": -1, "price_rel": 2.0, "quantity": 1, "accepted": True, "agent_id": 1, "order_id": 4},
            {"time": 1, "side": 1, "price_rel": -30.0, "quantity": 1, "accepted": False, "agent_id": 2, "order_id": 5},
        ],
        "lob": {
            "best_bid_relative": -20.0,
            "best_ask_relative": 2.0,
            "bids_relative": [-20.0, -30.0],
            "asks_relative": [2.0, 8.0],
        },
        "best_bid_relative": -20.0,
        "best_ask_relative": 2.0,
        "bids_relative": [-20.0, -30.0],
        "asks_relative": [2.0, 8.0],
        "spread_relative": 22.0,
        "mid_relative": -9.0,
        "lob_snapshots": [],
        "market_trades": [{"time": 1, "price": -1.0, "buyer_id": 0, "seller_id": 1, "agent_involved": False}],
        "total_trades": 1,
        "recent_trade_prices": [-1.0],
        "last_trade_price": -1.0,
        "current_pv": 50.0,
    },
    {
        "timestep": 3,
        "L": 6,
        "memory_start_time": 1,
        "transaction_count": 2,
        "orders": [
            {"time": 1, "side": 1, "price_rel": -1.0, "quantity": 1, "accepted": False, "agent_id": 3, "order_id": 6},
            {"time": 2, "side": -1, "price_rel": 30.0, "quantity": 1, "accepted": False, "agent_id": 4, "order_id": 7},
        ],
        "lob": {
            "best_bid_relative": -1.0,
            "best_ask_relative": 30.0,
            "bids_relative": [-1.0, -4.0],
            "asks_relative": [30.0, 40.0],
        },
        "best_bid_relative": -1.0,
        "best_ask_relative": 30.0,
        "bids_relative": [-1.0, -4.0],
        "asks_relative": [30.0, 40.0],
        "spread_relative": 31.0,
        "mid_relative": 14.5,
        "lob_snapshots": [],
        "market_trades": [],
        "total_trades": 0,
        "recent_trade_prices": [],
        "last_trade_price": None,
        "current_pv": 200.0,
    },
)


class StrategyVerifier:
    def __init__(self, function_name: str = "agent_strategy") -> None:
        self.function_name = function_name

    @staticmethod
    def _forbidden_import_error(code: str) -> Optional[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in _FORBIDDEN_NUMERIC_MODULES:
                        return (
                            f"Forbidden import '{alias.name}'. "
                            "Use only pure Python (no numpy/random)."
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", 1)[0]
                if root in _FORBIDDEN_NUMERIC_MODULES:
                    return (
                        f"Forbidden import from '{module}'. "
                        "Use only pure Python (no numpy/random)."
                    )
        return None

    @staticmethod
    def _forbidden_raw_lob_in_features_mode(code: str) -> Optional[str]:
        lowered = (code or "").lower()
        hit = sorted(
            token
            for token in _FORBIDDEN_RAW_LOB_TOKENS_IN_FEATURES_MODE
            if token in lowered
        )
        if not hit:
            return None
        return (
            "lob_input_mode=features forbids raw nested LOB fields "
            f"({', '.join(hit)}). Use market_history['features'] scalars "
            "(spread_rel, mid_rel, imbalance, depth_bid, depth_ask, trade_mom, "
            "log_pv, pv_minus_mid, has_book) or current_pv."
        )

    @staticmethod
    def _forbidden_raw_lob_in_hbl_parity_mode(code: str) -> Optional[str]:
        lowered = (code or "").lower()
        hit = sorted(
            token
            for token in _FORBIDDEN_RAW_LOB_TOKENS_IN_HBL_PARITY
            if token in lowered
        )
        if not hit:
            return None
        return (
            "lob_input_mode=hbl_parity forbids non-HBL fields "
            f"({', '.join(hit)}). Use orders / L / memory_start_time / "
            "transaction_count / best_bid_relative / best_ask_relative / "
            "current_pv only."
        )

    @staticmethod
    def _market_history_key_hits(code: str, tokens: FrozenSet[str]) -> List[str]:
        text = code or ""
        hit: List[str] = []
        for token in tokens:
            if re.search(
                rf"""market_history\s*(?:\[|\.get\()\s*['\"]{re.escape(token)}['\"]""",
                text,
                flags=re.IGNORECASE,
            ):
                hit.append(token)
        return sorted(hit)

    @staticmethod
    def _forbidden_raw_lob_in_book_orders_mode(code: str) -> Optional[str]:
        hit = StrategyVerifier._market_history_key_hits(
            code, _FORBIDDEN_RAW_LOB_TOKENS_IN_BOOK_ORDERS
        )
        if re.search(r"""market_history\s*(?:\[|\.get\()\s*['\"]L['\"]""", code or ""):
            hit = sorted(set(hit) | {"L"})
        if not hit:
            return None
        return (
            "lob_input_mode=book_orders forbids HBL/raw extras "
            f"({', '.join(hit)}). Use bids / asks / orders / best_bid / "
            "best_ask / current_pv / fundamental_estimate only "
            "(L is not provided). Local vars may compute spread from "
            "best_ask-best_bid; do not read spread_relative from market_history."
        )

    @staticmethod
    def _compile_strategy(code: str) -> Callable:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        fn = namespace.get("agent_strategy")
        if fn is None:
            raise ValueError("agent_strategy not found in generated code")
        return fn

    @staticmethod
    def _reject_invalid_on_prev_event(
        fn: Callable,
        *,
        omit_fundamental: bool = False,
    ) -> Optional[str]:
        probes = [
            (
                "own reject",
                _spread_book_dummy(
                    omit_fundamental=omit_fundamental,
                    prev_event={
                        "price_abs": 1001.0,
                        "price_rel": 1.0,
                        "side": 1,
                        "accepted": False,
                    },
                ),
            ),
            (
                "own accept",
                _spread_book_dummy(
                    omit_fundamental=omit_fundamental,
                    prev_event={
                        "price_abs": 999.0,
                        "price_rel": -1.0,
                        "side": 1,
                        "accepted": True,
                    },
                ),
            ),
        ]
        for name, state in probes:
            try:
                result = fn(state)
                parsed = parse_strategy_output(result, clip=False)
            except Exception as exc:
                return (
                    f"RuntimeError on book prev_event probe ({name}): "
                    f"{type(exc).__name__}: {exc}"
                )
            if parsed is None:
                return (
                    f"Book prev_event probe ({name}) returned {result!r}; "
                    "must be a finite shade in [0,1]."
                )
        return None

    @staticmethod
    def _reject_unshaded_on_missing_quotes(
        fn: Callable,
        *,
        mode: str = "classic",
        omit_fundamental: bool = False,
    ) -> Optional[str]:
        if mode == "zip_parity":
            factory = _spread_zip_parity_dummy
        elif mode == "book":
            factory = _spread_book_dummy
        else:
            factory = _spread_dummy_state

        def _state(**kwargs: Any) -> Dict[str, Any]:
            return factory(omit_fundamental=omit_fundamental, **kwargs)

        probes = [
            ("empty book", _state(best_bid=None, best_ask=None)),
            ("bid-only", _state(best_bid=-1.0, best_ask=None)),
            ("ask-only", _state(best_bid=None, best_ask=1.0)),
        ]
        if mode != "book":
            probes.append(
                (
                    "empty book with shouts",
                    _state(
                        best_bid=None,
                        best_ask=None,
                        shouts=[
                            {
                                "time": 0,
                                "price_abs": 999.0,
                                "price_rel": -1.0,
                                "side": 1,
                                "accepted": False,
                            },
                            {
                                "time": 1,
                                "price_abs": 1001.0,
                                "price_rel": 1.0,
                                "side": -1,
                                "accepted": True,
                            },
                        ],
                    ),
                )
            )
        for name, state in probes:
            try:
                result = fn(state)
                parsed = parse_strategy_output(result, clip=False)
            except Exception as exc:
                return (
                    f"RuntimeError on missing-quote probe ({name}): "
                    f"{type(exc).__name__}: {exc}"
                )
            if parsed is None:
                return (
                    f"Missing-quote probe ({name}): returned {result!r}, "
                    "which must be a finite numeric shade in [0,1]."
                )
        return None

    @staticmethod
    def _forbidden_menu_label_scaffolding(code: str) -> Optional[str]:
        src = code or ""
        patterns = (
            (
                r"""(?m)^\s*(?:principle|form_class|shade_error_direction)\s*=\s*['\"]""",
                "assignment",
            ),
            (
                r"""\bif\s+(?:principle|form_class|shade_error_direction)\s*==""",
                "dispatch",
            ),
        )
        for pattern, kind in patterns:
            if re.search(pattern, src):
                return (
                    f"Menu-label scaffolding ({kind}) is forbidden in agent_strategy. "
                    "Do not assign or branch on principle / form_class / "
                    "shade_error_direction strings — those are DIAGNOSIS labels only. "
                    "Write the shade math from market inputs."
                )
        return None

    @staticmethod
    def _forbidden_fundamental_in_omit_mode(code: str) -> Optional[str]:
        hit: List[str] = []
        for token in ("fundamental_estimate", "fundamental"):
            if re.search(
                rf"""market_state\s*(?:\[|\.get\()\s*['\"]{re.escape(token)}['\"]""",
                code or "",
                flags=re.IGNORECASE,
            ):
                hit.append(token)
        if not hit:
            return None
        return (
            "spread_omit_fundamental forbids reading "
            f"({', '.join(hit)}) from market_state. Use quotes, inventory, "
            "current_pv, prev_event/shouts only — do not invent F̂."
        )


    def verify(
        self,
        code: str,
        *,
        required_args: int = 1,
        input_mode: str = "pv",
        forbid_numpy_random: bool = False,
        parent_code: Optional[str] = None,
        spread_omit_fundamental: bool = False,
    ) -> Tuple[bool, Optional[str], Optional[Callable]]:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}", None
        except Exception as exc:
            return False, f"ParseError: {exc}", None

        from marketsim.LLM.reflection_pipeline.code_edit import (
            parent_helper_completeness_error,
        )

        completeness = parent_helper_completeness_error(parent_code, code)
        if completeness:
            return False, completeness, None

        if forbid_numpy_random:
            forbidden = self._forbidden_import_error(code)
            if forbidden:
                return False, forbidden, None

        if input_mode == "lob_features":
            forbidden_raw = self._forbidden_raw_lob_in_features_mode(code)
            if forbidden_raw:
                return False, forbidden_raw, None
        if input_mode == "lob_hbl_parity":
            forbidden_extra = self._forbidden_raw_lob_in_hbl_parity_mode(code)
            if forbidden_extra:
                return False, forbidden_extra, None
        if input_mode == "lob_book_orders":
            forbidden_extra = self._forbidden_raw_lob_in_book_orders_mode(code)
            if forbidden_extra:
                return False, forbidden_extra, None
        omit_f = bool(spread_omit_fundamental)
        if omit_f and input_mode in {"spread", "spread_zip_parity", "spread_book"}:
            forbidden_f = self._forbidden_fundamental_in_omit_mode(code)
            if forbidden_f:
                return False, forbidden_f, None
        menu_scaffold = self._forbidden_menu_label_scaffolding(code)
        if menu_scaffold:
            return False, menu_scaffold, None

        try:
            fn = self._compile_strategy(code)
        except Exception as exc:
            return False, f"CompileError: {exc}", None

        try:
            params = list(inspect.signature(fn).parameters.values())
            if len(params) != required_args:
                return (
                    False,
                    f"agent_strategy must accept exactly {required_args} argument(s)",
                    None,
                )
        except Exception as exc:
            return False, f"SignatureError: {exc}", None

        try:
            dummy_private_values = [100.0, 50.0, 25.0]

            if required_args == 1:
                if input_mode == "spread":
                    dummy_state = _spread_dummy_state(omit_fundamental=omit_f)
                    result = fn(dummy_state)
                elif input_mode == "spread_zip_parity":
                    dummy_state = _spread_zip_parity_dummy(omit_fundamental=omit_f)
                    result = fn(dummy_state)
                elif input_mode == "spread_book":
                    dummy_state = _spread_book_dummy(omit_fundamental=omit_f)
                    result = fn(dummy_state)
                elif input_mode == "lob_features":
                    result = fn(lob_features_dummy_history())
                elif input_mode == "lob_hbl_parity":
                    result = fn(lob_hbl_parity_dummy_history())
                elif input_mode == "lob_book_orders":
                    result = fn(lob_book_orders_dummy_history())
                elif input_mode == "lob":
                    prior_snapshot = {
                        "best_bid_relative": -10.0,
                        "best_ask_relative": 10.0,
                        "bids_relative": [-10.0, -20.0],
                        "asks_relative": [10.0, 20.0],
                    }
                    dummy_history = {
                        "timestep": 1,
                        "L": 6,
                        "memory_start_time": 0,
                        "transaction_count": 0,
                        "orders": [
                            {
                                "time": 0,
                                "side": 1,
                                "price_rel": -5.0,
                                "quantity": 1,
                                "accepted": False,
                                "agent_id": 1,
                                "order_id": 1,
                            },
                            {
                                "time": 0,
                                "side": -1,
                                "price_rel": 5.0,
                                "quantity": 1,
                                "accepted": True,
                                "agent_id": 2,
                                "order_id": 2,
                            },
                        ],
                        "market_trades": [],
                        "total_trades": 0,
                        "recent_trade_prices": [],
                        "last_trade_price": None,
                        "lob_snapshots": [prior_snapshot],
                        "lob": prior_snapshot,
                        "best_bid_relative": -10.0,
                        "best_ask_relative": 10.0,
                        "bids_relative": [-10.0, -20.0],
                        "asks_relative": [10.0, 20.0],
                        "spread_relative": 20.0,
                        "mid_relative": 0.0,
                        "current_pv": 100.0,
                    }
                    result = fn(dummy_history)
                else:
                    result = fn(dummy_private_values)
            elif required_args == 2 and input_mode == "pv":
                result = fn(dummy_private_values, 1000.0)
            elif required_args == 3 and input_mode == "pv":
                result = fn(dummy_private_values, 1000.0, 100.0)
            else:
                dummy_history = {
                    "timestep": 0,
                    "market_trades": [],
                    "total_trades": 0,
                    "recent_trade_prices": [],
                    "last_trade_price": None,
                    "lob": {
                        "best_bid_relative": -10.0,
                        "best_ask_relative": 10.0,
                        "bids_relative": [-10.0, -20.0],
                        "asks_relative": [10.0, 20.0],
                    },
                    "best_bid_relative": -10.0,
                    "best_ask_relative": 10.0,
                    "bids_relative": [-10.0, -20.0],
                    "asks_relative": [10.0, 20.0],
                    "spread_relative": 20.0,
                    "mid_relative": 0.0,
                }
                dummy_state = {
                    "current_pv": 100.0,
                    "best_bid": None,
                    "best_ask": None,
                    "units_available": 3,
                    "q_max": 3,
                    "fundamental_estimate": 1000.0,
                }
                result = fn(dummy_private_values, dummy_history, dummy_state)

            parsed = parse_strategy_output(result, clip=False)
            if parsed is None:
                return (
                    False,
                    (
                        f"Strategy returned {result!r}, which must be a finite "
                        "numeric shade in [0,1] (out-of-range values are rejected; "
                        "clip inside agent_strategy before returning)."
                    ),
                    None,
                )

            if input_mode in {"spread", "spread_zip_parity", "spread_book"}:
                mode = (
                    "zip_parity"
                    if input_mode == "spread_zip_parity"
                    else "book"
                    if input_mode == "spread_book"
                    else "classic"
                )
                missing_quote_err = self._reject_unshaded_on_missing_quotes(
                    fn,
                    mode=mode,
                    omit_fundamental=omit_f,
                )
                if missing_quote_err:
                    return False, missing_quote_err, None
                if mode == "book":
                    prev_event_err = self._reject_invalid_on_prev_event(
                        fn,
                        omit_fundamental=omit_f,
                    )
                    if prev_event_err:
                        return False, prev_event_err, None

        except Exception as exc:
            from marketsim.LLM.reflection_pipeline.learning_focused import (
                enrich_verify_error_hint,
            )

            return (
                False,
                enrich_verify_error_hint(
                    f"RuntimeError during execution with dummy data: "
                    f"{type(exc).__name__}: {exc}"
                ),
                None,
            )

        return True, None, fn




