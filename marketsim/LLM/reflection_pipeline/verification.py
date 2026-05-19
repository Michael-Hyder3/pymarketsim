from __future__ import annotations

import ast
from typing import Callable, Optional, Tuple, Any, Dict, List

from marketsim.LLM.utils import parse_strategy_output, safe_strategy_call


class StrategyVerifier:
    def __init__(self, function_name: str = "agent_strategy") -> None:
        self.function_name = function_name

    @staticmethod
    def _compile_strategy(code: str) -> Callable:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        fn = namespace.get("agent_strategy")
        if fn is None:
            raise ValueError("agent_strategy not found in generated code")
        return fn

    def verify(self, code: str) -> Tuple[bool, Optional[str], Optional[Callable]]:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc}", None
        except Exception as exc:
            return False, f"ParseError: {exc}", None

        try:
            fn = self._compile_strategy(code)
        except Exception as exc:
            return False, f"CompileError: {exc}", None

        try:
            dummy_private_values = [100.0, 50.0, 25.0]
            dummy_history = {"timestep": 0, "market_trades": [], "total_trades": 0}
            dummy_state = {
                "max_bid": 100.0,
                "best_bid": None,
                "best_ask": None,
                "units_available": 3,
                "q_max": 3,
                "fundamental_estimate": 1000.0,
            }
            result = safe_strategy_call(fn, dummy_private_values, dummy_history, dummy_state)
            parsed = parse_strategy_output(result)
            if parsed is None:
                return False, "Strategy output not parseable as numeric in [0,1]", None
        except Exception as exc:
            return False, f"RuntimeError: {exc}", None

        return True, None, fn
