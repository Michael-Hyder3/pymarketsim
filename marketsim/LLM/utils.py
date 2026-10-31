from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import inspect
import json
import warnings





def existing_path(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def load_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Dict[str, Any], indent: int = 2) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent)


def safe_strategy_call(
    strategy_func: Callable,
    rel_pvs: List[float],
    market_history: Dict[str, Any],
    market_state: Dict[str, Any],
    *,
    strategy_type: str = "full",
) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="Mean of empty slice.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="invalid value encountered in scalar divide",
            )
            if strategy_type == "spread":
                return strategy_func(market_state)
            if strategy_type == "lob":
                return strategy_func(market_history)
            if strategy_type == "pv":
                fund = None
                current_pv = None
                if isinstance(market_state, dict):
                    fund = market_state.get("fundamental_estimate")
                    current_pv = market_state.get("current_pv")
                if fund is None and isinstance(market_history, dict):
                    fund = market_history.get("fundamental_estimate")
                if current_pv is None and isinstance(market_history, dict):
                    current_pv = market_history.get("current_pv")
                try:
                    n_params = len(inspect.signature(strategy_func).parameters)
                except Exception:
                    n_params = 3
                if n_params <= 1:
                    return strategy_func(rel_pvs)
                if n_params == 2:
                    return strategy_func(rel_pvs, fund)
                return strategy_func(rel_pvs, fund, current_pv)
            params = list(inspect.signature(strategy_func).parameters.values())
            if len(params) <= 1:
                return strategy_func(rel_pvs)
            if len(params) == 2:
                return strategy_func(rel_pvs, market_history)
            return strategy_func(rel_pvs, market_history, market_state)
    except Exception:
        return None


def parse_strategy_output(
    result: Any,
    *,
    clip: bool = True,
) -> Optional[Tuple[float, int]]:
    if result is None:
        return None
    try:
        if isinstance(result, (int, float)):
            shade = float(result)
        elif isinstance(result, (list, tuple)) and len(result) >= 1:
            shade = float(result[0])
        else:
            return None
        if shade != shade or shade in (float("inf"), float("-inf")):
            return None
        if clip:
            shade = min(1.0, max(0.0, shade))
        elif shade < 0.0 or shade > 1.0:
            return None
        return shade, 1
    except Exception:
        return None
    return None
