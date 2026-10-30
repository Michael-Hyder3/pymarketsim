from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

BO_TRAIN_STEP_INDEX = 7777


def load_strategy_from_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy file not found at: {file_path}")
    spec = importlib.util.spec_from_file_location("strategy_module", str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent_strategy


def infer_strategy_type(strategy_fn) -> str:
    try:
        params = list(inspect.signature(strategy_fn).parameters.values())
    except Exception:
        return "full"
    if len(params) == 3:
        names = [(p.name or "").lower() for p in params]
        if (
            any("private" in n or n in {"pvs", "pv", "theta"} for n in names)
            and any("fundamental" in n or n in {"ftil", "f_tilde", "f"} for n in names)
            and any("current" in n or n in {"v", "pv_here", "unit_pv"} for n in names)
        ):
            return "pv"
        return "full"
    if len(params) >= 3:
        return "full"
    if len(params) == 2:
        names = [(p.name or "").lower() for p in params]
        if any("private" in n or n in {"pvs", "pv", "theta"} for n in names) and any(
            "fundamental" in n or n in {"ftil", "f_tilde", "f"} for n in names
        ):
            return "pv"
    if len(params) == 1:
        name = (params[0].name or "").lower()
        if "market_state" in name or name in {"state", "ms"}:
            return "spread"
        if "market_history" in name or "history" in name:
            return "lob"
        if "private_values" in name or name in {"pvs", "pv"}:
            return "pv"
    return "full"


def _dummy_strategy(market_state):
    return 0.5
