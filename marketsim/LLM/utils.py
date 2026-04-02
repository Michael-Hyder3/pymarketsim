from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json


def existing_path(path: Optional[str]) -> Optional[Path]:
    """Return a Path only when the input path exists on disk."""
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.exists() else None


def load_json_if_exists(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load JSON from an existing path; otherwise return None."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON payload to disk with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent)


def safe_strategy_call(
    strategy_func: Callable,
    rel_pvs: List[float],
    market_history: Dict[str, Any],
    market_state: Dict[str, Any],
) -> Any:
    """Invoke strategy function and return None on any runtime error."""
    try:
        return strategy_func(rel_pvs, market_history, market_state)
    except Exception:
        return None


def parse_strategy_output(result: Any) -> Optional[Tuple[float, int]]:
    """Parse strategy output into (nonnegative shade, positive quantity)."""
    if result is None:
        return None
    try:
        shade = max(0.0, float(result[0]))
        quantity = int(max(1, result[1]))
        return shade, quantity
    except Exception:
        return None
