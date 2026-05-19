from __future__ import annotations

import json
import random
from typing import Any, Dict, Optional

from marketsim.fourheap.constants import BUY, SELL
from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig


def _zi_strategy(private_values, market_history, market_state):
    return random.uniform(0.0, 1.0)


def generate_offline_example(
    *,
    config: ReflectionMarketConfig,
    seed: int,
    output_path: str,
    max_events: int = 40,
) -> Dict[str, Any]:
    order_type = config.llm_order_type
    strategy = _zi_strategy

    result = run_episode(
        llm_strategy_func=strategy,
        random_seed=seed,
        capture_detail=True,
        llm_action_mode="shade",
        **config.to_run_episode_kwargs(),
    )

    diagnostic = result.get("diagnostic_json")
    if diagnostic is None:
        raise RuntimeError("Diagnostic JSON missing from offline example run")

    events = diagnostic.get("events", [])
    if max_events > 0:
        diagnostic["events"] = events[:max_events]

    payload = {
        "agent_type": diagnostic.get("agent_type", "buyer" if order_type == BUY else "seller"),
        "agent_private_values_relative": diagnostic.get("agent_private_values_relative"),
        "timesteps": diagnostic.get("timesteps"),
        "events": diagnostic.get("events", []),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def load_offline_example(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
