from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def build_generation_prompt(
    *,
    parent_code: str,
    parent_reflection: str,
    batch_context: List[Dict[str, Any]],
    strategy_output_mode: str,
    market_description: str,
    offline_example: Optional[Dict[str, Any]] = None,
) -> str:
    context_json = json.dumps(batch_context, indent=2)
    requirements = [
        "Return exactly one Python function named agent_strategy.",
        "Signature: agent_strategy(private_values, market_history, market_state).",
        "Return a single float x in [0,1] representing the shading level.",
        "Assume the market only allows single-unit trades.",
        "Only use provided inputs; do not import external libraries.",
    ]
    if strategy_output_mode == "x_cap":
        requirements.append("Interpret x as x_cap (max shade); output should remain in [0,1].")
    else:
        requirements.append("Interpret x as shading fraction; output should remain in [0,1].")

    sections = [
        "You are creating a trading strategy for a continuous double auction.",
        "",
        "## Market Description",
        market_description.strip(),
        "",
        "## Input Schema",
        "market_history is a dict with:",
        "- 'timestep': int",
        "- 'market_trades': list of dicts with keys {'time', 'price'} (may be empty)",
        "- 'total_trades': int",
        "market_state is a dict with keys: 'best_bid', 'best_ask', 'units_available',",
        "'q_max', 'fundamental_estimate', 'max_bid', 'min_ask'.",
        "Use market_history['market_trades'] to compute trade-price stats;",
        "if empty, fall back to market_state['fundamental_estimate'] or 0.",
        "If best_bid or best_ask is None, avoid using them directly;",
        "use a safe fallback (e.g., fundamental_estimate) and skip division.",
        "If best_ask <= best_bid, treat spread as zero and avoid division.",
    ]
    if offline_example:
        example_text = json.dumps(offline_example, indent=2)
        sections += [
            "",
            "## Offline ZI Example (single agent perspective)",
            example_text,
            "Use this example to understand the data format and typical values.",
        ]
    if parent_code.strip():
        sections += [
            "",
            "## Parent Strategy",
            "```python",
            parent_code.strip(),
            "```",
        ]
        sections += [
            "",
            "## Parent Reflection",
            parent_reflection.strip() or "(none)",
        ]
    sections += [
        "",
        "## Batch Context Summary",
        context_json,
        "",
        "## Requirements",
        *[f"- {req}" for req in requirements],
        "",
        "Respond with exactly one ```python``` block.",
    ]
    return "\n".join(sections)


def build_debug_prompt(*, code: str, error: str) -> str:
    return "\n".join(
        [
            "The following strategy code failed verification.",
            "Fix the errors and return a corrected version.",
            "",
            "## Error",
            error.strip(),
            "",
            "## Code",
            "```python",
            code.strip(),
            "```",
            "",
            "Respond with exactly one ```python``` block.",
        ]
    )


def build_reflection_prompt(
    *,
    code: str,
    batch_summary: Dict[str, Any],
    reward: float,
) -> str:
    summary_text = json.dumps(batch_summary, indent=2)
    return "\n".join(
        [
            "You are a critic reviewing a trading strategy after evaluation.",
            "Provide a 3-part note with the exact headings:",
            "Environment:",
            "Performance:",
            "Fix:",
            "Keep each part concise and actionable.",
            "",
            "## Strategy",
            "```python",
            code.strip(),
            "```",
            "",
            f"## Batch Reward: {reward:.4f}",
            "",
            "## Batch Summary",
            summary_text,
        ]
    )
