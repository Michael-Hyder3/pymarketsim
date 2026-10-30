from __future__ import annotations

import ast
import json
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from openai import OpenAI

from marketsim.LLM.code_generator import CodeGenerator
from marketsim.LLM.code_storage import CodeStorage
from marketsim.LLM.lob_features import (
    DEFAULT_LOB_INPUT_MODE,
    is_lob_features_mode,
    normalize_lob_input_mode,
)
from marketsim.LLM.reflection_pipeline.benchmark_suite import FixedEvalSuite, build_fixed_eval_suite
from marketsim.LLM.reflection_pipeline.config import (
    ReflectionMarketConfig,
    ReflectionConfigSpace,
    build_default_config_pool,
    build_default_config_space,
    sample_stratified_home_set,
    DEFAULT_PARENT_SELECTION_MODE,
    DEFAULT_Q_EVAL_MODE,
    DEFAULT_Q_REWARD_MODE,
    DEFAULT_Q_UPDATE_MODE,
    DEFAULT_CRITIC_MODE,
    CRITIC_MODES,
    FULL_REFRESH_PARALLEL_WORKERS,
    INACTIVE_BEAT_GAP,
    INACTIVE_GATE_MIN_NODES,
    INACTIVE_SURPLUS_GAP,
    PARENT_SELECTION_MODES,
    ZI_Q_REWARD_MODES,
)
from marketsim.LLM.reflection_pipeline.evaluation import (
    build_case_card_payload,
    evaluate_batch,
)
from marketsim.LLM.reflection_pipeline.metrics import (
    build_critic_batch_summary,
    summarize_batch_performance,
    summarize_per_config_performance,
)
from marketsim.LLM.reflection_pipeline.parallel_eval import refresh_nodes_parallel
from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import evaluate_batch_with_zi_benchmarks
from marketsim.LLM.reflection_pipeline.prompts import (
    build_generation_prompt,
    build_debug_prompt,
    build_reflection_prompt,
    build_plateau_revision_prompt,
    build_hypothesis_jump_prompt,
    build_combination_revision_prompt,
    collect_hypothesis_pairs_from_reflections,
    compute_menu_coverage_stats,
    combination_generator_instructions,
    derive_family_id,
    hypothesis_jump_generator_instructions,
    hypothesis_pair_differs_from_parent,
    is_structured_principle_form,
    menu_hypothesis_pairs,
    diagnosis_action_generator_instructions,
    parse_diagnosis_action,
    parse_principle_form_fields,
    parse_same_failure_as_last_hold,
    plateau_escape_generator_instructions,
    set_principle_menu_preset,
    validate_structured_curriculum_reflection,
    validate_structured_principle_form_reflection,
)
from marketsim.LLM.reflection_pipeline.diagnosis_agreement import (
    validate_code_implements_diagnosis,
)
from marketsim.LLM.reflection_pipeline.hypothesis_quality import (
    score_hypothesis_adherence,
)
from marketsim.LLM.reflection_pipeline.tree import RefinementTree
from marketsim.LLM.feedback_loop import DEFAULT_MARKET_DESCRIPTION
from marketsim.LLM.reflection_pipeline.verification import StrategyVerifier

DEFAULT_STRATEGY_CODE = ""


@dataclass
class GenerationResult:
    code: Optional[str]
    strategy_fn: Optional[Any]
    parent_id: int
    success: bool
    error: Optional[str]
    code_file: Optional[str]
    strategy_type: str
    node_id: Optional[int] = None
    escape_mode: bool = False
    hypothesis_jump: bool = False
    combination_mode: bool = False
    parent_select_mode: Optional[str] = None
    combination_source_codes: Optional[List[str]] = None
    combination_source_family_ids: Optional[List[str]] = None


CRITIC_CONTEXT_LIMIT = 8192
CRITIC_MAX_OUTPUT_TOKENS = 640
CRITIC_PROMPT_SAFETY_TOKENS = 96
CRITIC_CHARS_PER_TOKEN = 3
CRITIC_MAX_PROMPT_CHARS = (
    CRITIC_CONTEXT_LIMIT - CRITIC_MAX_OUTPUT_TOKENS - CRITIC_PROMPT_SAFETY_TOKENS
) * CRITIC_CHARS_PER_TOKEN
# Per-tape shout cap for case-card payloads (worst + good). Keep small for 8k context.
CRITIC_INTERACTION_LIMITS = (8, 5, 3, 0)
GENERATOR_INTERACTION_LIMIT = 5
CRITIC_SYSTEM_PROMPT = (
    "Return a concise 5-part reflection for advantage-mode batch evaluation. "
    "Use headings Environment, Performance, Mechanism, Principle, Fix. "
    "Reference FAILED per-config unit tests by test_id. "
    "Mechanism must cite at least two trajectory moments. Fix must implement Principle."
)
CRITIC_SYSTEM_PROMPT_ZI = (
    "Return a concise 5-part reflection for ZI beat-fraction evaluation. "
    "Use headings Environment, Performance, Mechanism, Principle, Fix. "
    "Reference FAILED ZI benchmark unit tests by test_id. "
    "Mechanism must cite at least two trajectory moments. "
    "Fix describes a revision direction; do not paste full replacement code."
)
CRITIC_SYSTEM_PROMPT_ZI_ADVANTAGE = (
    "Return a concise 5-part reflection for ZI benchmark evaluation with advantage Q reward. "
    "Use headings Environment, Performance, Mechanism, Principle, Fix. "
    "Reference FAILED ZI benchmark unit tests by test_id. "
    "Mechanism must cite at least two trajectory moments. "
    "Fix describes a revision direction; do not paste full replacement code."
)
CRITIC_SYSTEM_PROMPT_STRUCTURED_ZI_CURRICULUM = (
    "Return a field-first DIAGNOSIS for PV-only curriculum evaluation. "
    "Start with the DIAGNOSIS block (requested_x_cap, shade_error_direction, error_type, "
    "fix_class, consensus). Then MECHANISM and FIX only (no Environment section). "
    "Copy computed values from Curriculum Facts unless you justify a change in MECHANISM. "
    "Reference FAILED tests by test_id. "
    "Propose general PV→x_cap improvements — never hardcode shade_hi floats or test_ids in code. "
    "Higher x_cap = more shade (less aggressive). PV-only: no LOB/trend stories."
)
CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM = (
    "Return a field-first DIAGNOSIS for a PV-only shading strategy. "
    "Start with the DIAGNOSIS block (principle, form_class, features_used, parent_form, "
    "why_parent_fails, diversity_vs_parent). Then MECHANISM and FIX only "
    "(no Environment/Performance/Principle essay sections). "
    "Pick principle and form_class from the allowed PV menus. "
    "Optimize batch advantage using market principles — do not teach shade levels via "
    "ZI shade_hi or shade_more/shade_less. Never hardcode shade floats or test_ids. "
    "Higher x_cap = more shade (less aggressive). PV-only: private_values (Θ), "
    "fundamental_estimate (F̃), and current_pv (unit being priced) are available — "
    "prefer principles that can use them when useful; ban LOB/bid-ask/trend stories."
)
CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_LOB = (
    "Return a field-first DIAGNOSIS for LOB-only principle/form refinement. "
    "Start with the DIAGNOSIS block (principle, form_class, features_used, parent_form, "
    "why_parent_fails, diversity_vs_parent). Then MECHANISM and FIX only "
    "(no Environment/Performance/Principle essay sections). "
    "Pick principle and form_class from the allowed LOB order-memory menus "
    "(fill_vs_reject / side_pressure / order_flow_imbalance / recent_aggressiveness / "
    "current_pv_vs_flow). Prefer market_history['orders'] (accepted/side/price_rel/L) "
    "over snapshot depth alone. "
    "Optimize batch advantage using market principles — do not teach shade levels via "
    "ZI shade_hi or shade_more/shade_less. Never hardcode shade floats or test_ids. "
    "Higher x_cap = more shade (less aggressive). LOB-only: use market_history fields; "
    "do not invent a full private_values schedule."
)
CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_SPREAD = (
    "Return a field-first DIAGNOSIS for a spread-only shading strategy. "
    "Start with DIAGNOSIS (principle, form_class, features_used, current_form, "
    "why_this_fails, action, proposed_change), then MECHANISM and FIX only. "
    "Pick principle and form_class from the allowed spread menus. "
    "Optimize batch advantage from the case cards — do not teach shade levels. "
    "Use market_state fields (quotes, inventory, PV, and when present shouts or "
    "own prev_event); do not invent a PV schedule or LOB depth."
)


def _structured_principle_form_system_prompt(*, strategy_type: str = "pv") -> str:
    if strategy_type == "lob":
        return CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_LOB
    if strategy_type == "spread":
        return CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_SPREAD
    return CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM


def _forced_menu_pair_diagnosis(
    *,
    principle: str,
    form_class: str,
    strategy_type: str = "pv",
    lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
) -> str:
    """Injected Parent Reflection DIAGNOSIS for menu-sweep forced pairs."""
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        features = "imbalance, spread_rel, log_pv"
    elif strategy_type == "lob":
        features = "bids_relative, asks_relative, spread_relative, current_pv"
    elif strategy_type == "spread":
        features = "best_bid, best_ask, units_available, current_pv"
    else:
        features = "pv_mean, pv_max, units_available"
    return (
        "DIAGNOSIS:\n"
        f"  principle: {principle}\n"
        f"  form_class: {form_class}\n"
        f"  features_used: {features}\n"
        "  parent_form: menu-sweep forced pair\n"
        "  why_parent_fails: systematic menu coverage on this market\n"
        f"  diversity_vs_parent: force principle={principle} form_class={form_class}\n"
        "MECHANISM: Menu-sweep warmup assigns this (principle, form_class) pair.\n"
        f"FIX: Implement principle={principle} with form_class={form_class}.\n"
    )


def _fallback_principle_form_diagnosis(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
) -> str:
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        return (
            "DIAGNOSIS:\n"
            "  principle: prior_spread_or_mid\n"
            "  form_class: ratio\n"
            "  features_used: spread_rel, mid_rel, log_pv\n"
            "  parent_form: unknown\n"
            "  why_parent_fails: unknown\n"
            "  diversity_vs_parent: change form vs parent\n"
            "MECHANISM: unknown\n"
            "FIX: unknown\n"
        )
    if strategy_type == "lob":
        return (
            "DIAGNOSIS:\n"
            "  principle: prior_spread_or_mid\n"
            "  form_class: ratio\n"
            "  features_used: spread_relative, mid_relative, current_pv\n"
            "  parent_form: unknown\n"
            "  why_parent_fails: unknown\n"
            "  diversity_vs_parent: change form vs parent\n"
            "MECHANISM: unknown\n"
            "FIX: unknown\n"
        )
    if strategy_type == "spread":
        return (
            "DIAGNOSIS:\n"
            "  principle: bid_ask_spread_width\n"
            "  form_class: ratio\n"
            "  features_used: best_bid, best_ask, current_pv\n"
            "  parent_form: unknown\n"
            "  why_parent_fails: unknown\n"
            "  diversity_vs_parent: change form vs parent\n"
            "MECHANISM: unknown\n"
            "FIX: unknown\n"
        )
    return (
        "DIAGNOSIS:\n"
        "  principle: pv_rank_or_distance_to_best\n"
        "  form_class: rank\n"
        "  features_used: unknown\n"
        "  parent_form: unknown\n"
        "  why_parent_fails: unknown\n"
        "  diversity_vs_parent: change form vs parent\n"
        "MECHANISM: unknown\n"
        "FIX: unknown\n"
    )


def _fallback_plateau_diagnosis(
    *,
    strategy_type: str = "pv",
    lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
) -> str:
    if strategy_type == "lob" and is_lob_features_mode(lob_input_mode):
        return (
            "DIAGNOSIS:\n"
            "  principle: book_imbalance\n"
            "  form_class: threshold\n"
            "  features_used: imbalance, spread_rel, has_book\n"
            "  parent_form: stalled parent form\n"
            "  why_parent_fails: local edits did not improve surplus\n"
            "  diversity_vs_parent: switch form_class vs stalled parent\n"
            "MECHANISM: Plateau escape fallback after critic failure.\n"
            "FIX: Implement a different form_class than the stalled parent.\n"
        )
    if strategy_type == "lob":
        return (
            "DIAGNOSIS:\n"
            "  principle: book_imbalance\n"
            "  form_class: threshold\n"
            "  features_used: bids_relative, asks_relative, spread_relative\n"
            "  parent_form: stalled parent form\n"
            "  why_parent_fails: local edits did not improve surplus\n"
            "  diversity_vs_parent: switch form_class vs stalled parent\n"
            "MECHANISM: Plateau escape fallback after critic failure.\n"
            "FIX: Implement a different form_class than the stalled parent.\n"
        )
    if strategy_type == "spread":
        return (
            "DIAGNOSIS:\n"
            "  principle: inventory_remaining_fraction\n"
            "  form_class: threshold\n"
            "  features_used: units_available, q_max, current_pv\n"
            "  parent_form: stalled parent form\n"
            "  why_parent_fails: local edits did not improve surplus\n"
            "  diversity_vs_parent: switch form_class vs stalled parent\n"
            "MECHANISM: Plateau escape fallback after critic failure.\n"
            "FIX: Implement a different form_class than the stalled parent.\n"
        )
    return (
        "DIAGNOSIS:\n"
        "  principle: pv_rank_or_distance_to_best\n"
        "  form_class: rank\n"
        "  features_used: pv_mean, pv_max\n"
        "  parent_form: stalled parent form\n"
        "  why_parent_fails: local edits did not improve surplus\n"
        "  diversity_vs_parent: switch form_class vs stalled parent\n"
        "MECHANISM: Plateau escape fallback after critic failure.\n"
        "FIX: Implement a different form_class than the stalled parent.\n"
    )


def _normalize_strategy_code(code: str) -> str:
    """AST-normalized form for duplicate detection (whitespace/comments ignored)."""
    stripped = code.strip()
    if not stripped:
        return ""
    try:
        tree = ast.parse(stripped)
        return ast.dump(tree, annotate_fields=False)
    except SyntaxError:
        return "".join(stripped.split())


def _strategy_code_equivalent(left: str, right: str) -> bool:
    left_norm = _normalize_strategy_code(left)
    right_norm = _normalize_strategy_code(right)
    return bool(left_norm) and left_norm == right_norm


# Reject codegen that matches parent, any sibling, or the most recent N tree nodes.
DUPLICATE_CODE_RECENT_NODE_LIMIT = 10
FORBIDDEN_STRATEGY_MAX_EXAMPLES = 6
FORBIDDEN_STRATEGY_MAX_CHARS = 1800
# After this many consecutive failed codegen/verify steps on a parent, cool it out of UCB.
PARENT_CODEGEN_FAIL_COOL_THRESHOLD = 2
# Mild temperature for duplicate recovery debug (local parent edits, not form jumps).
DUPLICATE_RECOVERY_TEMPERATURE = 0.3
# After this many consecutive non-improving children, revise the parent's hypothesis.
# Fire hypothesis replace sooner so stalled lineages do not burn many local edits.
PLATEAU_NON_IMPROVE_THRESHOLD = 4
# Minimum surplus gain (absolute) to count as an improvement vs parent.
PLATEAU_IMPROVE_SURPLUS_EPS = 1.0
# Probability of forcing the elite (best surplus / Q) parent instead of UCB.
ELITE_PARENT_BIAS_PROB = 0.25
# Probability of a UCB-parent hypothesis jump (new principle and/or form_class).
HYPOTHESIS_JUMP_PROB = 0.0
# Temperature for plateau-escape generation after hypothesis revision.
PLATEAU_ESCAPE_TEMPERATURE = 0.5
PLATEAU_ESCAPE_MODES = ("revision", "combination")
CODEGEN_RECOVERY_MODES = ("plan_guided", "legacy")
DEFAULT_CODEGEN_RECOVERY_MODE = "legacy"

# Short names for vLLM-served checkpoints (same path style as Llama).
MODEL_ALIASES = {
    "llama": "models/Meta-Llama-3-8B-Instruct",
    "qwen": "models/Qwen2.5-Coder-32B-Instruct",
}
DEFAULT_MODEL_ALIAS = "llama"


def resolve_model_name(model: Optional[str] = None) -> str:
    """Map --model alias (or path) to the OpenAI model id sent to vLLM."""
    key = (model or DEFAULT_MODEL_ALIAS).strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    return key


def _truncate_strategy_code(code: str, *, max_chars: int = FORBIDDEN_STRATEGY_MAX_CHARS) -> str:
    text = str(code or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n# ... truncated ..."


def _collect_forbidden_strategy_examples(
    parent: Any,
    tree: Any,
    *,
    max_examples: int = FORBIDDEN_STRATEGY_MAX_EXAMPLES,
    max_chars: int = FORBIDDEN_STRATEGY_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Sibling snippets only — dup recovery should re-implement under the parent, not ban it."""
    examples: List[Dict[str, Any]] = []
    sibling_ids = sorted(
        (int(sid) for sid in (getattr(parent, "children", None) or [])),
        reverse=True,
    )
    for sibling_id in sibling_ids:
        if len(examples) >= max_examples:
            break
        try:
            sibling = tree.get(sibling_id)
        except Exception:
            continue
        code = getattr(sibling, "code", None)
        if not code or not str(code).strip():
            continue
        examples.append(
            {
                "node_id": sibling_id,
                "role": "sibling",
                "code": _truncate_strategy_code(str(code), max_chars=max_chars),
            }
        )
    return examples


def _duplicate_strategy_rejection_message(
    code: str,
    *,
    parent: Any,
    tree: Any,
    recent_limit: int = DUPLICATE_CODE_RECENT_NODE_LIMIT,
) -> Optional[str]:
    """Return a rejection message if ``code`` matches parent, siblings, or recent nodes."""
    if not code or not str(code).strip():
        return None

    candidates: Dict[int, Any] = {}
    parent_id = int(getattr(parent, "node_id", -1))
    if parent_id > 0:
        candidates[parent_id] = parent

    sibling_ids = set(getattr(parent, "children", None) or [])
    for sibling_id in sibling_ids:
        try:
            candidates[int(sibling_id)] = tree.get(int(sibling_id))
        except Exception:
            continue

    evaluable_ids = []
    if hasattr(tree, "evaluable_node_ids"):
        evaluable_ids = [int(nid) for nid in tree.evaluable_node_ids()]
    recent_ids = sorted(evaluable_ids, reverse=True)[: max(0, int(recent_limit))]
    for node_id in recent_ids:
        try:
            candidates[int(node_id)] = tree.get(int(node_id))
        except Exception:
            continue

    for node_id, node in sorted(candidates.items()):
        other = getattr(node, "code", None)
        if not other or not str(other).strip():
            continue
        if not _strategy_code_equivalent(code, str(other)):
            continue
        if node_id == parent_id:
            role = "parent"
        elif node_id in sibling_ids:
            role = "sibling"
        else:
            role = "recent tree node"
        return (
            f"Generated strategy is identical or equivalent to existing {role} "
            f"(node {node_id}). Do not return a duplicate of any strategy already in "
            f"the tree (parent, siblings, or the last {int(recent_limit)} nodes). "
            "Re-implement under the parent principle+form with a distinct shade "
            "expression (features / normalization / branches) — not comments, "
            "renames, coefficient-only, or +ε damper tweaks."
        )
    return None


def _critic_system_prompt(
    q_reward_mode: str,
    critic_mode: str = "default",
    *,
    strategy_type: str = "pv",
) -> str:
    if critic_mode == "structured_zi_curriculum":
        return CRITIC_SYSTEM_PROMPT_STRUCTURED_ZI_CURRICULUM
    if critic_mode == "structured_principle_form":
        return _structured_principle_form_system_prompt(strategy_type=strategy_type)
    if q_reward_mode == "zi_beat_fraction":
        return CRITIC_SYSTEM_PROMPT_ZI
    if q_reward_mode == "zi_advantage":
        return CRITIC_SYSTEM_PROMPT_ZI_ADVANTAGE
    return CRITIC_SYSTEM_PROMPT


def _slim_llm_interaction_for_critic(event: Dict[str, Any]) -> Dict[str, Any]:
    """Drop legacy duplicate fields before serializing critic trajectory data."""
    slim: Dict[str, Any] = {
        "time": event.get("time"),
        "transacted": event.get("transacted"),
        "agent_action": event.get("agent_action"),
    }
    for key in ("pre_lob", "post_lob"):
        lob = event.get(key) or {}
        slim[key] = {
            "best_bid_relative": lob.get("best_bid_relative"),
            "best_ask_relative": lob.get("best_ask_relative"),
            "bids_relative": lob.get("bids_relative"),
            "asks_relative": lob.get("asks_relative"),
        }
    for key in ("trade_price_relative", "surplus_earned"):
        if key in event:
            slim[key] = event[key]
    return slim


def _compact_case_card_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """First market's operational card, no tape / worst tag."""
    if not isinstance(payload, dict):
        return None
    cards = payload.get("case_cards")
    if not cards:
        return None
    card = dict(cards[0])
    card.pop("worst", None)
    card.pop("prior_hold", None)
    return card


def _attach_prior_hold_card(
    batch_data: Dict[str, Any],
    card: Dict[str, Any],
) -> Dict[str, Any]:
    """Copy a last-hold stats card onto batch_data for critic/generator prompts."""
    out = dict(batch_data)
    prior = dict(card)
    prior["prior_hold"] = True
    prior.pop("worst", None)
    out["prior_hold_card"] = prior
    return out


def _compress_batch_data_for_critic(
    batch_data: Dict[str, Any],
    *,
    max_interactions: int = 8,
) -> Dict[str, Any]:
    """Compact per-market case cards plus worst/good episode tapes (no raw LOB dumps)."""
    if not isinstance(batch_data, dict) or not batch_data.get("configs"):
        payload: Dict[str, Any] = {"case_cards": [], "worst_market": None, "tapes": {}}
    else:
        payload = build_case_card_payload(
            batch_data,
            max_interactions=max_interactions,
        )
    prior = batch_data.get("prior_hold_card") if isinstance(batch_data, dict) else None
    if isinstance(prior, dict) and prior:
        payload["prior_hold_card"] = dict(prior)
    return payload


def _estimate_critic_prompt_tokens(*parts: str) -> int:
    return sum(len(part) for part in parts) // CRITIC_CHARS_PER_TOKEN + 1


def _is_critic_context_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "context length" in message
        or "max_tokens" in message
        or "max_completion_tokens" in message
    )


def _critic_max_output_tokens(*, system_content: str, prompt: str) -> int:
    est_input = _estimate_critic_prompt_tokens(system_content, prompt)
    remaining = CRITIC_CONTEXT_LIMIT - est_input - 32
    return max(64, min(CRITIC_MAX_OUTPUT_TOKENS, remaining))


def _prepare_critic_prompt(
    *,
    code: str,
    batch_summary: Dict[str, Any],
    batch_data: Dict[str, Any],
    reward: float,
    reward_variance: float,
    include_data: bool,
    q_reward_mode: str = "advantage",
    critic_mode: str = "default",
    strategy_type: str = "pv",
    run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    menu_exploration_enabled: bool = True,
    lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
    require_last_hold_comparison: Optional[bool] = None,
) -> Tuple[str, int, str]:
    system_prompt = _critic_system_prompt(
        q_reward_mode,
        critic_mode=critic_mode,
        strategy_type=strategy_type,
    )
    if require_last_hold_comparison is None:
        require_last_hold_comparison = bool(
            isinstance(batch_data, dict) and batch_data.get("prior_hold_card")
        )
    if not include_data:
        prompt = build_reflection_prompt(
            code=code,
            batch_summary=batch_summary,
            batch_data=batch_data if include_data else {},
            reward=reward,
            reward_variance=reward_variance,
            include_data=False,
            q_reward_mode=q_reward_mode,
            critic_mode=critic_mode,
            strategy_type=strategy_type,
            run_explored_pairs=run_explored_pairs,
            menu_exploration_enabled=menu_exploration_enabled,
            lob_input_mode=lob_input_mode,
            require_last_hold_comparison=require_last_hold_comparison,
        )
        return prompt, _critic_max_output_tokens(system_content=system_prompt, prompt=prompt), system_prompt

    for limit in CRITIC_INTERACTION_LIMITS:
        compressed = _compress_batch_data_for_critic(batch_data, max_interactions=limit)
        prompt = build_reflection_prompt(
            code=code,
            batch_summary=batch_summary,
            batch_data=compressed,
            reward=reward,
            reward_variance=reward_variance,
            include_data=True,
            q_reward_mode=q_reward_mode,
            critic_mode=critic_mode,
            strategy_type=strategy_type,
            run_explored_pairs=run_explored_pairs,
            menu_exploration_enabled=menu_exploration_enabled,
            lob_input_mode=lob_input_mode,
            require_last_hold_comparison=require_last_hold_comparison,
        )
        if len(prompt) <= CRITIC_MAX_PROMPT_CHARS:
            return prompt, _critic_max_output_tokens(system_content=system_prompt, prompt=prompt), system_prompt

    prompt = build_reflection_prompt(
        code=code,
        batch_summary=batch_summary,
        batch_data={},
        reward=reward,
        reward_variance=reward_variance,
        include_data=False,
        q_reward_mode=q_reward_mode,
        critic_mode=critic_mode,
        strategy_type=strategy_type,
        run_explored_pairs=run_explored_pairs,
        menu_exploration_enabled=menu_exploration_enabled,
        lob_input_mode=lob_input_mode,
        require_last_hold_comparison=require_last_hold_comparison,
    )
    return prompt, _critic_max_output_tokens(system_content=system_prompt, prompt=prompt), system_prompt


def _format_reward_stats(reward: float, reward_variance: float) -> str:
    return f"reward={reward:.4f} reward_var={reward_variance:.4f}"


def _performance_payload(
    *,
    reward: float,
    reward_variance: float,
    per_config_rewards: Dict[str, float],
    per_config_reward_variances: Dict[str, float],
    overall: Optional[Dict[str, Any]] = None,
    zi_benchmark_comparison: Optional[Dict[str, Any]] = None,
    batch_data: Optional[Dict[str, Any]] = None,
    reward_mode: str = "advantage",
) -> Dict[str, Any]:
    from marketsim.LLM.reflection_pipeline.advantage_config_feedback import (
        enrich_advantage_summary_with_config_tests,
    )

    payload: Dict[str, Any] = {
        "reward": float(reward),
        "reward_variance": float(reward_variance),
        "per_config_rewards": dict(per_config_rewards),
        "per_config_reward_variances": dict(per_config_reward_variances),
    }
    if overall is not None:
        payload["overall"] = overall
    if isinstance(batch_data, dict) and batch_data.get("configs"):
        metrics = summarize_per_config_performance(batch_data, reward_mode=reward_mode)
        for label, row in metrics.items():
            cfg = row.get("config")
            if isinstance(cfg, dict):
                row["config"] = {"label": cfg.get("label", label)}
        payload["per_config_metrics"] = metrics
    if zi_benchmark_comparison is not None:
        payload["zi_benchmark_comparison"] = zi_benchmark_comparison
    elif per_config_rewards:
        payload = enrich_advantage_summary_with_config_tests(payload)
    return payload


def _build_batch_summary_for_critic(
    pipeline: "ReflectionPipeline",
    *,
    reward: float,
    reward_variance: float,
    per_config_rewards: Dict[str, float],
    per_config_reward_variances: Dict[str, float],
    batch_data: Dict[str, Any],
) -> Dict[str, Any]:
    return build_critic_batch_summary(
        reward=reward,
        reward_variance=reward_variance,
        per_config_rewards=per_config_rewards,
        per_config_reward_variances=per_config_reward_variances,
        batch_data=batch_data,
        reward_mode=pipeline.reward_mode,
        per_config_breakdown=pipeline.critic_per_config_metrics,
    )


def _critic_prompt_inputs(
    pipeline: "ReflectionPipeline",
    *,
    reward: float,
    reward_variance: float,
    per_config_rewards: Dict[str, float],
    per_config_reward_variances: Dict[str, float],
    batch_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Any]:
    if not pipeline.critic_include_data:
        return {}, {}
    batch_summary = _build_batch_summary_for_critic(
        pipeline,
        reward=reward,
        reward_variance=reward_variance,
        per_config_rewards=per_config_rewards,
        per_config_reward_variances=per_config_reward_variances,
        batch_data=batch_data,
    )
    compressed_data = _compress_batch_data_for_critic(batch_data)
    return batch_summary, compressed_data


class StrategyGenerator:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._generator = CodeGenerator(base_url=base_url, api_key=api_key, model_name=model_name)

    def generate(
        self,
        *,
        parent_code: str,
        parent_reflection: str,
        strategy_output_mode: str,
        offline_example: Optional[Dict[str, Any]],
        strategy_type: str = "full",
        use_reflection: bool = True,
        parent_performance: Optional[Dict[str, Any]] = None,
        generator_batch_summary: Optional[Dict[str, Any]] = None,
        generator_batch_data: Optional[Dict[str, Any]] = None,
        sibling_code: Optional[str] = None,
        sibling_context: Optional[Dict[str, Any]] = None,
        q_reward_mode: str = "advantage",
        critic_mode: str = "default",
        forbid_numpy_random: bool = False,
        forbidden_strategies: Optional[List[Dict[str, Any]]] = None,
        extra_instructions: Optional[List[str]] = None,
        temperature: float = 0.2,
        lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
    ) -> str:
        prompt = build_generation_prompt(
            parent_code=parent_code,
            parent_reflection=parent_reflection,
            strategy_output_mode=strategy_output_mode,
            market_description=DEFAULT_MARKET_DESCRIPTION,
            offline_example=offline_example,
            strategy_type=strategy_type,
            use_reflection=use_reflection,
            parent_performance=parent_performance,
            generator_batch_summary=generator_batch_summary,
            generator_batch_data=generator_batch_data,
            sibling_code=sibling_code,
            sibling_context=sibling_context,
            q_reward_mode=q_reward_mode,
            critic_mode=critic_mode,
            forbid_numpy_random=forbid_numpy_random,
            forbidden_strategies=forbidden_strategies,
            extra_instructions=extra_instructions,
            lob_input_mode=lob_input_mode,
        )
        return self._generator.generate_code(prompt, temperature=float(temperature))

    def debug(
        self,
        *,
        code: str,
        error: str,
        forbidden_strategies: Optional[List[Dict[str, Any]]] = None,
        force_new_form: bool = False,
        temperature: float = 0.2,
        parent_reflection: Optional[str] = None,
        plan_guided_recovery: bool = False,
        critic_mode: str = "default",
        strategy_type: str = "pv",
    ) -> str:
        prompt = build_debug_prompt(
            code=code,
            error=error,
            forbidden_strategies=forbidden_strategies,
            force_new_form=force_new_form,
            parent_reflection=parent_reflection,
            plan_guided_recovery=plan_guided_recovery,
            critic_mode=critic_mode,
            strategy_type=strategy_type,
        )
        return self._generator.generate_code(prompt, temperature=float(temperature))


class ReflectionCritic:
    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
    DEFAULT_API_KEY = "dummy"
    DEFAULT_MODEL_NAME = "models/Meta-Llama-3-8B-Instruct"
    FALLBACK_BASE_URLS = [
        "http://127.0.0.1:8000/v1",
        "http://localhost:8000/v1",
    ]

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or self.DEFAULT_BASE_URL
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or self.DEFAULT_API_KEY
        self._base_url_candidates = [resolved_base_url] + [
            u for u in self.FALLBACK_BASE_URLS if u != resolved_base_url
        ]
        self._api_key = resolved_api_key
        self._model_name = model_name or os.getenv("OPENAI_MODEL") or self.DEFAULT_MODEL_NAME
        self._client = OpenAI(base_url=resolved_base_url, api_key=resolved_api_key)

    def generate(
        self,
        *,
        code: str,
        batch_summary: Dict[str, Any],
        batch_data: Any,
        reward: float,
        reward_variance: float = 0.0,
        include_data: bool = True,
        q_reward_mode: str = "advantage",
        critic_mode: str = "default",
        strategy_type: str = "pv",
        parent_reflection: Optional[str] = None,
        sibling_reflections: Optional[List[str]] = None,
        run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
        menu_exploration_enabled: bool = True,
        lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
    ) -> str:
        require_last_hold_comparison = bool(
            isinstance(batch_data, dict) and batch_data.get("prior_hold_card")
        )
        prompt, max_tokens, system_prompt = _prepare_critic_prompt(
            code=code,
            batch_summary=batch_summary,
            batch_data=batch_data if isinstance(batch_data, dict) else {},
            reward=reward,
            reward_variance=reward_variance,
            include_data=include_data,
            q_reward_mode=q_reward_mode,
            critic_mode=critic_mode,
            strategy_type=strategy_type,
            run_explored_pairs=run_explored_pairs,
            menu_exploration_enabled=menu_exploration_enabled,
            lob_input_mode=lob_input_mode,
            require_last_hold_comparison=require_last_hold_comparison,
        )
        multi_config = len(batch_summary.get("per_config_rewards") or {}) > 1
        last_error = None
        for candidate_url in self._base_url_candidates:
            try:
                self._client = OpenAI(base_url=candidate_url, api_key=self._api_key)
                validation_feedback: Optional[str] = None
                reflection = ""
                for attempt in range(2):
                    user_prompt = prompt
                    if validation_feedback:
                        user_prompt = (
                            f"{prompt}\n\n"
                            "## Revision Required\n"
                            f"Your prior reflection was rejected: {validation_feedback}\n"
                            "Reply again with a valid DIAGNOSIS block and MECHANISM/FIX only."
                        )
                    response = self._client.chat.completions.create(
                        model=self._model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=max_tokens,
                    )
                    reflection = response.choices[0].message.content.strip()
                    if critic_mode == "structured_zi_curriculum":
                        validation_error = validate_structured_curriculum_reflection(
                            reflection,
                            multi_config=multi_config,
                        )
                    elif critic_mode == "structured_principle_form":
                        validation_error = validate_structured_principle_form_reflection(
                            reflection,
                            strategy_type=strategy_type,
                            parent_reflection=parent_reflection,
                            sibling_reflections=sibling_reflections,
                            run_explored_pairs=run_explored_pairs,
                            require_last_hold_comparison=require_last_hold_comparison,
                            lob_input_mode=lob_input_mode,
                            spread_input_mode=spread_input_mode,
                        )
                    else:
                        return reflection
                    if not validation_error:
                        return reflection
                    if attempt == 0:
                        validation_feedback = validation_error
                        continue
                    return reflection
                return reflection
            except Exception as exc:
                last_error = exc
                if "Connection error" in str(exc):
                    continue
                if include_data and _is_critic_context_error(exc):
                    prompt, max_tokens, system_prompt = _prepare_critic_prompt(
                        code=code,
                        batch_summary=batch_summary,
                        batch_data={},
                        reward=reward,
                        reward_variance=reward_variance,
                        include_data=False,
                        q_reward_mode=q_reward_mode,
                        critic_mode=critic_mode,
                        strategy_type=strategy_type,
                        run_explored_pairs=run_explored_pairs,
                        menu_exploration_enabled=menu_exploration_enabled,
                        lob_input_mode=lob_input_mode,
                        require_last_hold_comparison=require_last_hold_comparison,
                    )
                    include_data = False
                    validation_feedback = None
                    continue
                raise
        if last_error is not None:
            raise last_error
        if critic_mode == "structured_principle_form":
            return _fallback_principle_form_diagnosis(
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
            )
        return (
            "DIAGNOSIS:\n"
            "  shade_error_direction: reshape_pv_map\n"
            "  error_type: pv_mapping_wrong\n"
            "  fix_class: change_pv_feature\n"
            "  consensus: n/a\n"
            "MECHANISM: unknown\n"
            "FIX: unknown\n"
        )

    def generate_plateau_revision(
        self,
        *,
        prompt: str,
        critic_mode: str,
        stalled_form_class: Optional[str] = None,
        strategy_type: str = "pv",
        temperature: float = 0.4,
        run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
        lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
    ) -> str:
        """Ask the critic for a revised principle/form after a refine plateau."""
        system_prompt = _critic_system_prompt(
            "zi_advantage", critic_mode, strategy_type=strategy_type
        )
        if critic_mode == "structured_principle_form":
            base = _structured_principle_form_system_prompt(strategy_type=strategy_type)
            system_prompt = (
                base
                + " This is a plateau revision: pick a different form_class than the "
                "stalled hypothesis and explain why local edits failed."
            )
        user_prompt = prompt
        # Hard cap prompt size for 8k-context models (leave room for system+output).
        max_user_chars = CRITIC_MAX_PROMPT_CHARS
        if len(user_prompt) > max_user_chars:
            user_prompt = (
                user_prompt[: max_user_chars - 80].rstrip()
                + "\n\n[...plateau prompt truncated for context window...]\n"
            )
        last_error = None
        for candidate_url in self._base_url_candidates:
            try:
                self._client = OpenAI(base_url=candidate_url, api_key=self._api_key)
                truncated_retry = False
                while True:
                    validation_feedback: Optional[str] = None
                    reflection = ""
                    try:
                        for attempt in range(2):
                            request_prompt = user_prompt
                            if validation_feedback:
                                request_prompt = (
                                    f"{user_prompt}\n\n"
                                    "## Revision Required\n"
                                    f"Your prior reflection was rejected: {validation_feedback}\n"
                                    "Reply again with a valid DIAGNOSIS that revises the stalled "
                                    "hypothesis (different form_class)."
                                )
                            response = self._client.chat.completions.create(
                                model=self._model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": request_prompt},
                                ],
                                temperature=float(temperature),
                                max_tokens=CRITIC_MAX_OUTPUT_TOKENS,
                            )
                            reflection = response.choices[0].message.content.strip()
                            if critic_mode != "structured_principle_form":
                                return reflection
                            validation_error = validate_structured_principle_form_reflection(
                                reflection,
                                strategy_type=strategy_type,
                                run_explored_pairs=run_explored_pairs,
                                lob_input_mode=lob_input_mode,
                                spread_input_mode=spread_input_mode,
                            )
                            if not validation_error and stalled_form_class:
                                fields = parse_principle_form_fields(reflection)
                                new_form = fields.get("form_class")
                                if new_form and new_form == stalled_form_class:
                                    validation_error = (
                                        f"form_class must differ from stalled form_class "
                                        f"({stalled_form_class})"
                                    )
                            if not validation_error:
                                return reflection
                            if attempt == 0:
                                validation_feedback = validation_error
                                continue
                            return reflection
                        return reflection
                    except Exception as inner_exc:
                        if (
                            _is_critic_context_error(inner_exc)
                            and not truncated_retry
                        ):
                            print(
                                "[reflection] plateau revision context overflow; "
                                "retrying with truncated prompt...",
                                flush=True,
                            )
                            user_prompt = (
                                user_prompt[: max(2000, max_user_chars // 3)].rstrip()
                                + "\n\n[...further truncated after context error...]\n"
                                "Reply with DIAGNOSIS (different form_class), "
                                "MECHANISM, FIX only.\n"
                            )
                            truncated_retry = True
                            continue
                        raise
            except Exception as exc:
                last_error = exc
                if "Connection error" in str(exc):
                    continue
                # Soft-fail below rather than killing the run.
                break
        if last_error is not None:
            print(
                f"[reflection] plateau revision failed ({last_error}); "
                "using heuristic escape DIAGNOSIS",
                flush=True,
            )
        return _fallback_plateau_diagnosis(
            strategy_type=strategy_type,
            lob_input_mode=lob_input_mode,
        )

    def generate_hypothesis_jump(
        self,
        *,
        prompt: str,
        critic_mode: str,
        parent_reflection: str,
        strategy_type: str = "pv",
        temperature: float = 0.4,
        run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
        lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
    ) -> str:
        """Ask the critic for a new principle/form exploration jump from a UCB parent."""
        system_prompt = _critic_system_prompt(
            "zi_advantage", critic_mode, strategy_type=strategy_type
        )
        if critic_mode == "structured_principle_form":
            base = _structured_principle_form_system_prompt(strategy_type=strategy_type)
            system_prompt = (
                base
                + " This is a hypothesis exploration jump: change at least one of "
                "principle or form_class vs the parent hypothesis."
            )
        user_prompt = prompt
        max_user_chars = CRITIC_MAX_PROMPT_CHARS
        if len(user_prompt) > max_user_chars:
            user_prompt = (
                user_prompt[: max_user_chars - 80].rstrip()
                + "\n\n[...hypothesis jump prompt truncated for context window...]\n"
            )
        last_error = None
        for candidate_url in self._base_url_candidates:
            try:
                self._client = OpenAI(base_url=candidate_url, api_key=self._api_key)
                truncated_retry = False
                while True:
                    validation_feedback: Optional[str] = None
                    reflection = ""
                    try:
                        for attempt in range(2):
                            request_prompt = user_prompt
                            if validation_feedback:
                                request_prompt = (
                                    f"{user_prompt}\n\n"
                                    "## Revision Required\n"
                                    f"Your prior reflection was rejected: {validation_feedback}\n"
                                    "Reply again with a valid DIAGNOSIS that changes at least "
                                    "one of principle or form_class vs the parent."
                                )
                            response = self._client.chat.completions.create(
                                model=self._model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": request_prompt},
                                ],
                                temperature=float(temperature),
                                max_tokens=CRITIC_MAX_OUTPUT_TOKENS,
                            )
                            reflection = response.choices[0].message.content.strip()
                            if critic_mode != "structured_principle_form":
                                return reflection
                            validation_error = validate_structured_principle_form_reflection(
                                reflection,
                                strategy_type=strategy_type,
                                run_explored_pairs=run_explored_pairs,
                                lob_input_mode=lob_input_mode,
                                spread_input_mode=spread_input_mode,
                            )
                            if not validation_error and not hypothesis_pair_differs_from_parent(
                                reflection, parent_reflection
                            ):
                                validation_error = (
                                    "hypothesis jump requires at least one of principle or "
                                    "form_class to differ from the parent hypothesis"
                                )
                            if not validation_error:
                                return reflection
                            if attempt == 0:
                                validation_feedback = validation_error
                                continue
                            return reflection
                        return reflection
                    except Exception as inner_exc:
                        if (
                            _is_critic_context_error(inner_exc)
                            and not truncated_retry
                        ):
                            print(
                                "[reflection] hypothesis jump context overflow; "
                                "retrying with truncated prompt...",
                                flush=True,
                            )
                            user_prompt = (
                                user_prompt[: max(2000, max_user_chars // 3)].rstrip()
                                + "\n\n[...further truncated after context error...]\n"
                                "Reply with DIAGNOSIS (change principle or form_class), "
                                "MECHANISM, FIX only.\n"
                            )
                            truncated_retry = True
                            continue
                        raise
            except Exception as exc:
                last_error = exc
                if "Connection error" in str(exc):
                    continue
                break
        if last_error is not None:
            print(
                f"[reflection] hypothesis jump failed ({last_error}); "
                "using heuristic jump DIAGNOSIS",
                flush=True,
            )
        return _fallback_principle_form_diagnosis(
            strategy_type=strategy_type,
            lob_input_mode=lob_input_mode,
        )

    def generate_combination_revision(
        self,
        *,
        prompt: str,
        critic_mode: str,
        strategy_type: str = "pv",
        temperature: float = 0.4,
        run_explored_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> str:
        """Ask the critic for a gated family-combination DIAGNOSIS on plateau escape."""
        system_prompt = _critic_system_prompt(
            "zi_advantage", critic_mode, strategy_type=strategy_type
        )
        if critic_mode == "structured_principle_form":
            base = _structured_principle_form_system_prompt(strategy_type=strategy_type)
            system_prompt = (
                base
                + " This is a plateau combination escape: propose a gated merge of two "
                "source families (conditional branches), not a coefficient blend."
            )
        user_prompt = prompt
        max_user_chars = CRITIC_MAX_PROMPT_CHARS
        if len(user_prompt) > max_user_chars:
            user_prompt = (
                user_prompt[: max_user_chars - 80].rstrip()
                + "\n\n[...combination prompt truncated for context window...]\n"
            )
        last_error = None
        for candidate_url in self._base_url_candidates:
            try:
                self._client = OpenAI(base_url=candidate_url, api_key=self._api_key)
                truncated_retry = False
                while True:
                    validation_feedback: Optional[str] = None
                    reflection = ""
                    try:
                        for attempt in range(2):
                            request_prompt = user_prompt
                            if validation_feedback:
                                request_prompt = (
                                    f"{user_prompt}\n\n"
                                    "## Revision Required\n"
                                    f"Your prior reflection was rejected: {validation_feedback}\n"
                                    "Reply again with a valid gated-combination DIAGNOSIS."
                                )
                            response = self._client.chat.completions.create(
                                model=self._model_name,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": request_prompt},
                                ],
                                temperature=float(temperature),
                                max_tokens=CRITIC_MAX_OUTPUT_TOKENS,
                            )
                            reflection = response.choices[0].message.content.strip()
                            if critic_mode != "structured_principle_form":
                                return reflection
                            validation_error = validate_structured_principle_form_reflection(
                                reflection,
                                strategy_type=strategy_type,
                                run_explored_pairs=run_explored_pairs,
                                lob_input_mode=lob_input_mode,
                                spread_input_mode=spread_input_mode,
                            )
                            if not validation_error:
                                return reflection
                            if attempt == 0:
                                validation_feedback = validation_error
                                continue
                            return reflection
                        return reflection
                    except Exception as inner_exc:
                        if (
                            _is_critic_context_error(inner_exc)
                            and not truncated_retry
                        ):
                            print(
                                "[reflection] combination revision context overflow; "
                                "retrying with truncated prompt...",
                                flush=True,
                            )
                            user_prompt = (
                                user_prompt[: max(2000, max_user_chars // 3)].rstrip()
                                + "\n\n[...further truncated after context error...]\n"
                                "Reply with gated-combination DIAGNOSIS, MECHANISM, FIX only.\n"
                            )
                            truncated_retry = True
                            continue
                        raise
            except Exception as exc:
                last_error = exc
                if "Connection error" in str(exc):
                    continue
                break
        if last_error is not None:
            print(
                f"[reflection] combination revision failed ({last_error}); "
                "using heuristic combination DIAGNOSIS",
                flush=True,
            )
        return (
            "DIAGNOSIS:\n"
            "  principle: inventory_remaining_units\n"
            "  form_class: piecewise_by_sign\n"
            "  features_used: pv_mean, pv_max, sign\n"
            "  parent_form: gate on pv_mean sign; branch A ratio else branch B ratio\n"
            "  why_parent_fails: single-family local edits stalled\n"
            "  diversity_vs_parent: gated merge of two source families\n"
            "MECHANISM: Plateau combination fallback after critic failure.\n"
            "FIX: Implement if-gate with distinct branch formulas from source families.\n"
        )


class ReflectionPipeline:
    def __init__(
        self,
        *,
        config_pool: Optional[List[ReflectionMarketConfig]] = None,
        config_space: Optional[ReflectionConfigSpace] = None,
        batch_size: int = 2,
        steps: int = 3,
        episodes_per_config: int = 2,
        exploration_constant: float = 1.0,
        max_debug_retries: int = 3,
        storage_dir: str = "llm_calls_reflection_pipeline",
        output_path: str = "results/reflection_pipeline_summary.json",
        seed: int = 123,
        strategy_output_mode: str = "x_cap",
        reward_mode: str = "advantage",
        fixed_strategy_code: Optional[str] = None,
        critic_enabled: bool = True,
        critic_include_data: bool = True,
        critic_per_config_metrics: bool = False,
        generator_include_critic_data: bool = False,
        generator_include_sibling: bool = False,
        run_dir: Optional[str] = None,
        initial_strategy_types: Optional[List[str]] = None,
        q_eval_mode: Optional[Literal["incremental", "full_refresh"]] = None,
        q_update_mode: Optional[Literal["replace", "average"]] = None,
        q_reward_mode: Optional[Literal["advantage", "zi_beat_fraction", "zi_advantage"]] = None,
        parent_selection_mode: Optional[Literal["auto", "ucb", "newest"]] = None,
        recipe_id: Optional[str] = None,
        inactive_gate_enabled: bool = True,
        inactive_gate_min_nodes: int = INACTIVE_GATE_MIN_NODES,
        inactive_surplus_gap: float = INACTIVE_SURPLUS_GAP,
        inactive_beat_gap: int = INACTIVE_BEAT_GAP,
        critic_mode: Optional[str] = None,
        zi_benchmark_shade_ranges: Optional[List[List[float]]] = None,
        forbid_numpy_random: bool = False,
        parent_codegen_fail_cool_threshold: int = PARENT_CODEGEN_FAIL_COOL_THRESHOLD,
        plateau_escape_enabled: bool = True,
        plateau_non_improve_threshold: int = PLATEAU_NON_IMPROVE_THRESHOLD,
        plateau_improve_surplus_eps: float = PLATEAU_IMPROVE_SURPLUS_EPS,
        elite_parent_bias_prob: float = ELITE_PARENT_BIAS_PROB,
        hypothesis_jump_prob: float = HYPOTHESIS_JUMP_PROB,
        plateau_escape_temperature: float = PLATEAU_ESCAPE_TEMPERATURE,
        plateau_escape_mode: str = "revision",
        principle_menu_preset: Optional[str] = None,
        menu_exploration_enabled: bool = True,
        codegen_recovery_mode: str = DEFAULT_CODEGEN_RECOVERY_MODE,
        lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
        model_name: Optional[str] = None,
        market_block_size: int = 0,
        q_suite_size: int = 3,
        menu_sweep_markets: int = 0,
        home_set_size: int = 0,
        resample_every: int = 0,
    ) -> None:
        if config_pool is not None:
            self._config_pool = config_pool
            self._config_space = None
        else:
            self._config_pool = None
            self._config_space = config_space or build_default_config_space()
        self.batch_size = max(1, int(batch_size))
        self.steps = max(1, int(steps))
        self.episodes_per_config = max(1, int(episodes_per_config))
        self.exploration_constant = float(exploration_constant)
        self.max_debug_retries = max(1, int(max_debug_retries))
        self.run_dir = run_dir
        resolved_storage_dir, resolved_output_path = self._resolve_run_paths(
            storage_dir=storage_dir,
            output_path=output_path,
        )
        self.storage_dir = resolved_storage_dir
        self.output_path = resolved_output_path
        self.seed = seed
        self.strategy_output_mode = strategy_output_mode
        self.reward_mode = reward_mode
        self.fixed_strategy_code = fixed_strategy_code
        self.critic_enabled = critic_enabled
        self.critic_include_data = bool(critic_include_data)
        self.critic_per_config_metrics = bool(critic_per_config_metrics)
        self.generator_include_critic_data = bool(generator_include_critic_data)
        self.generator_include_sibling = bool(generator_include_sibling)
        self.q_eval_mode = q_eval_mode or DEFAULT_Q_EVAL_MODE
        if self.q_eval_mode not in {"incremental", "full_refresh"}:
            raise ValueError(f"Unsupported q_eval_mode: {self.q_eval_mode}")
        self.q_update_mode = q_update_mode or DEFAULT_Q_UPDATE_MODE
        if self.q_update_mode not in {"replace", "average"}:
            raise ValueError(f"Unsupported q_update_mode: {self.q_update_mode}")
        self.q_reward_mode = q_reward_mode or DEFAULT_Q_REWARD_MODE
        if self.q_reward_mode not in {"advantage", "zi_beat_fraction", "zi_advantage"}:
            raise ValueError(f"Unsupported q_reward_mode: {self.q_reward_mode}")
        if self.q_reward_mode in ZI_Q_REWARD_MODES:
            # ZI Q modes need benchmark comparison context in generator prompts.
            self.generator_include_critic_data = True
        # Discrete beat-fraction Q is set once per node from that step's ZI eval;
        # skip full-tree Q refresh for efficiency / stability.
        if self.q_reward_mode == "zi_beat_fraction" and self.q_eval_mode == "full_refresh":
            print(
                "[reflection] zi_beat_fraction forces q_eval_mode=incremental "
                "(no full Q refresh)",
                flush=True,
            )
            self.q_eval_mode = "incremental"
        self.parent_selection_mode = parent_selection_mode or DEFAULT_PARENT_SELECTION_MODE
        if self.parent_selection_mode not in PARENT_SELECTION_MODES:
            raise ValueError(
                f"Unsupported parent_selection_mode: {self.parent_selection_mode}"
            )
        self.recipe_id = recipe_id
        self.inactive_gate_enabled = bool(inactive_gate_enabled)
        self.inactive_gate_min_nodes = max(1, int(inactive_gate_min_nodes))
        self.inactive_surplus_gap = float(inactive_surplus_gap)
        self.inactive_beat_gap = max(0, int(inactive_beat_gap))
        self.critic_mode = critic_mode or DEFAULT_CRITIC_MODE
        if self.critic_mode not in CRITIC_MODES:
            raise ValueError(f"Unsupported critic_mode: {self.critic_mode}")
        self.zi_benchmark_shade_ranges = (
            [list(r) for r in zi_benchmark_shade_ranges]
            if zi_benchmark_shade_ranges is not None
            else None
        )
        self.forbid_numpy_random = bool(forbid_numpy_random)
        self.parent_codegen_fail_cool_threshold = max(
            1, int(parent_codegen_fail_cool_threshold)
        )
        self.plateau_escape_enabled = bool(plateau_escape_enabled)
        self.plateau_non_improve_threshold = max(1, int(plateau_non_improve_threshold))
        self.plateau_improve_surplus_eps = float(plateau_improve_surplus_eps)
        self.elite_parent_bias_prob = min(1.0, max(0.0, float(elite_parent_bias_prob)))
        self.hypothesis_jump_prob = min(1.0, max(0.0, float(hypothesis_jump_prob)))
        self.plateau_escape_temperature = float(plateau_escape_temperature)
        resolved_escape_mode = str(plateau_escape_mode or "revision").strip().lower()
        if resolved_escape_mode not in PLATEAU_ESCAPE_MODES:
            raise ValueError(
                f"Unsupported plateau_escape_mode: {plateau_escape_mode!r} "
                f"(expected one of {PLATEAU_ESCAPE_MODES})"
            )
        self.plateau_escape_mode = resolved_escape_mode
        self.principle_menu_preset = (
            str(principle_menu_preset).strip() if principle_menu_preset else None
        ) or None
        set_principle_menu_preset(self.principle_menu_preset)
        self.menu_exploration_enabled = bool(menu_exploration_enabled)
        resolved_codegen_recovery = str(
            codegen_recovery_mode or DEFAULT_CODEGEN_RECOVERY_MODE
        ).strip().lower()
        if resolved_codegen_recovery not in CODEGEN_RECOVERY_MODES:
            raise ValueError(
                f"Unsupported codegen_recovery_mode: {codegen_recovery_mode!r} "
                f"(expected one of {CODEGEN_RECOVERY_MODES})"
            )
        self.codegen_recovery_mode = resolved_codegen_recovery
        self.lob_input_mode = normalize_lob_input_mode(lob_input_mode)
        self.model_name = resolve_model_name(model_name)
        self.market_block_size = max(0, int(market_block_size))
        self.q_suite_size = max(1, int(q_suite_size))
        self.menu_sweep_markets = max(0, int(menu_sweep_markets))
        self.home_set_size = max(0, int(home_set_size))
        self.resample_every = max(0, int(resample_every))
        if self.home_set_size > 0 and self._config_space is None:
            raise ValueError(
                "home_set_size>0 requires a config_space (do not use a fixed "
                "config_pool / --fixed-curriculum-market)."
            )
        if self.home_set_size > 0 and self.market_block_size > 0:
            raise ValueError(
                "home_set_size and market_block_size cannot be used together."
            )
        if self.home_set_size > 0 and self.menu_sweep_markets > 0:
            raise ValueError(
                "home_set_size and menu_sweep_markets cannot be used together."
            )
        if self.resample_every > 0 and self._config_space is None:
            raise ValueError(
                "resample_every>0 requires a config_space (do not use a fixed "
                "config_pool / --fixed-curriculum-market)."
            )
        if self.resample_every > 0 and self.home_set_size > 0:
            raise ValueError(
                "resample_every and home_set_size cannot be used together."
            )
        if self.resample_every > 0 and self.market_block_size > 0:
            raise ValueError(
                "resample_every and market_block_size cannot be used together."
            )
        if self.resample_every > 0 and self.menu_sweep_markets > 0:
            raise ValueError(
                "resample_every and menu_sweep_markets cannot be used together."
            )
        if self.market_block_size > 0 and self._config_space is None:
            raise ValueError(
                "market_block_size>0 requires a config_space (do not use a fixed "
                "config_pool / --fixed-curriculum-market)."
            )
        if self.menu_sweep_markets > 0:
            if self._config_space is None:
                raise ValueError(
                    "menu_sweep_markets>0 requires a config_space (do not use a fixed "
                    "config_pool / --fixed-curriculum-market)."
                )
            if self.principle_menu_preset != "min2":
                raise ValueError(
                    "menu_sweep_markets>0 requires --principle-menu-preset min2 "
                    f"(got {self.principle_menu_preset!r})."
                )
        # Menu-sweep warmup markets (sampled before block curriculum).
        self._sweep_markets: List[ReflectionMarketConfig] = []
        # Block curriculum: ordered train markets (one per block).
        self._block_markets: List[ReflectionMarketConfig] = []
        # Per-step overrides used by menu sweep (cleared after each step).
        self._force_hypothesis_pair: Optional[Tuple[str, str]] = None
        self._force_train_configs: Optional[List[ReflectionMarketConfig]] = None
        self._force_q_configs: Optional[List[ReflectionMarketConfig]] = None

        self._rng = random.Random(seed)
        self._home_set: List[ReflectionMarketConfig] = []
        if self.home_set_size > 0:
            self._home_set = sample_stratified_home_set(
                self._config_space,
                self._rng,
                size=self.home_set_size,
                exclude=build_default_config_pool(),
            )
            self.batch_size = self.home_set_size
            self.critic_per_config_metrics = True
            print(
                "[reflection] home set (frozen "
                f"{self.home_set_size} markets; critic sees opaque per-market rows):",
                flush=True,
            )
            for cfg in self._home_set:
                state = cfg.observable_state()
                print(
                    f"[reflection]   {cfg.label}: buy={state['num_zi_buy']} "
                    f"sell={state['num_zi_sell']} t={state['timesteps']} "
                    f"q_max={state['q_max']} shade={state['shade_range']} "
                    f"arrival={state['arrival_rate']}",
                    flush=True,
                )
        self._resample_markets: List[ReflectionMarketConfig] = []
        self._resample_needs_q_realign = False
        if self.resample_every > 0:
            self.batch_size = 1
            print(
                "[reflection] resample-every="
                f"{self.resample_every} (hold 1 sampled market for "
                f"{self.resample_every} steps; train=Q; then redraw; "
                "holdout fingerprints excluded)",
                flush=True,
            )
        self._tree = RefinementTree(
            initial_types=initial_strategy_types,
            lob_input_mode=self.lob_input_mode,
        )
        self._total_visits = 1
        self._storage = CodeStorage(self.storage_dir)
        self._verifier = StrategyVerifier()
        self._generator = StrategyGenerator(model_name=self.model_name)
        self._critic = (
            ReflectionCritic(model_name=self.model_name) if critic_enabled else None
        )
        self._q_refresh_history: List[Dict[str, Any]] = []

    def _store_node_performance(self, node_id: int, performance: Dict[str, Any]) -> None:
        self._tree.get(node_id).metadata["performance"] = performance

    def _store_node_generator_context(
        self,
        node_id: int,
        *,
        reward: float,
        reward_variance: float,
        per_config_rewards: Dict[str, float],
        per_config_reward_variances: Dict[str, float],
        batch_data: Optional[Dict[str, Any]],
    ) -> None:
        if not batch_data:
            return
        batch_summary = _build_batch_summary_for_critic(
            self,
            reward=reward,
            reward_variance=reward_variance,
            per_config_rewards=per_config_rewards,
            per_config_reward_variances=per_config_reward_variances,
            batch_data=batch_data,
        )
        compressed_data = _compress_batch_data_for_critic(
            batch_data,
            max_interactions=GENERATOR_INTERACTION_LIMIT,
        )
        node = self._tree.get(node_id)
        node.metadata["generator_batch_summary"] = batch_summary
        stored = dict(compressed_data)
        prior = stored.pop("prior_hold_card", None)
        if prior is None and isinstance(batch_data, dict):
            prior = batch_data.get("prior_hold_card")
        node.metadata["generator_batch_data"] = stored
        hold_card = _compact_case_card_from_payload(stored)
        if hold_card:
            node.metadata["hold_case_card"] = hold_card
        if isinstance(prior, dict) and prior:
            node.metadata["prior_hold_case_card"] = dict(prior)

    def _parent_generator_context_for_generation(
        self,
        parent_id: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not self.generator_include_critic_data:
            return None, None
        parent = self._tree.get(parent_id)
        summary = parent.metadata.get("generator_batch_summary")
        data = parent.metadata.get("generator_batch_data")
        if isinstance(summary, dict) and summary and isinstance(data, dict) and data:
            return summary, data
        return None, None

    def _most_recent_sibling(self, parent_id: int) -> Optional[Any]:
        if not self.generator_include_sibling:
            return None
        parent = self._tree.get(parent_id)
        if not parent.children:
            return None
        sibling_id = max(parent.children)
        return self._tree.get(sibling_id)

    @staticmethod
    def _sibling_context_for_generation(sibling: Any) -> Dict[str, Any]:
        performance = sibling.metadata.get("performance") or {}
        return {
            "node_id": sibling.node_id,
            "q_value": float(sibling.q_value),
            "reward": float(performance.get("reward", sibling.q_value)),
            "reward_variance": float(performance.get("reward_variance", 0.0)),
        }

    def _parent_performance_for_generation(self, parent_id: int) -> Dict[str, Any]:
        parent = self._tree.get(parent_id)
        performance = parent.metadata.get("performance")
        if isinstance(performance, dict) and performance:
            return dict(performance)
        return {
            "reward": float(parent.q_value),
            "reward_variance": 0.0,
            "per_config_rewards": {},
            "per_config_reward_variances": {},
        }

    def _resolve_run_paths(self, *, storage_dir: str, output_path: str) -> Tuple[str, str]:
        if self.run_dir:
            base_dir = Path(self.run_dir)
        else:
            default_storage = storage_dir == "llm_calls_reflection_pipeline"
            default_output = output_path == "results/reflection_pipeline_summary.json"
            if default_storage and default_output:
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_dir = Path(__file__).resolve().parent / "runs" / run_id
            else:
                return storage_dir, output_path

        base_dir.mkdir(parents=True, exist_ok=True)
        resolved_storage = str(base_dir / "llm_calls")
        resolved_output = str(base_dir / "reflection_pipeline_summary.json")
        return resolved_storage, resolved_output

    def _select_batch(self) -> List[ReflectionMarketConfig]:
        if self._config_pool is not None:
            if self.batch_size >= len(self._config_pool):
                return list(self._config_pool)
            return self._rng.sample(self._config_pool, self.batch_size)
        return self._config_space.sample_batch(self._rng, self.batch_size)

    def _block_index_for_step(self, step_index: int) -> int:
        """0-based step_index -> block index when market_block_size>0."""
        if self.market_block_size <= 0:
            return 0
        return max(0, int(step_index)) // self.market_block_size

    def _ensure_block_market(self, step_index: int) -> ReflectionMarketConfig:
        """Return the train market for this step, sampling new markets at block starts."""
        if self.market_block_size <= 0:
            raise RuntimeError("_ensure_block_market called with market_block_size=0")
        if self._config_space is None:
            raise RuntimeError("block curriculum requires config_space")
        block_idx = self._block_index_for_step(step_index)
        while len(self._block_markets) <= block_idx:
            cfg = self._config_space.sample(
                self._rng, batch_index=len(self._block_markets)
            )
            self._block_markets.append(cfg)
            print(
                f"[reflection] market block {len(self._block_markets) - 1}: "
                f"train={cfg.label}",
                flush=True,
            )
        return self._block_markets[block_idx]

    def _resample_slot_index(self, step_index: int) -> int:
        return max(0, int(step_index)) // self.resample_every

    def _holdout_fingerprints(self) -> set:
        return {cfg.fingerprint() for cfg in build_default_config_pool()}

    def _sample_resample_market(self, slot_index: int) -> ReflectionMarketConfig:
        if self._config_space is None:
            raise RuntimeError("resample-every requires config_space")
        blocked = self._holdout_fingerprints() | {
            cfg.fingerprint() for cfg in self._resample_markets
        }
        last: Optional[ReflectionMarketConfig] = None
        for _ in range(40):
            last = self._config_space.sample(self._rng, batch_index=slot_index)
            if last.fingerprint() not in blocked:
                return last
        assert last is not None
        return last

    def _ensure_resample_market(self, step_index: int) -> ReflectionMarketConfig:
        """Hold one sampled market for resample_every steps, then redraw."""
        if self.resample_every <= 0:
            raise RuntimeError("_ensure_resample_market called with resample_every=0")
        slot_index = self._resample_slot_index(step_index)
        while len(self._resample_markets) <= slot_index:
            if self._resample_markets:
                self._snapshot_prior_hold_cards()
            cfg = self._sample_resample_market(len(self._resample_markets))
            self._resample_markets.append(cfg)
            # Seed eval already aligned Q on the first market; later slots need a
            # realign so UCB/elite are not still ranked on the previous hold.
            self._resample_needs_q_realign = len(self._resample_markets) > 1
            state = cfg.observable_state()
            print(
                f"[reflection] resample slot {len(self._resample_markets) - 1}: "
                f"hold {cfg.label} for {self.resample_every} steps "
                f"(train=Q) buy={state['num_zi_buy']} sell={state['num_zi_sell']} "
                f"t={state['timesteps']} q_max={state['q_max']} "
                f"shade={state['shade_range']} arrival={state['arrival_rate']}",
                flush=True,
            )
        return self._resample_markets[slot_index]

    def _maybe_realign_q_for_resample(
        self,
        *,
        step_index: int,
        q_suite_configs: List[ReflectionMarketConfig],
    ) -> None:
        if not self._resample_needs_q_realign:
            return
        if self.q_eval_mode != "full_refresh":
            self._resample_needs_q_realign = False
            return
        if not self._tree.evaluable_node_ids():
            self._resample_needs_q_realign = False
            return
        suite = self._build_step_eval_suite(q_suite_configs, step_index=step_index)
        labels = [cfg.label for cfg in q_suite_configs]
        print(
            f"[reflection] resample slot {self._resample_slot_index(step_index)}: "
            f"realigning Q on {labels} before parent select "
            f"(suite {suite.suite_id})",
            flush=True,
        )
        started = time.perf_counter()
        refresh_results = self._refresh_all_nodes(
            suite=suite,
            include_batch_data=False,
            log_prefix="[reflection]",
        )
        refresh_entry = self._make_q_refresh_entry(
            step=step_index,
            phase="resample_realign",
            refresh_results=refresh_results,
            refresh_seconds=time.perf_counter() - started,
            eval_suite=suite,
        )
        self._q_refresh_history.append(refresh_entry)
        self._resample_needs_q_realign = False

    def _is_resample_transfer_step(self, step_index: Optional[int]) -> bool:
        """First step of a new resample hold (after slot 0)."""
        if self.resample_every <= 0 or step_index is None:
            return False
        idx = int(step_index)
        return idx > 0 and idx % self.resample_every == 0

    def _previous_resample_market(self) -> Optional[ReflectionMarketConfig]:
        if len(self._resample_markets) < 2:
            return None
        return self._resample_markets[-2]

    def _snapshot_prior_hold_cards(self) -> None:
        n = 0
        for node_id in self._tree.evaluable_node_ids():
            node = self._tree.get(node_id)
            card = node.metadata.get("hold_case_card")
            if isinstance(card, dict) and card:
                node.metadata["prior_hold_case_card"] = dict(card)
                n += 1
        if n:
            print(
                f"[reflection] snapshotted last-hold case cards for {n} node(s) "
                "(stats only; used on the first step of the next hold)",
                flush=True,
            )

    def _strategy_fn_from_node(self, node) -> Optional[Any]:
        strategy_type = str(node.metadata.get("type", "full") or "full")
        required_args, input_mode = self._strategy_type_params(strategy_type)
        ok, _err, fn = self._verifier.verify(
            node.code,
            required_args=required_args,
            input_mode=input_mode,
            forbid_numpy_random=self.forbid_numpy_random,
        )
        if not ok:
            return None
        return fn

    def _evaluate_compact_case_payload(
        self,
        *,
        strategy_fn: Any,
        config: ReflectionMarketConfig,
        step_index: int,
        strategy_type: str,
    ) -> Optional[Dict[str, Any]]:
        result = self._evaluate_strategy_batch(
            strategy_fn=strategy_fn,
            batch_configs=[config],
            step_index=step_index,
            strategy_type=strategy_type,
            include_batch_data=True,
        )
        if not result.batch_data:
            return None
        return _compress_batch_data_for_critic(
            result.batch_data,
            max_interactions=GENERATOR_INTERACTION_LIMIT,
        )

    def _attach_parent_transfer_cards(
        self,
        *,
        parent: Any,
        generator_batch_data: Optional[Dict[str, Any]],
        strategy_type: str,
        step_index: int,
    ) -> Optional[Dict[str, Any]]:
        """Parent: this-hold card (fresh eval) + last-hold card (stats only)."""
        data = dict(generator_batch_data) if isinstance(generator_batch_data, dict) else {}
        current_cfgs = list(self._force_train_configs or [])
        fn = self._strategy_fn_from_node(parent)
        if fn is not None and current_cfgs:
            print(
                f"[reflection] transfer test: parent {parent.node_id} on this hold "
                f"{current_cfgs[0].label} (generator; Q unchanged)",
                flush=True,
            )
            payload = self._evaluate_compact_case_payload(
                strategy_fn=fn,
                config=current_cfgs[0],
                step_index=step_index,
                strategy_type=strategy_type,
            )
            if payload:
                data = dict(payload)
                parent.metadata["generator_batch_data"] = dict(payload)
                card = _compact_case_card_from_payload(payload)
                if card:
                    parent.metadata["hold_case_card"] = card
        prior = parent.metadata.get("prior_hold_case_card")
        if isinstance(prior, dict) and prior:
            data = _attach_prior_hold_card(data, prior)
        return data or generator_batch_data

    def _second_test_prior_hold_card(
        self,
        *,
        strategy_fn: Any,
        strategy_type: str,
        step_index: int,
    ) -> Optional[Dict[str, Any]]:
        prev = self._previous_resample_market()
        if prev is None:
            return None
        print(
            f"[reflection] transfer test: same child on last hold {prev.label} "
            "(critic; stats only, Q unchanged)",
            flush=True,
        )
        payload = self._evaluate_compact_case_payload(
            strategy_fn=strategy_fn,
            config=prev,
            step_index=int(step_index) + 30_000,
            strategy_type=strategy_type,
        )
        return _compact_case_card_from_payload(payload)

    def _q_suite_history(self) -> List[ReflectionMarketConfig]:
        """Sweep markets (warmup) followed by block train markets."""
        return list(self._sweep_markets) + list(self._block_markets)

    def _q_suite_for_step(self, step_index: int) -> List[ReflectionMarketConfig]:
        """Rolling suite = last q_suite_size markets from sweep+block history."""
        self._ensure_block_market(step_index)
        history = self._q_suite_history()
        start = max(0, len(history) - self.q_suite_size)
        return list(history[start:])

    def _train_and_q_configs(
        self, step_index: int
    ) -> Tuple[List[ReflectionMarketConfig], List[ReflectionMarketConfig]]:
        """
        Return (train_configs, q_suite_configs).

        Force overrides (menu sweep) win when set. A frozen home set uses the
        same K markets for train and Q every step. resample_every holds one
        sampled market for K steps with train=Q, then redraws. With market
        blocks: train = current block market; Q = rolling suite over
        sweep+block history. Otherwise: both are the usual _select_batch()
        result.
        """
        if self._force_train_configs is not None:
            train = list(self._force_train_configs)
            suite = (
                list(self._force_q_configs)
                if self._force_q_configs is not None
                else train
            )
            return train, suite
        if self.home_set_size > 0:
            home = list(self._home_set)
            return home, home
        if self.resample_every > 0:
            cfg = self._ensure_resample_market(step_index)
            return [cfg], [cfg]
        if self.market_block_size > 0:
            train = [self._ensure_block_market(step_index)]
            suite = self._q_suite_for_step(step_index)
            return train, suite
        batch = self._select_batch()
        return batch, batch

    def _menu_sweep_pairs(self, strategy_type: str) -> List[Tuple[str, str]]:
        return sorted(menu_hypothesis_pairs(strategy_type))

    def _clear_step_force_overrides(self) -> None:
        self._force_hypothesis_pair = None
        self._force_train_configs = None
        self._force_q_configs = None

    def _build_step_eval_suite(
        self,
        batch_configs: List[ReflectionMarketConfig],
        *,
        step_index: int,
    ) -> FixedEvalSuite:
        return build_fixed_eval_suite(
            configs=batch_configs,
            episodes_per_config=self.episodes_per_config,
            base_seed=self.seed,
            step_index=step_index,
        )

    def _node_specs(self, node_ids: Optional[List[int]] = None) -> List[Tuple[int, str, str]]:
        ids = node_ids if node_ids is not None else self._tree.active_node_ids()
        specs: List[Tuple[int, str, str]] = []
        for node_id in ids:
            node = self._tree.get(node_id)
            strategy_type = str(node.metadata.get("type", "full"))
            specs.append((node_id, node.code, strategy_type))
        return specs

    def _store_creation_metrics(
        self,
        node_id: int,
        *,
        batch_data: Optional[Dict[str, Any]],
        comparison: Optional[Dict[str, Any]],
    ) -> None:
        node = self._tree.get(node_id)
        mean_surplus = None
        if comparison is not None and comparison.get("llm_mean_surplus") is not None:
            mean_surplus = float(comparison["llm_mean_surplus"])
        elif batch_data is not None:
            overall = summarize_batch_performance(batch_data, reward_mode=self.reward_mode)
            if overall.get("llm_surplus_mean") is not None:
                mean_surplus = float(overall["llm_surplus_mean"])
        if mean_surplus is not None:
            node.metadata["mean_surplus"] = mean_surplus
        if comparison is not None and comparison.get("beat_count") is not None:
            node.metadata["zi_beat_count"] = int(comparison["beat_count"])
            node.metadata["zi_total_benchmarks"] = int(comparison.get("total_benchmarks", 0))
            node.metadata["zi_beat_fraction"] = float(comparison.get("beat_fraction", 0.0))

    def _best_mean_surplus(self) -> Optional[float]:
        values = [
            float(node.metadata["mean_surplus"])
            for node in self._tree.nodes()
            if node.node_id != 0 and node.metadata.get("mean_surplus") is not None
        ]
        return max(values) if values else None

    def _best_zi_beat_count(self) -> Optional[int]:
        values = [
            int(node.metadata["zi_beat_count"])
            for node in self._tree.nodes()
            if node.node_id != 0 and node.metadata.get("zi_beat_count") is not None
        ]
        return max(values) if values else None

    def _maybe_mark_inactive(self, node_id: int) -> bool:
        """
        After enough nodes exist, freeze clearly-worse new nodes: skip future
        Q refresh and parent selection.
        """
        if not self.inactive_gate_enabled:
            return False
        evaluable = self._tree.evaluable_node_ids()
        if len(evaluable) <= self.inactive_gate_min_nodes:
            return False
        # Only gate nodes created after the min-node threshold.
        if node_id not in evaluable:
            return False
        # Rank by creation order (node ids increase).
        ordered = sorted(evaluable)
        if ordered.index(node_id) < self.inactive_gate_min_nodes:
            return False

        node = self._tree.get(node_id)
        if self.q_reward_mode == "zi_beat_fraction":
            beat = node.metadata.get("zi_beat_count")
            best_beat = self._best_zi_beat_count()
            if beat is None or best_beat is None:
                return False
            if int(beat) <= int(best_beat) - self.inactive_beat_gap:
                reason = (
                    f"zi_beat_count={beat} <= best_beat_count={best_beat} "
                    f"- gap={self.inactive_beat_gap}"
                )
                self._tree.mark_inactive(node_id, reason=reason)
                print(
                    f"[reflection] node {node_id} marked inactive ({reason}); "
                    "skipping future Q refresh and parent selection",
                    flush=True,
                )
                return True
            return False

        surplus = node.metadata.get("mean_surplus")
        best_surplus = self._best_mean_surplus()
        if surplus is None or best_surplus is None:
            return False
        if float(surplus) <= float(best_surplus) - self.inactive_surplus_gap:
            reason = (
                f"mean_surplus={float(surplus):.2f} <= best_mean_surplus="
                f"{float(best_surplus):.2f} - gap={self.inactive_surplus_gap:g}"
            )
            self._tree.mark_inactive(node_id, reason=reason)
            print(
                f"[reflection] node {node_id} marked inactive ({reason}); "
                "skipping future Q refresh and parent selection",
                flush=True,
            )
            return True
        return False

    def _note_parent_codegen_outcome(self, parent_id: int, *, success: bool) -> None:
        """Track consecutive failed refine steps; cool parent out of UCB after N fails."""
        if parent_id is None or int(parent_id) <= 0:
            return
        try:
            parent = self._tree.get(int(parent_id))
        except Exception:
            return
        if success:
            parent.metadata["codegen_fail_streak"] = 0
            return
        streak = int(parent.metadata.get("codegen_fail_streak", 0)) + 1
        parent.metadata["codegen_fail_streak"] = streak
        if streak < self.parent_codegen_fail_cool_threshold:
            print(
                f"[reflection] parent {parent_id} codegen/verify fail streak={streak}/"
                f"{self.parent_codegen_fail_cool_threshold}",
                flush=True,
            )
            return
        if bool(parent.metadata.get("inactive", False)):
            return
        reason = (
            f"codegen_fail_streak={streak} "
            f"(threshold={self.parent_codegen_fail_cool_threshold})"
        )
        self._tree.mark_inactive(int(parent_id), reason=reason)
        print(
            f"[reflection] parent {parent_id} marked inactive ({reason}); "
            "skipping future UCB parent selection",
            flush=True,
        )

    def _refresh_all_nodes(
        self,
        *,
        suite: FixedEvalSuite,
        node_ids: Optional[List[int]] = None,
        include_batch_data: bool = False,
        log_prefix: str = "[reflection]",
    ) -> List[Dict[str, Any]]:
        node_specs = self._node_specs(node_ids)
        if not node_specs:
            return []

        # zi_advantage: refresh only the continuous advantage Q; keep creation-time
        # ZI beat diagnostics frozen on each node (do not re-run ZI cohorts).
        refresh_q_mode = (
            "advantage" if self.q_reward_mode == "zi_advantage" else self.q_reward_mode
        )
        started = time.perf_counter()
        refresh_note = (
            "; advantage Q only, ZI benchmarks not refreshed"
            if self.q_reward_mode == "zi_advantage"
            else ""
        )
        print(
            f"{log_prefix} refreshing Q for {len(node_specs)} node(s) "
            f"on fixed suite {suite.suite_id} "
            f"(workers={FULL_REFRESH_PARALLEL_WORKERS}{refresh_note})...",
            flush=True,
        )
        results = refresh_nodes_parallel(
            node_specs=node_specs,
            suite=suite,
            episodes_per_config=self.episodes_per_config,
            base_seed=self.seed,
            reward_mode=self.reward_mode,
            strategy_output_mode=self.strategy_output_mode,
            parallel_workers=FULL_REFRESH_PARALLEL_WORKERS,
            include_batch_data=include_batch_data,
            q_reward_mode=refresh_q_mode,
            lob_input_mode=self.lob_input_mode,
        )
        q_map: Dict[int, float] = {}
        for result in results:
            node_id = int(result["node_id"])
            if result.get("success"):
                q_map[node_id] = float(result["reward"])
                batch_data = result.get("batch_data")
                overall = None
                if batch_data:
                    overall = summarize_batch_performance(
                        batch_data,
                        reward_mode=self.reward_mode,
                    )
                prior = self._tree.get(node_id).metadata.get("performance") or {}
                prior_zi = None
                if isinstance(prior, dict):
                    prior_zi = prior.get("zi_benchmark_comparison")
                performance_payload = _performance_payload(
                    reward=float(result["reward"]),
                    reward_variance=float(result.get("reward_variance", 0.0)),
                    per_config_rewards=result.get("per_config_rewards") or {},
                    per_config_reward_variances=result.get("per_config_reward_variances") or {},
                    overall=overall,
                    zi_benchmark_comparison=(
                        result.get("zi_benchmark_comparison")
                        or prior_zi
                    ),
                    batch_data=batch_data if isinstance(batch_data, dict) else None,
                    reward_mode=self.reward_mode,
                )
                self._store_node_performance(node_id, performance_payload)
                # Keep elite / inactive surplus aligned with the Q suite when available.
                suite_surplus = result.get("llm_mean_surplus")
                if suite_surplus is None and overall is not None:
                    suite_surplus = overall.get("llm_surplus_mean")
                if suite_surplus is not None:
                    self._tree.get(node_id).metadata["mean_surplus"] = float(suite_surplus)
                self._store_node_generator_context(
                    node_id,
                    reward=float(result["reward"]),
                    reward_variance=float(result.get("reward_variance", 0.0)),
                    per_config_rewards=result.get("per_config_rewards") or {},
                    per_config_reward_variances=result.get("per_config_reward_variances") or {},
                    batch_data=batch_data,
                )
            else:
                print(
                    f"{log_prefix} node {node_id} refresh FAILED: {result.get('error')}",
                    flush=True,
                )
        self._tree.apply_q_values(q_map, mode=self.q_update_mode)
        elapsed = time.perf_counter() - started
        print(
            f"{log_prefix} Q refresh done in {elapsed:.1f}s "
            f"(updated {len(q_map)} node(s))",
            flush=True,
        )
        return results

    def _make_q_refresh_entry(
        self,
        *,
        step: int,
        phase: str,
        refresh_results: List[Dict[str, Any]],
        refresh_seconds: float,
        new_node_id: Optional[int] = None,
        eval_suite: Optional[FixedEvalSuite] = None,
    ) -> Dict[str, Any]:
        q_values = {
            str(int(item["node_id"])): float(self._tree.get(int(item["node_id"])).q_value)
            for item in refresh_results
            if item.get("success")
        }
        q_eval_counts = {
            str(int(item["node_id"])): int(self._tree.get(int(item["node_id"])).metadata.get("q_eval_count", 0))
            for item in refresh_results
            if item.get("success")
        }
        q_variances = {
            str(int(item["node_id"])): float(item["reward_variance"])
            for item in refresh_results
            if item.get("success") and item.get("reward_variance") is not None
        }
        return {
            "step": step,
            "phase": phase,
            "new_node_id": new_node_id,
            "eval_suite": eval_suite.to_dict() if eval_suite is not None else None,
            "q_values": q_values,
            "q_eval_counts": q_eval_counts,
            "q_variances": q_variances,
            "refresh_seconds": refresh_seconds,
            "per_node": [
                {
                    "node_id": item["node_id"],
                    "success": item.get("success"),
                    "reward": item.get("reward"),
                    "reward_variance": item.get("reward_variance"),
                    "q_value": (
                        float(self._tree.get(int(item["node_id"])).q_value)
                        if item.get("success")
                        else None
                    ),
                    "q_eval_count": (
                        int(self._tree.get(int(item["node_id"])).metadata.get("q_eval_count", 0))
                        if item.get("success")
                        else None
                    ),
                    "error": item.get("error"),
                }
                for item in refresh_results
            ],
        }

    def _evaluate_on_suite(
        self,
        *,
        strategy_fn: Any,
        suite: FixedEvalSuite,
        strategy_type: str,
        include_batch_data: bool = True,
    ):
        return self._evaluate_strategy_batch(
            strategy_fn=strategy_fn,
            batch_configs=suite.configs,
            step_index=-1,
            strategy_type=strategy_type,
            include_batch_data=include_batch_data,
            episode_seeds=suite.episode_seeds,
        )

    def _evaluate_strategy_batch(
        self,
        *,
        strategy_fn: Any,
        batch_configs: List[ReflectionMarketConfig],
        step_index: int,
        strategy_type: str,
        include_batch_data: bool = True,
        episode_seeds: Optional[List[List[int]]] = None,
    ):
        common = dict(
            strategy_fn=strategy_fn,
            batch_configs=batch_configs,
            episodes_per_config=self.episodes_per_config,
            step_index=step_index,
            base_seed=self.seed,
            reward_mode=self.reward_mode,
            max_events_per_episode=20,
            strategy_output_mode=self.strategy_output_mode,
            strategy_type=strategy_type,
            episode_seeds=episode_seeds,
            include_batch_data=include_batch_data,
            lob_input_mode=self.lob_input_mode,
        )
        if self.q_reward_mode in ZI_Q_REWARD_MODES:
            return evaluate_batch_with_zi_benchmarks(
                **common,
                q_reward_mode=self.q_reward_mode,
                zi_benchmark_shade_ranges=self.zi_benchmark_shade_ranges,
            )
        return evaluate_batch(**common, special_agent_kind="llm")

    def _format_step_eval_reward(self, eval_result) -> str:
        base = _format_reward_stats(eval_result.reward, eval_result.reward_variance)
        if eval_result.zi_benchmark_comparison:
            comparison = eval_result.zi_benchmark_comparison
            zi_line = (
                f"zi_beat={comparison['beat_count']}/"
                f"{comparison['total_benchmarks']} "
                f"llm_mean_surplus={comparison['llm_mean_surplus']:.4f}"
            )
            if self.q_reward_mode == "zi_advantage" and eval_result.scalar_reward is not None:
                zi_line += f" beat_fraction={float(eval_result.scalar_reward):.4f}"
            if comparison.get("llm_mean_requested_x_cap") is not None:
                zi_line += (
                    f" llm_mean_x_cap={float(comparison['llm_mean_requested_x_cap']):.4f}"
                )
            if comparison.get("llm_mean_shade") is not None:
                zi_line += (
                    f" llm_trade_shade={float(comparison['llm_mean_shade']):.4f}"
                )
            return f"{base} {zi_line}"
        return base

    def _zi_comparison_from_batch(
        self, batch_data: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(batch_data, dict):
            return None
        comparison = batch_data.get("zi_benchmark_comparison")
        return comparison if isinstance(comparison, dict) else None

    def _apply_zi_node_eval(
        self,
        node_id: int,
        *,
        comparison: Optional[Dict[str, Any]],
    ) -> None:
        """Persist ZI beat counts for diagnostics; Q itself uses beat_fraction via UCB."""
        if self.q_reward_mode != "zi_beat_fraction" or not comparison:
            return
        node = self._tree.get(node_id)
        node.metadata["zi_beat_count"] = int(comparison.get("beat_count", 0))
        node.metadata["zi_total_benchmarks"] = int(comparison.get("total_benchmarks", 0))
        node.metadata["zi_beat_fraction"] = float(comparison.get("beat_fraction", 0.0))

    def _resolved_parent_selection_mode(self) -> str:
        # UCB is the only tree-search parent rule for all Q modes (including
        # discrete ZI beat fraction). "auto" remains an alias for UCB.
        if self.parent_selection_mode == "newest":
            return "newest"
        return "ucb"

    def _select_elite_parent(self):
        """Prefer highest mean_surplus among active nodes; tie-break by Q."""
        best = None
        best_key = None
        for node in self._tree.nodes():
            if node.node_id == 0 or node.visits <= 0:
                continue
            if bool(node.metadata.get("inactive", False)):
                continue
            surplus = node.metadata.get("mean_surplus")
            surplus_key = float(surplus) if surplus is not None else float("-inf")
            key = (surplus_key, float(node.q_value))
            if best is None or key > best_key:
                best = node
                best_key = key
        return best

    def _select_parent_node(self):
        mode = self._resolved_parent_selection_mode()
        if mode == "newest":
            return self._tree.select_newest(), "newest"
        if self.elite_parent_bias_prob > 0 and self._rng.random() < self.elite_parent_bias_prob:
            elite = self._select_elite_parent()
            if elite is not None:
                return elite, "elite"
        parent = self._tree.select_ucb(
            exploration=self.exploration_constant,
            total_visits=self._total_visits,
        )
        return parent, "ucb"

    def _log_parent_selection(self, parent, *, select_mode: str = "ucb") -> None:
        parent_type = parent.metadata.get("type", "full")
        streak = int(parent.metadata.get("non_improve_streak", 0))
        streak_note = (
            f", non_improve_streak={streak}/{self.plateau_non_improve_threshold}"
            if self.plateau_escape_enabled and streak > 0
            else ""
        )
        if select_mode == "newest":
            print(
                f"[reflection] newest-lineage parent node {parent.node_id} "
                f"(type={parent_type}, q={parent.q_value:.4f}, visits={parent.visits}"
                f"{streak_note})",
                flush=True,
            )
            return
        if select_mode == "elite":
            surplus = parent.metadata.get("mean_surplus")
            surplus_txt = f"{float(surplus):.2f}" if surplus is not None else "n/a"
            print(
                f"[reflection] elite parent node {parent.node_id} "
                f"(type={parent_type}, q={parent.q_value:.4f}, "
                f"surplus={surplus_txt}, visits={parent.visits}{streak_note})",
                flush=True,
            )
            return
        print(
            f"[reflection] UCB parent node {parent.node_id} (type={parent_type}, "
            f"q={parent.q_value:.4f}, visits={parent.visits}{streak_note})",
            flush=True,
        )

    def _run_explored_hypothesis_pairs(self, strategy_type: str) -> Set[Tuple[str, str]]:
        reflections: List[str] = []
        for node in self._tree.nodes():
            node_type = str((node.metadata or {}).get("type", "full") or "full")
            if node_type != strategy_type:
                continue
            if node.reflection:
                reflections.append(str(node.reflection))
        return collect_hypothesis_pairs_from_reflections(reflections)

    def _menu_coverage_summary(self) -> Dict[str, Any]:
        if not is_structured_principle_form(self.critic_mode):
            return {}
        reflections_by_type: Dict[str, List[str]] = {}
        for node in self._tree.nodes():
            if node.node_id == 0:
                continue
            strategy_type = str((node.metadata or {}).get("type", "pv") or "pv")
            if node.reflection:
                reflections_by_type.setdefault(strategy_type, []).append(str(node.reflection))
        return {
            strategy_type: compute_menu_coverage_stats(
                strategy_type=strategy_type,
                reflections=reflections,
            )
            for strategy_type, reflections in sorted(reflections_by_type.items())
        }

    def _should_plateau_escape(self, parent) -> bool:
        if not self.plateau_escape_enabled:
            return False
        streak = int(parent.metadata.get("non_improve_streak", 0))
        return streak >= self.plateau_non_improve_threshold

    def _generate_plateau_revision(self, parent) -> str:
        """Critic-led hypothesis revision used as the Parent Reflection for escape."""
        fields = parse_principle_form_fields(parent.reflection or "")
        summary, _data = self._parent_generator_context_for_generation(parent.node_id)
        performance = self._parent_performance_for_generation(parent.node_id)
        streak = int(parent.metadata.get("non_improve_streak", 0))
        strategy_type = str(parent.metadata.get("type", "pv") or "pv")
        explored_pairs = (
            self._run_explored_hypothesis_pairs(strategy_type)
            if self.menu_exploration_enabled
            else None
        )
        prompt = build_plateau_revision_prompt(
            parent_code=parent.code,
            parent_reflection=parent.reflection or "",
            parent_performance=performance,
            non_improve_streak=streak,
            critic_mode=self.critic_mode,
            generator_batch_summary=summary if isinstance(summary, dict) else None,
            strategy_type=strategy_type,
            run_explored_pairs=explored_pairs,
        )
        if self._critic is None:
            return _fallback_plateau_diagnosis(
                strategy_type=strategy_type,
                lob_input_mode=self.lob_input_mode,
            )
        return self._critic.generate_plateau_revision(
            prompt=prompt,
            critic_mode=self.critic_mode,
            stalled_form_class=fields.get("form_class"),
            strategy_type=strategy_type,
            temperature=max(0.2, self.plateau_escape_temperature - 0.1),
            run_explored_pairs=explored_pairs,
            lob_input_mode=self.lob_input_mode,
        )

    def _generate_hypothesis_jump_revision(self, parent) -> str:
        """Critic-led hypothesis jump: change principle and/or form_class from UCB parent."""
        summary, _data = self._parent_generator_context_for_generation(parent.node_id)
        performance = self._parent_performance_for_generation(parent.node_id)
        strategy_type = str(parent.metadata.get("type", "pv") or "pv")
        explored_pairs = (
            self._run_explored_hypothesis_pairs(strategy_type)
            if self.menu_exploration_enabled
            else None
        )
        prompt = build_hypothesis_jump_prompt(
            parent_code=parent.code,
            parent_reflection=parent.reflection or "",
            parent_performance=performance,
            critic_mode=self.critic_mode,
            generator_batch_summary=summary if isinstance(summary, dict) else None,
            strategy_type=strategy_type,
            run_explored_pairs=explored_pairs,
        )
        if self._critic is None:
            return _fallback_principle_form_diagnosis(
                strategy_type=strategy_type,
                lob_input_mode=self.lob_input_mode,
            )
        return self._critic.generate_hypothesis_jump(
            prompt=prompt,
            critic_mode=self.critic_mode,
            parent_reflection=parent.reflection or "",
            strategy_type=strategy_type,
            temperature=max(0.2, self.plateau_escape_temperature - 0.1),
            run_explored_pairs=explored_pairs,
            lob_input_mode=self.lob_input_mode,
        )

    def _node_family_id(self, node) -> str:
        return derive_family_id(
            reflection=getattr(node, "reflection", "") or "",
            code=getattr(node, "code", "") or "",
            metadata=getattr(node, "metadata", None) or {},
        )

    def _family_source_payload(self, node) -> Dict[str, Any]:
        fields = parse_principle_form_fields(getattr(node, "reflection", "") or "")
        meta = getattr(node, "metadata", None) or {}
        return {
            "node_id": int(node.node_id),
            "family_id": self._node_family_id(node),
            "form_class": meta.get("form_class") or fields.get("form_class") or "unknown",
            "principle": meta.get("principle") or fields.get("principle") or "unknown",
            "mean_surplus": meta.get("mean_surplus"),
            "code": getattr(node, "code", "") or "",
        }

    def _select_family_sources(self, parent, *, limit: int = 2) -> List[Dict[str, Any]]:
        """Pick top-performing distinct families from active nodes."""
        by_family: Dict[str, Dict[str, Any]] = {}
        for node in self._tree.nodes():
            if node.node_id == 0:
                continue
            if not (getattr(node, "code", None) or "").strip():
                continue
            if int(getattr(node, "visits", 0) or 0) <= 0:
                continue
            if bool((getattr(node, "metadata", None) or {}).get("inactive", False)):
                continue
            payload = self._family_source_payload(node)
            surplus = payload.get("mean_surplus")
            surplus_key = float(surplus) if surplus is not None else float("-inf")
            prev = by_family.get(payload["family_id"])
            if prev is None or surplus_key > float(prev.get("mean_surplus") or float("-inf")):
                payload["mean_surplus"] = surplus
                by_family[payload["family_id"]] = payload

        ranked = sorted(
            by_family.values(),
            key=lambda item: float(item.get("mean_surplus") or float("-inf")),
            reverse=True,
        )
        if not ranked:
            return [self._family_source_payload(parent)]

        parent_family = self._node_family_id(parent)
        selected: List[Dict[str, Any]] = []
        parent_payload = by_family.get(parent_family) or self._family_source_payload(parent)
        selected.append(parent_payload)
        for item in ranked:
            if item["family_id"] == parent_family:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
        if len(selected) < limit:
            for item in ranked:
                if any(s["family_id"] == item["family_id"] for s in selected):
                    continue
                selected.append(item)
                if len(selected) >= limit:
                    break
        return selected[:limit]

    def _generate_combination_revision(
        self,
        parent,
        family_sources: List[Dict[str, Any]],
    ) -> str:
        streak = int(parent.metadata.get("non_improve_streak", 0))
        strategy_type = str(parent.metadata.get("type", "pv") or "pv")
        explored_pairs = (
            self._run_explored_hypothesis_pairs(strategy_type)
            if self.menu_exploration_enabled
            else None
        )
        prompt = build_combination_revision_prompt(
            parent_code=parent.code,
            parent_reflection=parent.reflection or "",
            family_sources=family_sources,
            non_improve_streak=streak,
            critic_mode=self.critic_mode,
            strategy_type=strategy_type,
        )
        if self._critic is None:
            if strategy_type == "lob" and is_lob_features_mode(self.lob_input_mode):
                return (
                    "DIAGNOSIS:\n"
                    "  principle: book_imbalance\n"
                    "  form_class: piecewise_by_sign\n"
                    "  features_used: imbalance, spread_rel, has_book\n"
                    "  parent_form: gate on book imbalance; branch formulas from two families\n"
                    "  why_parent_fails: local refinements did not improve surplus\n"
                    "  diversity_vs_parent: gated merge of source families\n"
                    "MECHANISM: Critic disabled; heuristic combination escape.\n"
                    "FIX: Implement gated composition of the two source families.\n"
                )
            if strategy_type == "lob":
                return (
                    "DIAGNOSIS:\n"
                    "  principle: book_imbalance\n"
                    "  form_class: piecewise_by_sign\n"
                    "  features_used: bids_relative, asks_relative, spread_relative\n"
                    "  parent_form: gate on book imbalance; branch formulas from two families\n"
                    "  why_parent_fails: local refinements did not improve surplus\n"
                    "  diversity_vs_parent: gated merge of source families\n"
                    "MECHANISM: Critic disabled; heuristic combination escape.\n"
                    "FIX: Implement gated composition of the two source families.\n"
                )
            if strategy_type == "spread":
                return (
                    "DIAGNOSIS:\n"
                    "  principle: bid_ask_spread_width\n"
                    "  form_class: piecewise_by_sign\n"
                    "  features_used: best_bid, best_ask, units_available, current_pv\n"
                    "  parent_form: gate on spread width; branch formulas from two families\n"
                    "  why_parent_fails: local refinements did not improve surplus\n"
                    "  diversity_vs_parent: gated merge of source families\n"
                    "MECHANISM: Critic disabled; heuristic combination escape.\n"
                    "FIX: Implement gated composition of the two source families.\n"
                )
            return (
                "DIAGNOSIS:\n"
                "  principle: inventory_remaining_units\n"
                "  form_class: piecewise_by_sign\n"
                "  features_used: pv_mean, pv_max\n"
                "  parent_form: gate on pv_mean; branch formulas from two families\n"
                "  why_parent_fails: local refinements did not improve surplus\n"
                "  diversity_vs_parent: gated merge of source families\n"
                "MECHANISM: Critic disabled; heuristic combination escape.\n"
                "FIX: Implement gated composition of the two source families.\n"
            )
        return self._critic.generate_combination_revision(
            prompt=prompt,
            critic_mode=self.critic_mode,
            strategy_type=strategy_type,
            temperature=max(0.2, self.plateau_escape_temperature - 0.1),
            run_explored_pairs=explored_pairs,
        )

    def _store_node_family_metadata(self, node_id: int, *, reflection: str, code: str) -> None:
        try:
            node = self._tree.get(int(node_id))
        except Exception:
            return
        fields = parse_principle_form_fields(reflection or "")
        if fields.get("principle"):
            node.metadata["principle"] = fields["principle"]
        if fields.get("form_class"):
            node.metadata["form_class"] = fields["form_class"]
        node.metadata["family_id"] = derive_family_id(
            reflection=reflection or node.reflection or "",
            code=code or node.code or "",
            metadata=node.metadata,
        )

    def _note_parent_refine_outcome(
        self,
        parent_id: int,
        *,
        child_id: int,
        escape_mode: bool,
    ) -> None:
        """Update non-improve streak after a successful child evaluation."""
        if not self.plateau_escape_enabled or int(parent_id) <= 0:
            return
        try:
            parent = self._tree.get(int(parent_id))
            child = self._tree.get(int(child_id))
        except Exception:
            return

        parent_surplus = parent.metadata.get("mean_surplus")
        child_surplus = child.metadata.get("mean_surplus")
        improved = False
        if parent_surplus is not None and child_surplus is not None:
            improved = float(child_surplus) > (
                float(parent_surplus) + self.plateau_improve_surplus_eps
            )
        else:
            child_reward = (child.metadata.get("performance") or {}).get("reward")
            if child_reward is None:
                child_reward = child.q_value
            improved = float(child_reward) > (
                float(parent.q_value) + self.plateau_improve_surplus_eps
            )

        if escape_mode:
            parent.metadata["non_improve_streak"] = 0
            print(
                f"[reflection] parent {parent_id}: plateau escape complete; "
                f"non_improve_streak reset "
                f"(child_surplus={child_surplus}, parent_surplus={parent_surplus}, "
                f"improved={improved})",
                flush=True,
            )
            return

        if improved:
            parent.metadata["non_improve_streak"] = 0
            print(
                f"[reflection] parent {parent_id}: refine improved surplus; "
                f"non_improve_streak reset "
                f"(child={child_surplus}, parent={parent_surplus})",
                flush=True,
            )
            return

        streak = int(parent.metadata.get("non_improve_streak", 0)) + 1
        parent.metadata["non_improve_streak"] = streak
        print(
            f"[reflection] parent {parent_id}: no surplus improve "
            f"(child={child_surplus}, parent={parent_surplus}); "
            f"non_improve_streak={streak}/{self.plateau_non_improve_threshold}",
            flush=True,
        )

    def _evaluate_seed_nodes_full_refresh(
        self, suite: FixedEvalSuite
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        seed_ids = self._tree.seed_node_ids()
        if not seed_ids:
            return [], {}

        # zi_advantage: one ZI+advantage seed pass for diagnostics; later refreshes
        # recompute advantage Q only (see _refresh_all_nodes).
        if self.q_reward_mode == "zi_advantage":
            return self._evaluate_seed_nodes_on_suite_with_zi(suite)

        refresh_started = time.perf_counter()
        refresh_results = self._refresh_all_nodes(
            suite=suite,
            node_ids=seed_ids,
            include_batch_data=True,
            log_prefix="[reflection] seed",
        )
        refresh_seconds = time.perf_counter() - refresh_started
        refresh_entry = self._make_q_refresh_entry(
            step=0,
            phase="seed",
            refresh_results=refresh_results,
            refresh_seconds=refresh_seconds,
            new_node_id=None,
            eval_suite=suite,
        )
        results_by_id = {int(item["node_id"]): item for item in refresh_results}
        seed_evaluations: List[Dict[str, Any]] = []

        for node_id in seed_ids:
            node = self._tree.get(node_id)
            strategy_type = node.metadata.get("type", "full")
            refresh_result = results_by_id.get(node_id)
            if refresh_result is None or not refresh_result.get("success"):
                seed_evaluations.append(
                    {
                        "node_id": node_id,
                        "type": strategy_type,
                        "success": False,
                        "error": (
                            refresh_result.get("error")
                            if refresh_result is not None
                            else "Missing refresh result"
                        ),
                    }
                )
                continue

            reward = float(refresh_result["reward"])
            reward_variance = float(refresh_result.get("reward_variance", 0.0))
            per_config_rewards = refresh_result.get("per_config_rewards", {})
            per_config_reward_variances = refresh_result.get("per_config_reward_variances", {})
            batch_data = refresh_result.get("batch_data") or {"configs": []}
            performance = summarize_batch_performance(
                batch_data,
                reward_mode=self.reward_mode,
            )
            performance_payload = _performance_payload(
                reward=reward,
                reward_variance=reward_variance,
                per_config_rewards=per_config_rewards,
                per_config_reward_variances=per_config_reward_variances,
                overall=performance,
                batch_data=batch_data,
                reward_mode=self.reward_mode,
            )
            self._store_node_performance(node_id, performance_payload)
            self._store_node_generator_context(
                node_id,
                reward=reward,
                reward_variance=reward_variance,
                per_config_rewards=per_config_rewards,
                per_config_reward_variances=per_config_reward_variances,
                batch_data=batch_data,
            )
            reflection = ""
            if self.critic_enabled:
                batch_summary, compressed_data = _critic_prompt_inputs(
                    self,
                    reward=reward,
                    reward_variance=reward_variance,
                    per_config_rewards=per_config_rewards,
                    per_config_reward_variances=per_config_reward_variances,
                    batch_data=batch_data,
                )
                reflection = self._generate_reflection(
                    code=node.code,
                    reward=reward,
                    reward_variance=reward_variance,
                    batch_summary=batch_summary,
                    batch_data=compressed_data,
                    batch_data_full=batch_data,
                    strategy_type=str(strategy_type or "pv"),
                )
                self._tree.set_reflection(node_id, reflection)
            else:
                self._tree.set_reflection(node_id, "")
            print(
                f"[reflection] seed node {node_id} ({strategy_type}) "
                f"{_format_reward_stats(reward, reward_variance)} "
                f"per_config={per_config_rewards}",
                flush=True,
            )
            seed_evaluations.append(
                {
                    "node_id": node_id,
                    "type": strategy_type,
                    "success": True,
                    "reward": reward,
                    "reward_variance": reward_variance,
                    "per_config_rewards": per_config_rewards,
                    "per_config_reward_variances": per_config_reward_variances,
                    "performance": performance_payload,
                    "reflection": reflection,
                }
            )
        return seed_evaluations, refresh_entry

    def _evaluate_seed_nodes_on_suite_with_zi(
        self, suite: FixedEvalSuite
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Seed eval for zi_advantage: one ZI+advantage pass on the shared suite."""
        seed_ids = self._tree.seed_node_ids()
        seed_evaluations: List[Dict[str, Any]] = []
        refresh_results: List[Dict[str, Any]] = []
        refresh_started = time.perf_counter()
        print(
            f"[reflection] seed evaluating {len(seed_ids)} node(s) with ZI diagnostics "
            f"on suite {suite.suite_id} (Q=advantage; later refreshes skip ZI)...",
            flush=True,
        )
        for node_id in seed_ids:
            node = self._tree.get(node_id)
            strategy_type = node.metadata.get("type", "full")
            required_args, input_mode = self._strategy_type_params(strategy_type)
            ok, err, strategy_fn = self._verifier.verify(
                node.code,
                required_args=required_args,
                input_mode=input_mode,
                forbid_numpy_random=self.forbid_numpy_random,
            )
            if not ok or strategy_fn is None:
                seed_evaluations.append(
                    {
                        "node_id": node_id,
                        "type": strategy_type,
                        "success": False,
                        "error": err,
                    }
                )
                refresh_results.append(
                    {
                        "node_id": node_id,
                        "success": False,
                        "error": err,
                        "reward": float("-inf"),
                    }
                )
                continue

            eval_result = self._evaluate_on_suite(
                strategy_fn=strategy_fn,
                suite=suite,
                strategy_type=strategy_type,
                include_batch_data=True,
            )
            overall = summarize_batch_performance(
                eval_result.batch_data,
                reward_mode=self.reward_mode,
            )
            performance_payload = _performance_payload(
                reward=eval_result.reward,
                reward_variance=eval_result.reward_variance,
                per_config_rewards=eval_result.per_config_rewards,
                per_config_reward_variances=eval_result.per_config_reward_variances,
                overall=overall,
                zi_benchmark_comparison=eval_result.zi_benchmark_comparison,
                batch_data=eval_result.batch_data if isinstance(eval_result.batch_data, dict) else None,
                reward_mode=self.reward_mode,
            )
            self._store_node_performance(node_id, performance_payload)
            self._store_node_generator_context(
                node_id,
                reward=eval_result.reward,
                reward_variance=eval_result.reward_variance,
                per_config_rewards=eval_result.per_config_rewards,
                per_config_reward_variances=eval_result.per_config_reward_variances,
                batch_data=eval_result.batch_data,
            )
            self._store_creation_metrics(
                node_id,
                batch_data=eval_result.batch_data
                if isinstance(eval_result.batch_data, dict)
                else None,
                comparison=eval_result.zi_benchmark_comparison,
            )
            reflection = ""
            if self.critic_enabled:
                batch_summary, compressed_data = _critic_prompt_inputs(
                    self,
                    reward=eval_result.reward,
                    reward_variance=eval_result.reward_variance,
                    per_config_rewards=eval_result.per_config_rewards,
                    per_config_reward_variances=eval_result.per_config_reward_variances,
                    batch_data=eval_result.batch_data,
                )
                reflection = self._generate_reflection(
                    code=node.code,
                    reward=eval_result.reward,
                    reward_variance=eval_result.reward_variance,
                    batch_summary=batch_summary,
                    batch_data=compressed_data,
                    batch_data_full=eval_result.batch_data,
                    strategy_type=str(strategy_type or "pv"),
                )
                self._tree.set_reflection(node_id, reflection)
            else:
                self._tree.set_reflection(node_id, "")
            self._tree.update_q_value(
                node_id,
                eval_result.reward,
                mode=self.q_update_mode,
            )
            print(
                f"[reflection] seed node {node_id} ({strategy_type}) "
                f"{self._format_step_eval_reward(eval_result)} "
                f"per_config={eval_result.per_config_rewards}",
                flush=True,
            )
            if self.critic_enabled and reflection:
                print(
                    f"[reflection] seed node {node_id} reflection "
                    f"(preview: {reflection[:120].replace(chr(10), ' ')}...)",
                    flush=True,
                )
            seed_evaluations.append(
                {
                    "node_id": node_id,
                    "type": strategy_type,
                    "success": True,
                    "reward": eval_result.reward,
                    "reward_variance": eval_result.reward_variance,
                    "per_config_rewards": eval_result.per_config_rewards,
                    "per_config_reward_variances": eval_result.per_config_reward_variances,
                    "performance": performance_payload,
                    "reflection": reflection,
                    "zi_benchmark_comparison": eval_result.zi_benchmark_comparison,
                }
            )
            refresh_results.append(
                {
                    "node_id": node_id,
                    "success": True,
                    "error": None,
                    "reward": float(eval_result.reward),
                    "reward_variance": float(eval_result.reward_variance),
                    "per_config_rewards": eval_result.per_config_rewards,
                    "per_config_reward_variances": eval_result.per_config_reward_variances,
                    "zi_benchmark_comparison": eval_result.zi_benchmark_comparison,
                    "scalar_reward": eval_result.scalar_reward,
                }
            )

        refresh_seconds = time.perf_counter() - refresh_started
        refresh_entry = self._make_q_refresh_entry(
            step=0,
            phase="seed",
            refresh_results=refresh_results,
            refresh_seconds=refresh_seconds,
            new_node_id=None,
            eval_suite=suite,
        )
        return seed_evaluations, refresh_entry

    def _strategy_type_params(self, strategy_type: str) -> Tuple[int, str]:
        if strategy_type == "pv":
            return 3, "pv"
        requires_single_input = strategy_type in {"spread", "lob"}
        required_args = 1 if requires_single_input else 3
        if strategy_type == "lob" and is_lob_features_mode(self.lob_input_mode):
            input_mode = "lob_features"
        else:
            input_mode = strategy_type if requires_single_input else "full"
        return required_args, input_mode

    def _evaluate_seed_nodes(self, batch_configs: List[ReflectionMarketConfig]) -> List[Dict[str, Any]]:
        """Run batch evaluation on each initial seed node and set its q_value."""
        results: List[Dict[str, Any]] = []
        seed_ids = self._tree.seed_node_ids()
        if not seed_ids:
            return results

        print(f"[reflection] evaluating {len(seed_ids)} seed node(s)...", flush=True)
        for node_id in seed_ids:
            node = self._tree.get(node_id)
            strategy_type = node.metadata.get("type", "full")
            required_args, input_mode = self._strategy_type_params(strategy_type)
            ok, err, strategy_fn = self._verifier.verify(
                node.code,
                required_args=required_args,
                input_mode=input_mode,
                forbid_numpy_random=self.forbid_numpy_random,
            )
            if not ok or strategy_fn is None:
                print(
                    f"[reflection] seed node {node_id} ({strategy_type}) verify FAILED: {err}",
                    flush=True,
                )
                results.append(
                    {
                        "node_id": node_id,
                        "type": strategy_type,
                        "success": False,
                        "error": err,
                    }
                )
                continue

            eval_result = self._evaluate_strategy_batch(
                strategy_fn=strategy_fn,
                batch_configs=batch_configs,
                step_index=-1,
                strategy_type=strategy_type,
                include_batch_data=True,
            )
            overall = summarize_batch_performance(
                eval_result.batch_data,
                reward_mode=self.reward_mode,
            )
            performance_payload = _performance_payload(
                reward=eval_result.reward,
                reward_variance=eval_result.reward_variance,
                per_config_rewards=eval_result.per_config_rewards,
                per_config_reward_variances=eval_result.per_config_reward_variances,
                overall=overall,
                zi_benchmark_comparison=eval_result.zi_benchmark_comparison,
                batch_data=eval_result.batch_data if isinstance(eval_result.batch_data, dict) else None,
                reward_mode=self.reward_mode,
            )
            self._store_node_performance(node_id, performance_payload)
            self._store_node_generator_context(
                node_id,
                reward=eval_result.reward,
                reward_variance=eval_result.reward_variance,
                per_config_rewards=eval_result.per_config_rewards,
                per_config_reward_variances=eval_result.per_config_reward_variances,
                batch_data=eval_result.batch_data,
            )
            reflection = ""
            if self.critic_enabled:
                batch_summary, compressed_data = _critic_prompt_inputs(
                    self,
                    reward=eval_result.reward,
                    reward_variance=eval_result.reward_variance,
                    per_config_rewards=eval_result.per_config_rewards,
                    per_config_reward_variances=eval_result.per_config_reward_variances,
                    batch_data=eval_result.batch_data,
                )
                reflection = self._generate_reflection(
                    code=node.code,
                    reward=eval_result.reward,
                    reward_variance=eval_result.reward_variance,
                    batch_summary=batch_summary,
                    batch_data=compressed_data,
                    batch_data_full=eval_result.batch_data,
                    strategy_type=str(strategy_type or "pv"),
                )
                self._tree.set_reflection(node_id, reflection)
            else:
                self._tree.set_reflection(node_id, "")
            if self.q_reward_mode == "zi_beat_fraction":
                self._apply_zi_node_eval(
                    node_id,
                    comparison=eval_result.zi_benchmark_comparison,
                )
            self._tree.update_q_value(
                node_id,
                eval_result.reward,
                mode=self.q_update_mode,
            )
            print(
                f"[reflection] seed node {node_id} ({strategy_type}) "
                f"{self._format_step_eval_reward(eval_result)} "
                f"per_config={eval_result.per_config_rewards}",
                flush=True,
            )
            if self.critic_enabled and reflection:
                print(
                    f"[reflection] seed node {node_id} reflection "
                    f"(preview: {reflection[:120].replace(chr(10), ' ')}...)",
                    flush=True,
                )
            results.append(
                {
                    "node_id": node_id,
                    "type": strategy_type,
                    "success": True,
                    "reward": eval_result.reward,
                    "reward_variance": eval_result.reward_variance,
                    "per_config_rewards": eval_result.per_config_rewards,
                    "per_config_reward_variances": eval_result.per_config_reward_variances,
                    "performance": performance_payload,
                    "reflection": reflection,
                }
            )
        return results

    def _save_code(
        self,
        *,
        code: str,
        success: bool,
        error: Optional[str],
        task_description: str,
        step: int,
        attempt: int,
    ) -> Tuple[str, Optional[int]]:
        if success:
            node_id = self._tree.next_node_id()
            code_file = self._storage.save_code(
                code=code,
                function_name="agent_strategy",
                iteration=node_id,
                task_description=task_description,
                error_feedback=error,
                success=True,
            )
            return code_file, node_id

        code_file = self._storage.save_failed_attempt(
            code=code,
            function_name="agent_strategy",
            step=step,
            attempt=attempt,
            task_description=task_description,
            error_feedback=error,
        )
        return code_file, None

    def generate_and_verify(
        self, *, step: int, step_index: Optional[int] = None
    ) -> GenerationResult:
        parent, select_mode = self._select_parent_node()
        self._log_parent_selection(parent, select_mode=select_mode)
        parent_type = parent.metadata.get("type", "full")
        required_args, input_mode = self._strategy_type_params(parent_type)
        escape_mode = self._should_plateau_escape(parent)
        hypothesis_jump = False
        if self.fixed_strategy_code is not None:
            code = self.fixed_strategy_code
            ok, err, fn = self._verifier.verify(
                code,
                required_args=required_args,
                input_mode=input_mode,
                forbid_numpy_random=self.forbid_numpy_random,
            )
            code_file, node_id = self._save_code(
                code=code,
                success=bool(ok),
                error=err,
                task_description="fixed strategy",
                step=step,
                attempt=1,
            )
            return GenerationResult(
                code=code,
                strategy_fn=fn,
                parent_id=parent.node_id,
                success=bool(ok),
                error=err,
                code_file=code_file,
                strategy_type=parent_type,
                node_id=node_id,
                escape_mode=False,
                hypothesis_jump=False,
                parent_select_mode=select_mode,
            )

        parent_reflection = parent.reflection
        extra_instructions: Optional[List[str]] = None
        gen_temperature = 0.2
        combination_mode = False
        combination_source_codes: Optional[List[str]] = None
        combination_source_family_ids: Optional[List[str]] = None
        forced_pair = self._force_hypothesis_pair
        if forced_pair is not None:
            principle, form_class = forced_pair
            escape_mode = False
            hypothesis_jump = True
            parent_fields = parse_principle_form_fields(parent.reflection or "")
            parent_reflection = _forced_menu_pair_diagnosis(
                principle=principle,
                form_class=form_class,
                strategy_type=str(parent_type or "pv"),
                lob_input_mode=self.lob_input_mode,
            )
            extra_instructions = [
                "MENU SWEEP: implement the forced hypothesis pair from Parent Reflection.",
                f"REQUIRED principle={principle}, form_class={form_class}.",
                "Follow the Parent Reflection DIAGNOSIS — new structure/features/control "
                "flow as needed for this pair, not only constants.",
            ] + hypothesis_jump_generator_instructions(
                parent_principle=parent_fields.get("principle"),
                parent_form=parent_fields.get("form_class"),
            )
            gen_temperature = self.plateau_escape_temperature
            print(
                f"[reflection] menu-sweep forced pair "
                f"principle={principle} form_class={form_class} "
                f"(parent={parent.node_id})",
                flush=True,
            )
        elif escape_mode:
            if self.plateau_escape_mode == "combination":
                combination_mode = True
                family_sources = self._select_family_sources(parent)
                family_ids = [str(s.get("family_id") or "?") for s in family_sources]
                combination_source_family_ids = family_ids
                combination_source_codes = [
                    str(s.get("code") or "") for s in family_sources if s.get("code")
                ]
                print(
                    f"[reflection] plateau combination escape on parent {parent.node_id} "
                    f"(non_improve_streak="
                    f"{int(parent.metadata.get('non_improve_streak', 0))}/"
                    f"{self.plateau_non_improve_threshold}); "
                    f"families={family_ids}",
                    flush=True,
                )
                parent_reflection = self._generate_combination_revision(
                    parent,
                    family_sources,
                )
                extra_instructions = combination_generator_instructions(
                    family_a_id=family_ids[0] if family_ids else None,
                    family_b_id=family_ids[1] if len(family_ids) > 1 else None,
                    strategy_type=str(parent.metadata.get("type", "pv") or "pv"),
                )
                parent.metadata["combination_revision"] = parent_reflection
                parent.metadata["combination_source_family_ids"] = family_ids
            else:
                print(
                    f"[reflection] plateau escape on parent {parent.node_id} "
                    f"(non_improve_streak="
                    f"{int(parent.metadata.get('non_improve_streak', 0))}/"
                    f"{self.plateau_non_improve_threshold}): revising hypothesis...",
                    flush=True,
                )
                parent_reflection = self._generate_plateau_revision(parent)
                stalled = parse_principle_form_fields(parent.reflection or "")
                extra_instructions = plateau_escape_generator_instructions(
                    stalled_principle=stalled.get("principle"),
                    stalled_form=stalled.get("form_class"),
                )
            gen_temperature = self.plateau_escape_temperature
            parent.metadata["plateau_revision"] = parent_reflection
            parent.metadata["last_escape_step"] = int(step)
        elif (
            select_mode == "ucb"
            and self.hypothesis_jump_prob > 0
            and parent.reflection
            and self._rng.random() < self.hypothesis_jump_prob
        ):
            hypothesis_jump = True
            print(
                f"[reflection] hypothesis jump on UCB parent {parent.node_id} "
                f"(prob={self.hypothesis_jump_prob:.2f}): exploring new principle/form...",
                flush=True,
            )
            parent_reflection = self._generate_hypothesis_jump_revision(parent)
            parent_fields = parse_principle_form_fields(parent.reflection or "")
            extra_instructions = hypothesis_jump_generator_instructions(
                parent_principle=parent_fields.get("principle"),
                parent_form=parent_fields.get("form_class"),
            )
            gen_temperature = self.plateau_escape_temperature
            parent.metadata["hypothesis_jump_revision"] = parent_reflection
            parent.metadata["last_hypothesis_jump_step"] = int(step)

        action = parse_diagnosis_action(parent_reflection or "")
        same_fail = parse_same_failure_as_last_hold(parent_reflection or "")
        if action is None and same_fail == "yes":
            action = "refine"
        elif action is None and same_fail == "no":
            action = "replace"
        # Plateau / jump / menu-sweep already set extra_instructions; don't
        # contradict those with refine/replace from the injected diagnosis.
        if action and extra_instructions is None:
            parent_fields = parse_principle_form_fields(parent_reflection or "")
            extra_instructions = diagnosis_action_generator_instructions(
                action=action,
                parent_principle=parent_fields.get("principle"),
                parent_form=parent_fields.get("form_class"),
            )

        forbidden_strategies = _collect_forbidden_strategy_examples(parent, self._tree)

        generator_batch_summary, generator_batch_data = (
            self._parent_generator_context_for_generation(parent.node_id)
        )
        if (
            step_index is not None
            and self._is_resample_transfer_step(step_index)
            and self.generator_include_critic_data
        ):
            generator_batch_data = self._attach_parent_transfer_cards(
                parent=parent,
                generator_batch_data=generator_batch_data,
                strategy_type=parent_type,
                step_index=step_index,
            )

        print("[reflection] calling LLM for strategy code...", flush=True)
        code = ""
        last_error = None
        strategy_fn = None
        success = False
        code_file = None
        node_id: Optional[int] = None
        parent_performance = self._parent_performance_for_generation(parent.node_id)
        sibling = self._most_recent_sibling(parent.node_id)
        sibling_code = sibling.code if sibling is not None else None
        sibling_context = (
            self._sibling_context_for_generation(sibling)
            if sibling is not None
            else None
        )
        plan_guided_recovery = (
            self.codegen_recovery_mode == "plan_guided"
            and bool(parent_reflection)
            and is_structured_principle_form(self.critic_mode)
        )

        def _debug_codegen(
            *,
            error: str,
            current_code: str,
            force_new_form: bool = False,
            temperature: float = DUPLICATE_RECOVERY_TEMPERATURE,
        ) -> Optional[str]:
            """Retry debug codegen; return None instead of aborting the run."""
            last_dbg_error: Optional[str] = None
            for debug_attempt in range(self.max_debug_retries):
                try:
                    return self._generator.debug(
                        code=current_code,
                        error=error,
                        forbidden_strategies=forbidden_strategies or None,
                        force_new_form=force_new_form,
                        temperature=temperature,
                        parent_reflection=(
                            parent_reflection if plan_guided_recovery else None
                        ),
                        plan_guided_recovery=plan_guided_recovery,
                        critic_mode=self.critic_mode,
                        strategy_type=parent_type,
                    )
                except Exception as exc:
                    last_dbg_error = str(exc)
                    print(
                        f"[reflection] debug codegen failed "
                        f"(attempt {debug_attempt + 1}/{self.max_debug_retries}): "
                        f"{last_dbg_error}",
                        flush=True,
                    )
            print(
                "[reflection] debug codegen exhausted retries; skipping step",
                flush=True,
            )
            return None

        def _generate_once(*, temperature: float = 0.2) -> str:
            return self._generator.generate(
                parent_code=parent.code,
                parent_reflection=parent_reflection,
                strategy_output_mode=self.strategy_output_mode,
                offline_example=None,
                strategy_type=parent_type,
                use_reflection=self.critic_enabled or bool(parent_reflection),
                parent_performance=parent_performance,
                generator_batch_summary=generator_batch_summary,
                generator_batch_data=generator_batch_data,
                sibling_code=sibling_code,
                sibling_context=sibling_context,
                q_reward_mode=self.q_reward_mode,
                critic_mode=self.critic_mode,
                forbid_numpy_random=self.forbid_numpy_random,
                extra_instructions=extra_instructions,
                temperature=temperature,
                lob_input_mode=self.lob_input_mode,
            )

        for gen_attempt in range(self.max_debug_retries):
            try:
                code = _generate_once(temperature=gen_temperature)
                break
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"[reflection] codegen failed (attempt {gen_attempt + 1}): {last_error}",
                    flush=True,
                )
                if gen_attempt >= self.max_debug_retries - 1:
                    code_file, node_id = self._save_code(
                        code=code or "# code generation failed",
                        success=False,
                        error=last_error,
                        task_description="reflection step codegen failed",
                        step=step,
                        attempt=gen_attempt + 1,
                    )
                    self._note_parent_codegen_outcome(parent.node_id, success=False)
                    return GenerationResult(
                        code=code,
                        strategy_fn=None,
                        parent_id=parent.node_id,
                        success=False,
                        error=last_error,
                        code_file=code_file,
                        strategy_type=parent_type,
                        node_id=node_id,
                        escape_mode=escape_mode,
                        hypothesis_jump=hypothesis_jump,
                        parent_select_mode=select_mode,
                    )

        for attempt in range(self.max_debug_retries):
            ok, err, fn = self._verifier.verify(
                code,
                required_args=required_args,
                input_mode=input_mode,
                forbid_numpy_random=self.forbid_numpy_random,
            )
            code_file, node_id = self._save_code(
                code=code,
                success=bool(ok),
                error=err,
                task_description=f"reflection step attempt {attempt + 1}",
                step=step,
                attempt=attempt + 1,
            )
            if ok:
                dup_error = _duplicate_strategy_rejection_message(
                    code,
                    parent=parent,
                    tree=self._tree,
                )
                if dup_error:
                    ok = False
                    last_error = dup_error
                    print(
                        f"[reflection] verify rejected (attempt {attempt + 1}): "
                        f"{last_error}",
                        flush=True,
                    )
                    if attempt < self.max_debug_retries - 1:
                        recovery_label = (
                            "plan-guided debug (implement Parent Reflection DIAGNOSIS)"
                            if plan_guided_recovery
                            else "local parent edit (sibling clones forbidden)"
                        )
                        print(
                            f"[reflection] calling LLM debug for {recovery_label}...",
                            flush=True,
                        )
                        code = _debug_codegen(
                            error=last_error,
                            current_code=code,
                            force_new_form=bool(escape_mode or hypothesis_jump),
                            temperature=(
                                self.plateau_escape_temperature
                                if escape_mode or hypothesis_jump
                                else DUPLICATE_RECOVERY_TEMPERATURE
                            ),
                        )
                        if code is None:
                            last_error = last_error or "debug codegen failed after retries"
                            break
                        continue
                diag_error = None
                if plan_guided_recovery:
                    diag_error = validate_code_implements_diagnosis(
                        code,
                        parent_reflection or "",
                        strategy_type=str(parent_type or "pv"),
                    )
                if diag_error:
                    ok = False
                    last_error = diag_error
                    print(
                        f"[reflection] verify rejected (attempt {attempt + 1}): "
                        f"{last_error}",
                        flush=True,
                    )
                    if attempt < self.max_debug_retries - 1:
                        print(
                            "[reflection] calling LLM debug for plan-guided "
                            "diagnosis implementation...",
                            flush=True,
                        )
                        code = _debug_codegen(
                            error=last_error,
                            current_code=code,
                            force_new_form=bool(escape_mode or hypothesis_jump),
                            temperature=(
                                self.plateau_escape_temperature
                                if escape_mode or hypothesis_jump
                                else DUPLICATE_RECOVERY_TEMPERATURE
                            ),
                        )
                        if code is None:
                            last_error = last_error or "debug codegen failed after retries"
                            break
                        continue
                elif combination_mode and combination_source_codes:
                    combo_error = self._verifier.combination_behavior_distinct(
                        code,
                        combination_source_codes,
                        strategy_type=str(parent_type or "pv"),
                        lob_input_mode=self.lob_input_mode,
                    )
                    if combo_error:
                        ok = False
                        last_error = combo_error
                        print(
                            f"[reflection] verify rejected (attempt {attempt + 1}): "
                            f"{last_error}",
                            flush=True,
                        )
                        if attempt < self.max_debug_retries - 1:
                            print(
                                "[reflection] calling LLM debug for combination "
                                "distinctness...",
                                flush=True,
                            )
                            code = _debug_codegen(
                                error=last_error,
                                current_code=code,
                                force_new_form=True,
                                temperature=self.plateau_escape_temperature,
                            )
                            if code is None:
                                last_error = (
                                    last_error or "debug codegen failed after retries"
                                )
                                break
                            continue
                else:
                    success = True
                    strategy_fn = fn
                    last_error = None
                    print(f"[reflection] verify OK (attempt {attempt + 1})", flush=True)
                    break
            last_error = err or "Verification failed"
            print(
                f"[reflection] verify failed (attempt {attempt + 1}): {last_error}",
                flush=True,
            )
            if attempt < self.max_debug_retries - 1:
                print("[reflection] calling LLM debug...", flush=True)
                code = _debug_codegen(
                    error=last_error,
                    current_code=code,
                    temperature=DUPLICATE_RECOVERY_TEMPERATURE,
                )
                if code is None:
                    last_error = last_error or "debug codegen failed after retries"
                    break

        self._note_parent_codegen_outcome(parent.node_id, success=bool(success))
        return GenerationResult(
            code=code,
            strategy_fn=strategy_fn,
            parent_id=parent.node_id,
            success=success,
            error=last_error,
            code_file=code_file,
            strategy_type=parent_type,
            node_id=node_id,
            escape_mode=escape_mode,
            hypothesis_jump=hypothesis_jump,
            combination_mode=combination_mode,
            parent_select_mode=select_mode,
            combination_source_codes=combination_source_codes,
            combination_source_family_ids=combination_source_family_ids,
        )

    def _sibling_reflections_for_parent(self, parent_id: int) -> List[str]:
        parent = self._tree.get(parent_id)
        reflections: List[str] = []
        for child_id in getattr(parent, "children", []) or []:
            child = self._tree.get(child_id)
            if child is not None and child.reflection:
                reflections.append(str(child.reflection))
        return reflections

    def _generate_reflection(
        self,
        *,
        code: str,
        reward: float,
        reward_variance: float,
        batch_summary: Dict[str, Any],
        batch_data: Any,
        batch_data_full: Optional[Dict[str, Any]] = None,
        strategy_type: str = "pv",
        parent_id: Optional[int] = None,
    ) -> str:
        if self._critic is None:
            return ""
        critic_batch_data = batch_data_full if batch_data_full is not None else batch_data
        parent_reflection: Optional[str] = None
        sibling_reflections: Optional[List[str]] = None
        if parent_id is not None:
            parent_node = self._tree.get(parent_id)
            parent_reflection = getattr(parent_node, "reflection", "") or ""
            sibling_reflections = self._sibling_reflections_for_parent(parent_id)
        explored_pairs = (
            self._run_explored_hypothesis_pairs(strategy_type)
            if self.menu_exploration_enabled
            else None
        )
        return self._critic.generate(
            code=code,
            batch_summary=batch_summary,
            batch_data=critic_batch_data,
            reward=reward,
            reward_variance=reward_variance,
            include_data=self.critic_include_data,
            q_reward_mode=self.q_reward_mode,
            critic_mode=self.critic_mode,
            strategy_type=strategy_type,
            parent_reflection=parent_reflection,
            sibling_reflections=sibling_reflections,
            run_explored_pairs=explored_pairs,
            menu_exploration_enabled=self.menu_exploration_enabled,
            lob_input_mode=self.lob_input_mode,
        )

    def reflect_and_update(
        self,
        *,
        code: str,
        reward: float,
        reward_variance: float,
        batch_summary: Dict[str, Any],
        batch_data: Any,
        batch_data_full: Optional[Dict[str, Any]] = None,
        per_config_rewards: Optional[Dict[str, float]] = None,
        per_config_reward_variances: Optional[Dict[str, float]] = None,
        parent_id: int,
        escape_mode: bool = False,
        hypothesis_jump: bool = False,
        combination_mode: bool = False,
        combination_source_family_ids: Optional[List[str]] = None,
    ) -> str:
        parent_node = self._tree.get(parent_id)
        parent_type = parent_node.metadata.get("type", "full")
        reflection = ""
        if self.critic_enabled:
            reflection = self._generate_reflection(
                code=code,
                reward=reward,
                reward_variance=reward_variance,
                batch_summary=batch_summary,
                batch_data=batch_data,
                batch_data_full=batch_data_full,
                strategy_type=str(parent_type or "pv"),
                parent_id=parent_id,
            )

        parent_hypothesis = getattr(parent_node, "reflection", "") or ""
        adherence = score_hypothesis_adherence(
            code,
            parent_hypothesis,
            strategy_type=str(parent_type or "pv"),
        )
        child_metadata = {
            "type": parent_type,
            "escape_mode": bool(escape_mode),
            "hypothesis_jump": bool(hypothesis_jump),
            "hypothesis_adherence": adherence.to_dict(),
        }
        parent_desc = parent_node.metadata.get("desc")
        if combination_mode:
            child_metadata["combination_mode"] = True
            if combination_source_family_ids:
                child_metadata["source_family_ids"] = list(combination_source_family_ids)
        if parent_desc:
            child_metadata["desc"] = parent_desc

        child = self._tree.add_node(
            parent_id=parent_id,
            code=code,
            reflection=reflection,
            reward=reward,
            metadata=child_metadata,
            q_update_mode=self.q_update_mode,
        )
        self._store_node_family_metadata(child.node_id, reflection=reflection, code=code)
        overall = None
        if batch_data_full is not None:
            overall = summarize_batch_performance(
                batch_data_full,
                reward_mode=self.reward_mode,
            )
        performance_payload = _performance_payload(
            reward=reward,
            reward_variance=reward_variance,
            per_config_rewards=per_config_rewards or {},
            per_config_reward_variances=per_config_reward_variances or {},
            overall=overall,
            zi_benchmark_comparison=self._zi_comparison_from_batch(batch_data_full),
            batch_data=batch_data_full if isinstance(batch_data_full, dict) else None,
            reward_mode=self.reward_mode,
        )
        self._store_node_performance(child.node_id, performance_payload)
        self._store_node_generator_context(
            child.node_id,
            reward=reward,
            reward_variance=reward_variance,
            per_config_rewards=per_config_rewards or {},
            per_config_reward_variances=per_config_reward_variances or {},
            batch_data=batch_data_full,
        )
        self._apply_zi_node_eval(
            child.node_id,
            comparison=performance_payload.get("zi_benchmark_comparison"),
        )
        self._store_creation_metrics(
            child.node_id,
            batch_data=batch_data_full if isinstance(batch_data_full, dict) else None,
            comparison=performance_payload.get("zi_benchmark_comparison"),
        )
        self._note_parent_refine_outcome(
            parent_id,
            child_id=child.node_id,
            escape_mode=bool(escape_mode),
        )
        print(
            f"[reflection] child {child.node_id}: hypothesis adherence "
            f"overall={adherence.overall:.3f} "
            f"(principle_ok={adherence.principle_ok}, form_ok={adherence.form_class_ok}, "
            f"features={adherence.feature_coverage:.3f})",
            flush=True,
        )
        if adherence.violation:
            print(
                f"[reflection] child {child.node_id}: adherence note: {adherence.violation}",
                flush=True,
            )
        self._maybe_mark_inactive(child.node_id)
        self._total_visits += 1
        return reflection

    def _run_refinement_step(
        self,
        *,
        step_display: int,
        steps_total: int,
        step_index: int,
        codegen_step: int,
        phase: str,
        train_configs: List[ReflectionMarketConfig],
        q_suite_configs: List[ReflectionMarketConfig],
        block_idx: Optional[int] = None,
        sweep_market_index: Optional[int] = None,
        forced_pair: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        """One codegen → train eval → critic → Q refresh cycle."""
        batch_configs = train_configs
        header_bits = [
            f"\n[reflection] === {phase} step {step_display}/{steps_total} === "
            f"batch={[cfg.label for cfg in batch_configs]}"
        ]
        if phase == "menu_sweep":
            header_bits.append(
                f" sweep_mkt={sweep_market_index} "
                f"q_suite={[cfg.label for cfg in q_suite_configs]}"
            )
            if forced_pair is not None:
                header_bits.append(
                    f" force_pair={forced_pair[0]}+{forced_pair[1]}"
                )
        elif self.market_block_size > 0:
            header_bits.append(
                f" block={block_idx} q_suite={[cfg.label for cfg in q_suite_configs]}"
            )
        elif self.home_set_size > 0:
            header_bits.append(
                f" home_set={[cfg.label for cfg in train_configs]}"
            )
        elif self.resample_every > 0:
            header_bits.append(
                f" resample_slot={self._resample_slot_index(step_index)} "
                f"hold={[cfg.label for cfg in train_configs]}"
            )
            if self._is_resample_transfer_step(step_index):
                prev = self._previous_resample_market()
                header_bits.append(
                    f" transfer_test last={prev.label if prev is not None else '?'}"
                )
        print("".join(header_bits), flush=True)

        step_suite: Optional[FixedEvalSuite] = None
        q_suite: Optional[FixedEvalSuite] = None
        if self.q_eval_mode == "full_refresh":
            step_suite = self._build_step_eval_suite(train_configs, step_index=step_index)
            train_labels = [cfg.label for cfg in train_configs]
            q_labels = [cfg.label for cfg in q_suite_configs]
            q_suite = (
                step_suite
                if train_labels == q_labels
                else self._build_step_eval_suite(
                    q_suite_configs, step_index=step_index + 10_000
                )
            )
            print(
                f"[reflection] step {step_display} train suite {step_suite.suite_id} "
                f"(creation/critic); Q suite {q_suite.suite_id} "
                f"({len(q_suite_configs)} market(s))",
                flush=True,
            )

        self._force_hypothesis_pair = forced_pair
        self._force_train_configs = list(train_configs)
        self._force_q_configs = list(q_suite_configs)
        try:
            gen = self.generate_and_verify(step=codegen_step, step_index=step_index)
        finally:
            self._clear_step_force_overrides()

        if not gen.success or gen.strategy_fn is None:
            print(
                f"[reflection] step {step_display}: codegen/verify FAILED "
                f"(parent={gen.parent_id}, type={gen.strategy_type}): {gen.error}",
                flush=True,
            )
            return {
                "step": step_display,
                "phase": phase,
                "success": False,
                "error": gen.error,
                "parent_id": gen.parent_id,
                "code_file": gen.code_file,
                "sweep_market_index": sweep_market_index,
                "forced_pair": (
                    {"principle": forced_pair[0], "form_class": forced_pair[1]}
                    if forced_pair
                    else None
                ),
            }

        print(
            f"[reflection] step {step_display}: strategy OK "
            f"(parent={gen.parent_id}, type={gen.strategy_type}, code={gen.code_file})",
            flush=True,
        )
        print(f"[reflection] step {step_display}: evaluating batch...", flush=True)
        if self.q_eval_mode == "full_refresh":
            eval_result = self._evaluate_on_suite(
                strategy_fn=gen.strategy_fn,
                suite=step_suite,
                strategy_type=gen.strategy_type,
                include_batch_data=True,
            )
        else:
            eval_result = self._evaluate_strategy_batch(
                strategy_fn=gen.strategy_fn,
                batch_configs=batch_configs,
                step_index=step_index,
                strategy_type=gen.strategy_type,
                include_batch_data=True,
            )
        if (
            self._is_resample_transfer_step(step_index)
            and gen.strategy_fn is not None
            and isinstance(eval_result.batch_data, dict)
        ):
            prior_card = self._second_test_prior_hold_card(
                strategy_fn=gen.strategy_fn,
                strategy_type=gen.strategy_type,
                step_index=step_index,
            )
            if prior_card:
                eval_result.batch_data = _attach_prior_hold_card(
                    eval_result.batch_data, prior_card
                )
        batch_summary, compressed_data = _critic_prompt_inputs(
            self,
            reward=eval_result.reward,
            reward_variance=eval_result.reward_variance,
            per_config_rewards=eval_result.per_config_rewards,
            per_config_reward_variances=eval_result.per_config_reward_variances,
            batch_data=eval_result.batch_data,
        )
        print(
            f"[reflection] step {step_display}: "
            f"{self._format_step_eval_reward(eval_result)} "
            f"per_config={eval_result.per_config_rewards}",
            flush=True,
        )
        if self.critic_enabled:
            print(
                f"[reflection] step {step_display}: generating critic reflection...",
                flush=True,
            )
        else:
            print(
                f"[reflection] step {step_display}: recording metrics (no reflection)...",
                flush=True,
            )
        reflection = self.reflect_and_update(
            code=gen.code,
            reward=eval_result.reward,
            reward_variance=eval_result.reward_variance,
            batch_summary=batch_summary,
            batch_data=compressed_data,
            batch_data_full=eval_result.batch_data,
            per_config_rewards=eval_result.per_config_rewards,
            per_config_reward_variances=eval_result.per_config_reward_variances,
            parent_id=gen.parent_id,
            escape_mode=bool(gen.escape_mode),
            hypothesis_jump=bool(gen.hypothesis_jump),
            combination_mode=bool(gen.combination_mode),
            combination_source_family_ids=gen.combination_source_family_ids,
        )

        step_payload: Dict[str, Any] = {
            "step": step_display,
            "phase": phase,
            "success": True,
            "parent_id": gen.parent_id,
            "node_id": gen.node_id,
            "escape_mode": bool(gen.escape_mode),
            "hypothesis_jump": bool(gen.hypothesis_jump),
            "combination_mode": bool(gen.combination_mode),
            "parent_select_mode": gen.parent_select_mode,
            "reward": eval_result.reward,
            "reward_variance": eval_result.reward_variance,
            "per_config_rewards": eval_result.per_config_rewards,
            "per_config_reward_variances": eval_result.per_config_reward_variances,
            "performance": summarize_batch_performance(
                eval_result.batch_data,
                reward_mode=self.reward_mode,
            ),
            "reflection": reflection,
            "code_file": gen.code_file,
            "batch_configs": [cfg.observable_state() for cfg in batch_configs],
            "q_suite_configs": [cfg.observable_state() for cfg in q_suite_configs],
            "market_block_index": block_idx,
            "sweep_market_index": sweep_market_index,
            "forced_pair": (
                {"principle": forced_pair[0], "form_class": forced_pair[1]}
                if forced_pair
                else None
            ),
            "batch_data": eval_result.batch_data,
        }
        if eval_result.scalar_reward is not None:
            step_payload["scalar_reward"] = eval_result.scalar_reward
        if eval_result.zi_benchmark_comparison is not None:
            step_payload["zi_benchmark_comparison"] = eval_result.zi_benchmark_comparison

        if self.q_eval_mode == "full_refresh":
            refresh_started = time.perf_counter()
            refresh_results = self._refresh_all_nodes(
                suite=q_suite if q_suite is not None else step_suite,
                include_batch_data=False,
            )
            refresh_seconds = time.perf_counter() - refresh_started
            refresh_entry = self._make_q_refresh_entry(
                step=codegen_step,
                phase=phase,
                refresh_results=refresh_results,
                refresh_seconds=refresh_seconds,
                new_node_id=self._tree.evaluable_node_ids()[-1],
                eval_suite=q_suite if q_suite is not None else step_suite,
            )
            self._q_refresh_history.append(refresh_entry)
            step_payload["q_refresh"] = refresh_entry
            new_node_id = refresh_entry["new_node_id"]
            refreshed_node = self._tree.get(new_node_id)
            step_payload["reward"] = float(refreshed_node.q_value)
            step_payload["reward_variance"] = float(
                refresh_entry.get("q_variances", {}).get(str(new_node_id), 0.0)
            )
            print(
                f"[reflection] step {step_display}: refreshed Q for all nodes; "
                f"new node {new_node_id} "
                f"{_format_reward_stats(step_payload['reward'], step_payload['reward_variance'])}",
                flush=True,
            )

        if self.critic_enabled and reflection:
            print(
                f"[reflection] step {step_display}: reflection saved "
                f"(preview: {reflection[:120].replace(chr(10), ' ')}...)",
                flush=True,
            )
        else:
            print(
                f"[reflection] step {step_display}: node added with performance metrics only",
                flush=True,
            )
        return step_payload

    def _run_menu_sweep(self, *, codegen_step_start: int) -> Tuple[List[Dict[str, Any]], int]:
        """Sample K markets × all menu pairs; return (summary_entries, next_codegen_step)."""
        if self.menu_sweep_markets <= 0:
            return [], codegen_step_start
        if self._config_space is None:
            raise RuntimeError("menu sweep requires config_space")

        strategy_type = "spread"
        active_types = {
            str(self._tree.get(nid).metadata.get("type", "pv"))
            for nid in self._tree.active_node_ids()
        }
        if len(active_types) == 1:
            strategy_type = next(iter(active_types))
        elif "spread" in active_types:
            strategy_type = "spread"
        elif "lob" in active_types:
            strategy_type = "lob"
        elif "pv" in active_types:
            strategy_type = "pv"

        pairs = self._menu_sweep_pairs(strategy_type)
        total_steps = self.menu_sweep_markets * len(pairs)
        print(
            f"[reflection] menu sweep: markets={self.menu_sweep_markets} "
            f"pairs={len(pairs)} total_steps={total_steps} "
            f"strategy_type={strategy_type} "
            f"pair_order={[f'{p}+{f}' for p, f in pairs]}",
            flush=True,
        )
        summary: List[Dict[str, Any]] = []
        codegen_step = codegen_step_start
        display = 0
        for market_i in range(self.menu_sweep_markets):
            market = self._config_space.sample(
                self._rng, batch_index=1000 + market_i
            )
            self._sweep_markets.append(market)
            print(
                f"[reflection] menu sweep market {market_i}: train={market.label}",
                flush=True,
            )
            q_suite_configs = self._sweep_markets[-self.q_suite_size :]
            for pair in pairs:
                display += 1
                codegen_step += 1
                # Positive band above post-sweep steps (0..steps-1) and below
                # the +10_000 Q-suite offset; must keep seed_for in uint32 range.
                step_index = 5_000 + display - 1
                payload = self._run_refinement_step(
                    step_display=display,
                    steps_total=total_steps,
                    step_index=step_index,
                    codegen_step=codegen_step,
                    phase="menu_sweep",
                    train_configs=[market],
                    q_suite_configs=list(q_suite_configs),
                    block_idx=None,
                    sweep_market_index=market_i,
                    forced_pair=pair,
                )
                summary.append(payload)
        print(
            f"[reflection] menu sweep complete "
            f"({len(self._sweep_markets)} markets, {display} steps); "
            f"starting block/refinement phase for {self.steps} steps",
            flush=True,
        )
        return summary, codegen_step

    def run(self) -> Dict[str, Any]:
        summary: List[Dict[str, Any]] = []
        print(
            f"[reflection] run_dir={Path(self.output_path).parent} "
            f"steps={self.steps} batch_size={self.batch_size} "
            f"episodes_per_config={self.episodes_per_config} critic={self.critic_enabled} "
            f"critic_data={self.critic_include_data} "
            f"generator_critic_data={self.generator_include_critic_data} "
            f"generator_sibling={self.generator_include_sibling} "
            f"q_eval_mode={self.q_eval_mode} q_update_mode={self.q_update_mode} "
            f"q_reward_mode={self.q_reward_mode} critic_mode={self.critic_mode} "
            f"parent_select={self._resolved_parent_selection_mode()} "
            f"plateau_escape={self.plateau_escape_enabled} "
            f"plateau_escape_mode={self.plateau_escape_mode} "
            f"elite_bias={self.elite_parent_bias_prob:.2f} "
            f"hypothesis_jump={self.hypothesis_jump_prob:.2f}"
            + (f" recipe={self.recipe_id}" if self.recipe_id else "")
            + (
                f" menu_preset={self.principle_menu_preset}"
                if self.principle_menu_preset
                else ""
            )
            + f" menu_explore={self.menu_exploration_enabled}"
            + f" codegen_recovery={self.codegen_recovery_mode}"
            + f" lob_input_mode={self.lob_input_mode}"
            + f" market_block_size={self.market_block_size}"
            + f" q_suite_size={self.q_suite_size}"
            + f" menu_sweep_markets={self.menu_sweep_markets}"
            + f" home_set_size={self.home_set_size}"
            + f" resample_every={self.resample_every}"
            + f" model={self.model_name}",
            flush=True,
        )

        if self.menu_sweep_markets > 0 and self._config_space is not None:
            seed_cfg = self._config_space.sample(self._rng, batch_index=-1)
            seed_train = [seed_cfg]
            seed_q = seed_train
            seed_batch = seed_train
        elif self.q_eval_mode == "full_refresh":
            seed_train, seed_q = self._train_and_q_configs(step_index=0)
            seed_batch = seed_train
        else:
            seed_train, _seed_q = self._train_and_q_configs(step_index=0)
            seed_batch = seed_train
            seed_q = seed_train

        if self.q_eval_mode == "full_refresh":
            print(
                f"[reflection] seed-eval batch: {[cfg.label for cfg in seed_batch]}"
                + (
                    f" q_suite={[cfg.label for cfg in seed_q]}"
                    if self.market_block_size > 0
                    or self.menu_sweep_markets > 0
                    or self.home_set_size > 0
                    or self.resample_every > 0
                    else ""
                ),
                flush=True,
            )
            seed_suite = self._build_step_eval_suite(seed_q, step_index=-1)
            print(
                f"[reflection] seed eval suite {seed_suite.suite_id} "
                f"(shared seeds for all seed nodes)",
                flush=True,
            )
            seed_evaluations, seed_refresh_entry = self._evaluate_seed_nodes_full_refresh(
                seed_suite
            )
            self._q_refresh_history.append(seed_refresh_entry)
            seed_eval_suite = seed_suite.to_dict()
        else:
            print(
                f"[reflection] seed-eval batch: {[cfg.label for cfg in seed_batch]}",
                flush=True,
            )
            seed_evaluations = self._evaluate_seed_nodes(seed_batch)
            seed_eval_suite = None

        sweep_summary, codegen_step = self._run_menu_sweep(codegen_step_start=0)
        summary.extend(sweep_summary)

        for step_index in range(self.steps):
            train_configs, q_suite_configs = self._train_and_q_configs(step_index)
            self._maybe_realign_q_for_resample(
                step_index=step_index,
                q_suite_configs=q_suite_configs,
            )
            block_idx = (
                self._block_index_for_step(step_index)
                if self.market_block_size > 0
                else None
            )
            if self.resample_every > 0:
                phase = "resample"
            elif self.market_block_size > 0:
                phase = "block"
            else:
                phase = "step"
            codegen_step += 1
            step_payload = self._run_refinement_step(
                step_display=step_index + 1,
                steps_total=self.steps,
                step_index=step_index,
                codegen_step=codegen_step,
                phase=phase,
                train_configs=train_configs,
                q_suite_configs=q_suite_configs,
                block_idx=block_idx,
            )
            summary.append(step_payload)

        payload = {
            "params": {
                "batch_size": self.batch_size,
                "steps": self.steps,
                "episodes_per_config": self.episodes_per_config,
                "exploration_constant": self.exploration_constant,
                "max_debug_retries": self.max_debug_retries,
                "seed": self.seed,
                "strategy_output_mode": self.strategy_output_mode,
                "reward_mode": self.reward_mode,
                "critic_enabled": self.critic_enabled,
                "critic_include_data": self.critic_include_data,
                "critic_per_config_metrics": self.critic_per_config_metrics,
                "generator_include_critic_data": self.generator_include_critic_data,
                "generator_include_sibling": self.generator_include_sibling,
                "q_eval_mode": self.q_eval_mode,
                "q_update_mode": self.q_update_mode,
                "q_reward_mode": self.q_reward_mode,
                "parent_selection_mode": self.parent_selection_mode,
                "resolved_parent_selection_mode": self._resolved_parent_selection_mode(),
                "recipe_id": self.recipe_id,
                "inactive_gate_enabled": self.inactive_gate_enabled,
                "inactive_gate_min_nodes": self.inactive_gate_min_nodes,
                "inactive_surplus_gap": self.inactive_surplus_gap,
                "inactive_beat_gap": self.inactive_beat_gap,
                "critic_mode": self.critic_mode,
                "zi_benchmark_shade_ranges": self.zi_benchmark_shade_ranges,
                "forbid_numpy_random": self.forbid_numpy_random,
                "parent_codegen_fail_cool_threshold": self.parent_codegen_fail_cool_threshold,
                "plateau_escape_enabled": self.plateau_escape_enabled,
                "plateau_non_improve_threshold": self.plateau_non_improve_threshold,
                "plateau_improve_surplus_eps": self.plateau_improve_surplus_eps,
                "elite_parent_bias_prob": self.elite_parent_bias_prob,
                "hypothesis_jump_prob": self.hypothesis_jump_prob,
                "plateau_escape_temperature": self.plateau_escape_temperature,
                "plateau_escape_mode": self.plateau_escape_mode,
                "principle_menu_preset": self.principle_menu_preset,
                "menu_exploration_enabled": self.menu_exploration_enabled,
                "codegen_recovery_mode": self.codegen_recovery_mode,
                "lob_input_mode": self.lob_input_mode,
                "model_name": self.model_name,
                "market_block_size": self.market_block_size,
                "q_suite_size": self.q_suite_size,
                "menu_sweep_markets": self.menu_sweep_markets,
                "home_set_size": self.home_set_size,
                "home_set": [cfg.observable_state() for cfg in self._home_set],
                "resample_every": self.resample_every,
                "resample_markets": [
                    cfg.observable_state() for cfg in self._resample_markets
                ],
                "sweep_markets": [
                    cfg.observable_state() for cfg in self._sweep_markets
                ],
                "block_markets": [
                    cfg.observable_state() for cfg in self._block_markets
                ],
                "full_refresh_parallel_workers": FULL_REFRESH_PARALLEL_WORKERS,
                "config_space": (
                    self._config_space.to_dict()
                    if self._config_space is not None
                    else None
                ),
                "fixed_config_pool": (
                    [cfg.observable_state() for cfg in self._config_pool]
                    if self._config_pool is not None
                    else None
                ),
            },
            "run_dir": str(Path(self.output_path).parent),
            "seed_eval_batch": [cfg.observable_state() for cfg in seed_batch],
            "seed_evaluations": seed_evaluations,
            "seed_eval_suite": seed_eval_suite,
            "q_refresh_history": self._q_refresh_history,
            "summary": summary,
            "tree": self._tree.to_dict(),
        }
        menu_coverage = self._menu_coverage_summary()
        if menu_coverage:
            payload["menu_coverage"] = menu_coverage

        out_path = Path(self.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if menu_coverage:
            for strategy_type, stats in menu_coverage.items():
                print(
                    f"[reflection] menu coverage ({strategy_type}): "
                    f"{stats['explored_pair_count']}/{stats['menu_size']} pairs "
                    f"({100.0 * stats['coverage_fraction']:.1f}%), "
                    f"principle_entropy_norm={stats['principle_entropy_normalized']:.3f}, "
                    f"complete={stats['exploration_complete']}",
                    flush=True,
                )
        print(f"\n[reflection] done — summary written to {out_path}", flush=True)
        return payload


def run_reflection_pipeline(**kwargs: Any) -> Dict[str, Any]:
    pipeline = ReflectionPipeline(**kwargs)
    return pipeline.run()
