from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math

from marketsim.LLM.lob_features import (
    DEFAULT_LOB_INPUT_MODE,
    is_lob_features_mode,
    normalize_lob_input_mode,
)
from marketsim.LLM.reflection_pipeline.learning_focused import (
    DEFAULT_LOB_SEED_SCALE,
    LOB_SEED_SCALE_HBL,
    LOB_SEED_SCALE_HBL_SCAFFOLD,
    lob_order_memory_seed_code,
    lob_order_memory_seed_description,
    normalize_lob_seed_scale,
)


@dataclass
class TreeNode:
    node_id: int
    code: str
    reflection: str
    q_value: float
    visits: int
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


def _pv_seed_strategy() -> Dict[str, str]:
    code = "\n".join(
        [
            "def agent_strategy(private_values, fundamental_estimate, current_pv):",
            "    # Schema foothold only: bind o_PV fields, fixed light shade.",
            "    private_values = list(private_values or [])",
            "    fundamental_estimate = float(fundamental_estimate or 0.0)",
            "    current_pv = float(current_pv or 0.0)",
            "    _ = (private_values, fundamental_estimate, current_pv)",
            "    return 0.01",
        ]
    )
    return {
        "code": code,
        "type": "pv",
        "desc": "Schema foothold: bind schedule Θ, F̃, and current_pv; fixed shade 0.01",
    }


def _spread_seed_strategy(
    *,
    omit_fundamental: bool = False,
    spread_input_mode: str = "classic",
) -> Dict[str, str]:
    from marketsim.LLM.spread_input import is_spread_book_mode

    if is_spread_book_mode(spread_input_mode):
        bind_lines = [
            "def agent_strategy(market_state):",
            "    # Sp book foothold: bind o_Sp fields; fixed light shade.",
            "    best_bid = market_state.get('best_bid')",
            "    best_ask = market_state.get('best_ask')",
            "    bids = list(market_state.get('bids') or [])",
            "    asks = list(market_state.get('asks') or [])",
            "    units_available = float(market_state.get('units_available') or 0.0)",
            "    q_max = float(market_state.get('q_max') or 0.0)",
            "    current_pv = float(market_state.get('current_pv') or 0.0)",
            "    private_values = list(market_state.get('private_values') or [])",
            "    prev_event = market_state.get('prev_event')",
        ]
        unused_names = [
            "best_bid",
            "best_ask",
            "bids",
            "asks",
            "units_available",
            "q_max",
            "current_pv",
            "private_values",
        ]
        if not omit_fundamental:
            bind_lines += [
                "    fundamental_estimate = float(",
                "        market_state.get('fundamental_estimate') or 0.0",
                "    )",
            ]
            unused_names.append("fundamental_estimate")
        bind_lines += [
            "    if isinstance(prev_event, dict):",
            "        price_abs = float(prev_event.get('price_abs') or 0.0)",
            "        price_rel = float(prev_event.get('price_rel') or 0.0)",
            "        side = float(prev_event.get('side') or 0.0)",
            "        accepted = 1.0 if prev_event.get('accepted') else 0.0",
            "    else:",
            "        price_abs = 0.0",
            "        price_rel = 0.0",
            "        side = 0.0",
            "        accepted = 0.0",
            "    _ = (",
            *[f"        {name}," for name in unused_names],
            "        price_abs,",
            "        price_rel,",
            "        side,",
            "        accepted,",
            "    )",
            "    return 0.01",
        ]
        desc = (
            "Sp book foothold: bind quotes/depth/inventory/prev_event/PV; "
            "fixed shade 0.01"
            if omit_fundamental
            else "Sp book foothold: bind quotes/depth/inventory/prev_event/PV/F̃; "
            "fixed shade 0.01"
        )
        return {
            "code": "\n".join(bind_lines),
            "type": "spread",
            "desc": desc,
        }

    bind_lines = [
        "def agent_strategy(market_state):",
        "    # Schema foothold only: bind o_Sp fields, fixed light shade.",
        "    best_bid = market_state.get('best_bid')",
        "    best_ask = market_state.get('best_ask')",
        "    current_pv = float(market_state.get('current_pv') or 0.0)",
        "    private_values = list(market_state.get('private_values') or [])",
        "    units_available = float(market_state.get('units_available') or 0.0)",
        "    q_max = float(market_state.get('q_max') or 0.0)",
    ]
    unused_names = [
        "current_pv",
        "private_values",
        "units_available",
        "q_max",
    ]
    if not omit_fundamental:
        bind_lines += [
            "    fundamental_estimate = float(",
            "        market_state.get('fundamental_estimate') or 0.0",
            "    )",
        ]
        unused_names.append("fundamental_estimate")
    bind_lines += [
        "    shouts = list(market_state.get('shouts') or [])",
        "    _ = (",
        *[f"        {name}," for name in unused_names],
        "        shouts,",
        "        best_bid,",
        "        best_ask,",
        "    )",
        "    return 0.01",
    ]
    desc = (
        "Schema foothold: bind quotes, inventory, Θ, shouts; fixed shade 0.01"
        if omit_fundamental
        else "Schema foothold: bind quotes, inventory, Θ, F̃, shouts; fixed shade 0.01"
    )
    return {
        "code": "\n".join(bind_lines),
        "type": "spread",
        "desc": desc,
    }


class RefinementTree:
    def __init__(
        self,
        *,
        initial_types: Optional[List[str]] = None,
        lob_input_mode: str = DEFAULT_LOB_INPUT_MODE,
        lob_seed_scale: str = DEFAULT_LOB_SEED_SCALE,
        spread_input_mode: str = "classic",
        spread_omit_fundamental: bool = False,
    ) -> None:
        self._nodes: Dict[int, TreeNode] = {}
        resolved_lob_mode = normalize_lob_input_mode(lob_input_mode)
        resolved_seed_scale = normalize_lob_seed_scale(lob_seed_scale)
        omit_f = bool(spread_omit_fundamental)
        resolved_spread_mode = str(spread_input_mode or "classic")
        lob_raw_code = lob_order_memory_seed_code(
            lob_seed_scale=resolved_seed_scale,
            lob_input_mode=resolved_lob_mode,
        )
        lob_code = (
            lob_features_seed_strategy_code(resolved_lob_mode)
            if is_lob_features_mode(resolved_lob_mode)
            else lob_raw_code
        )
        lob_desc = (
            f"Limit-order-book scalar-feature heuristic ({resolved_lob_mode})"
            if is_lob_features_mode(resolved_lob_mode)
            else lob_order_memory_seed_description(
                lob_seed_scale=resolved_seed_scale,
                lob_input_mode=resolved_lob_mode,
            )
        )

        strategy_specs = {
            "pv": _pv_seed_strategy(),
            "spread": _spread_seed_strategy(
                omit_fundamental=omit_f,
                spread_input_mode=resolved_spread_mode,
            ),
            "lob": {
                "code": lob_code,
                "type": "lob",
                "desc": lob_desc,
            },
        }

        root = TreeNode(
            node_id=0,
            code="VIRTUAL_ROOT",
            reflection="Hub for initial strategy branches",
            q_value=0.0,
            visits=1,
            parent_id=None,
        )
        self._nodes[0] = root
        self._next_id = 1

        requested_types = initial_types or ["pv", "spread", "lob"]
        for strategy_type in requested_types:
            spec = strategy_specs.get(strategy_type)
            if spec is None:
                continue
            seed_meta = {
                "type": spec.get("type"),
                "desc": spec.get("desc"),
            }
            if (
                strategy_type == "lob"
                and resolved_seed_scale
                in (LOB_SEED_SCALE_HBL, LOB_SEED_SCALE_HBL_SCAFFOLD)
                and not is_lob_features_mode(resolved_lob_mode)
            ):
                seed_meta["seed_kind"] = (
                    "hbl_distilled"
                    if resolved_seed_scale == LOB_SEED_SCALE_HBL
                    else "hbl_scaffold"
                )
            self.add_node(
                parent_id=0,
                code=spec["code"],
                reflection=(
                    f"Strategy Type: {spec.get('type')}. "
                    f"Description: {spec.get('desc')}"
                ),
                reward=0.0,
                metadata=seed_meta,
            )
            seed_node = self._nodes[self._next_id - 1]
            seed_node.metadata["q_eval_count"] = 0

    def nodes(self) -> List[TreeNode]:
        return list(self._nodes.values())

    def get(self, node_id: int) -> TreeNode:
        return self._nodes[node_id]

    def seed_node_ids(self) -> List[int]:
        return list(self._nodes[0].children)

    def evaluable_node_ids(self) -> List[int]:
        return [node.node_id for node in self._nodes.values() if node.node_id != 0]

    def active_node_ids(self) -> List[int]:
        return [
            node_id
            for node_id in self.evaluable_node_ids()
            if not bool(self._nodes[node_id].metadata.get("inactive", False))
        ]

    def mark_inactive(self, node_id: int, *, reason: str) -> None:
        node = self._nodes[node_id]
        node.metadata["inactive"] = True
        node.metadata["inactive_reason"] = str(reason)

    def next_node_id(self) -> int:
        return self._next_id

    def apply_q_values(
        self,
        rewards: Dict[int, float],
        *,
        mode: str = "replace",
    ) -> None:
        for node_id, q_value in rewards.items():
            self.update_q_value(node_id, q_value, mode=mode)

    def update_q_value(self, node_id: int, q_value: float, *, mode: str = "replace") -> None:
        node = self._nodes[node_id]
        sample = float(q_value)
        if mode == "average":
            count = int(node.metadata.get("q_eval_count", 0))
            if count <= 0:
                node.q_value = sample
            else:
                node.q_value = (node.q_value * count + sample) / (count + 1)
            node.metadata["q_eval_count"] = count + 1
            return
        node.q_value = sample
        node.metadata["q_eval_count"] = int(node.metadata.get("q_eval_count", 0)) + 1




    def set_reflection(self, node_id: int, reflection: str) -> None:
        self._nodes[node_id].reflection = reflection

    def add_node(
        self,
        *,
        parent_id: int,
        code: str,
        reflection: str,
        reward: float,
        metadata: Dict = None,
        q_update_mode: str = "replace",
    ) -> TreeNode:
        node = TreeNode(
            node_id=self._next_id,
            code=code,
            reflection=reflection,
            q_value=0.0,
            visits=1,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._nodes[node.node_id] = node
        self._nodes[parent_id].children.append(node.node_id)
        if parent_id != 0:
            self._nodes[parent_id].visits += 1
        self._next_id += 1
        self.update_q_value(node.node_id, reward, mode=q_update_mode)
        return node

    def select_ucb(
        self,
        *,
        exploration: float,
        total_visits: int,
        max_children: int = 0,
    ) -> TreeNode:
        best_node = None
        best_score = -float("inf")
        child_cap = max(0, int(max_children))

        def _consider(respect_cap: bool) -> Optional[TreeNode]:
            nonlocal best_node, best_score
            best_node = None
            best_score = -float("inf")
            for node in self._nodes.values():
                if node.node_id == 0 or node.visits <= 0:
                    continue
                if bool(node.metadata.get("inactive", False)):
                    continue
                if respect_cap and child_cap > 0 and len(node.children) >= child_cap:
                    continue
                exploit = float(node.q_value)
                explore = exploration * math.sqrt(math.log(total_visits) / node.visits)
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_node = node
            return best_node

        chosen = _consider(respect_cap=True)
        if chosen is None and child_cap > 0:
            chosen = _consider(respect_cap=False)
        if chosen is not None:
            return chosen
        active = self.active_node_ids()
        if active:
            return self._nodes[active[0]]
        return self._nodes[1]



    def to_dict(self) -> Dict:
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "children": node.children,
                    "visits": node.visits,
                    "q_value": node.q_value,
                    "q_eval_count": node.metadata.get("q_eval_count", 0),
                    "ts_alpha": node.metadata.get("ts_alpha"),
                    "ts_beta": node.metadata.get("ts_beta"),
                    "type": node.metadata.get("type"),
                    "inactive": bool(node.metadata.get("inactive", False)),
                    "inactive_reason": node.metadata.get("inactive_reason"),
                }
                for node in self._nodes.values()
            ]
        }
