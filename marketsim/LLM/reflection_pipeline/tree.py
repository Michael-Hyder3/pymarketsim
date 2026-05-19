from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math


@dataclass
class TreeNode:
    node_id: int
    code: str
    reflection: str
    q_value: float
    visits: int
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)


class RefinementTree:
    def __init__(self, *, root_code: str, root_reflection: str = "") -> None:
        self._nodes: Dict[int, TreeNode] = {}
        root = TreeNode(
            node_id=0,
            code=root_code,
            reflection=root_reflection,
            q_value=0.0,
            visits=1,
            parent_id=None,
        )
        self._nodes[root.node_id] = root
        self._next_id = 1

    def nodes(self) -> List[TreeNode]:
        return list(self._nodes.values())

    def get(self, node_id: int) -> TreeNode:
        return self._nodes[node_id]

    def add_node(self, *, parent_id: int, code: str, reflection: str, reward: float) -> TreeNode:
        node = TreeNode(
            node_id=self._next_id,
            code=code,
            reflection=reflection,
            q_value=float(reward),
            visits=1,
            parent_id=parent_id,
        )
        self._nodes[node.node_id] = node
        self._nodes[parent_id].children.append(node.node_id)
        self._next_id += 1
        return node

    def select_ucb(self, *, exploration: float, total_visits: int) -> TreeNode:
        if total_visits <= 1:
            return self._nodes[0]
        best_node = None
        best_score = -float("inf")
        for node in self._nodes.values():
            if node.visits <= 0:
                continue
            exploit = node.q_value
            explore = exploration * math.sqrt(math.log(total_visits) / node.visits)
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_node = node
        if best_node is None:
            return self._nodes[0]
        return best_node

    def update_ancestors(self, *, start_node_id: int, reward: float) -> None:
        node_id = start_node_id
        while node_id is not None:
            node = self._nodes[node_id]
            node.visits += 1
            node.q_value = node.q_value + (reward - node.q_value) / node.visits
            node_id = node.parent_id

    def to_dict(self) -> Dict:
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "children": node.children,
                    "visits": node.visits,
                    "q_value": node.q_value,
                    "reflection": node.reflection,
                }
                for node in self._nodes.values()
            ]
        }
