from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from marketsim.LLM.reflection_pipeline.tree import RefinementTree

CompareMode = Literal["exact", "normalized", "code", "ast"]

# "code" compares parsed AST (ignores comments and formatting differences).
DEFAULT_COMPARE_MODE: CompareMode = "code"


@dataclass
class NodeStrategy:
    node_id: int
    code: str
    strategy_type: Optional[str]
    parent_id: Optional[int]
    q_value: Optional[float]
    visits: Optional[int]
    code_source: str
    return_signature: str = ""
    iteration_label: str = ""


@dataclass
class DuplicateGroup:
    key: str
    node_ids: List[int] = field(default_factory=list)
    nodes: List[NodeStrategy] = field(default_factory=list)


def _iteration_label(code_source: str) -> str:
    match = re.search(r"iteration_(\d+)", code_source)
    if match:
        # Successful saves use iteration_{node_id} (matches tree node id).
        return f"node{match.group(1)}"
    if code_source.startswith("builtin_seed:"):
        return code_source
    return Path(code_source).name


def _find_latest_summary() -> Optional[Path]:
    base_dir = Path(__file__).resolve().parent / "runs"
    runs = list(base_dir.glob("*/reflection_pipeline_summary.json"))
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def _builtin_seed_code_by_type(strategy_type: str) -> str:
    tree = RefinementTree(initial_types=[strategy_type])
    for node in tree.nodes():
        if node.node_id != 0 and node.parent_id == 0:
            return node.code
    raise KeyError(f"Unknown seed strategy type: {strategy_type}")


def _resolve_code_path(run_dir: Path, code_file: str) -> Path:
    path = Path(code_file)
    if path.is_file():
        return path
    candidate = run_dir / code_file
    if candidate.is_file():
        return candidate
    candidate = run_dir.parent.parent.parent / code_file
    if candidate.is_file():
        return candidate
    return path


def _tree_node_index(tree_snapshot: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(node["node_id"]): node for node in tree_snapshot.get("nodes", [])}


def _node_id_from_code_file(code_file: str) -> Optional[int]:
    match = re.search(r"iteration_(\d+)", code_file)
    if match:
        return int(match.group(1))
    return None


def _infer_new_node_id(
    *,
    step_data: Dict[str, Any],
    seed_ids: List[int],
) -> Optional[int]:
    explicit = step_data.get("node_id")
    if explicit is not None:
        return int(explicit)

    q_refresh = step_data.get("q_refresh") or {}
    new_node_id = q_refresh.get("new_node_id")
    if new_node_id is not None:
        return int(new_node_id)

    code_file = step_data.get("code_file") or ""
    from_iteration = _node_id_from_code_file(code_file)
    if from_iteration is not None:
        return from_iteration

    step = step_data.get("step")
    if step is None or not seed_ids:
        return None
    return max(seed_ids) + int(step)


def collect_node_strategies(
    *,
    summary_path: Path,
    data: Optional[Dict[str, Any]] = None,
) -> Tuple[List[NodeStrategy], List[str]]:
    payload = data if data is not None else json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = Path(payload.get("run_dir", summary_path.parent))
    if not run_dir.is_absolute():
        run_dir = (summary_path.parent / run_dir).resolve()
    if not run_dir.is_dir():
        run_dir = summary_path.parent.resolve()

    tree_snapshot = payload.get("tree") or {}
    node_index = _tree_node_index(tree_snapshot)
    seed_ids = sorted(
        node_id
        for node_id, node in node_index.items()
        if node_id != 0 and node.get("parent_id") == 0
    )

    strategies: List[NodeStrategy] = []
    warnings: List[str] = []

    for node_id in seed_ids:
        node = node_index[node_id]
        strategy_type = node.get("type") or "pv"
        try:
            code = _builtin_seed_code_by_type(str(strategy_type))
            source = f"builtin_seed:{strategy_type}"
        except KeyError:
            warnings.append(f"node {node_id}: missing builtin seed template for type={strategy_type}")
            continue
        strategies.append(
            NodeStrategy(
                node_id=node_id,
                code=code,
                strategy_type=str(strategy_type),
                parent_id=0,
                q_value=node.get("q_value"),
                visits=node.get("visits"),
                code_source=source,
                return_signature=extract_return_signature(code),
                iteration_label=_iteration_label(source),
            )
        )

    assigned_ids: set[int] = set(seed_ids)
    for step_data in payload.get("summary", []):
        if not step_data.get("success"):
            continue
        code_file = step_data.get("code_file")
        if not code_file:
            warnings.append(f"step {step_data.get('step')}: missing code_file")
            continue

        node_id = _infer_new_node_id(step_data=step_data, seed_ids=seed_ids)
        if node_id is None:
            warnings.append(f"step {step_data.get('step')}: could not infer new node id")
            continue
        if node_id in assigned_ids:
            warnings.append(f"step {step_data.get('step')}: duplicate assignment to node {node_id}")
        assigned_ids.add(node_id)

        code_path = _resolve_code_path(run_dir, str(code_file))
        if not code_path.is_file():
            warnings.append(f"node {node_id}: code file not found at {code_path}")
            continue

        node = node_index.get(node_id, {})
        code = code_path.read_text(encoding="utf-8")
        code_source = str(code_path)
        strategies.append(
            NodeStrategy(
                node_id=node_id,
                code=code,
                strategy_type=node.get("type"),
                parent_id=node.get("parent_id"),
                q_value=node.get("q_value"),
                visits=node.get("visits"),
                code_source=code_source,
                return_signature=extract_return_signature(code),
                iteration_label=_iteration_label(code_source),
            )
        )

    evaluable_ids = {node_id for node_id, node in node_index.items() if node_id != 0}
    loaded_ids = {item.node_id for item in strategies}
    missing = sorted(evaluable_ids - loaded_ids)
    for node_id in missing:
        warnings.append(f"node {node_id}: no strategy code loaded")

    strategies.sort(key=lambda item: item.node_id)
    return strategies, warnings


def _ast_canonical(code: str) -> str:
    """Unparsed source without comments; preserves literals and control flow."""
    tree = ast.parse(code)
    source = ast.unparse(tree).strip()
    lines = [line.rstrip() for line in source.splitlines()]
    return "\n".join(lines)


def _is_pv_mean_nonneg_test(test: ast.AST) -> bool:
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], (ast.GtE, ast.Gt)):
        return False
    left = test.left
    if isinstance(left, ast.Name) and left.id == "pv_mean":
        return True
    return False


def _first_return_expr(stmts: List[ast.stmt]) -> Optional[str]:
    for stmt in stmts:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            return ast.unparse(stmt.value).strip()
        if isinstance(stmt, ast.If):
            inner = _first_return_expr(stmt.body)
            if inner is not None:
                return inner
            inner = _first_return_expr(stmt.orelse)
            if inner is not None:
                return inner
    return None


def extract_return_signature(code: str) -> str:
    """Short summary of positive vs negative/fallback return expressions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "syntax_error"

    fn: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "agent_strategy":
            fn = node
            break
    if fn is None:
        return "no_agent_strategy"

    for index, stmt in enumerate(fn.body):
        if isinstance(stmt, ast.If) and _is_pv_mean_nonneg_test(stmt.test):
            positive = _first_return_expr(stmt.body) or "?"
            if stmt.orelse:
                negative = _first_return_expr(stmt.orelse) or "?"
                return f"pv>=0 -> {positive} | else -> {negative}"
            for later in fn.body[index + 1 :]:
                if isinstance(later, ast.Return) and later.value is not None:
                    fallback = ast.unparse(later.value).strip()
                    return f"pv>=0 -> {positive} | fallback -> {fallback}"
            return f"pv>=0 -> {positive} | fallback -> none"

    returns = [
        ast.unparse(stmt.value).strip()
        for stmt in fn.body
        if isinstance(stmt, ast.Return) and stmt.value is not None
    ]
    if returns:
        return "returns: " + "; ".join(returns)
    return "custom"


def canonicalize_code(code: str, mode: CompareMode) -> str:
    if mode == "exact":
        return code

    if mode in {"code", "ast"}:
        return _ast_canonical(code)

    stripped = textwrap.dedent(code).strip()
    lines = [line.rstrip() for line in stripped.splitlines()]
    normalized_lines: List[str] = []
    previous_blank = False
    for line in lines:
        if not line.strip():
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
            continue
        normalized_lines.append(line)
        previous_blank = False
    return "\n".join(normalized_lines).strip()


def strategy_key(code: str, mode: CompareMode) -> str:
    canonical = canonicalize_code(code, mode)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def group_duplicates(
    strategies: List[NodeStrategy],
    *,
    mode: CompareMode,
) -> Tuple[List[DuplicateGroup], Dict[int, str]]:
    groups: Dict[str, DuplicateGroup] = {}
    node_keys: Dict[int, str] = {}

    for item in strategies:
        key = strategy_key(item.code, mode)
        node_keys[item.node_id] = key
        if key not in groups:
            groups[key] = DuplicateGroup(key=key)
        groups[key].node_ids.append(item.node_id)
        groups[key].nodes.append(item)

    duplicate_groups = sorted(
        [group for group in groups.values() if len(group.node_ids) > 1],
        key=lambda group: (-len(group.node_ids), group.node_ids),
    )
    for group in duplicate_groups:
        group.node_ids.sort()
        group.nodes.sort(key=lambda item: item.node_id)
    return duplicate_groups, node_keys


def _code_preview(code: str, *, max_lines: int = 6, mode: CompareMode = "code") -> str:
    if mode in {"code", "ast"}:
        try:
            source = ast.unparse(ast.parse(code))
        except SyntaxError:
            source = code
        lines = source.strip().splitlines()
    else:
        lines = canonicalize_code(code, "normalized").splitlines()
    preview = lines[:max_lines]
    if len(lines) > max_lines:
        preview.append("...")
    return "\n".join(f"  {line}" for line in preview)


def build_report(
    *,
    summary_path: Path,
    strategies: List[NodeStrategy],
    duplicate_groups: List[DuplicateGroup],
    node_keys: Dict[int, str],
    warnings: List[str],
    mode: CompareMode,
) -> Dict[str, Any]:
    parent_child_duplicates: List[Dict[str, Any]] = []
    code_by_id = {item.node_id: item for item in strategies}
    for item in strategies:
        parent_id = item.parent_id
        if parent_id is None or parent_id == 0:
            continue
        parent = code_by_id.get(parent_id)
        if parent is None:
            continue
        if strategy_key(item.code, mode) == strategy_key(parent.code, mode):
            parent_child_duplicates.append(
                {
                    "child_id": item.node_id,
                    "parent_id": parent_id,
                    "child_q": item.q_value,
                    "parent_q": parent.q_value,
                }
            )

    unique_count = len(set(node_keys.values()))
    duplicate_node_count = sum(len(group.node_ids) for group in duplicate_groups)

    return {
        "summary_path": str(summary_path),
        "compare_mode": mode,
        "total_nodes": len(strategies),
        "unique_strategies": unique_count,
        "duplicate_groups_count": len(duplicate_groups),
        "duplicate_nodes_count": duplicate_node_count,
        "largest_group_size": max((len(group.node_ids) for group in duplicate_groups), default=0),
        "parent_child_duplicates": parent_child_duplicates,
        "duplicate_groups": [
            {
                "key": group.key,
                "node_ids": group.node_ids,
                "q_values": {
                    str(item.node_id): item.q_value for item in group.nodes
                },
                "types": {
                    str(item.node_id): item.strategy_type for item in group.nodes
                },
                "parent_ids": {
                    str(item.node_id): item.parent_id for item in group.nodes
                },
                "return_signatures": {
                    str(item.node_id): item.return_signature for item in group.nodes
                },
                "iteration_labels": {
                    str(item.node_id): item.iteration_label for item in group.nodes
                },
                "code_preview": _code_preview(group.nodes[0].code, mode=mode),
            }
            for group in duplicate_groups
        ],
        "warnings": warnings,
    }


def print_report(report: Dict[str, Any]) -> None:
    print(f"\nRun: {report['summary_path']}")
    mode = report["compare_mode"]
    mode_note = " (ignores comments; compares unparsed code)" if mode in {"code", "ast"} else ""
    print(f"Compare mode: {mode}{mode_note}")
    print(
        f"Nodes: {report['total_nodes']} | "
        f"Unique strategies: {report['unique_strategies']} | "
        f"Duplicate groups: {report['duplicate_groups_count']} | "
        f"Nodes in duplicate groups: {report['duplicate_nodes_count']}"
    )

    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    parent_child = report.get("parent_child_duplicates") or []
    if parent_child:
        print("\nParent-child duplicates (child code matches parent):")
        for item in parent_child:
            print(
                f"  child {item['child_id']} (q={item.get('child_q')}) "
                f"<- parent {item['parent_id']} (q={item.get('parent_q')})"
            )
    else:
        print("\nParent-child duplicates: none")

    groups = report.get("duplicate_groups") or []
    if not groups:
        print("\nNo duplicate strategy groups found.")
        return

    print("\nDuplicate strategy groups:")
    for index, group in enumerate(groups, start=1):
        node_ids = group["node_ids"]
        signatures = group.get("return_signatures") or {}
        iterations = group.get("iteration_labels") or {}
        sample_sig = signatures.get(str(node_ids[0]), "")
        print(
            f"\nGroup {index} ({len(node_ids)} nodes, key={group['key']}): {node_ids}\n"
            f"  Return pattern: {sample_sig}"
        )
        print("  Per node:")
        q_values = group.get("q_values") or {}
        parent_ids = group.get("parent_ids") or {}
        for node_id in node_ids:
            node_key = str(node_id)
            print(
                f"    N{node_id} ({iterations.get(node_key, '?')}) "
                f"q={q_values.get(node_key):.4f} parent={parent_ids.get(node_key)} "
                f"| {signatures.get(node_key, '?')}"
            )
        print("  Code preview:")
        print(group.get("code_preview", ""))


def analyze_duplicates(
    summary_path: Path,
    *,
    mode: CompareMode = DEFAULT_COMPARE_MODE,
) -> Dict[str, Any]:
    strategies, warnings = collect_node_strategies(summary_path=summary_path)
    duplicate_groups, node_keys = group_duplicates(strategies, mode=mode)
    return build_report(
        summary_path=summary_path,
        strategies=strategies,
        duplicate_groups=duplicate_groups,
        node_keys=node_keys,
        warnings=warnings,
        mode=mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate strategies across tree nodes")
    parser.add_argument(
        "summary_path",
        nargs="?",
        default=None,
        help="Path to reflection_pipeline_summary.json (default: latest run)",
    )
    parser.add_argument(
        "--mode",
        choices=["exact", "normalized", "code", "ast"],
        default=DEFAULT_COMPARE_MODE,
        help="exact | normalized (whitespace) | code/ast (ignore comments)",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write duplicate_strategies_report.json next to the summary file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_path:
        summary_path = Path(args.summary_path)
    else:
        latest = _find_latest_summary()
        if latest is None:
            print("Error: No summary JSON found under runs/. Pass a path explicitly.")
            sys.exit(1)
        summary_path = latest
        print(f"Using latest run: {summary_path}")

    if not summary_path.is_file():
        print(f"Error: Could not find file at {summary_path}")
        sys.exit(1)

    report = analyze_duplicates(summary_path, mode=args.mode)
    print_report(report)

    if args.write_json:
        out_path = summary_path.parent / "duplicate_strategies_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
