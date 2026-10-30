from __future__ import annotations

import ast
import re
from typing import FrozenSet, List, Optional, Set, Tuple

COMPLEX_PARENT_MIN_FUNCS = 2
COMPLEX_PARENT_MIN_LINES = 160

_DIFF_FENCE_RE = re.compile(r"```diff\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_PYTHON_FENCE_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\s*(.*?)\s*```", re.DOTALL)
_HUNK_HEADER_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")


def top_level_function_names(code: str) -> Set[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names




def loaded_name_ids(code: str) -> Set[str]:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
    return names


def parent_is_complex(parent_code: str) -> bool:
    text = parent_code or ""
    funcs = top_level_function_names(text)
    helpers = {n for n in funcs if n != "agent_strategy"}
    if len(helpers) >= 1:
        return True
    n_lines = len([ln for ln in text.splitlines() if ln.strip()])
    return n_lines >= COMPLEX_PARENT_MIN_LINES


def parent_helper_completeness_error(
    parent_code: Optional[str],
    child_code: str,
) -> Optional[str]:
    if not parent_code or not str(parent_code).strip():
        return None
    parent_funcs = top_level_function_names(parent_code)
    if not parent_funcs:
        return None
    child_funcs = top_level_function_names(child_code)
    parent_helpers = {n for n in parent_funcs if n != "agent_strategy"}
    if not parent_helpers:
        return None
    used = loaded_name_ids(child_code)
    missing = sorted(
        name
        for name in parent_helpers
        if name in used and name not in child_funcs
    )
    if not missing:
        return None
    return (
        "Parent-helper completeness failed: this strategy calls "
        f"{', '.join(missing)} (defined on the parent) but does not define "
        "them. Either (1) return a ```diff``` unified diff against the parent, "
        "or (2) return a COMPLETE ```python``` module that includes every "
        "parent helper you still call (copy helpers unchanged if needed)."
    )


def looks_like_unified_diff(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if raw.startswith(("---", "+++", "@@")):
        return True
    return bool(re.search(r"(?m)^@@\s+-\d+", raw))


def extract_code_or_diff_from_response(response_text: str) -> str:
    text = response_text or ""
    diff_matches = _DIFF_FENCE_RE.findall(text)
    if diff_matches:
        return diff_matches[0].strip()
    py_matches = _PYTHON_FENCE_RE.findall(text)
    if py_matches:
        return py_matches[0].strip()
    any_matches = _ANY_FENCE_RE.findall(text)
    for match in any_matches:
        stripped = match.strip()
        if looks_like_unified_diff(stripped):
            return stripped
        if stripped.startswith(("def ", "import ", "from ", "class ")):
            return stripped
    if any_matches:
        return any_matches[0].strip()
    stripped = text.strip()
    if stripped.startswith(("def ", "import ", "from ", "class ")) or looks_like_unified_diff(
        stripped
    ):
        return stripped
    raise ValueError(
        "No ```python``` or ```diff``` code block found in LLM response."
    )


def apply_unified_diff(parent_code: str, diff_text: str) -> str:
    parent_lines = (parent_code or "").splitlines(keepends=True)
    if parent_lines and not parent_lines[-1].endswith("\n"):
        parent_lines[-1] = parent_lines[-1] + "\n"
    hunks = _parse_unified_hunks(diff_text)
    if not hunks:
        raise ValueError("Unified diff contained no @@ hunks")

    result = list(parent_lines)
    for old_start, old_count, new_lines, old_block in reversed(hunks):
        start_idx = max(0, int(old_start) - 1)
        end_idx = start_idx + int(old_count)
        expected = result[start_idx:end_idx]
        if not _lines_match(expected, old_block):
            raise ValueError(
                f"Unified diff hunk at line {old_start} did not match parent "
                f"(expected {len(old_block)} line(s) starting at {old_start})"
            )
        result[start_idx:end_idx] = new_lines

    out = "".join(result)
    if out and not out.endswith("\n"):
        out += "\n"
    return out


def finalize_strategy_code(extracted: str, parent_code: str) -> str:
    body = (extracted or "").strip()
    if not body:
        raise ValueError("Empty strategy code")
    if looks_like_unified_diff(body):
        return apply_unified_diff(parent_code or "", body)
    return body if body.endswith("\n") else body + "\n"


def complex_parent_edit_instructions(parent_code: str) -> List[str]:
    if not parent_is_complex(parent_code):
        return []
    helpers = sorted(
        n for n in top_level_function_names(parent_code) if n != "agent_strategy"
    )
    helper_note = (
        f"Parent helpers to preserve unless intentionally removed: {', '.join(helpers)}."
        if helpers
        else "Parent is long — prefer a local edit, not a full rewrite from scratch."
    )
    return [
        "COMPLEX PARENT — prefer a surgical edit:",
        "Option A (preferred): respond with exactly one ```diff``` unified diff "
        "against the Parent Strategy (--- parent / +++ child, with @@ hunks). "
        "Only change the lines you mean to change; leave helpers intact.",
        "Option B: respond with exactly one COMPLETE ```python``` module that "
        "includes every helper function from the parent that you still call "
        "(copy helpers unchanged if you only edit agent_strategy).",
        "Do NOT return a stub that calls _belief_* / _optimal_* / other parent "
        "helpers without defining them.",
        helper_note,
    ]


def _parse_unified_hunks(
    diff_text: str,
) -> List[Tuple[int, int, List[str], List[str]]]:
    lines = (diff_text or "").splitlines(keepends=True)
    hunks: List[Tuple[int, int, List[str], List[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("---") or line.startswith("+++"):
            i += 1
            continue
        match = _HUNK_HEADER_RE.match(line.rstrip("\n"))
        if not match:
            i += 1
            continue
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        i += 1
        old_block: List[str] = []
        new_block: List[str] = []
        while i < len(lines):
            row = lines[i]
            if row.startswith("@@") or row.startswith("---") or row.startswith("+++"):
                break
            if row.startswith("\\"):
                i += 1
                continue
            if not row:
                i += 1
                continue
            tag = row[0]
            body = row[1:] if tag in " +-" else row
            if not body.endswith("\n"):
                body = body + "\n"
            if tag == " ":
                old_block.append(body)
                new_block.append(body)
            elif tag == "-":
                old_block.append(body)
            elif tag == "+":
                new_block.append(body)
            else:
                break
            i += 1
        hunks.append((old_start, old_count, new_block, old_block))
    return hunks


def _lines_match(actual: List[str], expected: List[str]) -> bool:
    if len(actual) != len(expected):
        if not expected and not actual:
            return True
        return False
    for a, e in zip(actual, expected):
        if a.rstrip("\n") != e.rstrip("\n"):
            return False
    return True
