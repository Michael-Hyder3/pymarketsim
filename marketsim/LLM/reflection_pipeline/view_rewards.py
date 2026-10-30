from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from marketsim.LLM.reflection_pipeline.metrics import summarize_batch_performance


def _find_latest_summary() -> Optional[Path]:
    base_dir = Path(__file__).resolve().parent / "runs"
    runs = list(base_dir.glob("*/reflection_pipeline_summary.json"))
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)


def _step_performance(step_data: Dict[str, Any], reward_mode: str) -> Optional[Dict[str, Any]]:
    if step_data.get("performance"):
        return step_data["performance"]
    batch_data = step_data.get("batch_data")
    if batch_data:
        return summarize_batch_performance(batch_data, reward_mode=reward_mode)
    return None


def _seed_performance_rows(seed_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seed_evaluations:
        if not seed.get("success"):
            continue
        perf = seed.get("performance")
        if perf is None:
            continue
        rows.append(
            {
                "label": f"seed node {seed.get('node_id')}",
                "node_id": seed.get("node_id"),
                **perf,
                "reward": seed.get("reward"),
                "reward_variance": seed.get("reward_variance"),
            }
        )
    return rows


def _format_float(value: Optional[float], width: int = 12) -> str:
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.4f}"


def _format_mean_var(mean_key: str, var_key: str, perf: Dict[str, Any], width: int = 12) -> str:
    mean = perf.get(mean_key)
    variance = perf.get(var_key)
    if mean is None:
        return f"{'N/A':>{width}}"
    if variance is None:
        return f"{mean:>{width}.4f}"
    return f"{mean:>{width}.4f} ({variance:.2f})"


def _print_llm_vs_zi_section(data: Dict[str, Any]) -> None:
    params = data.get("params", {})
    reward_mode = params.get("reward_mode", "advantage")
    summary = data.get("summary", [])
    seed_evaluations = data.get("seed_evaluations", [])

    print("\n" + "=" * 140)
    print("LLM vs SAME-SIDE ZI AGENTS  (values show mean (variance) where available)")
    print("=" * 140)
    print(
        f"{'Stage':<14} | {'LLM Surplus':<18} | {'ZI Surplus':<18} | "
        f"{'LLM-ZI':<18} | {'Baseline':<18} | {'Advantage':<18} | {'Q/Reward':<18}"
    )
    print("-" * 140)

    for row in _seed_performance_rows(seed_evaluations):
        print(
            f"{row['label']:<14} | "
            f"{_format_mean_var('llm_surplus_mean', 'llm_surplus_variance', row, 18)} | "
            f"{_format_mean_var('avg_zi_same_surplus_mean', 'avg_zi_same_surplus_variance', row, 18)} | "
            f"{_format_mean_var('llm_vs_zi_diff_mean', 'llm_vs_zi_diff_variance', row, 18)} | "
            f"{_format_mean_var('baseline_surplus_mean', 'baseline_surplus_variance', row, 18)} | "
            f"{_format_mean_var('advantage_mean', 'advantage_variance', row, 18)} | "
            f"{_format_float(row.get('reward'), 10)} ({row.get('reward_variance', 0.0):.2f})"
        )

    for step_data in summary:
        step = step_data.get("step", "?")
        if not step_data.get("success"):
            print(
                f"{'step ' + str(step):<14} | {'FAILED':<18} | {'':<18} | {'':<18} | "
                f"{'':<18} | {'':<18} | {'N/A':<18}"
            )
            continue

        perf = _step_performance(step_data, reward_mode)
        if perf is None:
            print(f"{'step ' + str(step):<14} | {'NO DATA':<18}")
            continue

        reward = step_data.get("reward")
        reward_variance = step_data.get("reward_variance", 0.0)
        print(
            f"{'step ' + str(step):<14} | "
            f"{_format_mean_var('llm_surplus_mean', 'llm_surplus_variance', perf, 18)} | "
            f"{_format_mean_var('avg_zi_same_surplus_mean', 'avg_zi_same_surplus_variance', perf, 18)} | "
            f"{_format_mean_var('llm_vs_zi_diff_mean', 'llm_vs_zi_diff_variance', perf, 18)} | "
            f"{_format_mean_var('baseline_surplus_mean', 'baseline_surplus_variance', perf, 18)} | "
            f"{_format_mean_var('advantage_mean', 'advantage_variance', perf, 18)} | "
            f"{_format_float(reward, 10)} ({reward_variance:.2f})"
        )

    print("-" * 140)
    print(
        "LLM-ZI = mean(llm_surplus) - mean(avg_zi_same_surplus) over batch episodes. "
        f"Q/Reward column uses pipeline reward mode ({reward_mode})."
    )


def _collect_q_refresh_events(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    history = data.get("q_refresh_history") or []
    if history:
        events.extend(history)
    else:
        for step_data in data.get("summary", []):
            q_refresh = step_data.get("q_refresh")
            if q_refresh:
                events.append(q_refresh)

    if not events:
        return events

    normalized: List[Dict[str, Any]] = []
    for event in events:
        phase = event.get("phase")
        if phase is None:
            step_num = event.get("step", 0)
            phase = "seed" if step_num == 0 else "step"
        normalized.append({**event, "phase": phase})
    return normalized


def _print_q_refresh_history(data: Dict[str, Any]) -> None:
    events = _collect_q_refresh_events(data)
    if not events:
        print("\n(No Q refresh history — run with --q-eval-mode full_refresh to populate.)")
        return

    node_ids = sorted(
        {
            int(node_id)
            for event in events
            for node_id in (event.get("q_values") or {}).keys()
        }
    )

    print("\n" + "=" * 140)
    print("Q VALUES BY REFRESH  (mean (variance))")
    print("=" * 140)

    header = f"{'Refresh':<12} | {'New Node':<8} | {'Seconds':<8}"
    for node_id in node_ids:
        header += f" | {'N' + str(node_id):>14}"
    print(header)
    print("-" * len(header))

    for event in events:
        phase = event.get("phase", "step")
        step = event.get("step", "?")
        label = f"{phase} {step}" if phase != "seed" else "seed"
        new_node = event.get("new_node_id")
        new_node_str = str(new_node) if new_node is not None else "-"
        seconds = event.get("refresh_seconds")
        seconds_str = f"{seconds:.1f}" if seconds is not None else "-"
        q_values = event.get("q_values") or {}
        q_variances = event.get("q_variances") or {}

        row = f"{label:<12} | {new_node_str:<8} | {seconds_str:<8}"
        for node_id in node_ids:
            q_val = q_values.get(str(node_id))
            q_var = q_variances.get(str(node_id))
            if q_val is None:
                cell = "N/A"
            elif q_var is None:
                cell = f"{float(q_val):.4f}"
            else:
                cell = f"{float(q_val):.4f} ({float(q_var):.2f})"
            row += f" | {cell:>14}"
        print(row)

    print("-" * len(header))
    print("Each row is one full_refresh pass: all nodes re-scored on the same sampled batch.")


def print_rewards(json_path: str) -> None:
    path = Path(json_path)
    if not path.exists():
        print(f"Error: Could not find file at {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_dir = data.get("run_dir", path.parent)
    params = data.get("params", {})
    print(f"\nRun: {run_dir}")
    print(
        f"Mode: {params.get('q_eval_mode', 'incremental')} | "
        f"reward_mode: {params.get('reward_mode')} | "
        f"q_update_mode: {params.get('q_update_mode', 'replace')} | "
        f"critic_per_config_metrics: {params.get('critic_per_config_metrics', False)} | "
        f"batch_size: {params.get('batch_size')} | "
        f"episodes_per_config: {params.get('episodes_per_config')}"
    )

    _print_llm_vs_zi_section(data)
    _print_q_refresh_history(data)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        latest = _find_latest_summary()
        if latest is None:
            print("Error: No summary JSON found under runs/. Pass a path explicitly.")
            sys.exit(1)
        target = str(latest)
        print(f"Using latest run: {target}")
    print_rewards(target)
