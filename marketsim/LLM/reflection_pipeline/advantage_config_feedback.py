from __future__ import annotations

from typing import Any, Dict, List


def enrich_advantage_summary_with_config_tests(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach soft unit-test style results for each market config in advantage mode.

    Tests never block the pipeline; they report PASS/FAIL with factual metrics only.
    Assertion: per-config mean advantage (LLM surplus minus baseline) > 0.
    """
    per_config_metrics = summary.get("per_config_metrics") or {}
    per_config_rewards = summary.get("per_config_rewards") or {}
    per_config_reward_variances = summary.get("per_config_reward_variances") or {}

    labels = sorted(set(per_config_rewards) | set(per_config_metrics))
    tests: List[Dict[str, Any]] = []

    for index, label in enumerate(labels):
        metrics = per_config_metrics.get(label, {})
        advantage_mean = float(
            per_config_rewards.get(
                label,
                metrics.get("advantage_mean", metrics.get("reward_mean", 0.0)),
            )
        )
        advantage_var = float(
            per_config_reward_variances.get(
                label,
                metrics.get("advantage_variance", metrics.get("reward_variance", 0.0)),
            )
        )
        passed = advantage_mean > 0.0
        test: Dict[str, Any] = {
            "test_id": f"config_{index}",
            "config_label": label,
            "status": "PASS" if passed else "FAIL",
            "advantage_mean": advantage_mean,
            "advantage_variance": advantage_var,
            "advantage_gap": float(advantage_mean),
            "assertion": f"advantage_mean ({advantage_mean:.2f}) > 0",
        }
        if metrics.get("llm_surplus_mean") is not None:
            test["llm_surplus_mean"] = float(metrics["llm_surplus_mean"])
        if metrics.get("baseline_surplus_mean") is not None:
            test["baseline_surplus_mean"] = float(metrics["baseline_surplus_mean"])
        if metrics.get("avg_shade_mean") is not None:
            test["avg_shade_mean"] = float(metrics["avg_shade_mean"])
        config = metrics.get("config")
        if isinstance(config, dict) and config:
            # LLM-facing: keep only the opaque label, not hyperparameters.
            test["config"] = {"label": config.get("label", label)}
        tests.append(test)

    failed = sorted(
        [t for t in tests if t["status"] == "FAIL"],
        key=lambda t: float(t["advantage_mean"]),
    )
    passed = [t for t in tests if t["status"] == "PASS"]

    summary["config_unit_tests"] = tests
    summary["failed_config_tests"] = failed
    summary["passed_config_tests"] = passed
    summary["config_tests_pass_count"] = len(passed)
    summary["config_tests_total"] = len(tests)
    return summary
