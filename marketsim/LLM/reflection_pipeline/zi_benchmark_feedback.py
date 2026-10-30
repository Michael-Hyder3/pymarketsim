from __future__ import annotations

from typing import Any, Dict, List


def _shade_hi(shade_range: List[float]) -> float:
    if not shade_range:
        return 0.0
    if len(shade_range) > 1:
        return float(shade_range[1])
    return float(shade_range[0])


def enrich_zi_comparison_with_benchmark_tests(comparison: Dict[str, Any]) -> Dict[str, Any]:
    llm_mean = float(comparison.get("llm_mean_surplus", 0.0))
    llm_mean_shade = comparison.get("llm_mean_shade")
    benchmarks = comparison.get("benchmarks") or []
    tests: List[Dict[str, Any]] = []

    for item in benchmarks:
        benchmark_index = int(item["benchmark_index"])
        shade_range = list(item.get("shade_range") or [])
        ladder_mode = str(item.get("ladder_mode") or "zi")
        label = str(item.get("label") or "")
        shade_hi = _shade_hi(shade_range)
        hbl_l = item.get("hbl_l")
        zi_mean = float(item["mean_surplus"])
        surplus_gap = llm_mean - zi_mean
        passed = bool(item.get("beat"))
        if ladder_mode == "standard":
            seat_desc = label or "standard_zip"
        elif ladder_mode == "hbl_L" or hbl_l is not None:
            seat_desc = label or f"hbl_L={int(hbl_l)}"
        elif shade_range:
            seat_desc = f"shade_range={shade_range}"
        else:
            seat_desc = label or f"bench_{benchmark_index}"
        test: Dict[str, Any] = {
            "test_id": f"bench_{benchmark_index}",
            "benchmark_index": benchmark_index,
            "shade_range": shade_range,
            "shade_hi": shade_hi,
            "ladder_mode": ladder_mode,
            "status": "PASS" if passed else "FAIL",
            "llm_mean_surplus": llm_mean,
            "zi_mean_surplus": zi_mean,
            "surplus_gap": float(surplus_gap),
            "assertion": (
                f"llm_mean_surplus ({llm_mean:.2f}) > zi_mean_surplus ({zi_mean:.2f}) "
                f"[{seat_desc}]"
            ),
        }
        if label:
            test["label"] = label
        if hbl_l is not None:
            test["hbl_l"] = int(hbl_l)
        if llm_mean_shade is not None:
            test["llm_mean_shade"] = float(llm_mean_shade)
        tests.append(test)

    failed = [t for t in tests if t["status"] == "FAIL"]
    passed = [t for t in tests if t["status"] == "PASS"]
    failed_sorted = sorted(failed, key=lambda t: float(t["surplus_gap"]))

    comparison["benchmark_tests"] = tests
    comparison["failed_tests"] = failed_sorted
    comparison["passed_tests"] = passed
    return comparison
