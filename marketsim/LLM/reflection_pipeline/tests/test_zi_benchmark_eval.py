from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import build_zi_beat_comparison


def test_aggregate_zi_episode_results():
    from marketsim.LLM.reflection_pipeline.zi_benchmark_eval import (
        aggregate_zi_episode_results,
    )

    ladder_specs = [
        {"shade_range": [0.0, 0.002], "ladder_mode": "zi", "label": "zi_cap=0.002"},
        {"shade_range": [0.0, 0.010], "ladder_mode": "zi", "label": "zi_cap=0.01"},
    ]
    results = aggregate_zi_episode_results(
        [
            {"benchmark_index": 0, "shade_range": [0.0, 0.002], "llm_surplus": 10.0},
            {"benchmark_index": 0, "shade_range": [0.0, 0.002], "llm_surplus": 20.0},
            {"benchmark_index": 1, "shade_range": [0.0, 0.010], "llm_surplus": 5.0},
            {"benchmark_index": 1, "shade_range": [0.0, 0.010], "llm_surplus": 15.0},
        ],
        ladder_specs,
    )
    assert results[0]["mean_surplus"] == 15.0
    assert results[1]["mean_surplus"] == 10.0
    assert results[0]["ladder_mode"] == "zi"


def test_build_zi_beat_comparison_counts_beats():
    comparison = build_zi_beat_comparison(
        llm_mean_surplus=100.0,
        benchmark_results=[
            {"benchmark_index": 0, "shade_range": [0.0, 0.002], "mean_surplus": 90.0},
            {"benchmark_index": 1, "shade_range": [0.0, 0.010], "mean_surplus": 100.0},
            {"benchmark_index": 2, "shade_range": [0.0, 0.020], "mean_surplus": 110.0},
        ],
    )
    assert comparison["beat_count"] == 1
    assert comparison["total_benchmarks"] == 3
    assert comparison["beat_fraction"] == 1 / 3
    assert comparison["beaten_shade_ranges"] == [[0.0, 0.002]]
    assert comparison["not_beaten_shade_ranges"] == [[0.0, 0.010], [0.0, 0.020]]


def test_build_zi_beat_comparison_all_beats():
    comparison = build_zi_beat_comparison(
        llm_mean_surplus=50.0,
        benchmark_results=[
            {"benchmark_index": i, "shade_range": [0.0, 0.01], "mean_surplus": 10.0}
            for i in range(10)
        ],
    )
    assert comparison["beat_count"] == 10
    assert comparison["beat_fraction"] == 1.0


def test_build_zi_beat_comparison_includes_standard_zip_seat():
    comparison = build_zi_beat_comparison(
        llm_mean_surplus=100.0,
        benchmark_results=[
            {
                "benchmark_index": 0,
                "shade_range": [0.0, 0.0015],
                "mean_surplus": 90.0,
                "ladder_mode": "fixed_mu",
                "label": "fixed_mu=0.0015",
            },
            {
                "benchmark_index": 1,
                "shade_range": [],
                "mean_surplus": 95.0,
                "ladder_mode": "standard",
                "label": "standard_zip",
            },
        ],
    )
    assert comparison["total_benchmarks"] == 2
    assert comparison["beat_count"] == 2
    assert comparison["benchmarks"][1]["ladder_mode"] == "standard"
    assert comparison["benchmarks"][1]["label"] == "standard_zip"


def test_build_zi_beat_comparison_empty():
    comparison = build_zi_beat_comparison(
        llm_mean_surplus=0.0,
        benchmark_results=[],
    )
    assert comparison["beat_fraction"] == 0.0
    assert comparison["total_benchmarks"] == 0
