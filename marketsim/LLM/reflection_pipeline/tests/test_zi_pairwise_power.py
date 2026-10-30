import random
import unittest

from marketsim.LLM.reflection_pipeline.config import DEFAULT_ZI_BENCHMARK_SHADE_RANGES
from marketsim.LLM.reflection_pipeline.zi_pairwise_power import (
    PairedEpisodePoint,
    analyze_paired_differences,
    empirical_power_curve,
    iter_shade_range_pairs,
    paired_points_from_shade_runs,
    sample_size_for_power,
    shade_ranges_equal,
    unique_shade_ranges,
)


class ZiPairwisePowerTest(unittest.TestCase):
    def test_unique_shade_ranges_dedupes_benchmark_grid(self):
        unique = unique_shade_ranges(DEFAULT_ZI_BENCHMARK_SHADE_RANGES)
        self.assertEqual(len(unique), 21)
        self.assertEqual(unique[0], [0.0, 0.001])
        self.assertEqual(unique[1], [0.0, 0.0015])
        self.assertEqual(unique[-1], [0.0, 1.0])

    def test_iter_shade_range_pairs_count(self):
        pairs = iter_shade_range_pairs(
            DEFAULT_ZI_BENCHMARK_SHADE_RANGES,
            include_self_pairs=True,
        )
        # 21 self + C(21,2)=210 cross = 231
        self.assertEqual(len(pairs), 231)
        self.assertEqual(sum(1 for a, b in pairs if shade_ranges_equal(a, b)), 21)
        self.assertEqual(
            len(iter_shade_range_pairs(
                DEFAULT_ZI_BENCHMARK_SHADE_RANGES,
                include_self_pairs=False,
            )),
            210,
        )

    def test_analyze_paired_differences_all_zero_for_self_pair_control(self):
        out = analyze_paired_differences(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            bootstrap_samples=10,
            power_ns=[5],
            rng=random.Random(2),
        )
        self.assertEqual(out["mean_difference"], 0.0)
        self.assertEqual(out["std_difference"], 0.0)
        self.assertIsNone(out["n_for_power_80"])

    def test_paired_points_from_shade_runs_aligns_episodes(self):
        runs = {
            "[0,0.001]": [
                {"episode_index": 0, "seed": 10, "surplus": 100.0},
                {"episode_index": 1, "seed": 11, "surplus": 110.0},
            ],
            "[0,0.002]": [
                {"episode_index": 0, "seed": 10, "surplus": 80.0},
                {"episode_index": 1, "seed": 11, "surplus": 95.0},
            ],
        }
        points = paired_points_from_shade_runs(
            config_label="cfg",
            shade_range_a=[0.0, 0.001],
            shade_range_b=[0.0, 0.002],
            shade_runs=runs,
        )
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0].difference, 20.0)
        self.assertAlmostEqual(points[1].difference, 15.0)

    def test_paired_points_from_shade_runs_self_pair_is_zero(self):
        runs = {
            "[0,0.001]": [
                {"episode_index": 0, "seed": 10, "surplus": 100.0},
                {"episode_index": 1, "seed": 11, "surplus": 110.0},
            ],
        }
        points = paired_points_from_shade_runs(
            config_label="cfg",
            shade_range_a=[0.0, 0.001],
            shade_range_b=[0.0, 0.001],
            shade_runs=runs,
        )
        self.assertTrue(all(point.difference == 0.0 for point in points))

    def test_analyze_paired_differences_detects_nonzero_mean(self):
        differences = [10.0, 12.0, 11.0, 9.0, 10.5, 11.5]
        out = analyze_paired_differences(
            differences,
            bootstrap_samples=50,
            power_ns=[5, 6],
            rng=random.Random(0),
        )
        self.assertEqual(out["n_episodes"], 6)
        self.assertAlmostEqual(out["mean_difference"], sum(differences) / 6.0)
        self.assertAlmostEqual(
            out["se_difference"],
            out["std_difference"] / (6 ** 0.5),
            places=9,
        )
        self.assertLess(out["paired_t_pvalue"], 0.01)

    def test_analyze_paired_differences_zero_mean_not_significant(self):
        differences = [1.0, -1.0, 0.5, -0.5, 0.0, 0.0]
        out = analyze_paired_differences(
            differences,
            bootstrap_samples=20,
            power_ns=[6],
            rng=random.Random(1),
        )
        self.assertGreater(out["paired_t_pvalue"], 0.05)

    def test_sample_size_for_power_increases_with_variance(self):
        low_var = sample_size_for_power(10.0, 5.0, power=0.8)
        high_var = sample_size_for_power(50.0, 5.0, power=0.8)
        self.assertIsNotNone(low_var)
        self.assertIsNotNone(high_var)
        self.assertGreater(high_var, low_var)

    def test_sample_size_for_power_none_when_no_effect(self):
        self.assertIsNone(sample_size_for_power(10.0, 0.0))

    def test_empirical_power_higher_for_large_effect(self):
        rng = random.Random(0)
        small_effect = [1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 0.0, 0.0]
        large_effect = [20.0, 18.0, 22.0, 19.0, 21.0, 20.0, 19.0, 21.0]
        small_power = empirical_power_curve(
            small_effect,
            ns=[8],
            bootstrap_samples=200,
            rng=rng,
        )
        large_power = empirical_power_curve(
            large_effect,
            ns=[8],
            bootstrap_samples=200,
            rng=random.Random(0),
        )
        self.assertGreater(large_power[8], small_power[8])


if __name__ == "__main__":
    unittest.main()
