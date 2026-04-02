import unittest
from llm_calls.iteration_2.suggest_shading_region import suggest_shading_region

class TestSuggestShadingRegion(unittest.TestCase):
    def test_positive_x(self):
        # Example private values
        private_values = [100, 95, 90, 85, 80]
        region = suggest_shading_region(private_values)
        self.assertIsInstance(region, list)
        self.assertEqual(region[0], 0)
        self.assertGreater(region[1], 0, "X (region[1]) should be positive")

    def test_zero_variance(self):
        private_values = [50, 50, 50, 50]
        region = suggest_shading_region(private_values)
        self.assertEqual(region[0], 0)
        self.assertGreater(region[1], 0, "X (region[1]) should be positive even if all values are the same")

    def test_negative_private_values(self):
        private_values = [-10, -20, -30]
        region = suggest_shading_region(private_values)
        self.assertEqual(region[0], 0)
        self.assertGreaterEqual(region[1], 0, "X (region[1]) should be non-negative")

    def test_mixed_positive_negative(self):
        # Test with mixed positive and negative values
        private_values = [100, 50, 0, -50, -100]
        region = suggest_shading_region(private_values)
        self.assertEqual(region[0], 0)
        # Even with mixed values, should return a reasonable region
        self.assertIsInstance(region[1], (int, float))
        self.assertGreaterEqual(region[1], 0)
        self.assertEqual(region[0], 0)
        self.assertGreater(region[1], 0, "X (region[1]) should be positive even for negative private values")

if __name__ == "__main__":
    unittest.main()
