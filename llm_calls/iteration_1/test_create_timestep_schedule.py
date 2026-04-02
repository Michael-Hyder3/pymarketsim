import unittest

from llm_calls.iteration_1.create_timestep_schedule import create_timestep_schedule


class TestCreateTimestepSchedule(unittest.TestCase):
    def test_returns_list_with_timesteps(self):
        private_values = [100, 95, 90, 85, 80]
        total_timesteps = 5
        schedule = create_timestep_schedule(private_values, total_timesteps)

        self.assertIsInstance(schedule, list)
        self.assertEqual(len(schedule), total_timesteps)

        for index, entry in enumerate(schedule):
            self.assertIsInstance(entry, dict)
            self.assertIn("timestep", entry)
            self.assertIn("shading_region", entry)
            self.assertEqual(entry["timestep"], index)

    def test_shading_region_structure(self):
        private_values = [50, 50, 50, 50]
        total_timesteps = 4
        schedule = create_timestep_schedule(private_values, total_timesteps)

        for entry in schedule:
            region = entry["shading_region"]
            self.assertIsInstance(region, list)
            self.assertEqual(len(region), 2)
            self.assertIsInstance(region[0], (int, float))
            self.assertIsInstance(region[1], (int, float))
            self.assertGreaterEqual(region[0], 0)
            self.assertGreaterEqual(region[1], 0)
            self.assertLessEqual(region[0], region[1])

    def test_handles_negative_values(self):
        private_values = [-10, -20, -30]
        total_timesteps = 3
        schedule = create_timestep_schedule(private_values, total_timesteps)

        for entry in schedule:
            region = entry["shading_region"]
            self.assertGreaterEqual(region[0], 0)
            self.assertGreaterEqual(region[1], 0)

    def test_coverage_of_timesteps(self):
        private_values = [100, 90]
        total_timesteps = 6
        schedule = create_timestep_schedule(private_values, total_timesteps)

        self.assertEqual(len(schedule), total_timesteps)
        self.assertEqual(schedule[0]["timestep"], 0)
        self.assertEqual(schedule[-1]["timestep"], total_timesteps - 1)


if __name__ == "__main__":
    unittest.main()
