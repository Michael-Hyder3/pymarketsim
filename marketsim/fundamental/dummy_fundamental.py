"""
Dummy fundamental model used by tests.

Provides a small, configurable wrapper that exposes the same interface
used by the rest of the codebase (`get_value_at`, `get_info`, etc.).
This consolidates the duplicate DummyFundamental definitions used in
multiple test files.
"""

class DummyFundamental:
    """Simple constant fundamental with configurable mean-reversion rate.

    Defaults mirror previous test behavior except `r` (mean reversion)
    is set to a small non-zero value so estimates are not constant.
    """

    def __init__(self, value=100000.0, final_time=100, r: float = 0.05):
        self.value = float(value)
        self.final_time = final_time
        self.mean = float(value)
        self.r = float(r)

    def get_value(self, time=None):
        return self.value

    def get_value_at(self, time):
        return self.value

    def get_fundamental_values(self):
        return {0: self.value}

    def get_final_fundamental(self):
        return self.value

    def get_r(self):
        return self.r

    def get_mean(self):
        return self.mean

    def get_info(self):
        return self.mean, self.r, self.final_time
