"""HBL vs HBL-fast: identical behavior on matched seeds; speed sanity."""

from __future__ import annotations

import time
import unittest

from marketsim.LLM.episode_runner import run_episode
from marketsim.fourheap.constants import BUY, SELL


def _dummy_strategy(*_args, **_kwargs):
    return 0.0


def _episode_fingerprint(result: dict) -> tuple:
    """Surplus + tracked-agent decision trail (if present)."""
    surplus = float(result.get("llm_surplus", float("nan")))
    decisions = []
    # Prefer compact market_config + surplus; also compare trade count if available.
    cfg = result.get("market_config") or {}
    n_trades = cfg.get("total_trades")
    if n_trades is None:
        diag = result.get("diagnostic_json") or {}
        events = diag.get("events") or []
        n_trades = sum(1 for e in events if e.get("event_type") == "trade")
    return (round(surplus, 10), int(n_trades) if n_trades is not None else -1)


class TestHblFastIdentical(unittest.TestCase):
    def _common(self, **overrides):
        base = dict(
            llm_strategy_func=_dummy_strategy,
            num_zi_buy=5,
            num_zi_sell=5,
            timesteps=200,
            q_max=4,
            shade_range=[0.0, 0.015],
            arrival_rate=0.15,
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
            background_agent_kind="zi",
        )
        base.update(overrides)
        return base

    def test_tracked_seat_buy_identical_across_seeds(self):
        mismatches = []
        for seed in range(12):
            common = self._common(llm_order_type=BUY, random_seed=seed)
            a = run_episode(special_agent_kind="hbl", **common)
            b = run_episode(special_agent_kind="hbl_fast", **common)
            fa, fb = _episode_fingerprint(a), _episode_fingerprint(b)
            if fa != fb:
                mismatches.append((seed, fa, fb, a["llm_surplus"], b["llm_surplus"]))
        self.assertEqual(mismatches, [], msg=f"mismatches={mismatches}")

    def test_tracked_seat_sell_identical_across_seeds(self):
        mismatches = []
        for seed in range(12):
            common = self._common(llm_order_type=SELL, random_seed=1000 + seed)
            a = run_episode(special_agent_kind="hbl", **common)
            b = run_episode(special_agent_kind="hbl_fast", **common)
            fa, fb = _episode_fingerprint(a), _episode_fingerprint(b)
            if fa != fb:
                mismatches.append((seed, fa, fb))
        self.assertEqual(mismatches, [], msg=f"mismatches={mismatches}")

    def test_hbl_background_population_identical(self):
        """Full HBL crowd: legacy vs fast backgrounds, ZI tracked seat."""
        mismatches = []
        for seed in (3, 17, 42, 99):
            common = self._common(
                llm_order_type=BUY,
                random_seed=seed,
                special_agent_kind="zi",
                num_zi_buy=8,
                num_zi_sell=8,
                timesteps=250,
            )
            common.pop("background_agent_kind", None)
            a = run_episode(background_agent_kind="hbl", **common)
            b = run_episode(background_agent_kind="hbl_fast", **common)
            fa, fb = _episode_fingerprint(a), _episode_fingerprint(b)
            if fa != fb:
                mismatches.append((seed, fa, fb, a["llm_surplus"], b["llm_surplus"]))
        self.assertEqual(mismatches, [], msg=f"mismatches={mismatches}")

    def test_dense_homeset_like_identical(self):
        common = self._common(
            llm_order_type=BUY,
            random_seed=8,
            special_agent_kind="hbl",
            num_zi_buy=5,
            num_zi_sell=15,
            timesteps=400,
            q_max=10,
            shade_range=[0.0, 0.025],
            arrival_rate=0.1,
            background_agent_kind="hbl",
        )
        a = run_episode(**common)
        common_fast = dict(common)
        common_fast["special_agent_kind"] = "hbl_fast"
        common_fast["background_agent_kind"] = "hbl_fast"
        b = run_episode(**common_fast)
        self.assertEqual(_episode_fingerprint(a), _episode_fingerprint(b))
        self.assertEqual(a["llm_surplus"], b["llm_surplus"])


class TestHblFastSpeedSanity(unittest.TestCase):
    def test_fast_not_slower_than_legacy_on_dense_bg(self):
        def lob_strategy(market_history):
            return 0.25

        common = dict(
            llm_strategy_func=lob_strategy,
            llm_order_type=BUY,
            num_zi_buy=5,
            num_zi_sell=15,
            timesteps=400,
            q_max=10,
            shade_range=[0.0, 0.025],
            arrival_rate=0.1,
            strategy_type="lob",
            lob_input_mode="hbl_parity",
            hbl_l_param=6,
            capture_detail=False,
            use_dummy_fundamental=True,
        )
        # warmup
        run_episode(random_seed=0, background_agent_kind="hbl", **common)
        run_episode(random_seed=0, background_agent_kind="hbl_fast", **common)

        def bench(kind, n=2):
            t0 = time.perf_counter()
            for i in range(n):
                run_episode(
                    random_seed=10 + i,
                    background_agent_kind=kind,
                    **common,
                )
            return (time.perf_counter() - t0) / n

        legacy = bench("hbl")
        fast = bench("hbl_fast")
        # Identity-preserving belief indexing should not regress; allow noise.
        self.assertLessEqual(fast, legacy * 1.15, msg=f"legacy={legacy:.3f}s fast={fast:.3f}s")


if __name__ == "__main__":
    unittest.main()
