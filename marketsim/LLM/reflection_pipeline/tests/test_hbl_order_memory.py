"""HBL ↔ LLM LOB order-memory parity."""

from __future__ import annotations

import unittest
from typing import Any, List
from unittest import mock

from marketsim.LLM.hbl_order_memory import (
    DEFAULT_HBL_L,
    hbl_memory_start_time,
    hbl_orders_in_memory,
    serialize_hbl_order_memory,
)
from marketsim.fourheap.constants import BUY, SELL


class _FakeOrder:
    def __init__(self, *, time, order_type, price, order_id, agent_id=0, quantity=1):
        self.time = time
        self.order_type = order_type
        self.price = price
        self.order_id = order_id
        self.agent_id = agent_id
        self.quantity = quantity


class _FakeMatched:
    def __init__(self, order, price=None):
        self.order = order
        self.price = float(order.price if price is None else price)


class _FakeMarket:
    def __init__(self, *, t_now: int, matched: List[Any], scheduled: dict):
        self._t = t_now
        self.matched_orders = matched
        self.event_queue = mock.Mock()
        self.event_queue.scheduled_activities = scheduled

    def get_time(self):
        return self._t


class HBLOrderMemoryParityTest(unittest.TestCase):
    def test_window_before_L_transactions_starts_at_zero(self):
        # Fewer than 2*L matched legs → memory starts at 0.
        o1 = _FakeOrder(time=3, order_type=BUY, price=-1.0, order_id=1)
        o2 = _FakeOrder(time=4, order_type=SELL, price=2.0, order_id=2)
        market = _FakeMarket(
            t_now=5,
            matched=[_FakeMatched(o1), _FakeMatched(o2)],
            scheduled={
                0: [_FakeOrder(time=0, order_type=BUY, price=-5.0, order_id=10)],
                1: [],
                2: [],
                3: [o1],
                4: [o2],
                5: [_FakeOrder(time=5, order_type=BUY, price=-3.0, order_id=11)],
            },
        )
        self.assertEqual(hbl_memory_start_time(market, L=6), 0)
        orders = hbl_orders_in_memory(market, L=6)
        self.assertEqual([o.order_id for o in orders], [10, 1, 2, 11])

    def test_window_uses_earliest_of_last_L_matched(self):
        # Build 2*L matched legs with earliest among last L at time 7.
        L = 2
        matched_orders = []
        scheduled = {t: [] for t in range(0, 12)}
        # Older matched pair at t=1 (outside last L window of 2 txns = 4 legs)
        early_b = _FakeOrder(time=1, order_type=BUY, price=-1.0, order_id=101)
        early_s = _FakeOrder(time=1, order_type=SELL, price=1.0, order_id=102)
        scheduled[1] = [early_b, early_s]
        matched_orders.extend([_FakeMatched(early_b), _FakeMatched(early_s)])
        # Last 2 transactions: legs at t=7,8,9,10 — earliest is 7
        for t, oid_b, oid_s in [(7, 201, 202), (8, 203, 204)]:
            b = _FakeOrder(time=t, order_type=BUY, price=-2.0, order_id=oid_b)
            s = _FakeOrder(time=t, order_type=SELL, price=2.0, order_id=oid_s)
            scheduled[t] = [b, s]
            matched_orders.extend([_FakeMatched(b), _FakeMatched(s)])
        # Extra unmatched shout after window start
        late = _FakeOrder(time=11, order_type=BUY, price=-4.0, order_id=300)
        scheduled[11] = [late]

        market = _FakeMarket(t_now=11, matched=matched_orders, scheduled=scheduled)
        self.assertEqual(hbl_memory_start_time(market, L=L), 7)
        orders = hbl_orders_in_memory(market, L=L)
        self.assertEqual([o.order_id for o in orders], [201, 202, 203, 204, 300])

        ser = serialize_hbl_order_memory(market, L=L)
        self.assertEqual(ser["L"], L)
        self.assertEqual(ser["memory_start_time"], 7)
        self.assertEqual(ser["transaction_count"], 3)
        self.assertEqual(len(ser["orders"]), 5)
        by_id = {row["order_id"]: row for row in ser["orders"]}
        self.assertTrue(by_id[201]["accepted"])
        self.assertFalse(by_id[300]["accepted"])
        self.assertEqual(by_id[201]["side"], BUY)
        self.assertEqual(by_id[202]["side"], SELL)
        self.assertEqual(by_id[201]["match_price_rel"], -2.0)
        self.assertIsNone(by_id[300]["match_price_rel"])

    def test_serialize_uses_match_price_not_submit_price(self):
        b = _FakeOrder(time=1, order_type=BUY, price=-10.0, order_id=1)
        s = _FakeOrder(time=1, order_type=SELL, price=10.0, order_id=2)
        market = _FakeMarket(
            t_now=1,
            matched=[_FakeMatched(b, price=-3.0), _FakeMatched(s, price=-3.0)],
            scheduled={1: [b, s]},
        )
        ser = serialize_hbl_order_memory(market, L=1)
        by_id = {row["order_id"]: row for row in ser["orders"]}
        self.assertEqual(by_id[1]["price_rel"], -10.0)
        self.assertEqual(by_id[1]["match_price_rel"], -3.0)
        self.assertEqual(by_id[2]["match_price_rel"], -3.0)

    def test_default_L_matches_hbl_default(self):
        self.assertEqual(DEFAULT_HBL_L, 6)


if __name__ == "__main__":
    unittest.main()
