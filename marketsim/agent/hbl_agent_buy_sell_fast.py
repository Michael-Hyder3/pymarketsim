from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from marketsim.agent.hbl_agent_buy_sell import HBLAgentBuy, HBLAgentSell
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order


def _matched_order_index(market: Any) -> Dict[int, Any]:
    index: Dict[int, Any] = {}
    for matched_order in getattr(market, "matched_orders", None) or []:
        oid = matched_order.order.order_id
        if oid not in index:
            index[oid] = matched_order
    return index


def _buyer_withdrawal_times(orders: Sequence[Any]) -> List[Optional[int]]:
    n = len(orders)
    out: List[Optional[int]] = [None] * n
    for i, order in enumerate(orders):
        for j in range(i + 1, n):
            later = orders[j]
            if (
                later.agent_id == order.agent_id
                and later.order_id != order.order_id
                and later.time > order.time
            ):
                out[i] = int(later.time)
                break
    return out


def _seller_withdrawal_times(orders: Sequence[Any]) -> List[Optional[int]]:
    n = len(orders)
    out: List[Optional[int]] = [None] * n
    for i, order in enumerate(orders):
        for j in range(i + 1, n):
            later = orders[j]
            if later.agent_id == order.agent_id:
                out[i] = int(later.time)
                break
    return out


class HBLAgentBuyFast(HBLAgentBuy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matched_index_cache: Optional[Dict[int, Any]] = None
        self._withdrawal_cache: Optional[List[Optional[int]]] = None


    def _ensure_matched_index(self) -> Dict[int, Any]:
        if self._matched_index_cache is None:
            self._matched_index_cache = _matched_order_index(self.market)
        return self._matched_index_cache

    def fast_belief_function(self, p, side, orders):
        matched = self._ensure_matched_index()
        TBL = 0
        AL = 0
        for order in orders:
            if order.price <= p and order.order_type == SELL:
                AL += order.quantity
            matched_order = matched.get(order.order_id)
            if matched_order is not None:
                if (
                    matched_order.order.order_type == BUY
                    and matched_order.price <= p
                ):
                    TBL += order.quantity
        return AL + TBL == 0

    def belief_function(self, p, side, orders):
        current_time = self.market.get_time()
        matched = self._ensure_matched_index()
        if self._withdrawal_cache is None or len(self._withdrawal_cache) != len(
            orders
        ):
            self._withdrawal_cache = _buyer_withdrawal_times(orders)
        withdrawal = self._withdrawal_cache

        TBL = 0
        AL = 0
        RBG = 0
        for ind, order in enumerate(orders):
            if order.price - p <= 0 and order.order_type == SELL:
                AL += order.quantity
            matched_order = matched.get(order.order_id)
            if matched_order is not None:
                if (
                    matched_order.order.order_type == BUY
                    and matched_order.price - p <= 0
                ):
                    TBL += order.quantity
            else:
                if order.order_type == BUY and order.price - p >= 0:
                    latest_order_time = withdrawal[ind]
                    if latest_order_time is None:
                        alive_time = current_time - order.time
                        if alive_time >= self.grace_period:
                            RBG += order.quantity
                        else:
                            RBG += (alive_time / self.grace_period) * order.quantity
                    else:
                        time_till_withdrawal = latest_order_time - order.time
                        if time_till_withdrawal >= self.grace_period:
                            RBG += order.quantity
                        else:
                            RBG += (
                                time_till_withdrawal / self.grace_period
                            ) * order.quantity

        if TBL + AL == 0:
            return 0
        return (TBL + AL) / (TBL + AL + RBG)

    def determine_optimal_price(self, side):
        self._matched_index_cache = _matched_order_index(self.market)
        self._withdrawal_cache = None
        try:
            return super().determine_optimal_price(side)
        finally:
            self._matched_index_cache = None
            self._withdrawal_cache = None


class HBLAgentSellFast(HBLAgentSell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._matched_index_cache: Optional[Dict[int, Any]] = None
        self._withdrawal_cache: Optional[List[Optional[int]]] = None


    def _ensure_matched_index(self) -> Dict[int, Any]:
        if self._matched_index_cache is None:
            self._matched_index_cache = _matched_order_index(self.market)
        return self._matched_index_cache

    def fast_belief_function(self, p, side, orders):
        matched = self._ensure_matched_index()
        TAG = 0
        BG = 0
        for order in orders:
            if order.price >= p and order.order_type == BUY:
                BG += order.quantity
            matched_order = matched.get(order.order_id)
            if matched_order is not None:
                if (
                    matched_order.order.order_type == SELL
                    and matched_order.price >= p
                ):
                    TAG += order.quantity
        return BG + TAG == 0

    def belief_function(self, p, side, orders):
        current_time = self.market.get_time()
        matched = self._ensure_matched_index()
        if self._withdrawal_cache is None or len(self._withdrawal_cache) != len(
            orders
        ):
            self._withdrawal_cache = _seller_withdrawal_times(orders)
        withdrawal = self._withdrawal_cache

        TAG = 0
        BG = 0
        RAL = 0

        for order in orders:
            if order.price - p >= 0 and order.order_type == BUY:
                BG += order.quantity

        for ind, order in enumerate(orders):
            matched_order = matched.get(order.order_id)
            if matched_order is not None:
                if (
                    matched_order.order.order_type == SELL
                    and matched_order.price - p >= 0
                ):
                    TAG += order.quantity
            else:
                if order.order_type == SELL and order.price - p <= 0:
                    latest_order_time = withdrawal[ind]
                    if latest_order_time is None:
                        alive_time = current_time - order.time
                        if alive_time >= self.grace_period:
                            RAL += order.quantity
                        else:
                            RAL += (alive_time / self.grace_period) * order.quantity
                    else:
                        time_till_withdrawal = latest_order_time - order.time
                        if time_till_withdrawal >= self.grace_period:
                            RAL += order.quantity
                        else:
                            RAL += (
                                time_till_withdrawal / self.grace_period
                            ) * order.quantity

        if TAG + BG == 0:
            return 0
        return (TAG + BG) / (TAG + BG + RAL)

    def determine_optimal_price(self, side):
        self._matched_index_cache = _matched_order_index(self.market)
        self._withdrawal_cache = None
        try:
            return super().determine_optimal_price(side)
        finally:
            self._matched_index_cache = None
            self._withdrawal_cache = None
