from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from marketsim.fourheap.constants import BUY, SELL

DEFAULT_HBL_L = 6

HBL_EXACT_SEED_FLAG = "HBL_EXACT_SEED"


def strategy_requests_hbl_exact_seed(strategy_func: Any) -> bool:
    if strategy_func is None:
        return False
    globals_dict = getattr(strategy_func, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return False
    return bool(globals_dict.get(HBL_EXACT_SEED_FLAG))


def hbl_memory_start_time(market: Any, L: int = DEFAULT_HBL_L) -> int:
    L = max(1, int(L))
    matched = list(getattr(market, "matched_orders", None) or [])
    if len(matched) < 2 * L:
        return 0
    last_matched_order_ind = len(matched) - L * 2
    earliest = min(
        matched[last_matched_order_ind:],
        key=lambda matched_order: matched_order.order.time,
    ).order.time
    return int(earliest)


def hbl_orders_in_memory(market: Any, L: int = DEFAULT_HBL_L) -> List[Any]:
    L = max(1, int(L))
    t0 = hbl_memory_start_time(market, L)
    t_now = int(market.get_time())
    scheduled = market.event_queue.scheduled_activities
    orders: List[Any] = []
    for time in range(t0, t_now + 1):
        orders.extend(list(scheduled[time]))
    return orders


def hbl_order_memory_partition(
    market: Any, L: int = DEFAULT_HBL_L
) -> Tuple[List[Any], List[Any], List[Any]]:
    last_L_orders = hbl_orders_in_memory(market, L)
    buy_orders = [order for order in last_L_orders if order.order_type == BUY]
    sell_orders = [order for order in last_L_orders if order.order_type == SELL]
    return last_L_orders, buy_orders, sell_orders




def serialize_hbl_order_memory(
    market: Any,
    *,
    L: int = DEFAULT_HBL_L,
) -> Dict[str, Any]:
    L = max(1, int(L))
    orders = hbl_orders_in_memory(market, L)
    match_price_by_oid: Dict[int, float] = {}
    for mo in getattr(market, "matched_orders", None) or []:
        try:
            oid = int(mo.order.order_id)
            match_price_by_oid[oid] = float(mo.price)
        except Exception:
            continue
    serialized: List[Dict[str, Any]] = []
    for order in orders:
        oid = int(order.order_id)
        accepted = oid in match_price_by_oid
        serialized.append(
            {
                "time": int(order.time),
                "side": int(order.order_type),
                "price_rel": float(order.price),
                "quantity": int(getattr(order, "quantity", 1) or 1),
                "accepted": accepted,
                "match_price_rel": (
                    float(match_price_by_oid[oid]) if accepted else None
                ),
                "agent_id": int(order.agent_id),
                "order_id": oid,
            }
        )
    txn_count = len(getattr(market, "matched_orders", None) or []) // 2
    return {
        "L": L,
        "memory_start_time": hbl_memory_start_time(market, L),
        "transaction_count": int(txn_count),
        "orders": serialized,
    }
