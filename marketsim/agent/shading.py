from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order


ShadeSpec = Union[List[float], List[Tuple[float, List[float]]]]


def resolve_shade_at_time(
    shade_spec: ShadeSpec,
    *,
    current_time: int,
    num_transactions: int = 0,
    schedule_mode: str = "time",
) -> List[float]:
    if not shade_spec:
        return [0.0, 1.0]

    if isinstance(shade_spec[0], (list, tuple)):
        current_trigger = current_time if schedule_mode == "time" else num_transactions
        applicable_shade = [0.0, 1.0]
        for trigger, shade in shade_spec:
            if trigger <= current_trigger:
                applicable_shade = list(shade)
            else:
                break
        return applicable_shade

    return list(shade_spec)


def _clamp_unit_fraction(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _shade_bounds(shade_range: List[float]) -> Tuple[float, float]:
    if not shade_range:
        return 0.0, 1.0
    if len(shade_range) == 1:
        cap = _clamp_unit_fraction(shade_range[0])
        return 0.0, cap
    lo = _clamp_unit_fraction(min(shade_range[0], shade_range[1]))
    hi = _clamp_unit_fraction(max(shade_range[0], shade_range[1]))
    return lo, hi


def sample_shade_fraction(
    shade_range: List[float],
    rng: random.Random,
    *,
    mode: str = "range",
) -> float:
    lo, hi = _shade_bounds(shade_range)
    if hi <= 0.0:
        return 0.0
    sampling = (mode or "range").lower()
    if sampling == "x_cap":
        return rng.uniform(0.0, hi)
    if hi <= lo:
        return lo
    return rng.uniform(lo, hi)


def zi_limit_price_buy(
    *,
    buyer_value: float,
    fundamental_now: float,
    shade_spec: ShadeSpec,
    current_time: int,
    num_transactions: int,
    schedule_mode: str,
    rng: random.Random,
    sampling_mode: str = "range",
) -> Tuple[float, float]:
    current_shade = resolve_shade_at_time(
        shade_spec,
        current_time=current_time,
        num_transactions=num_transactions,
        schedule_mode=schedule_mode,
    )
    shade_fraction = sample_shade_fraction(
        current_shade,
        rng,
        mode=sampling_mode,
    )
    full_value = fundamental_now + float(buyer_value)
    shade_amount = shade_fraction * max(0.0, full_value)
    price = float(buyer_value) - shade_amount
    return price, shade_fraction


def zi_limit_price_sell(
    *,
    seller_cost: float,
    fundamental_now: float,
    shade_spec: ShadeSpec,
    current_time: int,
    num_transactions: int,
    schedule_mode: str,
    rng: random.Random,
    sampling_mode: str = "range",
) -> Tuple[float, float]:
    current_shade = resolve_shade_at_time(
        shade_spec,
        current_time=current_time,
        num_transactions=num_transactions,
        schedule_mode=schedule_mode,
    )
    shade_fraction = sample_shade_fraction(
        current_shade,
        rng,
        mode=sampling_mode,
    )
    full_value = fundamental_now + float(seller_cost)
    shade_amount = shade_fraction * max(0.0, full_value)
    shaded_full_value = full_value + shade_amount
    price = shaded_full_value - fundamental_now
    return price, shade_fraction


def maybe_take_best_quote_buy(
    *,
    market,
    buyer_value: float,
    eta: float,
    agent_id: int,
    time: int,
    rng: random.Random,
) -> Optional[Order]:
    best_ask = market.order_book.get_best_ask()
    if np.isinf(best_ask):
        return None
    surplus_from_taking = float(buyer_value) - float(best_ask)
    requested_surplus = float(buyer_value)
    if requested_surplus <= 0 or surplus_from_taking < eta * requested_surplus:
        return None
    return Order(
        price=float(best_ask),
        quantity=1,
        agent_id=agent_id,
        time=time,
        order_type=BUY,
        order_id=rng.randint(1, 10_000_000),
    )


def maybe_take_best_quote_sell(
    *,
    market,
    seller_cost: float,
    eta: float,
    agent_id: int,
    time: int,
    rng: random.Random,
) -> Optional[Order]:
    best_bid = market.order_book.get_best_bid()
    if np.isinf(best_bid):
        return None
    surplus_from_taking = float(best_bid) - float(seller_cost)
    requested_surplus = -float(seller_cost)
    if requested_surplus <= 0 or surplus_from_taking < eta * requested_surplus:
        return None
    return Order(
        price=float(best_bid),
        quantity=1,
        agent_id=agent_id,
        time=time,
        order_type=SELL,
        order_id=rng.randint(1, 10_000_000),
    )
