"""
Episode evaluation: Run an LLM agent inside the real mixed-market simulator.

The LLM agent subclasses ZIAgentBuy/ZIAgentSell so it gets the same
SeparatedPrivateValues draw as every other ZI agent. Its take_action()
calls the LLM strategy instead of random shading, but all other mechanics
(surplus via pv.consume_marginal, update_position, order expiry) are
identical to mixed_market_simulator.run_mixed_agent_test.

Price convention: all order prices are RELATIVE to fundamental (~0, ±500).
"""

import random
from typing import Callable, Dict, List, Any, Optional, Set

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.LLM.mixed_market_simulator import _pair_matched_orders, _trade_price_from_orders
from marketsim.LLM.utils import safe_strategy_call, parse_strategy_output


# ---------------------------------------------------------------------------
# LOB snapshot helper
# ---------------------------------------------------------------------------

def _lob_snapshot(market, n: int = 4) -> dict:
    """
    Capture the current limit order book state as relative prices.

    All prices in the market are already stored relative to fundamental (~0),
    so no offset subtraction is needed.  Returns:
      best_bid_relative / best_ask_relative : float | null
      bids_relative / asks_relative         : list of top-n prices
    """
    ob = market.order_book
    best_bid = ob.get_best_bid()
    best_ask = ob.get_best_ask()

    # order_dict maps order_id -> Order; Order.price is the relative price
    raw_bids = sorted(
        [o.price for o in ob.buy_unmatched.order_dict.values()],
        reverse=True,
    )[:n]
    raw_asks = sorted(
        [o.price for o in ob.sell_unmatched.order_dict.values()]
    )[:n]

    def rel(p):
        return None if np.isinf(p) else round(float(p), 2)

    return {
        "best_bid_relative": rel(best_bid),
        "best_ask_relative": rel(best_ask),
        "bids_relative": [rel(p) for p in raw_bids if not np.isinf(p)],
        "asks_relative": [rel(p) for p in raw_asks if not np.isinf(p)],
    }


# ---------------------------------------------------------------------------
# LLM Agent wrappers
# ---------------------------------------------------------------------------

class LLMBuyAgent(ZIAgentBuy):
    """
    ZIAgentBuy whose bid is set by an LLM strategy function.

    Inherits SeparatedPrivateValues from ZIAgentBuy — same real pv draw.
    The strategy receives relative private values from self.pv and returns
    a relative bid price.
    """
    def __init__(self, agent_id, market, q_max, pv_var, obs_noise_var, strategy_func,
                 capture_detail: bool = False, generator=None):
        super().__init__(
            agent_id=agent_id, market=market, q_max=q_max,
            shade=[0, 0], pv_var=pv_var, obs_noise_var=obs_noise_var,
            generator=generator,
        )
        self.strategy_func = strategy_func
        self.capture_detail = capture_detail
        self.trades_detail: List[Dict] = []
        self._market_trades_window: List[Dict] = []
        # Pending order tracking for capture_detail: maps order_id -> submitted_price_relative
        self._pending_orders: Dict[int, float] = {}

    def take_action(self):
        if self.position >= self.q_max:
            return []

        # Real relative private value for this unit (same as ZI would use)
        rel_valuation = self.pv.value_for_exchange(self.position, BUY)

        t = self.market.get_time()
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()

        # Expose relative private values schedule to strategy
        rel_pvs = [self.pv.value_for_exchange(pos, BUY) for pos in range(self.q_max)]

        market_state = {
            "best_bid": None if np.isinf(best_bid) else float(best_bid),
            "best_ask": None if np.isinf(best_ask) else float(best_ask),
            "units_available": self.q_max - self.position,
            "q_max": self.q_max,
            "fundamental_estimate": float(self.estimate_fundamental()),
            "max_bid": float(rel_valuation),   # relative; bid = max_bid - shade
            "min_ask": float(rel_valuation),
        }
        market_history = {
            "timestep": t,
            "market_trades": self._market_trades_window[-10:],
            "total_trades": len(self._market_trades_window),
        }

        result = safe_strategy_call(self.strategy_func, rel_pvs, market_history, market_state)
        parsed = parse_strategy_output(result)
        if parsed is None:
            return []

        # shade is used directly as the bid shading amount
        # bid = private_value - shade  (rationality guaranteed when shade >= 0)
        shade, qty = parsed
        rel_price = rel_valuation - shade
        oid = random.randint(1, 10_000_000)
        if self.capture_detail:
            self._pending_orders[oid] = shade
        return [Order(price=rel_price, quantity=qty, agent_id=self.agent_id,
                      time=t, order_type=BUY, order_id=oid)]


class LLMSellAgent(ZIAgentSell):
    """
    ZIAgentSell whose ask is set by an LLM strategy function.
    """
    def __init__(self, agent_id, market, q_max, pv_var, obs_noise_var, strategy_func,
                 capture_detail: bool = False, generator=None):
        super().__init__(
            agent_id=agent_id, market=market, q_max=q_max,
            shade=[0, 0], pv_var=pv_var, obs_noise_var=obs_noise_var,
            generator=generator,
        )
        self.strategy_func = strategy_func
        self.capture_detail = capture_detail
        self.trades_detail: List[Dict] = []
        self._market_trades_window: List[Dict] = []
        self._pending_orders: Dict[int, float] = {}

    def take_action(self):
        if self.position <= -self.q_max:
            return []

        rel_valuation = self.pv.value_for_exchange(self.position, SELL)
        t = self.market.get_time()
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()

        rel_pvs = [self.pv.value_for_exchange(-pos, SELL) for pos in range(self.q_max)]

        market_state = {
            "best_bid": None if np.isinf(best_bid) else float(best_bid),
            "best_ask": None if np.isinf(best_ask) else float(best_ask),
            "units_available": self.q_max + self.position,
            "q_max": self.q_max,
            "fundamental_estimate": float(self.estimate_fundamental()),
            "max_bid": float(rel_valuation),
            "min_ask": float(rel_valuation),
        }
        market_history = {
            "timestep": t,
            "market_trades": self._market_trades_window[-10:],
            "total_trades": len(self._market_trades_window),
        }

        result = safe_strategy_call(self.strategy_func, rel_pvs, market_history, market_state)
        parsed = parse_strategy_output(result)
        if parsed is None:
            return []

        # shade is used directly as the ask shading amount
        # ask = private_value + shade  (rationality guaranteed when shade >= 0)
        shade, qty = parsed
        rel_price = rel_valuation + shade
        oid = random.randint(1, 10_000_000)
        if self.capture_detail:
            self._pending_orders[oid] = shade
        return [Order(price=rel_price, quantity=qty, agent_id=self.agent_id,
                      time=t, order_type=SELL, order_id=oid)]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _apply_fixed_private_values(agent, order_type: int, values: Optional[List[float]], q_max: int) -> None:
    """
    Override the LLM agent's private-value schedule when a fixed schedule is provided.

    For buyer runs, values are interpreted as the buyer's marginal values for units
    0..q_max-1 on the relative-price scale. For seller runs, values are interpreted
    as seller marginal costs.
    """
    if values is None:
        return

    fixed_vals = [float(v) for v in values]
    if len(fixed_vals) != q_max:
        raise ValueError(
            f"llm_private_values must have length q_max ({q_max}), got {len(fixed_vals)}"
        )

    import torch

    tensor_vals = torch.tensor(fixed_vals, dtype=torch.float32)
    if order_type == BUY:
        agent.pv.buyer_values = tensor_vals
        agent.pv.extra_buy = float(tensor_vals[-1].item())
    else:
        agent.pv.seller_costs = tensor_vals
        agent.pv.extra_sell = float(tensor_vals[-1].item())


def run_episode(
    llm_strategy_func: Callable,
    llm_order_type: int,
    llm_private_values: Optional[List[float]] = None,
    num_zi_buy: int = 5,
    num_zi_sell: int = 5,
    timesteps: int = 1000,
    q_max: int = 10,
    pv_var: float = 1e5,
    shade_range: Optional[List[float]] = None,
    fundamental_value: float = 100000.0,
    mean_reversion_rate: float = 0.05,
    shock_var: float = 1e5,
    obs_noise_var: float = 1e6,
    arrival_rate: float = 0.1,
    random_seed: Optional[int] = None,
    capture_detail: bool = False,
) -> Dict[str, Any]:
    """
    Run one episode using the real mixed-market simulator.

    One LLMBuyAgent (or LLMSellAgent) is injected alongside the ZI agents.
    Its private values are drawn from SeparatedPrivateValues(pv_var) — the
    same distribution as every other agent in the simulation.
    Surplus is calculated identically to mixed_market_simulator via pv.consume_marginal.

    When capture_detail=True the returned dict includes a "diagnostic_json" key
    with a unified event log (sorted by time):
      - event_type="trade": every trade (agent-involved or background ZI-ZI),
        with LOB snapshot and trade price.
      - event_type="agent_order": LLM agent order submissions that did NOT
        result in a trade that same timestep (order sat in book unexecuted).
    No opponent internals (pv_var, shade_range, agent counts) are included.
    """
    from marketsim.market.market import Market
    from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting

    if shade_range is None:
        shade_range = [0, 500]

    # ── Scoped RNGs — never touch global state ───────────────────────────
    import torch as _torch
    torch_gen = _torch.Generator()
    if random_seed is not None:
        torch_gen.manual_seed(random_seed)
    else:
        torch_gen.seed()

    rng = np.random.default_rng(random_seed)
    py_rng = random.Random(random_seed)
    # ────────────────────────────────────────────────────────────────────

    fundamental = LazyGaussianMeanReverting(
        final_time=timesteps, mean=fundamental_value,
        r=mean_reversion_rate, shock_var=shock_var,
        generator=torch_gen,
    )
    market = Market(fundamental=fundamental, time_steps=timesteps)


    agents: Dict[str, object] = {}
    id_to_agent: Dict[int, object] = {}
    agent_types: Dict[int, str] = {}
    next_id = 0

    for _ in range(num_zi_buy):
        ag = ZIAgentBuy(agent_id=next_id, market=market, q_max=q_max,
                        shade=shade_range, pv_var=pv_var, obs_noise_var=obs_noise_var,
                        generator=torch_gen, episode_seed=random_seed or 0)
        agents[f"zi_buy_{next_id}"] = ag
        id_to_agent[next_id] = ag
        agent_types[next_id] = "zi_buy"
        next_id += 1

    for _ in range(num_zi_sell):
        ag = ZIAgentSell(agent_id=next_id, market=market, q_max=q_max,
                         shade=shade_range, pv_var=pv_var, obs_noise_var=obs_noise_var,
                         generator=torch_gen, episode_seed=random_seed or 0)
        agents[f"zi_sell_{next_id}"] = ag
        id_to_agent[next_id] = ag
        agent_types[next_id] = "zi_sell"
        next_id += 1

    llm_agent_id = next_id
    if llm_order_type == BUY:
        llm_agent = LLMBuyAgent(
            agent_id=llm_agent_id, market=market, q_max=q_max,
            pv_var=pv_var, obs_noise_var=obs_noise_var,
            strategy_func=llm_strategy_func,
            capture_detail=capture_detail,
            generator=torch_gen,
        )
        agents[f"llm_buy_{llm_agent_id}"] = llm_agent
        agent_types[llm_agent_id] = "llm_buy"
    else:
        llm_agent = LLMSellAgent(
            agent_id=llm_agent_id, market=market, q_max=q_max,
            pv_var=pv_var, obs_noise_var=obs_noise_var,
            strategy_func=llm_strategy_func,
            capture_detail=capture_detail,
            generator=torch_gen,
        )
        agents[f"llm_sell_{llm_agent_id}"] = llm_agent
        agent_types[llm_agent_id] = "llm_sell"
    id_to_agent[llm_agent_id] = llm_agent

    # Optional fixed private-value schedule (part of the RL state in phase 1).
    _apply_fixed_private_values(llm_agent, llm_order_type, llm_private_values, q_max)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    total_trades = 0
    agent_surplus = {aid: 0.0 for aid in id_to_agent}
    agent_last_order_time = {aid: -1 for aid in id_to_agent}
    order_lifetime = int(1.0 / arrival_rate) if arrival_rate > 0 else None
    llm_shade_sum = 0.0
    llm_shade_count = 0

    # capture_detail accumulators
    all_events: List[Dict] = []
    llm_traded_timesteps: Set[int] = set()

    for t in range(timesteps):
        market.event_queue.set_time(t)
        _ = market.get_fundamental_value()

        if order_lifetime is not None:
            for aid in id_to_agent:
                last_t = agent_last_order_time[aid]
                if last_t >= 0 and (t - last_t) >= order_lifetime:
                    market.withdraw_all(aid)
                    agent_last_order_time[aid] = -1

        active = [(name, ag) for name, ag in agents.items()
                  if rng.random() < arrival_rate]
        py_rng.shuffle(active)

        # Track LLM order submission this timestep (for non-trade event recording)
        llm_submitted_this_t: Optional[Dict] = None

        for _, ag in active:
            aid = ag.get_id()
            market.withdraw_all(aid)
            orders = ag.take_action()
            market.add_orders(orders)
            agent_last_order_time[aid] = t

            # Record LLM agent's order submission (LOB captured after order is added)
            if capture_detail and aid == llm_agent_id and orders:
                order_id = orders[0].order_id
                shade_val = llm_agent._pending_orders.get(order_id, 0.0)
                lob = _lob_snapshot(market)
                llm_submitted_this_t = {
                    "time": t,
                    "event_type": "agent_order",
                    "agent_transacted": False,
                    "agent_bid": True,
                    "agent_action": {
                        "shade": round(float(shade_val), 4),
                    },
                    "lob": lob,
                }

        matched_orders = market.step()
        paired_trades = _pair_matched_orders(matched_orders)

        for trade in paired_trades.values():
            buyer_order = trade.get("buyer_order")
            seller_order = trade.get("seller_order")
            trade_price, price_setter_side = _trade_price_from_orders(buyer_order, seller_order)
            if trade_price is None:
                continue

            trade_price_shifted = trade_price + fundamental_value

            buyer_id = trade.get("buyer_id")
            seller_id = trade.get("seller_id")
            if buyer_id is None or seller_id is None:
                continue

            buyer = id_to_agent.get(buyer_id)
            seller = id_to_agent.get(seller_id)
            if buyer is None or seller is None:
                continue

            buyer_pos = buyer.position
            seller_pos = seller.position

            buyer_value = buyer.pv.consume_marginal(buyer.position, BUY) + fundamental_value
            seller_cost = seller.pv.consume_marginal(seller.position, SELL) + fundamental_value

            buyer_surplus_val = buyer_value - trade_price_shifted
            seller_surplus_val = trade_price_shifted - seller_cost
            agent_surplus[buyer_id] += buyer_surplus_val
            agent_surplus[seller_id] += seller_surplus_val

            if isinstance(buyer, LLMBuyAgent):
                rel_val = buyer.pv.value_for_exchange(buyer_pos, BUY)
                shade = float(rel_val - trade_price)
                llm_shade_sum += shade
                llm_shade_count += 1
                buyer._market_trades_window.append({"time": t, "price": float(trade_price)})
                if buyer.capture_detail:
                    llm_traded_timesteps.add(t)
                    shade_val = buyer._pending_orders.pop(
                        next(iter(buyer._pending_orders), None), shade
                    ) if buyer._pending_orders else shade
                    lob = _lob_snapshot(market)
                    all_events.append({
                        "time": t,
                        "event_type": "trade",
                        "agent_transacted": True,
                        "agent_bid": True,
                        "trade_price_relative": round(float(trade_price), 4),
                        "agent_action": {
                            "shade": round(float(shade_val), 4),
                            "surplus_earned": round(float(buyer_surplus_val), 4),
                        },
                        "lob": lob,
                    })

            elif isinstance(seller, LLMSellAgent):
                rel_val = seller.pv.value_for_exchange(seller_pos, SELL)
                shade = float(trade_price - rel_val)
                llm_shade_sum += shade
                llm_shade_count += 1
                seller._market_trades_window.append({"time": t, "price": float(trade_price)})
                if seller.capture_detail:
                    llm_traded_timesteps.add(t)
                    shade_val = seller._pending_orders.pop(
                        next(iter(seller._pending_orders), None), shade
                    ) if seller._pending_orders else shade
                    lob = _lob_snapshot(market)
                    all_events.append({
                        "time": t,
                        "event_type": "trade",
                        "agent_transacted": True,
                        "agent_bid": True,
                        "trade_price_relative": round(float(trade_price), 4),
                        "agent_action": {
                            "shade": round(float(shade_val), 4),
                            "surplus_earned": round(float(seller_surplus_val), 4),
                        },
                        "lob": lob,
                    })

            elif capture_detail:
                # Background ZI-ZI trade — useful LOB + price context for the LLM
                lob = _lob_snapshot(market)
                all_events.append({
                    "time": t,
                    "event_type": "trade",
                    "agent_transacted": False,
                    "agent_bid": False,
                    "trade_price_relative": round(float(trade_price), 4),
                    "lob": lob,
                })

            buyer.update_position(1, -trade_price_shifted)
            seller.update_position(-1, trade_price_shifted)
            total_trades += 1

        # If the LLM submitted an order this timestep but didn't execute a trade,
        # record the submission as an "agent_order" event (order sat in book).
        if capture_detail and llm_submitted_this_t is not None:
            if t not in llm_traded_timesteps:
                all_events.append(llm_submitted_this_t)

    llm_surplus = agent_surplus[llm_agent_id]
    llm_units_executed = len(llm_agent._market_trades_window)
    per_unit_surplus = llm_surplus / llm_units_executed if llm_units_executed > 0 else 0.0
    avg_shade = llm_shade_sum / llm_shade_count if llm_shade_count > 0 else None

    same_side_type = "zi_buy" if llm_order_type == BUY else "zi_sell"
    zi_same_surpluses = [
        agent_surplus[aid] for aid, t in agent_types.items() if t == same_side_type
    ]
    avg_zi_same_surplus = sum(zi_same_surpluses) / len(zi_same_surpluses) if zi_same_surpluses else 0.0

    llm_rel_pvs = [llm_agent.pv.value_for_exchange(pos, llm_order_type) for pos in range(q_max)]

    # Consumer surplus: sum of all positive private values in the agent's endowment.
    # This is the theoretical maximum value available, independent of how many units
    # were actually executed — so it is fixed per seed and identical across strategies.
    consumer_surplus = sum(pv for pv in llm_rel_pvs if pv > 0)

    result: Dict[str, Any] = {
        "llm_surplus": llm_surplus,
        "consumer_surplus": consumer_surplus,
        "llm_units_executed": llm_units_executed,
        "llm_units_available": q_max,
        "per_unit_surplus": per_unit_surplus,
        "avg_shade": avg_shade,
        "avg_zi_same_surplus": avg_zi_same_surplus,
        "llm_private_values_relative": llm_rel_pvs,
        "episode_stats": {
            "total_trades": total_trades,
            "execution_ratio": llm_units_executed / max(1, q_max),
        },
        # Internal — stripped before sending to LLM
        "market_config": {
            "num_zi_buy": num_zi_buy,
            "num_zi_sell": num_zi_sell,
            "timesteps": timesteps,
            "q_max": q_max,
            "pv_var": pv_var,
            "shade_range": shade_range,
        },
    }

    # Unified diagnostic JSON — only populated when capture_detail=True.
    # Schema: {agent_type, total_transactions, agent_transactions,
    #          agent_total_surplus, timesteps, events: [...]}
    # Events sorted by time; no opponent internals.
    if capture_detail:
        agent_type_str = "buyer" if llm_order_type == BUY else "seller"
        all_events_sorted = sorted(all_events, key=lambda e: e["time"])
        result["diagnostic_json"] = {
            "agent_type": agent_type_str,
            "agent_private_values_relative": llm_rel_pvs,  # agent's own pv schedule
            "total_transactions": total_trades,
            "agent_transactions": llm_units_executed,
            "agent_total_surplus": round(llm_surplus, 4),
            "timesteps": timesteps,
            "events": all_events_sorted,
        }

    return result
