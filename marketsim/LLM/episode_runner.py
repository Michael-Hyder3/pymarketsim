import random
from typing import Callable, Dict, List, Any, Optional, Tuple

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.LLM.background_agents import (
    DEFAULT_BACKGROUND_AGENT_KIND,
    normalize_background_agent_kind,
)
from marketsim.LLM.mixed_market_simulator import _pair_matched_orders, _trade_price_from_orders
from marketsim.LLM.utils import safe_strategy_call, parse_strategy_output
from marketsim.LLM.hbl_order_memory import (
    DEFAULT_HBL_L,
    serialize_hbl_order_memory,
    strategy_requests_hbl_exact_seed,
)
from marketsim.LLM.lob_features import (
    is_lob_book_orders_mode,
    is_lob_hbl_parity_mode,
    normalize_lob_input_mode,
)
from marketsim.LLM.spread_input import (
    DEFAULT_SPREAD_INPUT_MODE,
    build_spread_market_state,
    is_spread_book_mode,
    is_spread_zip_parity_mode,
    normalize_spread_input_mode,
    spread_pricing_fundamental,
)


def _shade_from_relative_limit_buy(
    *,
    relative_price: float,
    private_value: float,
    fundamental_abs: float,
) -> float:
    denom = float(fundamental_abs) + float(private_value)
    if denom <= 0.0:
        return 0.0
    return min(1.0, max(0.0, (float(private_value) - float(relative_price)) / denom))




def _ensure_hbl_buy_proxy(
    agent: "LLMBuyAgent",
    *,
    shade_range: Optional[List[float]],
    episode_seed: int,
):
    if getattr(agent, "_hbl_proxy", None) is not None:
        return agent._hbl_proxy
    from marketsim.agent.hbl_agent_buy_sell_fast import HBLAgentBuyFast

    shade = list(shade_range) if shade_range is not None else [0.0, 0.8]
    proxy = HBLAgentBuyFast(
        agent_id=agent.agent_id,
        market=agent.market,
        q_max=agent.q_max,
        shade=shade,
        L=int(agent.lob_memory_L),
        pv_var=float(agent.pv_var),
        arrival_rate=float(agent.arrival_rate),
        obs_noise_var=float(getattr(agent, "obs_noise_var", 0.0) or 0.0),
        generator=getattr(agent, "_generator", None),
        episode_seed=int(episode_seed),
    )
    proxy.pv = agent.pv
    agent._hbl_proxy = proxy
    return proxy




def _take_action_hbl_exact_buy(agent: "LLMBuyAgent") -> List:
    if agent.position >= agent.q_max:
        return []
    proxy = _ensure_hbl_buy_proxy(
        agent,
        shade_range=agent.hbl_shade_range,
        episode_seed=agent.hbl_episode_seed,
    )
    proxy.pv = agent.pv
    proxy.position = int(agent.position)
    proxy.cash = float(getattr(agent, "cash", 0.0) or 0.0)
    proxy.consumed_buy_positions = set(range(int(agent.position)))
    orders = proxy.take_action(seed=0)
    decision = getattr(proxy, "_last_decision", None) or {}
    shade = decision.get("shade_fraction")
    if shade is None and orders:
        rel_valuation = float(agent.pv.value_for_exchange(agent.position, BUY))
        fund_abs = float(agent.estimate_fundamental())
        shade = _shade_from_relative_limit_buy(
            relative_price=float(orders[0].price),
            private_value=rel_valuation,
            fundamental_abs=fund_abs,
        )
    if shade is not None:
        requested_x = min(1.0, max(0.0, float(shade)))
        agent._requested_x_sum += requested_x
        agent._requested_x_count += 1
        if agent.capture_detail and orders:
            agent._pending_orders[int(orders[0].order_id)] = requested_x
    return orders




def _broadcast_zip_shout(
    id_to_agent: Dict[int, object],
    *,
    price_abs: float,
    side: int,
    accepted: bool,
    price_rel: Optional[float] = None,
) -> None:
    for ag in id_to_agent.values():
        observe = getattr(ag, "observe_zip_shout", None)
        if callable(observe):
            kwargs = dict(
                price_abs=float(price_abs),
                side=int(side),
                accepted=bool(accepted),
            )
            if price_rel is not None:
                kwargs["price_rel"] = float(price_rel)
            observe(**kwargs)


def _append_llm_shout_log(
    llm_agent: object,
    *,
    time: int,
    price_abs: float,
    price_rel: float,
    side: int,
    accepted: bool,
    is_own: bool = False,
) -> None:
    record = {
        "time": int(time),
        "price_abs": float(price_abs),
        "price_rel": float(price_rel),
        "side": int(side),
        "accepted": bool(accepted),
    }
    log = getattr(llm_agent, "_shout_log", None)
    if log is not None:
        log.append(dict(record))
    if is_own:
        own_log = getattr(llm_agent, "_own_event_log", None)
        if own_log is not None:
            own_log.append(dict(record))


def _build_spread_arrival_schedule(
    *,
    agents: Dict[str, object],
    timesteps: int,
    arrival_rate: float,
    rng: np.random.RandomState,
    py_rng: random.Random,
) -> Dict[int, object]:
    schedule: Dict[int, object] = {}
    if arrival_rate <= 0:
        return schedule

    for t in range(timesteps):
        arrivals = [
            (name, ag)
            for name, ag in agents.items()
            if rng.rand() < arrival_rate
        ]
        py_rng.shuffle(arrivals)
        slot = t
        for _, ag in arrivals:
            while slot < timesteps and slot in schedule:
                slot += 1
            if slot >= timesteps:
                break
            schedule[slot] = ag
            slot += 1
    return schedule


def _lob_snapshot(market, n: Optional[int] = None) -> dict:
    ob = market.order_book
    best_bid = ob.get_best_bid()
    best_ask = ob.get_best_ask()

    bid_orders = sorted(
        ob.buy_unmatched.order_dict.values(),
        key=lambda o: (-float(o.price), int(o.time)),
    )
    ask_orders = sorted(
        ob.sell_unmatched.order_dict.values(),
        key=lambda o: (float(o.price), int(o.time)),
    )
    if n is not None:
        bid_orders = bid_orders[: max(0, int(n))]
        ask_orders = ask_orders[: max(0, int(n))]

    def rel(p):
        return None if np.isinf(p) else round(float(p), 2)

    def resting(orders):
        out = []
        for o in orders:
            price = rel(o.price)
            if price is None:
                continue
            out.append(
                {
                    "price_rel": price,
                    "quantity": float(o.quantity),
                }
            )
        return out

    bids = resting(bid_orders)
    asks = resting(ask_orders)
    return {
        "best_bid_relative": rel(best_bid),
        "best_ask_relative": rel(best_ask),
        "bids": bids,
        "asks": asks,
        "bids_relative": [float(e["price_rel"]) for e in bids],
        "asks_relative": [float(e["price_rel"]) for e in asks],
    }


def _empty_lob_snapshot() -> Dict[str, Any]:
    return {
        "best_bid_relative": None,
        "best_ask_relative": None,
        "bids": [],
        "asks": [],
        "bids_relative": [],
        "asks_relative": [],
    }


def _lob_derived_fields(lob: Dict[str, Any]) -> Dict[str, Any]:
    best_bid_rel = lob.get("best_bid_relative")
    best_ask_rel = lob.get("best_ask_relative")
    spread_rel = (
        float(best_ask_rel - best_bid_rel)
        if best_bid_rel is not None and best_ask_rel is not None
        else None
    )
    mid_rel = (
        float(0.5 * (best_bid_rel + best_ask_rel))
        if best_bid_rel is not None and best_ask_rel is not None
        else None
    )
    return {
        "best_bid_relative": best_bid_rel,
        "best_ask_relative": best_ask_rel,
        "bids": list(lob.get("bids") or []),
        "asks": list(lob.get("asks") or []),
        "bids_relative": lob.get("bids_relative", []),
        "asks_relative": lob.get("asks_relative", []),
        "spread_relative": spread_rel,
        "mid_relative": mid_rel,
    }


def _build_market_history(
    *,
    timestep: int,
    lob: Dict[str, Any],
    market_trades_window: List[Dict[str, Any]],
    lob_snapshots: Optional[List[Dict[str, Any]]] = None,
    hbl_order_memory: Optional[Dict[str, Any]] = None,
    current_pv: Optional[float] = None,
    lob_input_mode: str = "raw",
    grace_period: Optional[float] = None,
    side: Optional[int] = None,
    fundamental_estimate: Optional[float] = None,
    pricing_fundamental: Optional[float] = None,
) -> Dict[str, Any]:
    from marketsim.LLM.lob_features import apply_lob_input_mode

    recent_trades = market_trades_window[-10:]
    recent_prices = [float(tr["price"]) for tr in recent_trades if "price" in tr]
    derived = _lob_derived_fields(lob)
    history: Dict[str, Any] = {
        "timestep": timestep,
        "market_trades": recent_trades,
        "total_trades": len(market_trades_window),
        "recent_trade_prices": recent_prices,
        "last_trade_price": (recent_prices[-1] if recent_prices else None),
        "lob": lob,
        **derived,
    }
    if lob_snapshots is not None:
        history["lob_snapshots"] = lob_snapshots
    if hbl_order_memory is not None:
        history["L"] = int(hbl_order_memory.get("L", DEFAULT_HBL_L))
        history["memory_start_time"] = int(hbl_order_memory.get("memory_start_time", 0))
        history["transaction_count"] = int(hbl_order_memory.get("transaction_count", 0))
        history["orders"] = list(hbl_order_memory.get("orders") or [])
    if current_pv is not None:
        history["current_pv"] = float(current_pv)
    if grace_period is not None:
        history["grace_period"] = float(grace_period)
    if side is not None:
        history["side"] = int(side)
    if fundamental_estimate is not None:
        history["fundamental_estimate"] = float(fundamental_estimate)
    if pricing_fundamental is not None:
        history["pricing_fundamental"] = float(pricing_fundamental)
    return apply_lob_input_mode(history, lob_input_mode=lob_input_mode)


def _record_post_step_lob_history(agent: object, market: Any, *, max_snapshots: int = 20) -> None:
    history = getattr(agent, "_lob_history", None)
    if history is None:
        return
    history.append(_lob_snapshot(market))
    if len(history) > max_snapshots:
        del history[:-max_snapshots]


def _llm_action_payload(
    *,
    llm_agent: object,
    order: Order,
) -> Dict[str, Any]:
    agent_is_buy = order.order_type == BUY
    shade_val = None
    pending = getattr(llm_agent, "_pending_orders", None)
    if isinstance(pending, dict):
        shade_val = pending.get(order.order_id)
    try:
        pv_here = float(llm_agent.pv.value_for_exchange(llm_agent.position, order.order_type))
    except Exception:
        pv_here = None
    return {
        "agent_bid": bool(agent_is_buy),
        "price_relative": round(float(order.price), 4),
        "shade": (None if shade_val is None else round(float(shade_val), 4)),
        "position": int(getattr(llm_agent, "position", 0)),
        "current_pv": pv_here,
    }


def _finalize_llm_order_event(
    *,
    pending: Dict[str, Any],
    market: Any,
    transacted: bool,
    trade_price_relative: Optional[float] = None,
    surplus_earned: Optional[float] = None,
) -> Dict[str, Any]:
    post_lob = _lob_snapshot(market)
    event = {
        "time": pending["time"],
        "event_type": "llm_order",
        "pre_lob": pending["pre_lob"],
        "agent_action": pending["agent_action"],
        "post_lob": post_lob,
        "transacted": bool(transacted),
        "agent_transacted": bool(transacted),
        "agent_bid": pending["agent_action"]["agent_bid"],
        "lob": post_lob,
    }
    if trade_price_relative is not None:
        event["trade_price_relative"] = round(float(trade_price_relative), 4)
    if surplus_earned is not None:
        event["surplus_earned"] = round(float(surplus_earned), 4)
    return event


def _collect_market_valuations(
    *,
    buyers: List[Any],
    sellers: List[Any],
) -> Tuple[List[float], List[float]]:
    bids: List[float] = []
    asks: List[float] = []
    for buyer in buyers:
        vals = buyer.pv.buyer_values
        bids.extend([float(v.item()) for v in vals])
    for seller in sellers:
        vals = seller.pv.seller_costs
        asks.extend([float(v.item()) for v in vals])
    return bids, asks


def _equilibrium_price(bids: List[float], asks: List[float]) -> Optional[float]:
    if not bids or not asks:
        return None
    bids_sorted = sorted(bids, reverse=True)
    asks_sorted = sorted(asks)
    k_star = None
    for i in range(min(len(bids_sorted), len(asks_sorted))):
        if bids_sorted[i] >= asks_sorted[i]:
            k_star = i
    if k_star is None:
        return None
    return 0.5 * (bids_sorted[k_star] + asks_sorted[k_star])


def _baseline_surplus(
    *,
    llm_values: List[float],
    equilibrium_price: Optional[float],
    order_type: int,
) -> float:
    if equilibrium_price is None:
        return 0.0
    if order_type == BUY:
        return float(
            sum(
                val - equilibrium_price
                for val in llm_values
                if val >= equilibrium_price
            )
        )
    return float(
        sum(
            equilibrium_price - val
            for val in llm_values
            if val <= equilibrium_price
        )
    )


class LLMBuyAgent(ZIAgentBuy):
    def __init__(self, agent_id, market, q_max, pv_var, obs_noise_var, strategy_func,
                 capture_detail: bool = False, generator=None, action_mode: str = "shade",
                 strategy_type: str = "full", lob_input_mode: str = "raw",
                 lob_memory_L: int = DEFAULT_HBL_L,
                 spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
                 spread_omit_fundamental: bool = False,
                 arrival_rate: float = 0.1,
                 hbl_exact_seed: bool = False,
                 hbl_shade_range: Optional[List[float]] = None,
                 hbl_episode_seed: int = 0):
        super().__init__(
            agent_id=agent_id, market=market, q_max=q_max,
            shade=[0, 0], pv_var=pv_var, obs_noise_var=obs_noise_var,
            generator=generator,
        )
        self.strategy_func = strategy_func
        self.capture_detail = capture_detail
        self.action_mode = (action_mode or "shade").lower()
        self.strategy_type = strategy_type
        self.lob_input_mode = normalize_lob_input_mode(lob_input_mode)
        self.lob_memory_L = max(1, int(lob_memory_L))
        self.spread_input_mode = normalize_spread_input_mode(spread_input_mode)
        self.spread_omit_fundamental = bool(spread_omit_fundamental)
        arr = float(arrival_rate) if arrival_rate else 0.1
        self.arrival_rate = arr if arr > 0.0 else 0.1
        self.grace_period = 1.0 / self.arrival_rate
        self.hbl_exact_seed = bool(hbl_exact_seed) or strategy_requests_hbl_exact_seed(
            strategy_func
        )
        self.hbl_shade_range = (
            list(hbl_shade_range) if hbl_shade_range is not None else None
        )
        self.hbl_episode_seed = int(hbl_episode_seed or 0)
        self._hbl_proxy = None
        self.trades_detail: List[Dict] = []
        self._market_trades_window: List[Dict] = []
        self._lob_history: List[Dict[str, Any]] = []
        self._shout_log: List[Dict[str, Any]] = []
        self._own_event_log: List[Dict[str, Any]] = []
        self._pending_orders: Dict[int, float] = {}
        self._requested_x_sum = 0.0
        self._requested_x_count = 0

    def take_action(self):
        if self.hbl_exact_seed:
            return _take_action_hbl_exact_buy(self)

        if self.position >= self.q_max:
            return []

        rel_valuation = self.pv.value_for_exchange(self.position, BUY)

        t = self.market.get_time()
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()

        rel_pvs = [self.pv.value_for_exchange(pos, BUY) for pos in range(self.q_max)]

        best_bid_f = None if np.isinf(best_bid) else float(best_bid)
        best_ask_f = None if np.isinf(best_ask) else float(best_ask)
        fundamental_estimate = float(self.estimate_fundamental())
        fundamental_true = float(self.market.get_fundamental_value())
        units_available = self.q_max - self.position
        if self.strategy_type == "spread":
            prev_event = (
                self._own_event_log[-1] if self._own_event_log else None
            )
            live_lob = (
                _lob_snapshot(self.market)
                if is_spread_book_mode(self.spread_input_mode)
                else None
            )
            market_state = build_spread_market_state(
                best_bid=best_bid_f,
                best_ask=best_ask_f,
                units_available=units_available,
                q_max=self.q_max,
                current_pv=float(rel_valuation),
                fundamental_estimate=fundamental_estimate,
                fundamental_true=fundamental_true,
                shouts=list(self._shout_log),
                spread_input_mode=self.spread_input_mode,
                prev_event=prev_event,
                bids=(live_lob or {}).get("bids"),
                asks=(live_lob or {}).get("asks"),
                omit_fundamental=self.spread_omit_fundamental,
                private_values=list(rel_pvs),
            )
        else:
            market_state = {
                "best_bid": best_bid_f,
                "best_ask": best_ask_f,
                "units_available": self.q_max - self.position,
                "q_max": self.q_max,
                "fundamental_estimate": fundamental_estimate,
                "current_pv": float(rel_valuation),
            }
        if self.strategy_type == "lob":
            if is_lob_hbl_parity_mode(self.lob_input_mode) or is_lob_book_orders_mode(
                self.lob_input_mode
            ):
                decision_lob = _lob_snapshot(self.market)
            else:
                decision_lob = (
                    self._lob_history[-1]
                    if self._lob_history
                    else _empty_lob_snapshot()
                )
            hbl_mem = serialize_hbl_order_memory(self.market, L=self.lob_memory_L)
            fund_est_rel = None
            pricing_f = None
            if is_lob_book_orders_mode(self.lob_input_mode):
                pricing_f = float(fundamental_estimate)
                fund_est_for_hist = pricing_f
            elif is_lob_hbl_parity_mode(self.lob_input_mode):
                pricing_f = float(fundamental_estimate)
                fund_est_for_hist = pricing_f
            else:
                fund_est_for_hist = None
            market_history = _build_market_history(
                timestep=t,
                lob=decision_lob,
                market_trades_window=self._market_trades_window,
                lob_snapshots=list(self._lob_history[-10:]),
                hbl_order_memory=hbl_mem,
                current_pv=float(rel_valuation),
                lob_input_mode=self.lob_input_mode,
                grace_period=float(self.grace_period),
                side=int(BUY),
                fundamental_estimate=fund_est_for_hist,
                pricing_fundamental=pricing_f,
            )
        else:
            market_history = _build_market_history(
                timestep=t,
                lob=_lob_snapshot(self.market),
                market_trades_window=self._market_trades_window,
            )

        result = safe_strategy_call(
            self.strategy_func,
            rel_pvs,
            market_history,
            market_state,
            strategy_type=self.strategy_type,
        )
        parsed = parse_strategy_output(result)
        if parsed is None:
            return []

        raw_val, qty = parsed
        requested_x = min(1.0, max(0.0, float(raw_val)))
        self._requested_x_sum += requested_x
        self._requested_x_count += 1
        shade_fraction = requested_x
        if self.action_mode == "x_cap":
            sample_rng = random.Random((self.agent_id * 1_000_003) + int(t) + int(self.position))
            shade_fraction = sample_rng.uniform(0.0, requested_x)
        if self.strategy_type == "spread":
            F_price = spread_pricing_fundamental(
                fundamental_estimate=fundamental_estimate,
                fundamental_true=fundamental_true,
                spread_input_mode=self.spread_input_mode,
            )
        else:
            F_price = fundamental_estimate
        self._last_pricing_fundamental = float(F_price)
        full_value = F_price + rel_valuation
        shaded_full_value = full_value - (shade_fraction * full_value)
        rel_price = shaded_full_value - F_price
        oid = random.randint(1, 10_000_000)
        if self.capture_detail:
            self._pending_orders[oid] = shade_fraction
        return [Order(price=rel_price, quantity=qty, agent_id=self.agent_id,
                      time=t, order_type=BUY, order_id=oid)]


class LLMSellAgent(ZIAgentSell):
    def __init__(self, agent_id, market, q_max, pv_var, obs_noise_var, strategy_func,
                 capture_detail: bool = False, generator=None, action_mode: str = "shade",
                 strategy_type: str = "full", lob_input_mode: str = "raw",
                 lob_memory_L: int = DEFAULT_HBL_L,
                 spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
                 spread_omit_fundamental: bool = False,
                 arrival_rate: float = 0.1,
                 hbl_exact_seed: bool = False,
                 hbl_shade_range: Optional[List[float]] = None,
                 hbl_episode_seed: int = 0):
        super().__init__(
            agent_id=agent_id, market=market, q_max=q_max,
            shade=[0, 0], pv_var=pv_var, obs_noise_var=obs_noise_var,
            generator=generator,
        )
        self.strategy_func = strategy_func
        self.capture_detail = capture_detail
        self.action_mode = (action_mode or "shade").lower()
        self.strategy_type = strategy_type
        self.lob_input_mode = normalize_lob_input_mode(lob_input_mode)
        self.lob_memory_L = max(1, int(lob_memory_L))
        self.spread_input_mode = normalize_spread_input_mode(spread_input_mode)
        self.spread_omit_fundamental = bool(spread_omit_fundamental)
        arr = float(arrival_rate) if arrival_rate else 0.1
        self.arrival_rate = arr if arr > 0.0 else 0.1
        self.grace_period = 1.0 / self.arrival_rate
        self.hbl_exact_seed = bool(hbl_exact_seed) or strategy_requests_hbl_exact_seed(
            strategy_func
        )
        self.hbl_shade_range = (
            list(hbl_shade_range) if hbl_shade_range is not None else None
        )
        self.hbl_episode_seed = int(hbl_episode_seed or 0)
        self._hbl_proxy = None
        self.trades_detail: List[Dict] = []
        self._market_trades_window: List[Dict] = []
        self._lob_history: List[Dict[str, Any]] = []
        self._shout_log: List[Dict[str, Any]] = []
        self._own_event_log: List[Dict[str, Any]] = []
        self._pending_orders: Dict[int, float] = {}
        self._requested_x_sum = 0.0
        self._requested_x_count = 0

    def take_action(self):
        if self.hbl_exact_seed:
            return _take_action_hbl_exact_sell(self)

        if self.position <= -self.q_max:
            return []

        rel_valuation = self.pv.value_for_exchange(self.position, SELL)
        t = self.market.get_time()
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()

        rel_pvs = [self.pv.value_for_exchange(-pos, SELL) for pos in range(self.q_max)]

        best_bid_f = None if np.isinf(best_bid) else float(best_bid)
        best_ask_f = None if np.isinf(best_ask) else float(best_ask)
        fundamental_estimate = float(self.estimate_fundamental())
        fundamental_true = float(self.market.get_fundamental_value())
        units_available = self.q_max + self.position
        if self.strategy_type == "spread":
            prev_event = (
                self._own_event_log[-1] if self._own_event_log else None
            )
            live_lob = (
                _lob_snapshot(self.market)
                if is_spread_book_mode(self.spread_input_mode)
                else None
            )
            market_state = build_spread_market_state(
                best_bid=best_bid_f,
                best_ask=best_ask_f,
                units_available=units_available,
                q_max=self.q_max,
                current_pv=float(rel_valuation),
                fundamental_estimate=fundamental_estimate,
                fundamental_true=fundamental_true,
                shouts=list(self._shout_log),
                spread_input_mode=self.spread_input_mode,
                prev_event=prev_event,
                bids=(live_lob or {}).get("bids"),
                asks=(live_lob or {}).get("asks"),
                omit_fundamental=self.spread_omit_fundamental,
                private_values=list(rel_pvs),
            )
        else:
            market_state = {
                "best_bid": best_bid_f,
                "best_ask": best_ask_f,
                "units_available": self.q_max + self.position,
                "q_max": self.q_max,
                "fundamental_estimate": fundamental_estimate,
                "current_pv": float(rel_valuation),
            }
        if self.strategy_type == "lob":
            if is_lob_hbl_parity_mode(self.lob_input_mode) or is_lob_book_orders_mode(
                self.lob_input_mode
            ):
                decision_lob = _lob_snapshot(self.market)
            else:
                decision_lob = (
                    self._lob_history[-1]
                    if self._lob_history
                    else _empty_lob_snapshot()
                )
            hbl_mem = serialize_hbl_order_memory(self.market, L=self.lob_memory_L)
            fund_est_rel = None
            pricing_f = None
            if is_lob_book_orders_mode(self.lob_input_mode):
                pricing_f = float(fundamental_estimate)
                fund_est_for_hist = pricing_f
            elif is_lob_hbl_parity_mode(self.lob_input_mode):
                pricing_f = float(fundamental_estimate)
                fund_est_for_hist = pricing_f
            else:
                fund_est_for_hist = None
            market_history = _build_market_history(
                timestep=t,
                lob=decision_lob,
                market_trades_window=self._market_trades_window,
                lob_snapshots=list(self._lob_history[-10:]),
                hbl_order_memory=hbl_mem,
                current_pv=float(rel_valuation),
                lob_input_mode=self.lob_input_mode,
                grace_period=float(self.grace_period),
                side=int(SELL),
                fundamental_estimate=fund_est_for_hist,
                pricing_fundamental=pricing_f,
            )
        else:
            market_history = _build_market_history(
                timestep=t,
                lob=_lob_snapshot(self.market),
                market_trades_window=self._market_trades_window,
            )

        result = safe_strategy_call(
            self.strategy_func,
            rel_pvs,
            market_history,
            market_state,
            strategy_type=self.strategy_type,
        )
        parsed = parse_strategy_output(result)
        if parsed is None:
            return []

        raw_val, qty = parsed
        requested_x = min(1.0, max(0.0, float(raw_val)))
        self._requested_x_sum += requested_x
        self._requested_x_count += 1
        shade_fraction = requested_x
        if self.action_mode == "x_cap":
            sample_rng = random.Random((self.agent_id * 1_000_003) + int(t) + int(abs(self.position)))
            shade_fraction = sample_rng.uniform(0.0, requested_x)
        if self.strategy_type == "spread":
            F_price = spread_pricing_fundamental(
                fundamental_estimate=fundamental_estimate,
                fundamental_true=fundamental_true,
                spread_input_mode=self.spread_input_mode,
            )
        else:
            F_price = fundamental_estimate
        self._last_pricing_fundamental = float(F_price)
        full_value = F_price + rel_valuation
        shaded_full_value = full_value + (shade_fraction * full_value)
        rel_price = shaded_full_value - F_price
        oid = random.randint(1, 10_000_000)
        if self.capture_detail:
            self._pending_orders[oid] = shade_fraction
        return [Order(price=rel_price, quantity=qty, agent_id=self.agent_id,
                      time=t, order_type=SELL, order_id=oid)]


def _apply_fixed_private_values(agent, order_type: int, values: Optional[List[float]], q_max: int) -> None:
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
    llm_action_mode: str = "shade",
    use_dummy_fundamental: bool = False,
    strategy_type: str = "full",
    lob_input_mode: str = "raw",
    spread_input_mode: str = DEFAULT_SPREAD_INPUT_MODE,
    spread_omit_fundamental: bool = False,
    background_agent_kind: str = DEFAULT_BACKGROUND_AGENT_KIND,
    background_zip_fixed_mu: Optional[float] = None,
    background_zip_initial_mu: Optional[float] = None,
    background_zip_zi_warmup_steps: int = 25,
    background_strategy_func: Optional[Callable] = None,
    background_strategy_type: str = "pv",
    background_strategy_output_mode: str = "shade",
    special_agent_kind: str = "llm",
    special_agent_shade_range: Optional[List[float]] = None,
    special_agent_shade_sampling: str = "range",
    bo_optimizer_strategy: str = "ucb",
    bo_action_mode: str = "fixed_shade",
    bo_fixed_x: Optional[float] = None,
    hbl_l_param: int = 6,
    special_agent_hbl_l: Optional[int] = None,
) -> Dict[str, Any]:
    from marketsim.market.market import Market
    from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
    from marketsim.fundamental.dummy_fundamental import DummyFundamental

    if shade_range is None:
        shade_range = [0, 500]

    _kind = (special_agent_kind or "llm").lower()
    _background = normalize_background_agent_kind(background_agent_kind)
    _strategy_type = str(strategy_type or "full").strip().lower()
    if _background == "zip":
        _background = "zip_noisy"
    if _kind == "zip":
        _kind = "zip_noisy"
    _tracked_hbl_l = (
        max(1, int(special_agent_hbl_l))
        if special_agent_hbl_l is not None
        else max(1, int(hbl_l_param))
    )

    import torch as _torch
    torch_gen = _torch.Generator()
    if random_seed is not None:
        torch_gen.manual_seed(random_seed)
    else:
        torch_gen.seed()

    if random_seed is not None:
        rng = np.random.RandomState(int(random_seed))
        py_rng = random.Random(int(random_seed))
    else:
        rng = np.random.RandomState()
        py_rng = random.Random()

    if use_dummy_fundamental:
        fundamental = DummyFundamental(value=fundamental_value, final_time=timesteps)
    else:
        lazy_kwargs = dict(
            final_time=timesteps,
            mean=fundamental_value,
            r=mean_reversion_rate,
            shock_var=shock_var,
        )
        lazy_kwargs["generator"] = torch_gen
        fundamental = LazyGaussianMeanReverting(**lazy_kwargs)
    market = Market(fundamental=fundamental, time_steps=timesteps, rand_seed=random_seed)


    agents: Dict[str, object] = {}
    id_to_agent: Dict[int, object] = {}
    agent_types: Dict[int, str] = {}
    next_id = 0

    if _background in ("zip", "zip_noisy", "zip_fixed"):
        from marketsim.agent.zip_agent_buy_sell import (
            ZIPAgentBuy,
            ZIPAgentBuyNoisy,
            ZIPAgentSell,
            ZIPAgentSellNoisy,
            shade_hi_to_fixed_zip_mu,
            shade_range_to_zip_mu_init,
        )

        _fixed_bg = _background == "zip_fixed"
        buy_cls = (
            ZIPAgentBuyNoisy
            if _background in ("zip_noisy", "zip_fixed")
            else ZIPAgentBuy
        )
        sell_cls = (
            ZIPAgentSellNoisy
            if _background in ("zip_noisy", "zip_fixed")
            else ZIPAgentSell
        )
        tag = "zip_fixed" if _fixed_bg else (
            "zip_noisy" if _background == "zip_noisy" else "zip"
        )
        buy_mu = shade_range_to_zip_mu_init(shade_range, side=BUY)
        sell_mu = shade_range_to_zip_mu_init(shade_range, side=SELL)
        shade = list(shade_range or [0.0, 0.015])
        s_hi = float(shade[-1] if shade else 0.015)
        if _fixed_bg and background_zip_fixed_mu is not None:
            s_hi = abs(float(background_zip_fixed_mu))
        _zip_warmup = max(0, int(background_zip_zi_warmup_steps))
        _init_mu = (
            abs(float(background_zip_initial_mu))
            if background_zip_initial_mu is not None
            else None
        )

        for _ in range(num_zi_buy):
            zip_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                pv_var=pv_var,
                obs_noise_var=obs_noise_var,
                generator=torch_gen,
                episode_seed=random_seed or 0,
                zi_warmup_steps=_zip_warmup,
                shade=shade,
            )
            if _fixed_bg:
                zip_kwargs["initial_mu"] = shade_hi_to_fixed_zip_mu(s_hi, side=BUY)
                zip_kwargs["freeze_margin"] = True
            elif _init_mu is not None:
                zip_kwargs["initial_mu"] = shade_hi_to_fixed_zip_mu(
                    _init_mu, side=BUY
                )
                zip_kwargs["freeze_margin"] = False
            else:
                zip_kwargs["mu_init_range"] = buy_mu
            ag = buy_cls(**zip_kwargs)
            agents[f"{tag}_buy_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = f"{tag}_buy"
            next_id += 1

        for _ in range(num_zi_sell):
            zip_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                pv_var=pv_var,
                obs_noise_var=obs_noise_var,
                generator=torch_gen,
                episode_seed=random_seed or 0,
                zi_warmup_steps=_zip_warmup,
                shade=shade,
            )
            if _fixed_bg:
                zip_kwargs["initial_mu"] = shade_hi_to_fixed_zip_mu(s_hi, side=SELL)
                zip_kwargs["freeze_margin"] = True
            elif _init_mu is not None:
                zip_kwargs["initial_mu"] = shade_hi_to_fixed_zip_mu(
                    _init_mu, side=SELL
                )
                zip_kwargs["freeze_margin"] = False
            else:
                zip_kwargs["mu_init_range"] = sell_mu
            ag = sell_cls(**zip_kwargs)
            agents[f"{tag}_sell_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = f"{tag}_sell"
            next_id += 1
    elif _background == "hbl":
        from marketsim.agent.hbl_agent_buy_sell import HBLAgentBuy, HBLAgentSell

        for _ in range(num_zi_buy):
            hbl_bg_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                shade=shade_range,
                L=hbl_l_param,
                pv_var=pv_var,
                arrival_rate=arrival_rate,
                obs_noise_var=obs_noise_var,
                generator=torch_gen,
                episode_seed=random_seed or 0,
            )
            ag = HBLAgentBuy(**hbl_bg_kwargs)
            agents[f"hbl_buy_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "hbl_buy"
            next_id += 1

        for _ in range(num_zi_sell):
            hbl_bg_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                shade=shade_range,
                L=hbl_l_param,
                pv_var=pv_var,
                arrival_rate=arrival_rate,
                obs_noise_var=obs_noise_var,
                generator=torch_gen,
                episode_seed=random_seed or 0,
            )
            ag = HBLAgentSell(**hbl_bg_kwargs)
            agents[f"hbl_sell_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "hbl_sell"
            next_id += 1
    elif _background == "hbl_fast":
        from marketsim.agent.hbl_agent_buy_sell_fast import (
            HBLAgentBuyFast,
            HBLAgentSellFast,
        )

        for _ in range(num_zi_buy):
            hbl_bg_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                shade=shade_range,
                L=hbl_l_param,
                pv_var=pv_var,
                arrival_rate=arrival_rate,
                obs_noise_var=obs_noise_var,
                generator=torch_gen,
                episode_seed=random_seed or 0,
            )
            ag = HBLAgentBuyFast(**hbl_bg_kwargs)
            agents[f"hbl_buy_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "hbl_buy"
            next_id += 1

        for _ in range(num_zi_sell):
            hbl_bg_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                shade=shade_range,
                L=hbl_l_param,
                pv_var=pv_var,
                arrival_rate=arrival_rate,
                obs_noise_var=obs_noise_var,
                generator=torch_gen,
                episode_seed=random_seed or 0,
            )
            ag = HBLAgentSellFast(**hbl_bg_kwargs)
            agents[f"hbl_sell_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "hbl_sell"
            next_id += 1
    elif _background == "llm":
        if background_strategy_func is None:
            raise ValueError(
                "background_agent_kind='llm' requires background_strategy_func"
            )
        _bg_stype = str(background_strategy_type or "pv").strip().lower()
        _bg_mode = str(background_strategy_output_mode or "shade").strip().lower()
        _bg_hbl_exact = strategy_requests_hbl_exact_seed(background_strategy_func)
        for _ in range(num_zi_buy):
            ag = LLMBuyAgent(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                pv_var=pv_var,
                obs_noise_var=obs_noise_var,
                strategy_func=background_strategy_func,
                capture_detail=False,
                generator=torch_gen,
                action_mode=_bg_mode,
                strategy_type=_bg_stype,
                lob_input_mode=lob_input_mode,
                lob_memory_L=hbl_l_param,
                spread_input_mode=spread_input_mode,
                spread_omit_fundamental=spread_omit_fundamental,
                arrival_rate=arrival_rate,
                hbl_exact_seed=_bg_hbl_exact,
                hbl_shade_range=shade_range,
                hbl_episode_seed=random_seed or 0,
            )
            agents[f"llm_bg_buy_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "llm_bg_buy"
            next_id += 1
        for _ in range(num_zi_sell):
            ag = LLMSellAgent(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                pv_var=pv_var,
                obs_noise_var=obs_noise_var,
                strategy_func=background_strategy_func,
                capture_detail=False,
                generator=torch_gen,
                action_mode=_bg_mode,
                strategy_type=_bg_stype,
                lob_input_mode=lob_input_mode,
                lob_memory_L=hbl_l_param,
                spread_input_mode=spread_input_mode,
                spread_omit_fundamental=spread_omit_fundamental,
                arrival_rate=arrival_rate,
                hbl_exact_seed=_bg_hbl_exact,
                hbl_shade_range=shade_range,
                hbl_episode_seed=random_seed or 0,
            )
            agents[f"llm_bg_sell_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "llm_bg_sell"
            next_id += 1
    else:
        for _ in range(num_zi_buy):
            zi_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                shade=shade_range,
                pv_var=pv_var,
                obs_noise_var=obs_noise_var,
            )
            zi_kwargs["generator"] = torch_gen
            zi_kwargs["episode_seed"] = random_seed or 0
            ag = ZIAgentBuy(**zi_kwargs)
            agents[f"zi_buy_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "zi_buy"
            next_id += 1

        for _ in range(num_zi_sell):
            zi_kwargs = dict(
                agent_id=next_id,
                market=market,
                q_max=q_max,
                shade=shade_range,
                pv_var=pv_var,
                obs_noise_var=obs_noise_var,
            )
            zi_kwargs["generator"] = torch_gen
            zi_kwargs["episode_seed"] = random_seed or 0
            ag = ZIAgentSell(**zi_kwargs)
            agents[f"zi_sell_{next_id}"] = ag
            id_to_agent[next_id] = ag
            agent_types[next_id] = "zi_sell"
            next_id += 1

    llm_agent_id = next_id

    if _kind == "llm":
        hbl_exact = strategy_requests_hbl_exact_seed(llm_strategy_func)
        if llm_order_type == BUY:
            llm_agent = LLMBuyAgent(
                agent_id=llm_agent_id, market=market, q_max=q_max,
                pv_var=pv_var, obs_noise_var=obs_noise_var,
                strategy_func=llm_strategy_func,
                capture_detail=capture_detail,
                generator=torch_gen,
                action_mode=llm_action_mode,
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
                lob_memory_L=hbl_l_param,
                spread_input_mode=spread_input_mode,
                spread_omit_fundamental=spread_omit_fundamental,
                arrival_rate=arrival_rate,
                hbl_exact_seed=hbl_exact,
                hbl_shade_range=shade_range,
                hbl_episode_seed=random_seed or 0,
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
                action_mode=llm_action_mode,
                strategy_type=strategy_type,
                lob_input_mode=lob_input_mode,
                lob_memory_L=hbl_l_param,
                spread_input_mode=spread_input_mode,
                spread_omit_fundamental=spread_omit_fundamental,
                arrival_rate=arrival_rate,
                hbl_exact_seed=hbl_exact,
                hbl_shade_range=shade_range,
                hbl_episode_seed=random_seed or 0,
            )
            agents[f"llm_sell_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_sell"

    elif _kind == "zi":
        from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy as _ZIB, ZIAgentSell as _ZIS
        tracked_shade = special_agent_shade_range if special_agent_shade_range is not None else shade_range
        if llm_order_type == BUY:
            llm_agent = _ZIB(
                agent_id=llm_agent_id, market=market, q_max=q_max,
                shade=tracked_shade, pv_var=pv_var, obs_noise_var=obs_noise_var,
                generator=torch_gen, episode_seed=random_seed or 0,
                shade_sampling_mode=special_agent_shade_sampling,
            )
            agents[f"zi_buy_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_buy"
        else:
            llm_agent = _ZIS(
                agent_id=llm_agent_id, market=market, q_max=q_max,
                shade=tracked_shade, pv_var=pv_var, obs_noise_var=obs_noise_var,
                generator=torch_gen, episode_seed=random_seed or 0,
                shade_sampling_mode=special_agent_shade_sampling,
            )
            agents[f"zi_sell_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_sell"

    elif _kind == "bo":
        from marketsim.agent.bo_agent_buy_sell import BOAgentBuy, BOAgentSell
        if llm_order_type == BUY:
            llm_agent = BOAgentBuy(
                agent_id=llm_agent_id, market=market, q_max=q_max,
                shade=shade_range, pv_var=pv_var, obs_noise_var=obs_noise_var,
                optimizer_strategy=bo_optimizer_strategy,
                action_mode=bo_action_mode,
                fixed_x=bo_fixed_x,
                generator=torch_gen,
                episode_seed=random_seed or 0,
            )
            agents[f"bo_buy_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_buy"
        else:
            llm_agent = BOAgentSell(
                agent_id=llm_agent_id, market=market, q_max=q_max,
                shade=shade_range, pv_var=pv_var, obs_noise_var=obs_noise_var,
                optimizer_strategy=bo_optimizer_strategy,
                action_mode=bo_action_mode,
                fixed_x=bo_fixed_x,
                generator=torch_gen,
                episode_seed=random_seed or 0,
            )
            agents[f"bo_sell_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_sell"

    elif _kind == "hbl":
        from marketsim.agent.hbl_agent_buy_sell import HBLAgentBuy, HBLAgentSell
        hbl_kwargs = dict(
            agent_id=llm_agent_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            L=_tracked_hbl_l,
            pv_var=pv_var,
            obs_noise_var=obs_noise_var,
            arrival_rate=arrival_rate,
            episode_seed=random_seed or 0,
        )
        hbl_kwargs["generator"] = torch_gen
        if llm_order_type == BUY:
            llm_agent = HBLAgentBuy(**hbl_kwargs)
            agents[f"hbl_buy_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_buy"
        else:
            llm_agent = HBLAgentSell(**hbl_kwargs)
            agents[f"hbl_sell_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_sell"

    elif _kind == "hbl_fast":
        from marketsim.agent.hbl_agent_buy_sell_fast import (
            HBLAgentBuyFast,
            HBLAgentSellFast,
        )
        hbl_kwargs = dict(
            agent_id=llm_agent_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            L=_tracked_hbl_l,
            pv_var=pv_var,
            obs_noise_var=obs_noise_var,
            arrival_rate=arrival_rate,
            episode_seed=random_seed or 0,
        )
        hbl_kwargs["generator"] = torch_gen
        if llm_order_type == BUY:
            llm_agent = HBLAgentBuyFast(**hbl_kwargs)
            agents[f"hbl_buy_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_buy"
        else:
            llm_agent = HBLAgentSellFast(**hbl_kwargs)
            agents[f"hbl_sell_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_sell"

    elif _kind in ("zip", "zip_noisy"):
        from marketsim.agent.zip_agent_buy_sell import (
            ZIPAgentBuy,
            ZIPAgentBuyNoisy,
            ZIPAgentSell,
            ZIPAgentSellNoisy,
            shade_hi_to_fixed_zip_mu,
        )
        zip_kwargs = dict(
            agent_id=llm_agent_id,
            market=market,
            q_max=q_max,
            pv_var=pv_var,
            obs_noise_var=obs_noise_var,
            generator=torch_gen,
            episode_seed=random_seed or 0,
            shade=list(shade_range or [0.0, 0.015]),
        )
        if special_agent_shade_range is not None:
            shade = list(special_agent_shade_range)
            s_hi = float(shade[-1] if shade else 0.0)
            side = BUY if llm_order_type == BUY else SELL
            zip_kwargs["initial_mu"] = shade_hi_to_fixed_zip_mu(s_hi, side=side)
            zip_kwargs["freeze_margin"] = True
            zip_kwargs["zi_warmup_steps"] = 0
        else:
            zip_kwargs["zi_warmup_steps"] = max(
                0, int(background_zip_zi_warmup_steps)
            )
            if background_zip_initial_mu is not None:
                side = BUY if llm_order_type == BUY else SELL
                zip_kwargs["initial_mu"] = shade_hi_to_fixed_zip_mu(
                    abs(float(background_zip_initial_mu)), side=side
                )
                zip_kwargs["freeze_margin"] = False
        if _kind == "zip_noisy":
            buy_cls, sell_cls, tag = ZIPAgentBuyNoisy, ZIPAgentSellNoisy, "zip_noisy"
        else:
            buy_cls, sell_cls, tag = ZIPAgentBuy, ZIPAgentSell, "zip"
        if llm_order_type == BUY:
            llm_agent = buy_cls(**zip_kwargs)
            agents[f"{tag}_buy_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_buy"
        else:
            llm_agent = sell_cls(**zip_kwargs)
            agents[f"{tag}_sell_{llm_agent_id}"] = llm_agent
            agent_types[llm_agent_id] = "llm_sell"

    else:
        raise ValueError(
            f"Unknown special_agent_kind: {_kind!r}. "
            "Must be 'llm', 'zi', 'bo', 'hbl', 'hbl_fast', 'zip', or 'zip_noisy'."
        )

    id_to_agent[llm_agent_id] = llm_agent

    _apply_fixed_private_values(llm_agent, llm_order_type, llm_private_values, q_max)
    if isinstance(llm_agent, (LLMBuyAgent, LLMSellAgent)) and getattr(
        llm_agent, "hbl_exact_seed", False
    ):
        proxy = getattr(llm_agent, "_hbl_proxy", None)
        if proxy is not None:
            proxy.pv = llm_agent.pv

    buyers = [id_to_agent[aid] for aid, t in agent_types.items() if t.endswith("_buy")]
    sellers = [id_to_agent[aid] for aid, t in agent_types.items() if t.endswith("_sell")]
    bids, asks = _collect_market_valuations(buyers=buyers, sellers=sellers)
    equilibrium_price = _equilibrium_price(bids, asks)
    if llm_order_type == BUY:
        llm_values = [float(v.item()) for v in llm_agent.pv.buyer_values]
    else:
        llm_values = [float(v.item()) for v in llm_agent.pv.seller_costs]
    baseline_surplus = _baseline_surplus(
        llm_values=llm_values,
        equilibrium_price=equilibrium_price,
        order_type=llm_order_type,
    )

    total_trades = 0
    agent_surplus = {aid: 0.0 for aid in id_to_agent}
    agent_last_order_time = {aid: -1 for aid in id_to_agent}
    order_lifetime = int(1.0 / arrival_rate) if arrival_rate > 0 else None
    llm_shade_sum = 0.0
    llm_shade_count = 0
    llm_transacted_private_values_sum = 0.0
    llm_amount_spent = 0.0
    llm_transacted_units_signed = 0

    all_events: List[Dict] = []
    arrival_schedule = _build_spread_arrival_schedule(
        agents=agents,
        timesteps=timesteps,
        arrival_rate=arrival_rate,
        rng=rng,
        py_rng=py_rng,
    )
    log_llm_orders = capture_detail and _kind == "llm" and isinstance(
        llm_agent, (LLMBuyAgent, LLMSellAgent)
    )

    for t in range(timesteps):
        market.event_queue.set_time(t)
        _ = market.get_fundamental_value()

        if order_lifetime is not None:
            for aid in id_to_agent:
                last_t = agent_last_order_time[aid]
                if last_t >= 0 and (t - last_t) >= order_lifetime:
                    market.withdraw_all(aid)
                    agent_last_order_time[aid] = -1

        pending_llm_event: Optional[Dict[str, Any]] = None
        scheduled_agent = arrival_schedule.get(t)
        orders: List = []
        aid: Optional[int] = None
        if scheduled_agent is not None:
            aid = scheduled_agent.get_id()
            market.withdraw_all(aid)
            pre_lob = _lob_snapshot(market) if log_llm_orders and aid == llm_agent_id else None
            orders = scheduled_agent.take_action()
            if orders:
                market.add_orders(orders)
                agent_last_order_time[aid] = t
                if log_llm_orders and aid == llm_agent_id:
                    pending_llm_event = {
                        "time": t,
                        "pre_lob": pre_lob,
                        "agent_action": _llm_action_payload(
                            llm_agent=llm_agent,
                            order=orders[0],
                        ),
                    }

        matched_orders = market.step()
        paired_trades = _pair_matched_orders(matched_orders)
        llm_trade_price_relative: Optional[float] = None
        llm_trade_surplus: Optional[float] = None
        llm_traded_this_step = False
        shout_trade_price_rel: Optional[float] = None

        for trade in paired_trades.values():
            buyer_order = trade.get("buyer_order")
            seller_order = trade.get("seller_order")
            trade_price, price_setter_side = _trade_price_from_orders(buyer_order, seller_order)
            if trade_price is None:
                continue

            fundamental_at_t = float(market.get_fundamental_value())
            trade_price_shifted = trade_price + fundamental_at_t

            buyer_id = trade.get("buyer_id")
            seller_id = trade.get("seller_id")
            if buyer_id is None or seller_id is None:
                continue

            buyer = id_to_agent.get(buyer_id)
            seller = id_to_agent.get(seller_id)
            if buyer is None or seller is None:
                continue

            if isinstance(llm_agent, (LLMBuyAgent, LLMSellAgent)):
                llm_agent._market_trades_window.append(
                    {
                        "time": int(t),
                        "price": float(trade_price),
                        "buyer_id": int(buyer_id),
                        "seller_id": int(seller_id),
                        "agent_involved": bool(
                            buyer_id == llm_agent_id or seller_id == llm_agent_id
                        ),
                    }
                )

            buyer_pos = buyer.position
            seller_pos = seller.position

            buyer_value = buyer.pv.consume_marginal(buyer.position, BUY) + fundamental_at_t
            seller_cost = seller.pv.consume_marginal(seller.position, SELL) + fundamental_at_t

            buyer_surplus_val = buyer_value - trade_price_shifted
            seller_surplus_val = trade_price_shifted - seller_cost
            agent_surplus[buyer_id] += buyer_surplus_val
            agent_surplus[seller_id] += seller_surplus_val

            if buyer_id == llm_agent_id:
                llm_transacted_private_values_sum += float(buyer_value - fundamental_at_t)
                llm_amount_spent += float(trade_price_shifted)
                llm_transacted_units_signed += 1
            elif seller_id == llm_agent_id:
                llm_transacted_private_values_sum += float(seller_cost - fundamental_at_t)
                llm_amount_spent -= float(trade_price_shifted)
                llm_transacted_units_signed -= 1

            if isinstance(buyer, LLMBuyAgent):
                rel_val = buyer.pv.value_for_exchange(buyer_pos, BUY)
                full_value = fundamental_at_t + rel_val
                shade_amount = float(rel_val - trade_price)
                shade_fraction = shade_amount / max(1e-12, float(full_value))
                shade_fraction = min(1.0, max(0.0, shade_fraction))
                llm_shade_sum += shade_fraction
                llm_shade_count += 1
                if buyer_id == llm_agent_id and pending_llm_event is not None and not llm_traded_this_step:
                    llm_traded_this_step = True
                    llm_trade_price_relative = float(trade_price)
                    llm_trade_surplus = float(buyer_surplus_val)

            elif isinstance(seller, LLMSellAgent):
                rel_val = seller.pv.value_for_exchange(seller_pos, SELL)
                full_value = fundamental_at_t + rel_val
                shade_amount = float(trade_price - rel_val)
                shade_fraction = shade_amount / max(1e-12, max(0.0, float(full_value)))
                shade_fraction = min(1.0, max(0.0, shade_fraction))
                llm_shade_sum += shade_fraction
                llm_shade_count += 1
                if seller_id == llm_agent_id and pending_llm_event is not None and not llm_traded_this_step:
                    llm_traded_this_step = True
                    llm_trade_price_relative = float(trade_price)
                    llm_trade_surplus = float(seller_surplus_val)

            if hasattr(buyer, "consumed_buy_positions"):
                buyer.consumed_buy_positions.add(buyer_pos)
            if hasattr(seller, "consumed_sell_positions"):
                seller.consumed_sell_positions.add(abs(seller_pos))

            if aid is not None and (buyer_id == aid or seller_id == aid):
                shout_trade_price_rel = float(trade_price)

            buyer.update_position(1, -trade_price_shifted)
            seller.update_position(-1, trade_price_shifted)
            total_trades += 1

        if orders and aid is not None:
            order0 = orders[0]
            F_broadcast = float(market.get_fundamental_value())
            side = int(order0.order_type)
            if shout_trade_price_rel is not None:
                price_rel = float(shout_trade_price_rel)
                accepted = True
            else:
                price_rel = float(order0.price)
                accepted = False
            price_abs_broadcast = price_rel + F_broadcast
            _broadcast_zip_shout(
                id_to_agent,
                price_abs=price_abs_broadcast,
                side=side,
                accepted=accepted,
                price_rel=price_rel,
            )
            if isinstance(llm_agent, (LLMBuyAgent, LLMSellAgent)):
                F_log = getattr(llm_agent, "_last_pricing_fundamental", None)
                if F_log is None:
                    F_log = float(llm_agent.estimate_fundamental())
                _append_llm_shout_log(
                    llm_agent,
                    time=int(t),
                    price_abs=price_rel + float(F_log),
                    price_rel=price_rel,
                    side=side,
                    accepted=accepted,
                    is_own=(aid == llm_agent_id),
                )

        if pending_llm_event is not None:
            all_events.append(
                _finalize_llm_order_event(
                    pending=pending_llm_event,
                    market=market,
                    transacted=llm_traded_this_step,
                    trade_price_relative=llm_trade_price_relative,
                    surplus_earned=llm_trade_surplus,
                )
            )

        if isinstance(llm_agent, (LLMBuyAgent, LLMSellAgent)):
            _record_post_step_lob_history(llm_agent, market)

    llm_surplus_trade_based = float(agent_surplus[llm_agent_id])
    llm_surplus = llm_surplus_trade_based
    llm_surplus_mtm = llm_surplus_trade_based
    llm_units_executed = abs(llm_transacted_units_signed)
    per_unit_surplus = llm_surplus / llm_units_executed if llm_units_executed > 0 else 0.0
    avg_shade = llm_shade_sum / llm_shade_count if llm_shade_count > 0 else None
    avg_requested_x_cap = None
    if isinstance(llm_agent, (LLMBuyAgent, LLMSellAgent)) and llm_agent._requested_x_count > 0:
        avg_requested_x_cap = float(
            llm_agent._requested_x_sum / float(llm_agent._requested_x_count)
        )
    llm_terminal_fundamental = float(market.get_final_fundamental())
    llm_commodities_worth_at_terminal = float(llm_transacted_units_signed * llm_terminal_fundamental)
    llm_profit_worth_minus_spent = float(llm_commodities_worth_at_terminal - llm_amount_spent)
    llm_total_surplus_private_plus_profit = float(
        llm_transacted_private_values_sum + llm_profit_worth_minus_spent
    )

    same_side_type = "zi_buy" if llm_order_type == BUY else "zi_sell"
    zi_same_surpluses = [
        agent_surplus[aid] for aid, t in agent_types.items() if t == same_side_type
    ]
    avg_zi_same_surplus = sum(zi_same_surpluses) / len(zi_same_surpluses) if zi_same_surpluses else 0.0

    bg_buy_surpluses: List[float] = []
    bg_sell_surpluses: List[float] = []
    bg_buy_advantages: List[float] = []
    bg_sell_advantages: List[float] = []
    for aid, t in agent_types.items():
        if aid == llm_agent_id:
            continue
        ag = id_to_agent[aid]
        sur = float(agent_surplus[aid])
        if str(t).endswith("_buy"):
            vals = [float(v.item()) for v in ag.pv.buyer_values]
            bl = _baseline_surplus(
                llm_values=vals,
                equilibrium_price=equilibrium_price,
                order_type=BUY,
            )
            bg_buy_surpluses.append(sur)
            bg_buy_advantages.append(sur - float(bl))
        elif str(t).endswith("_sell"):
            vals = [float(v.item()) for v in ag.pv.seller_costs]
            bl = _baseline_surplus(
                llm_values=vals,
                equilibrium_price=equilibrium_price,
                order_type=SELL,
            )
            bg_sell_surpluses.append(sur)
            bg_sell_advantages.append(sur - float(bl))
    bg_all_surpluses = bg_buy_surpluses + bg_sell_surpluses
    bg_all_advantages = bg_buy_advantages + bg_sell_advantages
    background_surplus_total = float(sum(bg_all_surpluses))
    background_surplus_mean = (
        background_surplus_total / len(bg_all_surpluses) if bg_all_surpluses else 0.0
    )
    background_surplus_mean_buy = (
        float(sum(bg_buy_surpluses) / len(bg_buy_surpluses)) if bg_buy_surpluses else 0.0
    )
    background_surplus_mean_sell = (
        float(sum(bg_sell_surpluses) / len(bg_sell_surpluses)) if bg_sell_surpluses else 0.0
    )
    background_advantage_total = float(sum(bg_all_advantages))
    background_advantage_mean = (
        background_advantage_total / len(bg_all_advantages) if bg_all_advantages else 0.0
    )
    background_advantage_mean_buy = (
        float(sum(bg_buy_advantages) / len(bg_buy_advantages))
        if bg_buy_advantages
        else 0.0
    )
    background_advantage_mean_sell = (
        float(sum(bg_sell_advantages) / len(bg_sell_advantages))
        if bg_sell_advantages
        else 0.0
    )
    llm_advantage = float(llm_surplus) - float(baseline_surplus)

    llm_rel_pvs = [llm_agent.pv.value_for_exchange(pos, llm_order_type) for pos in range(q_max)]

    consumer_surplus = sum(pv for pv in llm_rel_pvs if pv > 0)

    result: Dict[str, Any] = {
        "llm_surplus": llm_surplus,
        "llm_surplus_trade_based": llm_surplus_trade_based,
        "llm_surplus_mtm": llm_surplus_mtm,
        "baseline_surplus": float(baseline_surplus),
        "llm_advantage": float(llm_advantage),
        "consumer_surplus": consumer_surplus,
        "llm_units_executed": llm_units_executed,
        "llm_units_available": q_max,
        "per_unit_surplus": per_unit_surplus,
        "avg_shade": avg_shade,
        "avg_requested_x_cap": avg_requested_x_cap,
        "avg_zi_same_surplus": avg_zi_same_surplus,
        "background_surplus_total": background_surplus_total,
        "background_surplus_mean": float(background_surplus_mean),
        "background_surplus_mean_buy": float(background_surplus_mean_buy),
        "background_surplus_mean_sell": float(background_surplus_mean_sell),
        "background_surplus_n": int(len(bg_all_surpluses)),
        "background_surplus_n_buy": int(len(bg_buy_surpluses)),
        "background_surplus_n_sell": int(len(bg_sell_surpluses)),
        "background_advantage_total": background_advantage_total,
        "background_advantage_mean": float(background_advantage_mean),
        "background_advantage_mean_buy": float(background_advantage_mean_buy),
        "background_advantage_mean_sell": float(background_advantage_mean_sell),
        "llm_private_values_relative": llm_rel_pvs,
        "llm_transacted_private_values_sum": llm_transacted_private_values_sum,
        "llm_transacted_units_signed": llm_transacted_units_signed,
        "llm_terminal_fundamental": llm_terminal_fundamental,
        "llm_commodities_worth_at_terminal": llm_commodities_worth_at_terminal,
        "llm_amount_spent": llm_amount_spent,
        "llm_profit_worth_minus_spent": llm_profit_worth_minus_spent,
        "llm_total_surplus_private_plus_profit": llm_total_surplus_private_plus_profit,
        "episode_stats": {
            "total_trades": total_trades,
            "execution_ratio": llm_units_executed / max(1, q_max),
        },
        "market_config": {
            "num_zi_buy": num_zi_buy,
            "num_zi_sell": num_zi_sell,
            "timesteps": timesteps,
            "q_max": q_max,
            "pv_var": pv_var,
            "shade_range": shade_range,
            "background_agent_kind": _background,
            "background_zip_zi_warmup_steps": int(background_zip_zi_warmup_steps),
        },
    }

    if capture_detail:
        agent_type_str = "buyer" if llm_order_type == BUY else "seller"
        all_events_sorted = sorted(all_events, key=lambda e: e["time"])
        result["diagnostic_json"] = {
            "agent_type": agent_type_str,
            "agent_private_values_relative": llm_rel_pvs,
            "total_transactions": total_trades,
            "agent_transactions": llm_units_executed,
            "agent_total_surplus": round(llm_surplus, 4),
            "agent_transacted_private_values_sum": round(llm_transacted_private_values_sum, 4),
            "agent_terminal_fundamental": round(llm_terminal_fundamental, 4),
            "agent_commodities_worth_at_terminal": round(llm_commodities_worth_at_terminal, 4),
            "agent_amount_spent": round(llm_amount_spent, 4),
            "agent_profit_worth_minus_spent": round(llm_profit_worth_minus_spent, 4),
            "agent_total_surplus_private_plus_profit": round(llm_total_surplus_private_plus_profit, 4),
            "timesteps": timesteps,
            "events": all_events_sorted,
        }

    return result
