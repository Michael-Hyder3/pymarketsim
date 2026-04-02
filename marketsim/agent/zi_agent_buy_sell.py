import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List, Union, Tuple
import numpy as np


def _resolve_shade_at_time(shade_spec: Union[List, Tuple], current_time: int, num_transactions: int = 0, schedule_mode: str = "time") -> List:
    """
    Resolve the shading range based on current time or transaction count.
    
    Args:
        shade_spec: Either:
            - A simple list [min, max] (static shading)
            - A list of tuples [(time1, [min1, max1]), (time2, [min2, max2]), ...]
              indicating when to switch shading ranges
        current_time: Current timestep
        num_transactions: Number of transactions completed (for transaction-based scheduling)
        schedule_mode: "time" for time-based scheduling, "transactions" for transaction-based
    
    Returns:
        [min, max] shading range to use
    """
    # Check if this is a schedule (list of tuples) or a simple shade
    if not shade_spec or len(shade_spec) == 0:
        return [0, 500]
    
    # If first element is a list/tuple (not a number), treat as schedule
    if isinstance(shade_spec[0], (list, tuple)):
        # It's a schedule: [(trigger1, shade1), (trigger2, shade2), ...]
        current_trigger = current_time if schedule_mode == "time" else num_transactions
        
        # Find the latest trigger that applies
        applicable_shade = [0, 500]  # Default fallback
        for trigger, shade in shade_spec:
            if trigger <= current_trigger:
                applicable_shade = shade
            else:
                break
        return applicable_shade
    else:
        # It's a simple shade [min, max]
        return shade_spec


class ZIAgentBuy(Agent):
    """Zero-intelligence agent that ONLY submits buy orders."""
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: Union[List, List[Tuple]], pv_var: float, eta: float = 1.0, obs_noise_var: float = 0.0, shade_schedule_mode: str = "time", generator=None, episode_seed: int = 0):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="buyer", generator=generator)
        self.position = 0
        self.shade = shade
        self.shade_schedule_mode = shade_schedule_mode
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        # Scoped RNG — unique per (agent, episode) so agents never clobber each other or global state
        agent_seed = (episode_seed * 10000 + agent_id) & 0xFFFFFFFF
        self._rng = random.Random(agent_seed)
        self._np_rng = np.random.default_rng(agent_seed)

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        
        # Add observation noise
        if self.obs_noise_var > 0:
            estimate += self._np_rng.normal(0, np.sqrt(self.obs_noise_var))
        
        return estimate

    def take_action(self):
        side = BUY  # Always buy
        # Don't submit buys beyond q_max position
        if getattr(self, 'position', 0) >= self.q_max:
            return []
        
        # Get buyer's value for NEXT unit (position q+1)
        # Per paper: θ^{q+1}_i when buying
        try:
            buyer_value = self.pv.value_for_exchange(self.position, BUY)
        except Exception:
            return []
        
        t = self.market.get_time()
        self._rng.seed(t + self.agent_id)
        
        # Check η threshold: if we can get η fraction of surplus by taking best ask, do it
        best_ask = self.market.order_book.get_best_ask()
        if not np.isinf(best_ask):
            # Surplus from taking ask: (θ^{q+1}) - best_ask
            # Requested surplus: θ^{q+1} (buyer's value for this unit)
            surplus_from_taking = buyer_value - best_ask
            requested_surplus = buyer_value
            if requested_surplus > 0 and surplus_from_taking >= self.eta * requested_surplus:
                # Take the ask instead of submitting limit order
                order = Order(
                    price=best_ask,
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=side,
                    order_id=self._rng.randint(1, 10000000)
                )
                return [order]
        # Otherwise, submit limit order with random shading
        # Shade range is [0, Rmax]; buyers shade DOWN by that amount
        # Resolve shading based on time or transactions
        current_shade = _resolve_shade_at_time(
            self.shade,
            current_time=t,
            num_transactions=len(self.market.matched_orders),
            schedule_mode=self.shade_schedule_mode
        )
        shade_max = current_shade[1] if len(current_shade) > 1 else current_shade[0]
        price = buyer_value - self._rng.uniform(0, shade_max)
        order = Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=self._rng.randint(1, 10000000)
        )
        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'ZIBuy{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="buyer")


class ZIAgentSell(Agent):
    """Zero-intelligence agent that ONLY submits sell orders."""
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: Union[List, List[Tuple]], pv_var: float, eta: float = 1.0, obs_noise_var: float = 0.0, shade_schedule_mode: str = "time", generator=None, episode_seed: int = 0):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="seller", generator=generator)
        self.position = 0
        self.shade = shade
        self.shade_schedule_mode = shade_schedule_mode
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        agent_seed = (episode_seed * 10000 + agent_id) & 0xFFFFFFFF
        self._rng = random.Random(agent_seed)
        self._np_rng = np.random.default_rng(agent_seed)

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1-r)**(T-t)

        estimate = (1-rho)*mean + rho*val
        
        # Add observation noise
        if self.obs_noise_var > 0:
            estimate += self._np_rng.normal(0, np.sqrt(self.obs_noise_var))
        
        return estimate

    def take_action(self):
        side = SELL  # Always sell
        # Don't submit sells beyond q_max (limit absolute position)
        if abs(getattr(self, 'position', 0)) >= self.q_max:
            return []
        
        # Get seller's cost for CURRENT unit (position q)
        # Per paper: θ^q_i when selling (note: θ^q is negative for sellers)
        try:
            seller_cost = self.pv.value_for_exchange(self.position, SELL)
        except Exception:
            return []
        
        t = self.market.get_time()
        self._rng.seed(t + self.agent_id)
        
        # Check η threshold: if we can get η fraction of surplus by taking best bid, do it
        best_bid = self.market.order_book.get_best_bid()
        if not np.isinf(best_bid):
            # Surplus from taking bid: best_bid - seller_cost
            # Requested surplus: -seller_cost (seller's profit margin, note θ^q is negative)
            surplus_from_taking = best_bid - seller_cost
            requested_surplus = -seller_cost
            if requested_surplus > 0 and surplus_from_taking >= self.eta * requested_surplus:
                # Take the bid instead of submitting limit order
                order = Order(
                    price=best_bid,
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=side,
                    order_id=self._rng.randint(1, 10000000)
                )
                return [order]
        # Otherwise, submit limit order with random shading
        # Shade range is [0, Rmax]; sellers shade UP by that amount
        # Resolve shading based on time or transactions
        current_shade = _resolve_shade_at_time(
            self.shade,
            current_time=t,
            num_transactions=len(self.market.matched_orders),
            schedule_mode=self.shade_schedule_mode
        )
        shade_max = current_shade[1] if len(current_shade) > 1 else current_shade[0]
        price = seller_cost + self._rng.uniform(0, shade_max)
        order = Order(
            price=price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=side,
            order_id=self._rng.randint(1, 10000000)
        )
        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def __str__(self):
        return f'ZISell{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self.position = 0
        self.cash = 0
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="seller")
