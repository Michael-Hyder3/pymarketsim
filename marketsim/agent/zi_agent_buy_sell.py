import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List, Union, Tuple
import numpy as np

from marketsim.agent.fundamental_estimate import absolute_fundamental_estimate
from marketsim.agent.shading import (
    maybe_take_best_quote_buy,
    maybe_take_best_quote_sell,
    resolve_shade_at_time,
    zi_limit_price_buy,
    zi_limit_price_sell,
)

_resolve_shade_at_time = resolve_shade_at_time


class ZIAgentBuy(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: Union[List, List[Tuple]], pv_var: float, eta: float = 1.0, obs_noise_var: float = 0.0, shade_schedule_mode: str = "time", shade_sampling_mode: str = "range", generator=None, episode_seed: int = 0):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="buyer", generator=generator)
        self.position = 0
        self.shade = shade
        self.shade_schedule_mode = shade_schedule_mode
        self.shade_sampling_mode = shade_sampling_mode
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        agent_seed = (episode_seed * 10000 + agent_id) & 0xFFFFFFFF
        self._rng = random.Random(agent_seed)
        self._np_rng = np.random.default_rng(agent_seed)

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        return absolute_fundamental_estimate(
            self.market,
            obs_noise_var=self.obs_noise_var,
            np_rng=self._np_rng,
        )

    def take_action(self):
        side = BUY
        if getattr(self, 'position', 0) >= self.q_max:
            return []

        try:
            buyer_value = self.pv.value_for_exchange(self.position, BUY)
        except Exception:
            return []

        t = self.market.get_time()
        self._rng.seed(t + self.agent_id)

        lifted = maybe_take_best_quote_buy(
            market=self.market,
            buyer_value=buyer_value,
            eta=self.eta,
            agent_id=self.get_id(),
            time=t,
            rng=self._rng,
        )
        if lifted is not None:
            return [lifted]

        fundamental_now = float(self.estimate_fundamental())
        price, _ = zi_limit_price_buy(
            buyer_value=buyer_value,
            fundamental_now=fundamental_now,
            shade_spec=self.shade,
            current_time=t,
            num_transactions=len(self.market.matched_orders),
            schedule_mode=self.shade_schedule_mode,
            rng=self._rng,
            sampling_mode=self.shade_sampling_mode,
        )
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





class ZIAgentSell(Agent):
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: Union[List, List[Tuple]], pv_var: float, eta: float = 1.0, obs_noise_var: float = 0.0, shade_schedule_mode: str = "time", shade_sampling_mode: str = "range", generator=None, episode_seed: int = 0):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="seller", generator=generator)
        self.position = 0
        self.shade = shade
        self.shade_schedule_mode = shade_schedule_mode
        self.shade_sampling_mode = shade_sampling_mode
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        agent_seed = (episode_seed * 10000 + agent_id) & 0xFFFFFFFF
        self._rng = random.Random(agent_seed)
        self._np_rng = np.random.default_rng(agent_seed)

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        return absolute_fundamental_estimate(
            self.market,
            obs_noise_var=self.obs_noise_var,
            np_rng=self._np_rng,
        )

    def take_action(self):
        side = SELL
        if abs(getattr(self, 'position', 0)) >= self.q_max:
            return []

        try:
            seller_cost = self.pv.value_for_exchange(self.position, SELL)
        except Exception:
            return []

        t = self.market.get_time()
        self._rng.seed(t + self.agent_id)

        lifted = maybe_take_best_quote_sell(
            market=self.market,
            seller_cost=seller_cost,
            eta=self.eta,
            agent_id=self.get_id(),
            time=t,
            rng=self._rng,
        )
        if lifted is not None:
            return [lifted]

        fundamental_now = float(self.estimate_fundamental())
        price, _ = zi_limit_price_sell(
            seller_cost=seller_cost,
            fundamental_now=fundamental_now,
            shade_spec=self.shade,
            current_time=t,
            num_transactions=len(self.market.matched_orders),
            schedule_mode=self.shade_schedule_mode,
            rng=self._rng,
            sampling_mode=self.shade_sampling_mode,
        )
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



