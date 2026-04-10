import random
from typing import List

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fourheap.constants import BUY, SELL


class _BayesianShadeOptimizer:
    def __init__(
        self,
        initial_x: float = 250.0,
        initial_step: float = 150.0,
        length_scale: float = 50.0,
        beta: float = 2.0,
        strategy: str = "ucb",
        rng_seed: int = 0,
    ):
        self.initial_x = max(0.0, float(initial_x))
        self.initial_step = max(1e-6, float(initial_step))
        self.length_scale = max(1e-6, float(length_scale))
        self.beta = float(beta)
        self.strategy = (strategy or "ucb").lower()
        self.observed_shades: List[float] = []
        self.observed_rewards: List[float] = []
        self._rng = np.random.default_rng(rng_seed)

    def _posterior_mean_std(self, candidates: np.ndarray):
        x_obs = np.asarray(self.observed_shades, dtype=float)
        y_obs = np.asarray(self.observed_rewards, dtype=float)

        if len(x_obs) == 0:
            mean = np.zeros_like(candidates)
            std = np.ones_like(candidates) * self.initial_step
            return mean, std

        mean = np.zeros_like(candidates)
        std = np.zeros_like(candidates)
        for i, x in enumerate(candidates):
            distance = (x - x_obs) / self.length_scale
            weights = np.exp(-0.5 * (distance ** 2))
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                mean[i] = 0.0
                std[i] = self.initial_step
            else:
                weighted_mean = float(np.dot(weights, y_obs) / weight_sum)
                weighted_var = float(np.dot(weights, (y_obs - weighted_mean) ** 2) / weight_sum)
                mean[i] = weighted_mean
                std[i] = float(np.sqrt(max(weighted_var, 1e-6)) + 1.0 / np.sqrt(weight_sum))

        return mean, std

    def _candidate_grid(self) -> np.ndarray:
        if len(self.observed_shades) == 0:
            upper = max(self.initial_x * 2.0, self.initial_step * 2.0, 100.0)
            return np.linspace(0.0, upper, 81)

        x_obs = np.asarray(self.observed_shades, dtype=float)
        best_idx = int(np.argmax(self.observed_rewards))
        best_x = float(x_obs[best_idx])
        median_x = float(np.median(x_obs))
        spread = float(np.std(x_obs)) if len(x_obs) > 1 else 0.0
        decay = 1.0 / np.sqrt(max(1.0, float(len(x_obs))))
        local_step = max(self.initial_step * decay, self.initial_step * 0.15, spread * 0.2, 10.0)

        upper = max(
            self.initial_x + 6.0 * self.initial_step,
            best_x + 6.0 * local_step,
            median_x + 4.0 * local_step,
            100.0,
        )
        upper = min(upper, self.initial_x + 12.0 * self.initial_step)
        full = np.linspace(0.0, upper, 81)
        local_low = max(0.0, best_x - 4.0 * local_step)
        local_high = best_x + 4.0 * local_step
        local = np.linspace(local_low, local_high, 41)
        return np.unique(np.concatenate([full, local]))

    def suggest(self) -> float:
        candidates = self._candidate_grid()
        mean, std = self._posterior_mean_std(candidates)

        if self.strategy == "thompson":
            sampled = self._rng.normal(loc=mean, scale=std)
            return float(candidates[int(np.argmax(sampled))])

        score = mean + self.beta * std
        return float(candidates[int(np.argmax(score))])

    def observe(self, shade: float, reward: float):
        self.observed_shades.append(float(shade))
        self.observed_rewards.append(float(reward))


class BOAgentBuy(Agent):
    def __init__(
        self,
        agent_id: int,
        market: Market,
        q_max: int,
        shade: List,
        pv_var: float,
        eta: float = 1.0,
        obs_noise_var: float = 0.0,
        optimizer_strategy: str = "ucb",
        fixed_x: float = None,
        action_mode: str = "x_cap",
        generator=None,
        episode_seed: int = 0,
    ):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="buyer", generator=generator)
        self.position = 0
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        self.fixed_x = None if fixed_x is None else max(0.0, float(fixed_x))
        self.action_mode = (action_mode or "x_cap").lower()
        if self.action_mode not in {"x_cap", "fixed_shade"}:
            raise ValueError("action_mode must be one of {'x_cap', 'fixed_shade'}")
        shade_min = max(0.0, float(shade[0])) if len(shade) > 0 else 0.0
        shade_max = max(shade_min, float(shade[1])) if len(shade) > 1 else shade_min + 500.0
        initial_x = 0.5 * (shade_min + shade_max)
        initial_step = max(abs(shade_max - shade_min) * 0.5, 10.0)
        self._optimizer = _BayesianShadeOptimizer(
            initial_x=initial_x,
            initial_step=initial_step,
            strategy=optimizer_strategy,
            rng_seed=(episode_seed * 10000 + agent_id + 17) & 0xFFFFFFFF,
        )
        agent_seed = (episode_seed * 10000 + agent_id) & 0xFFFFFFFF
        self._rng = random.Random(agent_seed)
        self._current_round_x = None
        self._current_round_realized_shade = None
        self._current_round_reward = 0.0
        self._current_round_value = None
        self.x_history: List[float] = []
        self.realized_shade_history: List[float] = []
        self.shade_history = self.x_history
        self.reward_history: List[float] = []
        self.round_history: List[dict] = []
        self.best_x = None
        self.best_shade = None
        self.best_reward = float('-inf')
        self._current_round_time = None
        self._current_round_order_price = None
        self._current_round_best_quote = None
        self._current_round_trade_count = 0

    def _finalize_previous_round(self):
        if self._current_round_x is None:
            return
        x_value = float(self._current_round_x)
        reward = float(self._current_round_reward)
        if self.fixed_x is None:
            self._optimizer.observe(x_value, reward)
        self.x_history.append(x_value)
        self.realized_shade_history.append(float(self._current_round_realized_shade or 0.0))
        self.reward_history.append(reward)
        self.round_history.append(
            {
                "time": self._current_round_time,
                "x": x_value,
                "realized_shade": float(self._current_round_realized_shade or 0.0),
                "order_price": self._current_round_order_price,
                "best_quote": self._current_round_best_quote,
                "trades": int(self._current_round_trade_count),
                "reward": reward,
            }
        )
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_x = x_value
            self.best_shade = x_value
        self._current_round_x = None
        self._current_round_realized_shade = None
        self._current_round_reward = 0.0
        self._current_round_value = None
        self._current_round_time = None
        self._current_round_order_price = None
        self._current_round_best_quote = None
        self._current_round_trade_count = 0

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self):
        self._finalize_previous_round()
        if self.position >= self.q_max:
            return []

        try:
            buyer_value = self.pv.value_for_exchange(self.position, BUY)
        except Exception:
            return []

        t = self.market.get_time()
        self._rng.seed(t + self.agent_id)
        fundamental_now = float(self.market.get_fundamental_value())
        if self.fixed_x is None:
            x_cap = max(0.0, float(self._optimizer.suggest()))
        else:
            x_cap = float(self.fixed_x)
        if self.action_mode == "x_cap":
            realized_shade = self._rng.uniform(0.0, x_cap)
        else:
            realized_shade = x_cap
        order_price = buyer_value - realized_shade

        best_ask = self.market.order_book.get_best_ask()
        self._current_round_x = float(x_cap)
        self._current_round_realized_shade = float(realized_shade)
        self._current_round_reward = 0.0
        self._current_round_value = float(buyer_value + fundamental_now)
        self._current_round_time = int(t)
        self._current_round_order_price = float(order_price)
        self._current_round_best_quote = None if np.isinf(best_ask) else float(best_ask)
        self._current_round_trade_count = 0
        return [
            Order(
                price=order_price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=BUY,
                order_id=self._rng.randint(1, 10000000),
            )
        ]

    def update_position(self, q, p):
        if q > 0 and self._current_round_x is not None and self._current_round_value is not None:
            trade_price = -float(p)
            self._current_round_reward += float(self._current_round_value - trade_price)
            self._current_round_trade_count += int(q)

        self.position += q
        self.cash += p

    def __str__(self):
        return f'BOBuy{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self._finalize_previous_round()
        self.position = 0
        self.cash = 0
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="buyer")
        self._current_round_value = None


class BOAgentSell(Agent):
    def __init__(
        self,
        agent_id: int,
        market: Market,
        q_max: int,
        shade: List,
        pv_var: float,
        eta: float = 1.0,
        obs_noise_var: float = 0.0,
        optimizer_strategy: str = "ucb",
        fixed_x: float = None,
        action_mode: str = "x_cap",
        generator=None,
        episode_seed: int = 0,
    ):
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="seller", generator=generator)
        self.position = 0
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        self.fixed_x = None if fixed_x is None else max(0.0, float(fixed_x))
        self.action_mode = (action_mode or "x_cap").lower()
        if self.action_mode not in {"x_cap", "fixed_shade"}:
            raise ValueError("action_mode must be one of {'x_cap', 'fixed_shade'}")
        shade_min = max(0.0, float(shade[0])) if len(shade) > 0 else 0.0
        shade_max = max(shade_min, float(shade[1])) if len(shade) > 1 else shade_min + 500.0
        initial_x = 0.5 * (shade_min + shade_max)
        initial_step = max(abs(shade_max - shade_min) * 0.5, 10.0)
        self._optimizer = _BayesianShadeOptimizer(
            initial_x=initial_x,
            initial_step=initial_step,
            strategy=optimizer_strategy,
            rng_seed=(episode_seed * 10000 + agent_id + 29) & 0xFFFFFFFF,
        )
        agent_seed = (episode_seed * 10000 + agent_id) & 0xFFFFFFFF
        self._rng = random.Random(agent_seed)
        self._current_round_x = None
        self._current_round_realized_shade = None
        self._current_round_reward = 0.0
        self._current_round_cost = None
        self.x_history: List[float] = []
        self.realized_shade_history: List[float] = []
        self.shade_history = self.x_history
        self.reward_history: List[float] = []
        self.round_history: List[dict] = []
        self.best_x = None
        self.best_shade = None
        self.best_reward = float('-inf')
        self._current_round_time = None
        self._current_round_order_price = None
        self._current_round_best_quote = None
        self._current_round_trade_count = 0

    def _finalize_previous_round(self):
        if self._current_round_x is None:
            return
        x_value = float(self._current_round_x)
        reward = float(self._current_round_reward)
        if self.fixed_x is None:
            self._optimizer.observe(x_value, reward)
        self.x_history.append(x_value)
        self.realized_shade_history.append(float(self._current_round_realized_shade or 0.0))
        self.reward_history.append(reward)
        self.round_history.append(
            {
                "time": self._current_round_time,
                "x": x_value,
                "realized_shade": float(self._current_round_realized_shade or 0.0),
                "order_price": self._current_round_order_price,
                "best_quote": self._current_round_best_quote,
                "trades": int(self._current_round_trade_count),
                "reward": reward,
            }
        )
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_x = x_value
            self.best_shade = x_value
        self._current_round_x = None
        self._current_round_realized_shade = None
        self._current_round_reward = 0.0
        self._current_round_cost = None
        self._current_round_time = None
        self._current_round_order_price = None
        self._current_round_best_quote = None
        self._current_round_trade_count = 0

    def get_id(self) -> int:
        return self.agent_id

    def take_action(self):
        self._finalize_previous_round()
        if abs(self.position) >= self.q_max:
            return []

        try:
            seller_cost = self.pv.value_for_exchange(self.position, SELL)
        except Exception:
            return []

        t = self.market.get_time()
        self._rng.seed(t + self.agent_id)
        fundamental_now = float(self.market.get_fundamental_value())
        if self.fixed_x is None:
            x_cap = max(0.0, float(self._optimizer.suggest()))
        else:
            x_cap = float(self.fixed_x)
        if self.action_mode == "x_cap":
            realized_shade = self._rng.uniform(0.0, x_cap)
        else:
            realized_shade = x_cap
        order_price = seller_cost + realized_shade

        best_bid = self.market.order_book.get_best_bid()
        self._current_round_x = float(x_cap)
        self._current_round_realized_shade = float(realized_shade)
        self._current_round_reward = 0.0
        self._current_round_cost = float(seller_cost + fundamental_now)
        self._current_round_time = int(t)
        self._current_round_order_price = float(order_price)
        self._current_round_best_quote = None if np.isinf(best_bid) else float(best_bid)
        self._current_round_trade_count = 0
        return [
            Order(
                price=order_price,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=SELL,
                order_id=self._rng.randint(1, 10000000),
            )
        ]

    def update_position(self, q, p):
        if q < 0 and self._current_round_x is not None and self._current_round_cost is not None:
            trade_price = float(p)
            self._current_round_reward += float(trade_price - self._current_round_cost)
            self._current_round_trade_count += int(abs(q))

        self.position += q
        self.cash += p

    def __str__(self):
        return f'BOSell{self.agent_id}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self):
        self._finalize_previous_round()
        self.position = 0
        self.cash = 0
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="seller")
        self._current_round_cost = None
