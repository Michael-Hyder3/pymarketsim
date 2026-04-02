"""
LLM-First Zero Intelligence Agent - Adaptive Pricing Heuristic

This agent implements a heuristic strategy designed to maximize surplus by adaptively
adjusting willingness to pay based on market observations. It uses a simple learning
mechanism that responds to seller offers in the market.

Strategy:
1. Initialize WTP to the highest private value
2. Observe the best ask (lowest seller offer) at each timestep
3. Update WTP based on comparison with best ask:
   - If best ask < current WTP: increase WTP by adjustment_rate %
   - If best ask > current WTP: decrease WTP by adjustment_rate %
4. Submit buy orders at current WTP if there are valid sells below it
5. Stop buying once q_max position is reached
"""

import random
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List
import numpy as np


class LLMFirstZIAgentBuy(Agent):
    """
    LLM-First Zero Intelligence buyer agent with adaptive WTP heuristic.
    
    This agent learns to adjust its willingness to pay based on observed market offers,
    implementing a simple adaptive strategy to maximize surplus acquisition.
    """
    
    def __init__(
        self,
        agent_id: int,
        market: Market,
        q_max: int,
        shade: List,
        pv_var: float,
        eta: float = 1.0,
        obs_noise_var: float = 0.0,
        adjustment_rate: float = 0.01,
        debug: bool = False
    ):
        """
        Initialize the LLM-First ZI buyer agent.
        
        Args:
            agent_id: Unique identifier for the agent
            market: Market instance for order book access
            q_max: Maximum position quantity
            shade: Price shading range [min, max]
            pv_var: Variance of private values distribution
            eta: Threshold parameter for immediate execution (inherited from ZI)
            obs_noise_var: Observation noise variance
            adjustment_rate: Rate at which to adjust WTP based on market (default 1%)
            debug: Enable debug output
        """
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="buyer")
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        self.adjustment_rate = adjustment_rate
        self.debug = debug
        
        # Initialize WTP to the highest private value (first unit's value)
        self.current_wtp = self.pv.value_for_exchange(0, BUY)
        self.max_wtp = self.current_wtp  # Track the highest value
        
        # Track historical WTP for analysis
        self.wtp_history = [self.current_wtp]
        self.observation_history = []

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        """Estimate the fundamental value of the asset."""
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1 - r) ** (T - t)
        estimate = (1 - rho) * mean + rho * val
        
        # Add observation noise
        if self.obs_noise_var > 0:
            estimate += np.random.normal(0, np.sqrt(self.obs_noise_var))
        
        return estimate

    def _update_wtp_based_on_market(self):
        """
        Update WTP based on observed market conditions.
        
        Rule:
        - If best ask < current WTP: increase WTP by adjustment_rate %
        - If best ask > current WTP: decrease WTP by adjustment_rate %
        - If no best ask (no sellers): maintain WTP
        """
        best_ask = self.market.order_book.get_best_ask()
        
        if np.isinf(best_ask):
            # No sellers in the market; maintain current WTP
            observation = None
        else:
            observation = best_ask
            
            if best_ask < self.current_wtp:
                # Sellers are offering below our WTP; increase WTP to be more competitive
                adjustment = self.current_wtp * self.adjustment_rate
                self.current_wtp += adjustment
                if self.debug:
                    print(f"Agent {self.agent_id} (t={self.market.get_time()}): "
                          f"Best ask {best_ask:.2f} < WTP {self.current_wtp - adjustment:.2f}, "
                          f"increasing WTP to {self.current_wtp:.2f}")
                
            elif best_ask > self.current_wtp:
                # Sellers are asking above our WTP; decrease WTP to save cash
                adjustment = self.current_wtp * self.adjustment_rate
                self.current_wtp -= adjustment
                if self.debug:
                    print(f"Agent {self.agent_id} (t={self.market.get_time()}): "
                          f"Best ask {best_ask:.2f} > WTP {self.current_wtp + adjustment:.2f}, "
                          f"decreasing WTP to {self.current_wtp:.2f}")
        
        self.observation_history.append(observation)
        self.wtp_history.append(self.current_wtp)

    def take_action(self):
        """
        Determine trading action for this timestep.
        
        Process:
        1. Check if at maximum position capacity
        2. Update WTP based on market observations
        3. Get private value for next unit
        4. Submit buy order at current WTP
        """
        # Don't submit buys beyond q_max position
        if self.position >= self.q_max:
            return []
        
        # Update WTP based on observed market conditions
        self._update_wtp_based_on_market()
        
        # Get private value for the next unit we want to buy
        try:
            buyer_value = self.pv.value_for_exchange(self.position, BUY)
        except Exception:
            return []
        
        t = self.market.get_time()
        random.seed(t + self.agent_id)
        
        # Estimate fundamental value
        fundamental_estimate = self.estimate_fundamental()
        
        # Use current WTP, but ensure it doesn't exceed our fundamental estimate + value
        # This prevents excessive overbidding
        wtp_price = min(self.current_wtp, fundamental_estimate + buyer_value)
        
        # Check η threshold: if we can get η fraction of surplus by taking best ask, do it
        best_ask = self.market.order_book.get_best_ask()
        if not np.isinf(best_ask) and best_ask <= wtp_price:
            # There's a seller offering at or below our WTP; take it immediately
            order = Order(
                price=best_ask,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=BUY,
                order_id=random.randint(1, 10000000)
            )
            if self.debug:
                print(f"Agent {self.agent_id} (t={t}): Taking ask at {best_ask:.2f}")
            return [order]
        
        # Otherwise, submit limit order at our current WTP
        order = Order(
            price=wtp_price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=BUY,
            order_id=random.randint(1, 10000000)
        )
        
        if self.debug:
            print(f"Agent {self.agent_id} (t={t}): Submitting buy limit at {wtp_price:.2f}")
        
        return [order]

    def update_position(self, q, p):
        """Update position and cash after a trade."""
        self.position += q
        self.cash += p

    def __str__(self):
        return f'LLMFirstZIBuy{self.agent_id}'

    def get_pos_value(self) -> float:
        """Get total value of current position."""
        return self.pv.value_at_position(self.position)

    def reset(self):
        """Reset agent state for a new simulation."""
        self.position = 0
        self.cash = 0
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="buyer")
        self.current_wtp = self.pv.value_for_exchange(0, BUY)
        self.max_wtp = self.current_wtp
        self.wtp_history = [self.current_wtp]
        self.observation_history = []


class LLMFirstZIAgentSell(Agent):
    """
    LLM-First Zero Intelligence seller agent with adaptive ask price heuristic.
    
    Mirror implementation for sellers: adaptively adjusts willingness to sell
    based on observed buyer offers in the market.
    """
    
    def __init__(
        self,
        agent_id: int,
        market: Market,
        q_max: int,
        shade: List,
        pv_var: float,
        eta: float = 1.0,
        obs_noise_var: float = 0.0,
        adjustment_rate: float = 0.01,
        debug: bool = False
    ):
        """
        Initialize the LLM-First ZI seller agent.
        
        Args:
            agent_id: Unique identifier for the agent
            market: Market instance for order book access
            q_max: Maximum position quantity (negative)
            shade: Price shading range [min, max]
            pv_var: Variance of private values distribution
            eta: Threshold parameter for immediate execution
            obs_noise_var: Observation noise variance
            adjustment_rate: Rate at which to adjust ask based on market (default 1%)
            debug: Enable debug output
        """
        self.agent_id = agent_id
        self.market = market
        self.q_max = q_max
        self.pv_var = pv_var
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="seller")
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.eta = eta
        self.obs_noise_var = obs_noise_var
        self.adjustment_rate = adjustment_rate
        self.debug = debug
        
        # Initialize ask price (willingness to sell) to the lowest private cost (first unit's cost)
        # For sellers, private values are negative (costs), so first is the highest (least negative)
        self.current_ask = self.pv.value_for_exchange(0, SELL)
        self.min_ask = self.current_ask  # Track the lowest ask price
        
        # Track historical ask for analysis
        self.ask_history = [self.current_ask]
        self.observation_history = []

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        """Estimate the fundamental value of the asset."""
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()

        rho = (1 - r) ** (T - t)
        estimate = (1 - rho) * mean + rho * val
        
        # Add observation noise
        if self.obs_noise_var > 0:
            estimate += np.random.normal(0, np.sqrt(self.obs_noise_var))
        
        return estimate

    def _update_ask_based_on_market(self):
        """
        Update ask price based on observed market conditions.
        
        Rule (for sellers):
        - If best bid > current ask: decrease ask by adjustment_rate %
        - If best bid < current ask: increase ask by adjustment_rate %
        - If no best bid (no buyers): maintain ask
        """
        best_bid = self.market.order_book.get_best_bid()
        
        if np.isinf(best_bid):
            # No buyers in the market; maintain current ask
            observation = None
        else:
            observation = best_bid
            
            if best_bid > self.current_ask:
                # Buyers are bidding above our ask; decrease ask to be more competitive
                adjustment = self.current_ask * self.adjustment_rate
                self.current_ask -= adjustment
                if self.debug:
                    print(f"Agent {self.agent_id} (t={self.market.get_time()}): "
                          f"Best bid {best_bid:.2f} > ask {self.current_ask + adjustment:.2f}, "
                          f"decreasing ask to {self.current_ask:.2f}")
                
            elif best_bid < self.current_ask:
                # Buyers are bidding below our ask; increase ask to get better price
                adjustment = self.current_ask * self.adjustment_rate
                self.current_ask += adjustment
                if self.debug:
                    print(f"Agent {self.agent_id} (t={self.market.get_time()}): "
                          f"Best bid {best_bid:.2f} < ask {self.current_ask - adjustment:.2f}, "
                          f"increasing ask to {self.current_ask:.2f}")
        
        self.observation_history.append(observation)
        self.ask_history.append(self.current_ask)

    def take_action(self):
        """
        Determine trading action for this timestep.
        
        Process:
        1. Check if at maximum short position capacity
        2. Update ask price based on market observations
        3. Get private cost for next unit
        4. Submit sell order at current ask
        """
        # Don't submit sells beyond q_max (limit absolute position)
        if abs(self.position) >= self.q_max:
            return []
        
        # Update ask price based on observed market conditions
        self._update_ask_based_on_market()
        
        # Get seller's cost for the next unit
        try:
            seller_cost = self.pv.value_for_exchange(self.position, SELL)
        except Exception:
            return []
        
        t = self.market.get_time()
        random.seed(t + self.agent_id)
        
        # Estimate fundamental value
        fundamental_estimate = self.estimate_fundamental()
        
        # Use current ask, but ensure it's above our fundamental estimate + cost
        # This prevents underselling
        ask_price = max(self.current_ask, fundamental_estimate + seller_cost)
        
        # Check η threshold: if we can get η fraction of surplus by taking best bid, do it
        best_bid = self.market.order_book.get_best_bid()
        if not np.isinf(best_bid) and best_bid >= ask_price:
            # There's a buyer offering at or above our ask; take it immediately
            order = Order(
                price=best_bid,
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=SELL,
                order_id=random.randint(1, 10000000)
            )
            if self.debug:
                print(f"Agent {self.agent_id} (t={t}): Taking bid at {best_bid:.2f}")
            return [order]
        
        # Otherwise, submit limit order at our current ask
        order = Order(
            price=ask_price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=SELL,
            order_id=random.randint(1, 10000000)
        )
        
        if self.debug:
            print(f"Agent {self.agent_id} (t={t}): Submitting sell limit at {ask_price:.2f}")
        
        return [order]

    def update_position(self, q, p):
        """Update position and cash after a trade."""
        self.position += q
        self.cash += p

    def __str__(self):
        return f'LLMFirstZISell{self.agent_id}'

    def get_pos_value(self) -> float:
        """Get total value of current position."""
        return self.pv.value_at_position(self.position)

    def reset(self):
        """Reset agent state for a new simulation."""
        self.position = 0
        self.cash = 0
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="seller")
        self.current_ask = self.pv.value_for_exchange(0, SELL)
        self.min_ask = self.current_ask
        self.ask_history = [self.current_ask]
        self.observation_history = []
