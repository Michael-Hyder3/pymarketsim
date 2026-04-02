"""
HBL Buy/Sell Agents - Split version of HBL agent for buying and selling only.
Uses SeparatedPrivateValues with relative pricing (base_price=0).
Directly adapted from hbl_agent.py for BUY and SELL-only operations.
"""

import random
import sys
import scipy as sp
import numpy as np
from marketsim.agent.agent import Agent
from marketsim.market.market import Market
from marketsim.fourheap.order import Order
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fourheap.constants import BUY, SELL
from typing import List
from fastcubicspline import FCS


class HBLAgentBuy(Agent):
    """HBL Agent that only submits buy orders. Adapted from HBLAgent."""
    
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, L: int, 
                 pv_var: float, arrival_rate: float, obs_noise_var: float = 0.0, debug: bool = False):
        self.agent_id = agent_id
        self.market = market
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="buyer")
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.L = L
        self.grace_period = 1 / arrival_rate
        self.lower_bound_mem = 0
        
        # HBL tuning parameters
        self.buy_upper_mid_shade = 99/100
        self.buy_half_shade = 1/2
        
        self.q_max = q_max
        self.pv_var = pv_var
        self.obs_noise_var = obs_noise_var
        self.debug = debug
        
        # Track which positions have been consumed in actual trades
        # Each position (0, 1, 2 for q_max=3) can only be used ONCE
        self.consumed_buy_positions = set()  # Tracks which BUY positions have been traded

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()
        rho = (1 - r) ** (T - t)
        estimate = (1 - rho) * mean + rho * val
        if self.obs_noise_var > 0:
            estimate += np.random.normal(0, np.sqrt(self.obs_noise_var))
        return estimate

    def get_last_trade_time_step(self):
        if len(self.market.matched_orders) < 2 * self.L:
            return 0
        last_matched_order_ind = len(self.market.matched_orders) - self.L * 2
        earliest_order = min(
            self.market.matched_orders[last_matched_order_ind:],
            key=lambda matched_order: matched_order.order.time
        ).order.time
        return earliest_order

    def fast_belief_function(self, p, side, orders):
        TBL = 0  # Transact bids less or equal
        AL = 0  # Asks less or equal
        for ind, order in enumerate(orders):
            if order.price <= p and order.order_type == SELL:
                AL += order.quantity
            for matched_order in self.market.matched_orders:
                if order.order_id == matched_order.order.order_id:
                    if matched_order.order.order_type == BUY and matched_order.price <= p:
                        TBL += order.quantity
                    break
        return AL + TBL == 0

    def find_worst_order(self, side, order_mem, orders: List[Order]):
        beginning = 0
        end = len(order_mem) - 1
        while beginning < end:
            mid = (beginning + end) // 2
            mid_belief = self.fast_belief_function(order_mem[mid].price, side, orders)
            if mid != len(order_mem) - 1:
                if mid_belief:
                    if not self.fast_belief_function(order_mem[mid + 1].price, side, orders):
                        return order_mem[mid].price, 0
                    if beginning == mid and mid_belief:
                        return order_mem[mid + 1].price, 0
                    beginning = mid
                else:
                    end = mid
            else:
                return order_mem[0].price, self.belief_function(order_mem[0].price, side, orders)
        return order_mem[0].price, self.belief_function(order_mem[0].price, side, orders)

    def belief_function(self, p, side, orders):
        current_time = self.market.get_time()
        TBL = 0  # Transact bids less or equal
        AL = 0  # Asks less or equal
        RBG = 0  # Rejected bids greater or equal
        
        for ind, order in enumerate(orders):
            if order.price - p <= 0 and order.order_type == SELL:
                AL += order.quantity
            found_matched = False
            for matched_order in self.market.matched_orders:
                if order.order_id == matched_order.order.order_id:
                    if matched_order.order.order_type == BUY and matched_order.price - p <= 0:
                        TBL += order.quantity
                    found_matched = True
                    break
            if not found_matched:
                if order.order_type == BUY and order.price - p >= 0:
                    withdrawn = False
                    latest_order_time = 0
                    for i in range(ind + 1, len(orders)):
                        if orders[i].agent_id == order.agent_id and orders[i].order_id != order.order_id and orders[i].time > order.time:
                            latest_order_time = orders[i].time
                            withdrawn = True
                            break
                    if not withdrawn:
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
                            RBG += (time_till_withdrawal / self.grace_period) * order.quantity
        
        if TBL + AL == 0:
            return 0
        else:
            return (TBL + AL) / (TBL + AL + RBG)

    def get_order_list(self):
        self.lower_bound_mem = self.get_last_trade_time_step()
        buy_orders_memory = []
        sell_orders_memory = []
        last_L_orders = []
        for time in range(self.lower_bound_mem, self.market.get_time() + 1):
            last_L_orders.extend(self.market.event_queue.scheduled_activities[time])
        buy_orders_memory = [order for order in last_L_orders if order.order_type == BUY]
        sell_orders_memory = [order for order in last_L_orders if order.order_type == SELL]
        return last_L_orders, buy_orders_memory, sell_orders_memory

    def determine_optimal_price(self, side):
        last_L_orders, buy_orders_memory, sell_orders_memory = self.get_order_list()
        last_L_orders = np.array(last_L_orders)
        estimate = self.estimate_fundamental()
        buy_orders_memory = sorted(buy_orders_memory, key=lambda order: order.price)
        sell_orders_memory = sorted(sell_orders_memory, key=lambda order: order.price)
        best_ask = float(self.market.order_book.sell_unmatched.peek())
        best_buy = float(self.market.order_book.buy_unmatched.peek())
        
        spline_interp_objects = [[], []]
        private_value = self.pv.value_for_exchange(self.position, BUY)
        best_buy_belief = self.belief_function(best_buy, BUY, last_L_orders)
        best_ask_belief = 1
        
        def interpolate(bound1, bound2, bound1Belief, bound2Belief):
            cs = FCS(bound1, bound2, [bound1Belief, bound2Belief])
            spline_interp_objects[0].append(cs)
            spline_interp_objects[1].append((bound1, bound2))

        def expected_surplus_max():
            def optimize(price):
                for i in range(len(spline_interp_objects[0])):
                    if spline_interp_objects[1][i][0] <= price <= spline_interp_objects[1][i][1]:
                        return -((estimate + private_value - price) * spline_interp_objects[0][i](price))
                return 0

            if len(spline_interp_objects[1]) == 0:
                return best_ask, 0

            lb = min(spline_interp_objects[1], key=lambda bound_pair: bound_pair[0])[0]
            ub = max(spline_interp_objects[1], key=lambda bound_pair: bound_pair[1])[1]

            test_points = np.linspace(lb, ub, 40)
            vOptimize = np.vectorize(optimize)
            point_surpluses = vOptimize(test_points)
            min_index = np.argmin(point_surpluses)
            min_survey = test_points[min_index]
            
            max_x = sp.optimize.minimize(vOptimize, min_survey, bounds=[[lb, ub]])
            return max_x.x.item(), -max_x.fun

        buy_high = float(buy_orders_memory[-1].price)
        buy_high_belief = self.belief_function(buy_high, BUY, last_L_orders)
        buy_low, buy_low_belief = self.find_worst_order(BUY, buy_orders_memory, last_L_orders)
        optimal_price = (0, -sys.maxsize)

        if buy_high >= best_ask:
            buy_high = best_ask
            buy_high_belief = best_ask_belief
            buy_low = min(buy_high, buy_low)
            buy_low_belief = min(buy_high_belief, buy_low_belief)
        
        if buy_high >= best_buy:
            if best_ask != buy_high:
                interpolate(buy_high, best_ask, buy_high_belief, 1)
            if best_buy >= buy_low:
                buy_mid = buy_low + self.buy_upper_mid_shade * abs(best_buy - buy_low)
                buy_mid_belief = self.belief_function(buy_mid, BUY, last_L_orders)
                buy_half = buy_low + self.buy_half_shade * abs(best_buy - buy_low)
                buy_half_belief = self.belief_function(buy_half, BUY, last_L_orders)
                if best_buy != buy_high:
                    interpolate(best_buy, buy_high, best_buy_belief, buy_high_belief)
                if best_buy != buy_mid:
                    interpolate(buy_low, buy_half, buy_low_belief, buy_half_belief)
                    interpolate(buy_half, buy_mid, buy_half_belief, buy_mid_belief)
                    interpolate(buy_mid, best_buy, buy_mid_belief, best_buy_belief)
                if buy_low_belief > 0:
                    lower_bound = max(buy_low - 2 * (buy_high - buy_low) - 1, 0)
                    interpolate(lower_bound, buy_low, 0, buy_low_belief)
            elif best_buy < buy_low:
                if buy_high != buy_low:
                    interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
                if buy_low != best_buy:
                    interpolate(best_buy, buy_low, best_buy_belief, buy_low_belief)
                if best_buy_belief > 0:
                    lower_bound = max(best_buy - 2 * (buy_high - best_buy) - 1, 0)
                    buy_mid = lower_bound + self.buy_upper_mid_shade * abs(best_buy - lower_bound)
                    buy_mid_belief = self.belief_function(buy_mid, BUY, last_L_orders)
                    buy_half = lower_bound + self.buy_half_shade * abs(best_buy - lower_bound)
                    buy_half_belief = self.belief_function(buy_half, BUY, last_L_orders)
                    interpolate(buy_mid, best_buy, buy_mid_belief, best_buy_belief)
                    interpolate(buy_half, buy_mid, buy_half_belief, buy_mid_belief)
                    interpolate(lower_bound, buy_half, 0, buy_half_belief)
        elif buy_high < best_buy:
            buy_mid = buy_high + self.buy_upper_mid_shade * abs(best_buy - buy_high)
            buy_mid_belief = self.belief_function(buy_mid, BUY, last_L_orders)
            buy_half = buy_high + self.buy_half_shade * abs(best_buy - buy_high)
            buy_half_belief = self.belief_function(buy_half, BUY, last_L_orders)
            if best_ask != best_buy:
                interpolate(best_buy, best_ask, best_buy_belief, best_ask_belief)
            if best_buy != buy_high:
                interpolate(buy_high, buy_half, buy_high_belief, buy_half_belief)
                interpolate(buy_half, buy_mid, buy_half_belief, buy_mid_belief)
                interpolate(buy_mid, best_buy, buy_mid_belief, best_buy_belief)
            if buy_high != buy_low:
                interpolate(buy_low, buy_high, buy_low_belief, buy_high_belief)
            if buy_low_belief > 0:
                lower_bound = max(buy_low - 2 * (buy_high - buy_low) - 1, 0)
                interpolate(lower_bound, buy_low, 0, buy_low_belief)

        optimal_price = expected_surplus_max()

        if optimal_price[0] > estimate + private_value:
            return estimate + private_value, -1
        
        return optimal_price[0], optimal_price[1]

    def take_action(self, seed=0):
        t = self.market.get_time()
        random.seed(t + self.agent_id + seed)
        estimate = self.estimate_fundamental()
        transaction_count = len(self.market.matched_orders) // 2
        
        # CHECK: Has this position already been consumed in a trade?
        # If so, do NOT trade (each position can only be used once)
        if self.position in self.consumed_buy_positions:
            if self.debug:
                print(f"[HBLBuy{self.agent_id} t={t}] SKIPPED: position {self.position} already consumed")
            return []
        
        private_value = self.pv.value_for_exchange(self.position, BUY)
        best_bid = self.market.order_book.buy_unmatched.peek() if self.market.order_book.buy_unmatched.peek_order() is not None else None
        best_ask = self.market.order_book.sell_unmatched.peek() if self.market.order_book.sell_unmatched.peek_order() is not None else None
        
        if transaction_count >= self.L and best_bid is not None and best_ask is not None:
            if getattr(self, 'position', 0) >= self.q_max:
                if self.debug:
                    print(f"[HBLBuy{self.agent_id} t={t}] SKIPPED: position={self.position} >= q_max={self.q_max}")
                return []
            opt_price, opt_price_est_surplus = self.determine_optimal_price(BUY)
            used_fallback = False
            
            if self.debug:
                print(f"[HBLBuy{self.agent_id} t={t}] HBL MODE: estimate={estimate:.2f}, pv={private_value:.2f}, "
                      f"best_bid={best_bid:.2f}, best_ask={best_ask:.2f}, opt_price={opt_price:.2f}, "
                      f"exp_surplus={opt_price_est_surplus:.2f}, position={self.position}, txn_count={transaction_count}")
        else:
            # ZI Agent fallback (exact ZI behavior)
            used_fallback = True
            if getattr(self, 'position', 0) >= self.q_max:
                if self.debug:
                    print(f"[HBLBuy{self.agent_id} t={t}] SKIPPED: position={self.position} >= q_max={self.q_max}")
                return []
            shade_max = self.shade[1] if len(self.shade) > 1 else self.shade[0]
            shade_amount = random.uniform(0, shade_max)
            opt_price = estimate + private_value - shade_amount
            
            if self.debug:
                print(f"[HBLBuy{self.agent_id} t={t}] ZI FALLBACK: estimate={estimate:.2f}, pv={private_value:.2f}, "
                      f"shade={shade_amount:.2f}, opt_price={opt_price:.2f}, position={self.position}, "
                      f"txn_count={transaction_count}, best_bid={best_bid}, best_ask={best_ask}")
        
        order = Order(
            price=opt_price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=BUY,
            order_id=random.randint(1, 10000000)
        )
        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def reset(self):
        self.position = 0
        self.cash = 0
        self.consumed_buy_positions = set()  # Reset consumed positions tracking
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="buyer")

    def __str__(self):
        return f'HBLBuy{self.agent_id}'


class HBLAgentSell(Agent):
    """HBL Agent that only submits sell orders. Adapted from HBLAgent."""
    
    def __init__(self, agent_id: int, market: Market, q_max: int, shade: List, L: int, 
                 pv_var: float, arrival_rate: float, obs_noise_var: float = 0.0, debug: bool = False):
        self.agent_id = agent_id
        self.market = market
        self.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="seller")
        self.position = 0
        self.shade = shade
        self.cash = 0
        self.L = L
        self.grace_period = 1 / arrival_rate
        self.lower_bound_mem = 0
        
        # HBL tuning parameters
        self.sell_upper_mid_shade = 99/100
        self.sell_half_shade = 1/2
        
        self.q_max = q_max
        self.pv_var = pv_var
        self.obs_noise_var = obs_noise_var
        self.debug = debug
        
        # Track which positions have been consumed in actual trades
        # Each position (0, 1, 2 for q_max=3) can only be used ONCE
        self.consumed_sell_positions = set()  # Tracks which SELL positions have been traded

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self):
        mean, r, T = self.market.get_info()
        t = self.market.get_time()
        val = self.market.get_fundamental_value()
        rho = (1 - r) ** (T - t)
        estimate = (1 - rho) * mean + rho * val
        if self.obs_noise_var > 0:
            estimate += np.random.normal(0, np.sqrt(self.obs_noise_var))
        return estimate

    def get_last_trade_time_step(self):
        if len(self.market.matched_orders) < 2 * self.L:
            return 0
        last_matched_order_ind = len(self.market.matched_orders) - self.L * 2
        earliest_order = min(
            self.market.matched_orders[last_matched_order_ind:],
            key=lambda matched_order: matched_order.order.time
        ).order.time
        return earliest_order

    def fast_belief_function(self, p, side, orders):
        TAG = 0  # Transact ask greater or equal
        BG = 0  # Bid greater or equal
        for ind, order in enumerate(orders):
            if order.price >= p and order.order_type == BUY:
                BG += order.quantity
            for matched_order in self.market.matched_orders:
                if order.order_id == matched_order.order.order_id:
                    if matched_order.order.order_type == SELL and matched_order.price >= p:
                        TAG += order.quantity
                    break
        return BG + TAG == 0

    def find_worst_order(self, side, order_mem, orders: List[Order]):
        beginning = 0
        end = len(order_mem) - 1
        while beginning < end:
            mid = (beginning + end) // 2
            mid_belief = self.fast_belief_function(order_mem[mid].price, side, orders)
            if mid != len(order_mem) - 1:
                if mid_belief:
                    if not self.fast_belief_function(order_mem[mid + 1].price, side, orders):
                        return order_mem[mid].price, 0
                    if beginning == mid and mid_belief:
                        return order_mem[mid + 1].price, 0
                    beginning = mid
                else:
                    end = mid
            else:
                return order_mem[0].price, self.belief_function(order_mem[0].price, side, orders)
        return order_mem[0].price, self.belief_function(order_mem[0].price, side, orders)

    def belief_function(self, p, side, orders):
        current_time = self.market.get_time()
        TAG = 0  # Transact ask greater or equal
        BG = 0  # Bid greater or equal
        RAL = 0  # Reject ask less or equal

        for order in orders:
            if order.price - p >= 0 and order.order_type == BUY:
                BG += order.quantity

        for ind, order in enumerate(orders):
            found_matched = False
            for matched_order in self.market.matched_orders:
                if order.order_id == matched_order.order.order_id:
                    if matched_order.order.order_type == SELL and matched_order.price - p >= 0:
                        TAG += order.quantity
                    found_matched = True
                    break
            if not found_matched:
                if order.order_type == SELL and order.price - p <= 0:
                    withdrawn = False
                    latest_order_time = 0
                    for i in range(ind + 1, len(orders)):
                        if orders[i].agent_id == order.agent_id:
                            latest_order_time = orders[i].time
                            withdrawn = True
                            break
                    if not withdrawn:
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
                            RAL += (time_till_withdrawal / self.grace_period) * order.quantity
        
        if TAG + BG == 0:
            return 0
        else:
            return (TAG + BG) / (TAG + BG + RAL)

    def get_order_list(self):
        self.lower_bound_mem = self.get_last_trade_time_step()
        buy_orders_memory = []
        sell_orders_memory = []
        last_L_orders = []
        for time in range(self.lower_bound_mem, self.market.get_time() + 1):
            last_L_orders.extend(self.market.event_queue.scheduled_activities[time])
        buy_orders_memory = [order for order in last_L_orders if order.order_type == BUY]
        sell_orders_memory = [order for order in last_L_orders if order.order_type == SELL]
        return last_L_orders, buy_orders_memory, sell_orders_memory

    def determine_optimal_price(self, side):
        last_L_orders, buy_orders_memory, sell_orders_memory = self.get_order_list()
        last_L_orders = np.array(last_L_orders)
        estimate = self.estimate_fundamental()
        buy_orders_memory = sorted(buy_orders_memory, key=lambda order: order.price)
        sell_orders_memory = sorted(sell_orders_memory, key=lambda order: order.price)
        best_ask = float(self.market.order_book.sell_unmatched.peek())
        best_buy = float(self.market.order_book.buy_unmatched.peek())
        
        spline_interp_objects = [[], []]
        private_value = self.pv.value_for_exchange(self.position, SELL)
        best_buy_belief = 1
        best_ask_belief = self.belief_function(best_ask, SELL, last_L_orders)
        sell_high, sell_high_belief = self.find_worst_order(SELL, sorted(sell_orders_memory, key=lambda order: order.price, reverse=True), last_L_orders)
        sell_low = float(sell_orders_memory[0].price)
        sell_low_belief = self.belief_function(sell_low, SELL, last_L_orders)
        
        def interpolate(bound1, bound2, bound1Belief, bound2Belief):
            cs = FCS(bound1, bound2, [bound1Belief, bound2Belief])
            spline_interp_objects[0].append(cs)
            spline_interp_objects[1].append((bound1, bound2))
            
        def expected_surplus_max():
            def optimize(price):
                for i in range(len(spline_interp_objects[0])):
                    if spline_interp_objects[1][i][0] <= price <= spline_interp_objects[1][i][1]:
                        return -((price - (estimate + private_value)) * spline_interp_objects[0][i](price))
                return 0

            if len(spline_interp_objects[1]) == 0:
                return best_buy, 0

            lb = min(spline_interp_objects[1], key=lambda bound_pair: bound_pair[0])[0]
            ub = max(spline_interp_objects[1], key=lambda bound_pair: bound_pair[1])[1]
            test_points = np.linspace(lb, ub, 40)
            vOptimize = np.vectorize(optimize)
            point_surpluses = vOptimize(test_points)
            min_index = np.argmin(point_surpluses)
            min_survey = test_points[min_index]
            max_x = sp.optimize.minimize(vOptimize, min_survey, bounds=[[lb, ub]])
            return max_x.x.item(), -max_x.fun

        if best_buy > sell_low:
            sell_low = best_buy
            sell_low_belief = 1
            sell_high = max(sell_high, sell_low)
            sell_high_belief = min(sell_high_belief, sell_low_belief)

        if sell_low <= best_ask:
            if sell_low != best_buy:
                interpolate(best_buy, sell_low, best_buy_belief, sell_low_belief)
            if best_ask <= sell_high:
                if sell_low != best_ask:
                    sell_mid = sell_low + self.sell_upper_mid_shade * abs(best_ask - sell_low)
                    sell_mid_belief = self.belief_function(sell_mid, SELL, last_L_orders)
                    sell_half = sell_low + self.sell_half_shade * abs(best_ask - sell_low)
                    sell_half_belief = self.belief_function(sell_half, SELL, last_L_orders)
                    if sell_low != sell_half:
                        interpolate(sell_low, sell_half, sell_low_belief, sell_half_belief)
                    if sell_half != sell_mid:
                        interpolate(sell_half, sell_mid, sell_half_belief, sell_mid_belief)
                    if sell_mid != best_ask:
                        interpolate(sell_mid, best_ask, sell_mid_belief, best_ask_belief)
                if best_ask != sell_high:
                    interpolate(best_ask, sell_high, best_ask_belief, sell_high_belief)
                if sell_high_belief > 0:
                    upper_bound = sell_high + 2 * (sell_high - best_buy) + 1
                    interpolate(sell_high, upper_bound, sell_high_belief, 0)
            elif best_ask > sell_high:
                if sell_low != sell_high:
                    interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
                if sell_high != best_ask:
                    sell_mid = sell_high + self.sell_upper_mid_shade * abs(best_ask - sell_high)
                    sell_mid_belief = self.belief_function(sell_mid, SELL, last_L_orders)
                    sell_half = sell_high + self.sell_half_shade * abs(best_ask - sell_high)
                    sell_half_belief = self.belief_function(sell_half, SELL, last_L_orders)
                    if sell_high != sell_half:
                        interpolate(sell_high, sell_half, sell_high_belief, sell_half_belief)
                    if sell_half != sell_mid:
                        interpolate(sell_half, sell_mid, sell_half_belief, sell_mid_belief)
                    if sell_mid != best_ask:
                        interpolate(sell_mid, best_ask, sell_mid_belief, best_ask_belief)
                if best_ask_belief > 0:
                    upper_bound = best_ask + 2 * (best_ask - best_buy) + 1
                    interpolate(best_ask, upper_bound, best_ask_belief, 0)
        elif sell_low > best_ask:
            if best_buy != best_ask:
                sell_mid = best_buy + self.sell_upper_mid_shade * abs(best_ask - best_buy)
                sell_mid_belief = self.belief_function(sell_mid, SELL, last_L_orders)
                sell_half = best_buy + self.sell_half_shade * abs(best_ask - best_buy)
                sell_half_belief = self.belief_function(sell_half, SELL, last_L_orders)
                interpolate(best_buy, sell_half, best_buy_belief, sell_half_belief)
                interpolate(sell_half, sell_mid, sell_half_belief, sell_mid_belief)
                interpolate(sell_mid, best_ask, sell_low_belief, best_ask_belief)
            if best_ask != sell_low:
                interpolate(best_ask, sell_low, best_ask_belief, sell_low_belief)
            if sell_low != sell_high:
                interpolate(sell_low, sell_high, sell_low_belief, sell_high_belief)
            if sell_high_belief > 0:
                upper_bound = sell_high + 2 * (sell_high - best_buy) + 1
                interpolate(sell_high, upper_bound, sell_high_belief, 0)
        
        optimal_price = expected_surplus_max()

        if optimal_price[0] < estimate + private_value:
            return estimate + private_value, 0
        
        return optimal_price[0], optimal_price[1]

    def take_action(self, seed=0):
        t = self.market.get_time()
        random.seed(t + self.agent_id + seed)
        estimate = self.estimate_fundamental()
        transaction_count = len(self.market.matched_orders) // 2
        
        # CHECK: Has this position already been consumed in a trade?
        # If so, do NOT trade (each position can only be used once)
        sell_index = abs(self.position)
        if sell_index in self.consumed_sell_positions:
            if self.debug:
                print(f"[HBLSell{self.agent_id} t={t}] SKIPPED: position {self.position} already consumed")
            return []
        
        private_value = self.pv.value_for_exchange(self.position, SELL)
        best_bid = self.market.order_book.buy_unmatched.peek() if self.market.order_book.buy_unmatched.peek_order() is not None else None
        best_ask = self.market.order_book.sell_unmatched.peek() if self.market.order_book.sell_unmatched.peek_order() is not None else None
        
        if transaction_count >= self.L and best_bid is not None and best_ask is not None:
            if getattr(self, 'position', 0) <= -self.q_max:
                if self.debug:
                    print(f"[HBLSell{self.agent_id} t={t}] SKIPPED: position={self.position} <= -q_max={-self.q_max}")
                return []
            opt_price, opt_price_est_surplus = self.determine_optimal_price(SELL)
            used_fallback = False
            
            if self.debug:
                print(f"[HBLSell{self.agent_id} t={t}] HBL MODE: estimate={estimate:.2f}, pv={private_value:.2f}, "
                      f"best_bid={best_bid:.2f}, best_ask={best_ask:.2f}, opt_price={opt_price:.2f}, "
                      f"exp_surplus={opt_price_est_surplus:.2f}, position={self.position}, txn_count={transaction_count}")
        else:
            # ZI Agent fallback (exact ZI behavior)
            used_fallback = True
            if getattr(self, 'position', 0) <= -self.q_max:
                if self.debug:
                    print(f"[HBLSell{self.agent_id} t={t}] SKIPPED: position={self.position} <= -q_max={-self.q_max}")
                return []
            shade_amount = random.uniform(0, self.shade[1])
            opt_price = estimate + private_value + shade_amount
            
            if self.debug:
                print(f"[HBLSell{self.agent_id} t={t}] ZI FALLBACK: estimate={estimate:.2f}, pv={private_value:.2f}, "
                      f"shade={shade_amount:.2f}, opt_price={opt_price:.2f}, position={self.position}, "
                      f"txn_count={transaction_count}, best_bid={best_bid}, best_ask={best_ask}")
        
        order = Order(
            price=opt_price,
            quantity=1,
            agent_id=self.get_id(),
            time=t,
            order_type=SELL,
            order_id=random.randint(1, 10000000)
        )
        return [order]

    def update_position(self, q, p):
        self.position += q
        self.cash += p

    def reset(self):
        self.position = 0
        self.cash = 0
        self.consumed_sell_positions = set()  # Reset consumed positions tracking
        self.pv = SeparatedPrivateValues(self.q_max, val_var=self.pv_var, base_price=0.0, role="seller")

    def __str__(self):
        return f'HBLSell{self.agent_id}'
