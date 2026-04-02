"""
Test the market order matching with simple buyers and sellers.
"""

import numpy as np
from marketsim.market.market import Market
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
import torch

# Create a simple market
fundamental = LazyGaussianMeanReverting(
    final_time=10,
    mean=100000.0,
    r=0.05,
    shock_var=1e5
)

market = Market(fundamental=fundamental, time_steps=10)
market.event_queue.set_time(0)

# Create 2 buyers and 2 sellers
buyers = []
sellers = []

for i in range(2):
    buyer = ZIAgentBuy(
        agent_id=i,
        market=market,
        q_max=1,
        shade=[-5000, 5000],
        pv_var=5e6,
        obs_noise_var=1e3
    )
    # Set private values manually
    pv_buy = SeparatedPrivateValues(1, val_var=5e6, base_price=0.0)
    pv_buy.buyer_values = torch.tensor([5000.0])  # High value
    buyer.pv = pv_buy
    buyers.append(buyer)
    
    seller = ZIAgentSell(
        agent_id=100+i,
        market=market,
        q_max=1,
        shade=[-5000, 5000],
        pv_var=5e6,
        obs_noise_var=1e3
    )
    # Set private values manually
    pv_sell = SeparatedPrivateValues(1, val_var=5e6, base_price=0.0)
    pv_sell.seller_costs = torch.tensor([-5000.0])  # Low cost
    seller.pv = pv_sell
    sellers.append(seller)

print("=" * 80)
print("TESTING MARKET ORDER MATCHING")
print("=" * 80)

print(f"\nBuyer 0: value = 5000, should place a bid")
print(f"Seller 0: cost = -5000, should place an ask")
print(f"These should definitely match!\n")

# Get orders from agents
buyer_orders = []
seller_orders = []

for buyer in buyers:
    orders = buyer.take_action()
    if orders:
        buyer_orders.extend(orders)
        print(f"Buyer {buyer.get_id()} placed bid: {orders[0].price:.2f}")
    else:
        print(f"Buyer {buyer.get_id()} placed no order")

for seller in sellers:
    orders = seller.take_action()
    if orders:
        seller_orders.extend(orders)
        print(f"Seller {seller.get_id()} placed ask: {orders[0].price:.2f}")
    else:
        print(f"Seller {seller.get_id()} placed no order")

print(f"\nAdding {len(buyer_orders)} buy orders and {len(seller_orders)} sell orders to market...")

for order in buyer_orders + seller_orders:
    market.add_orders([order])

print(f"Order book state: {market.order_book}")

print(f"\nCalling market.step()...")
matched_orders = market.step()

print(f"Matched orders: {len(matched_orders)}")
for mo in matched_orders:
    print(f"  Order: {mo}")
