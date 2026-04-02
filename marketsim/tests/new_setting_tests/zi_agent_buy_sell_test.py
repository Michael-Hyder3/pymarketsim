import matplotlib.pyplot as plt
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.fourheap.constants import BUY, SELL
import numpy as np
from marketsim.private_values.separated_private_values import SeparatedPrivateValues

# Simulation parameters
num_buy_agents = 10
num_sell_agents = 10
sim_time = 500
q_max = 10
pv_var = 5e6
shade = [-200, 200]
mean = 1e5
r = 0.05
shock_var = 1e5

# Create market and fundamental
fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
market = Market(fundamental=fundamental, time_steps=sim_time)

# Create agents
agents = {}
for i in range(num_buy_agents):
    agents[f'buy_{i}'] = ZIAgentBuy(agent_id=i, market=market, q_max=q_max, shade=shade, pv_var=pv_var)
    # override PV to separated absolute-price PV and make agent ignore fundamental estimate
    agents[f'buy_{i}'].pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=100000.0, role="buyer")
    agents[f'buy_{i}'].estimate_fundamental = lambda: 0.0
for i in range(num_sell_agents):
    agents[f'sell_{i}'] = ZIAgentSell(agent_id=num_buy_agents + i, market=market, q_max=q_max, shade=shade, pv_var=pv_var)
    agents[f'sell_{i}'].pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=100000.0, role="seller")
    agents[f'sell_{i}'].estimate_fundamental = lambda: 0.0

# Run simulation
time_series = []
prices = []
volumes = []

for t in range(sim_time):
    market.event_queue.set_time(t)
    # Each agent acts once per timestep
    for agent_id, agent in agents.items():
        market.withdraw_all(agent.get_id())
        orders = agent.take_action()
        market.add_orders(orders)
    # Step market and record matched orders
    matched_orders = market.step()
    for mo in matched_orders:
        time_series.append(mo.time)
        prices.append(mo.price)
        volumes.append(mo.order.quantity)
"""
# Plot transactions over time
plt.figure(figsize=(10, 6))
plt.scatter(time_series, prices, s=np.array(volumes)*20, alpha=0.7, c='blue', label='Transaction')
plt.xlabel('Time')
plt.ylabel('Transaction Price')
plt.title('ZI Buy/Sell Agent Transactions Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# --- Offline Game Trajectory Simulation ---
print("\n--- Offline Game Trajectory Simulation (4 buyers, 4 sellers, 100 steps) ---\n")

num_buy_agents = 4
num_sell_agents = 4
sim_time = 100
q_max = 3
pv_var = 10000
shade = [-200, 200]
mean = 1000
r = 0.05
shock_var = 1e5

fundamental = LazyGaussianMeanReverting(mean=mean, final_time=sim_time, r=r, shock_var=shock_var)
market = Market(fundamental=fundamental, time_steps=sim_time)

agents = {}
buy_private_values = {}
sell_private_values = {}
agent_shading = {}
id_to_agent = {}
for i in range(num_buy_agents):
    agent = ZIAgentBuy(agent_id=i, market=market, q_max=q_max, shade=shade, pv_var=pv_var)
    agents[f'buy_{i}'] = agent
    id_to_agent[agent.get_id()] = agent
    # override PVs to separated absolute price PVs and force agents to ignore fundamental
    agent.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=100000.0, role="buyer")
    agent.estimate_fundamental = lambda: 0.0
    buy_private_values[f'buy_{i}'] = agent.pv.buyer_values.clone()
    agent_shading[f'buy_{i}'] = agent.shade
for i in range(num_sell_agents):
    agent = ZIAgentSell(agent_id=num_buy_agents + i, market=market, q_max=q_max, shade=shade, pv_var=pv_var)
    agents[f'sell_{i}'] = agent
    id_to_agent[agent.get_id()] = agent
    agent.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=100000.0, role="seller")
    agent.estimate_fundamental = lambda: 0.0
    sell_private_values[f'sell_{i}'] = agent.pv.seller_costs.clone()
    agent_shading[f'sell_{i}'] = agent.shade

# Run simulation and record full trajectory
order_book_history = []
transaction_history = []

for t in range(sim_time):
    market.event_queue.set_time(t)
    for agent_id, agent in agents.items():
        market.withdraw_all(agent.get_id())
        orders = agent.take_action()
        market.add_orders(orders)
    # Record order book state before matching
    bids = [(order.price, order.quantity) for _, oid in market.order_book.buy_unmatched.heap if oid not in market.order_book.buy_unmatched.deleted_ids for order in [market.order_book.buy_unmatched.order_dict[oid]]]
    asks = [(order.price, order.quantity) for _, oid in market.order_book.sell_unmatched.heap if oid not in market.order_book.sell_unmatched.deleted_ids for order in [market.order_book.sell_unmatched.order_dict[oid]]]
    order_book_state = {
        'time': t,
        'bids': bids,
        'asks': asks
    }
    order_book_history.append(order_book_state)
    # Step market and record matched orders
    matched_orders = market.step()

    # Pair matched orders (buy+sell) into single records and update agents' PVs
    paired = {}
    for mo in matched_orders:
        key = (mo.time, mo.price, mo.order.quantity)
        if key not in paired:
            paired[key] = {
                'time': mo.time,
                'price': mo.price,
                'quantity': mo.order.quantity,
                'buyer_id': None,
                'seller_id': None
            }
        if mo.order.order_type == BUY:
            paired[key]['buyer_id'] = mo.order.agent_id
        else:
            paired[key]['seller_id'] = mo.order.agent_id

    # Process paired trades: record and consume marginal private values
    for rec in paired.values():
        transaction_history.append(rec)
        price = rec['price']
        # Buyer side
        if rec['buyer_id'] is not None:
            buyer = id_to_agent.get(rec['buyer_id'])
            if buyer is not None:
                # consume marginal based on current position, then update position/cash
                buyer.pv.consume_marginal(buyer.position, BUY)
                buyer.update_position(1, -price)
        # Seller side
        if rec['seller_id'] is not None:
            seller = id_to_agent.get(rec['seller_id'])
            if seller is not None:
                seller.pv.consume_marginal(seller.position, SELL)
                seller.update_position(-1, price)

# Output summary
print(f"Participants: {len(agents)}")
print(f"Buyers: {list(buy_private_values.keys())}")
print(f"Sellers: {list(sell_private_values.keys())}")
print(f"Agent shading: {agent_shading}")
print("\nPrivate values (buyers):")
for aid, vals in buy_private_values.items():
    print(f"  {aid}: {vals}")
print("\nPrivate values (sellers):")
for aid, vals in sell_private_values.items():
    print(f"  {aid}: {vals}")
print(f"\nTotal transactions: {len(transaction_history)}")
print("First 10 transactions:")
for tx in transaction_history[:10]:
    print(tx)
print(f"\nOrder book history (first 5 timesteps):")
for ob in order_book_history[:5]:
    print(ob)
