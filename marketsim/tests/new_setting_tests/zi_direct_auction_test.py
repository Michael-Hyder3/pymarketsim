"""
Direct Auction Test using ZIAgentBuy and ZIAgentSell.
Constant fundamental value. Agents interact through private values + shading.
"""

from marketsim.market.market import Market
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.fourheap.constants import BUY, SELL
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
import matplotlib.pyplot as plt
import numpy as np
import os
import random


class DummyFundamental:
    """Dummy fundamental that returns a constant value (not used in pricing)."""
    def __init__(self, value=100000, final_time=100):
        self.value = value
        self.final_time = final_time
        self.mean = value
        self.r = 0.0

    def get_value(self):
        return self.value

    def get_value_at(self, time):
        return self.value

    def get_fundamental_values(self):
        return {0: self.value}

    def get_final_fundamental(self):
        return self.value

    def get_r(self):
        return self.r

    def get_mean(self):
        return self.mean

    def get_info(self):
        """Returns (mean, r, final_time) as expected by agents."""
        return self.mean, self.r, self.final_time


def run_direct_auction_with_agents(num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps):
    """
    Run a direct auction using ZIAgentBuy and ZIAgentSell.
    Agents bid/ask based on private values + shading.
    """
    # Create a dummy market with no real fundamental (agents won't use it)
    # set fundamental mean to 0 so agents' estimates don't offset absolute PVs
    dummy_fundamental = DummyFundamental(0, final_time=timesteps)
    market = Market(fundamental=dummy_fundamental, time_steps=timesteps)

    # Create agents
    agents = {}
    id_to_agent = {}
    
    for i in range(num_buyers):
        agent = ZIAgentBuy(agent_id=i, market=market, q_max=q_max, shade=shade_range, pv_var=pv_var)
        agents[f'buy_{i}'] = agent
        id_to_agent[agent.get_id()] = agent
    for i in range(num_sellers):
        agent = ZIAgentSell(agent_id=num_buyers + i, market=market, q_max=q_max, shade=shade_range, pv_var=pv_var)
        agents[f'sell_{i}'] = agent
        id_to_agent[agent.get_id()] = agent

    # Override all agents' private values with separated absolute-price private values
    base_price = 100000.0
    for name, a in agents.items():
        role = "buyer" if name.startswith("buy_") else "seller"
        a.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=base_price, role=role)
        a.estimate_fundamental = lambda: 0.0

    transactions = []
    pv_consumption_log = []

    print(f"\n--- Direct Auction with ZI Agents: {num_buyers} buyers, {num_sellers} sellers, {timesteps} timesteps ---\n")
    print("Initial Private Values (first 2 buyers, first 2 sellers):\n")
    for i in range(min(4, num_buyers)):
        print(f"Buyer {i} values (desc): {agents[f'buy_{i}'].pv.buyer_values}")
    for i in range(min(4, num_sellers)):
        print(f"Seller {i} costs (asc): {agents[f'sell_{i}'].pv.seller_costs}")

    # compute initial medians across agents (before any trades)
    buy_vals = []
    sell_vals = []
    for name, a in agents.items():
        if name.startswith('buy_'):
            buy_vals.append(a.pv.buyer_values.numpy())
        else:
            sell_vals.append(a.pv.seller_costs.numpy())

    if len(buy_vals) > 0:
        all_buy = np.concatenate(buy_vals)
        median_buy = float(np.median(all_buy))
    else:
        median_buy = 0.0

    if len(sell_vals) > 0:
        all_sell = np.concatenate(sell_vals)
        median_sell = float(np.median(all_sell))
    else:
        median_sell = 0.0

    # combined median across all buyer values and seller costs
    try:
        combined_vals = np.concatenate([all_buy, all_sell]) if (len(buy_vals) > 0 and len(sell_vals) > 0) else (all_buy if len(buy_vals) > 0 else (all_sell if len(sell_vals) > 0 else np.array([0.0])))
        median_combined = float(np.median(combined_vals))
    except Exception:
        median_combined = 0.0

    # Run simulation
    for t in range(timesteps):
        market.event_queue.set_time(t)

        # Each agent submits orders. Shuffle order each timestep so neither side
        # (buyers vs sellers) always has time priority.
        agent_items = list(agents.items())
        random.shuffle(agent_items)
        for agent_name, agent in agent_items:
            market.withdraw_all(agent.get_id())
            orders = agent.take_action()
            market.add_orders(orders)

        # Record unmatched book before matching
        bids = [
            (order.price, order.quantity, order.agent_id)
            for _, oid in market.order_book.buy_unmatched.heap
            if oid not in market.order_book.buy_unmatched.deleted_ids
            for order in [market.order_book.buy_unmatched.order_dict[oid]]
        ]
        asks = [
            (order.price, order.quantity, order.agent_id)
            for _, oid in market.order_book.sell_unmatched.heap
            if oid not in market.order_book.sell_unmatched.deleted_ids
            for order in [market.order_book.sell_unmatched.order_dict[oid]]
        ]

        # Step market and match orders
        matched_orders = market.step()

        # Pair matched orders and capture the original orders so we can
        # set the transaction price to the price of the order that arrived first.
        paired_trades = {}
        for mo in matched_orders:
            key = (mo.time, mo.order.quantity)
            if key not in paired_trades:
                paired_trades[key] = {
                    'time': mo.time,
                    'quantity': mo.order.quantity,
                    'buyer_id': None,
                    'seller_id': None,
                    'buyer_order': None,
                    'seller_order': None
                }
            if mo.order.order_type == BUY:
                paired_trades[key]['buyer_id'] = mo.order.agent_id
                paired_trades[key]['buyer_order'] = mo.order
            else:
                paired_trades[key]['seller_id'] = mo.order.agent_id
                paired_trades[key]['seller_order'] = mo.order

        # Process paired trades: determine trade price by earlier order,
        # consume PVs and log events
        for trade in paired_trades.values():
            # determine trade price from earlier order arrival
            buyer_order = trade.get('buyer_order')
            seller_order = trade.get('seller_order')
            if buyer_order is not None and seller_order is not None:
                # earlier by time wins; if same timestep, use arrival sequence
                if buyer_order.time < seller_order.time:
                    trade_price = buyer_order.price
                elif buyer_order.time > seller_order.time:
                    trade_price = seller_order.price
                else:
                    b_seq = getattr(buyer_order, '_arrival_seq', 0)
                    s_seq = getattr(seller_order, '_arrival_seq', 0)
                    if b_seq <= s_seq:
                        trade_price = buyer_order.price
                    else:
                        trade_price = seller_order.price
            elif buyer_order is not None:
                trade_price = buyer_order.price
            elif seller_order is not None:
                trade_price = seller_order.price
            else:
                trade_price = None

            # record transaction (use trade_price)
            rec = {
                'time': trade['time'],
                'buyer_id': trade['buyer_id'],
                'seller_id': trade['seller_id'],
                'price': trade_price,
                'quantity': trade['quantity']
            }
            transactions.append(rec)

            # Buyer side
            if trade['buyer_id'] is not None:
                buyer = id_to_agent.get(trade['buyer_id'])
                if buyer is not None:
                    # log the buyer-side PVs explicitly so consumed marginal is visible
                    try:
                        buyer_pv_before = buyer.pv.buyer_values.clone()
                    except Exception:
                        buyer_pv_before = buyer.pv.values.clone()
                    buyer_pos_before = buyer.position
                    consumed_val = buyer.pv.consume_marginal(buyer_pos_before, BUY)
                    buyer.update_position(1, -trade_price)

                    try:
                        buyer_pv_after = buyer.pv.buyer_values.clone()
                    except Exception:
                        buyer_pv_after = buyer.pv.values.clone()

                    pv_consumption_log.append({
                        'time': t,
                        'agent_id': trade['buyer_id'],
                        'side': 'BUY',
                        'pv_before': buyer_pv_before,
                        'pv_after': buyer_pv_after,
                        'consumed_value': consumed_val,
                        'position_before': buyer_pos_before,
                        'position_after': buyer.position,
                        'price': trade_price
                    })

            # Seller side
            if trade['seller_id'] is not None:
                seller = id_to_agent.get(trade['seller_id'])
                if seller is not None:
                    try:
                        seller_pv_before = seller.pv.seller_costs.clone()
                    except Exception:
                        seller_pv_before = seller.pv.values.clone()
                    seller_pos_before = seller.position
                    consumed_val = seller.pv.consume_marginal(seller_pos_before, SELL)
                    seller.update_position(-1, trade_price)

                    try:
                        seller_pv_after = seller.pv.seller_costs.clone()
                    except Exception:
                        seller_pv_after = seller.pv.values.clone()

                    pv_consumption_log.append({
                        'time': t,
                        'agent_id': trade['seller_id'],
                        'side': 'SELL',
                        'pv_before': seller_pv_before,
                        'pv_after': seller_pv_after,
                        'consumed_value': consumed_val,
                        'position_before': seller_pos_before,
                        'position_after': seller.position,
                        'price': trade_price
                    })

    # Print summary
    print(f"\nTotal transactions: {len(transactions)}")
    print("\nFirst 10 transactions:")
    for i, tx in enumerate(transactions[:10]):
        print(f"  {i}: time={tx['time']}, buyer={tx['buyer_id']}, seller={tx['seller_id']}, price={tx['price']:.2f}")

    print("\n\nPrivate Value Consumption Log (first 10 events):")
    for i, event in enumerate(pv_consumption_log[:10]):
        print(f"\nEvent {i}:")
        print(f"  Time: {event['time']}, Agent: {event['agent_id']} ({event['side']}), Price: {event['price']:.2f}")
        print(f"  Position: {event['position_before']} -> {event['position_after']}")
        print(f"  Consumed marginal value: {event['consumed_value']:.2f}")
        print(f"  PV before: {event['pv_before']}")
        print(f"  PV after:  {event['pv_after']}")

    print(f"\n\nFinal Agent Summary:")
    print("\nBuyers:")
    for i in range(min(4, num_buyers)):
        agent = agents[f'buy_{i}']
        print(f"  Buyer {i}: position={agent.position}, remaining PV len={len(agent.pv.values)}")
    print("\nSellers:")
    for i in range(min(4, num_sellers)):
        agent = agents[f'sell_{i}']
        print(f"  Seller {i}: position={agent.position}, remaining PV len={len(agent.pv.values)}")

    return transactions, pv_consumption_log, agents, median_buy, median_sell, median_combined


if __name__ == '__main__':
    # Simulation parameters
    num_buyers = 4
    num_sellers = 4
    q_max = 3
    pv_var = 1e5  # Smaller variance for more reasonable valuations
    shade_range = [0, 100]  # Shade range (buyers down, sellers up)
    timesteps = 100

    transactions, pv_log, agents, median_buy, median_sell, median_combined = run_direct_auction_with_agents(
        num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps
    )

    # Plot transaction prices over time and add median lines
    times = [tx['time'] for tx in transactions]
    prices = [tx['price'] for tx in transactions]

    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(10, 6))
    if len(times) > 0:
        plt.scatter(times, prices, s=20, alpha=0.7, label='Transactions')
    #plt.axhline(median_buy, color='green', linestyle='--', label='Median buyer value')
    #plt.axhline(median_sell, color='red', linestyle='--', label='Median seller cost')
    plt.axhline(median_combined, color='blue', linestyle=':', label='Median combined')
    plt.xlabel('Time')
    plt.ylabel('Transaction Price')
    plt.title('ZI Direct Auction: Transaction Prices over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/zi_direct_transactions.png')
    try:
        plt.show()
    except Exception:
        pass
