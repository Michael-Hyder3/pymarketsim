"""
HBL Agent test with HBLAgentBuy and HBLAgentSell using absolute private values.
Tests the simulator with HBL agents, transaction logging, and social welfare computation.
"""

import random
import numpy as np
from marketsim.market.market import Market
from marketsim.agent.hbl_agent_buy_sell import HBLAgentBuy, HBLAgentSell
from marketsim.fourheap.constants import BUY, SELL
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fundamental.dummy_fundamental import DummyFundamental


# Use centralized DummyFundamental from marketsim.fundamental.dummy_fundamental


def run_hbl_test(num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, 
                 L=4, arrival_rate=0.1, fundamental_value=100000.0, shared_pv=None,
                 fundamental=None, debug=False, debug_steps=10, obs_noise_var=0.0):
    """
    Run a simulation with HBLAgentBuy and HBLAgentSell using absolute private values.
    
    Args:
        num_buyers: number of HBL buyer agents
        num_sellers: number of HBL seller agents
        q_max: max quantity each agent can trade
        pv_var: variance of private values
        shade_range: [min_shade, max_shade] for agent price shading
        timesteps: number of simulation timesteps
        L: memory parameter for HBL agents
        arrival_rate: arrival rate for grace period calculation
        fundamental_value: base value for absolute private values
        shared_pv: optional dict of pre-generated private values (keys: 'buy_i', 'sell_i')
        fundamental: optional fundamental model (if None, uses DummyFundamental)
        debug: whether to capture per-timestep order diagnostics
        debug_steps: number of initial timesteps to capture when debug is True
        obs_noise_var: observation noise variance for fundamental estimates (default 0.0)
    
    Returns:
        agents: dict of all agents
        transactions: list of transaction records with prices and PVs
        eq_price: equilibrium price from remaining marginals
        initial_pv: dict of initial private values (before any consumption)
        debug_info (if debug=True): list of per-timestep order summaries
    """
    # Create market with fundamental
    if fundamental is None:
        dummy_fundamental = DummyFundamental(value=fundamental_value, final_time=timesteps)
        market = Market(fundamental=dummy_fundamental, time_steps=timesteps)
    else:
        market = Market(fundamental=fundamental, time_steps=timesteps)

    # Use relative private values (base_price = 0) to match comparison_test
    pv_base_price = 0.0

    # Create agents with private values
    agents = {}
    id_to_agent = {}
    initial_pv = {} if shared_pv is None else shared_pv.copy()

    for i in range(num_buyers):
        agent = HBLAgentBuy(
            agent_id=i,
            market=market,
            q_max=q_max,
            shade=shade_range,
            L=L,
            pv_var=pv_var,
            arrival_rate=arrival_rate,
            obs_noise_var=obs_noise_var
        )
        # Create or use shared private values
        if shared_pv is not None and f'buy_{i}' in shared_pv:
            # Create a SeparatedPrivateValues object from the shared values
            import torch
            pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=pv_base_price, role="buyer")
            pv.buyer_values = torch.tensor(shared_pv[f'buy_{i}'])
            agent.pv = pv
        else:
            agent.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=pv_base_price, role="buyer")
            if shared_pv is None:
                initial_pv[f'buy_{i}'] = agent.pv.buyer_values.numpy().copy()
        
        agents[f'hbl_buy_{i}'] = agent
        id_to_agent[agent.get_id()] = agent

    for i in range(num_sellers):
        agent = HBLAgentSell(
            agent_id=num_buyers + i,
            market=market,
            q_max=q_max,
            shade=shade_range,
            L=L,
            pv_var=pv_var,
            arrival_rate=arrival_rate,
            obs_noise_var=obs_noise_var
        )
        # Create or use shared private values
        if shared_pv is not None and f'sell_{i}' in shared_pv:
            # Create a SeparatedPrivateValues object from the shared values
            import torch
            pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=pv_base_price, role="seller")
            pv.seller_costs = torch.tensor(shared_pv[f'sell_{i}'])
            agent.pv = pv
        else:
            agent.pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=pv_base_price, role="seller")
            if shared_pv is None:
                initial_pv[f'sell_{i}'] = agent.pv.seller_costs.numpy().copy()
        
        agents[f'hbl_sell_{i}'] = agent
        id_to_agent[agent.get_id()] = agent

    # Print initial private values
    print(f"\n--- HBL Agent Test: {num_buyers} HBL buyers, {num_sellers} HBL sellers, {timesteps} timesteps ---")
    print(f"L={L}, arrival_rate={arrival_rate}\n")
    # Commented out for performance
    # print("Initial Private Values:")
    # for i in range(min(4, num_buyers)):
    #     agent = agents[f'hbl_buy_{i}']
    #     print(f"  HBL Buyer {i}: values (desc) = {agent.pv.buyer_values.numpy()}")
    # for i in range(min(4, num_sellers)):
    #     agent = agents[f'hbl_sell_{i}']
    #     print(f"  HBL Seller {i}: costs (asc) = {agent.pv.seller_costs.numpy()}")
    # print()

    transactions = []
    debug_info = []
    agent_pvs_before = {}  # store initial PVs for social welfare calculation

    # Record initial private values per agent
    for agent_name, agent in agents.items():
        try:
            if agent_name.startswith('hbl_buy_'):
                agent_pvs_before[agent.get_id()] = {
                    'side': 'BUY',
                    'values': agent.pv.buyer_values.numpy().copy()
                }
            else:
                agent_pvs_before[agent.get_id()] = {
                    'side': 'SELL',
                    'values': agent.pv.seller_costs.numpy().copy()
                }
        except Exception as e:
            print(f"Warning: could not record initial PVs for {agent_name}: {e}")

    # Run simulation
    for t in range(timesteps):
        market.event_queue.set_time(t)
        
        # Pre-generate fundamental value for this timestep to avoid lazy generation issues
        _ = market.get_fundamental_value()

        # Determine which agents act this timestep using Poisson sampling
        active_agents = []
        for agent_name, agent in agents.items():
            if np.random.rand() < arrival_rate:
                active_agents.append((agent_name, agent))
        random.shuffle(active_agents)

        step_orders = []
        agents_with_no_orders = 0

        for agent_name, agent in active_agents:
            market.withdraw_all(agent.get_id())
            orders = agent.take_action()
            market.add_orders(orders)
            if debug and t < debug_steps:
                if not orders:
                    agents_with_no_orders += 1
                else:
                    step_orders.extend(orders)

        if debug and t < debug_steps:
            buy_prices = [o.price for o in step_orders if o.order_type == BUY]
            sell_prices = [o.price for o in step_orders if o.order_type == SELL]
            best_bid = max(buy_prices) if buy_prices else None
            best_ask = min(sell_prices) if sell_prices else None
            debug_info.append({
                'time': t,
                'total_orders': len(step_orders),
                'num_buy_orders': len(buy_prices),
                'num_sell_orders': len(sell_prices),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'agents_with_no_orders': agents_with_no_orders,
                'crossing_possible': (best_bid is not None and best_ask is not None and best_bid >= best_ask)
            })

        # Step market and match orders
        matched_orders = market.step()

        # Pair matched orders
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

        # Process paired trades: determine price by earlier order arrival
        for trade in paired_trades.values():
            buyer_order = trade.get('buyer_order')
            seller_order = trade.get('seller_order')

            # Determine transaction price from earlier order
            if buyer_order is not None and seller_order is not None:
                if buyer_order.time < seller_order.time:
                    trade_price = buyer_order.price
                    price_setter_side = 'BUY'
                elif buyer_order.time > seller_order.time:
                    trade_price = seller_order.price
                    price_setter_side = 'SELL'
                else:
                    b_seq = getattr(buyer_order, '_arrival_seq', 0)
                    s_seq = getattr(seller_order, '_arrival_seq', 0)
                    if b_seq <= s_seq:
                        trade_price = buyer_order.price
                        price_setter_side = 'BUY'
                    else:
                        trade_price = seller_order.price
                        price_setter_side = 'SELL'
            elif buyer_order is not None:
                trade_price = buyer_order.price
                price_setter_side = 'BUY'
            elif seller_order is not None:
                trade_price = seller_order.price
                price_setter_side = 'SELL'
            else:
                continue

            buyer_id = trade.get('buyer_id')
            seller_id = trade.get('seller_id')

            if buyer_id is None or seller_id is None:
                continue

            buyer = id_to_agent.get(buyer_id)
            seller = id_to_agent.get(seller_id)

            if buyer is None or seller is None:
                continue

            # Get buyer's value and seller's cost BEFORE updating positions
            # Use current position to index into private values
            buyer_position_at_trade = buyer.position
            seller_position_at_trade = seller.position
            buyer_value = buyer.pv.consume_marginal(buyer.position, BUY)
            seller_cost = seller.pv.consume_marginal(seller.position, SELL)

            # MARK these positions as consumed (each can only be used once)
            if hasattr(buyer, 'consumed_buy_positions'):
                buyer.consumed_buy_positions.add(buyer_position_at_trade)
            if hasattr(seller, 'consumed_sell_positions'):
                seller.consumed_sell_positions.add(abs(seller_position_at_trade))

            # Update positions AFTER getting the values
            buyer.update_position(1, -trade_price)
            seller.update_position(-1, trade_price)

            # Calculate surplus using relative private values
            # Private values are relative (offsets from fundamental value)
            # Convert trade price to relative terms
            fundamental_at_t = market.get_fundamental_value()
            relative_price = trade_price - fundamental_at_t
            
            # buyer surplus: (value + fundamental) - price = value - (price - fundamental)
            # seller surplus: price - (cost + fundamental) = (price - fundamental) - cost
            buyer_surplus = buyer_value - relative_price if buyer_value is not None else None
            seller_surplus = relative_price - seller_cost if seller_cost is not None else None
            total_surplus = (buyer_surplus + seller_surplus) if (buyer_surplus is not None and seller_surplus is not None) else None

            transaction = {
                'time': t,
                'buyer_id': buyer_id,
                'seller_id': seller_id,
                'buyer_position': buyer_position_at_trade,
                'seller_position': seller_position_at_trade,
                'buyer_value': buyer_value,
                'seller_cost': seller_cost,
                'price': trade_price,
                'price_setter_side': price_setter_side,
                'buyer_surplus': buyer_surplus,
                'seller_surplus': seller_surplus,
                'total_surplus': total_surplus
            }
            transactions.append(transaction)

    # Print transaction summary
    print(f"\nTotal transactions: {len(transactions)}\n")
    # Commented out for performance - detailed transaction output is expensive
    # print("Transaction Details (all trades):")
    # print(f"{'Time':<6} {'Buyer':<8} {'Seller':<8} {'Buyer Val':<12} {'Seller Cost':<12} {'Price':<12} {'Buyer Surplus':<15} {'Seller Surplus':<15} {'Total Surplus':<15}")
    # print("-" * 120)
    # for tx in transactions:
    #     print(
    #         f"{tx['time']:<6} "
    #         f"{tx['buyer_id']:<8} "
    #         f"{tx['seller_id']:<8} "
    #         f"{tx['buyer_value']:<12.2f}" if tx['buyer_value'] is not None else f"{'None':<12} ",
    #         end=""
    #     )
    #     print(
    #         f"{tx['seller_cost']:<12.2f}" if tx['seller_cost'] is not None else f"{'None':<12} ",
    #         end=""
    #     )
    #     print(
    #         f"{tx['price']:<12.2f} "
    #         f"{tx['buyer_surplus']:<15.2f}" if tx['buyer_surplus'] is not None else f"{'None':<15} ",
    #         end=""
    #     )
    #     print(
    #         f"{tx['seller_surplus']:<15.2f}" if tx['seller_surplus'] is not None else f"{'None':<15} ",
    #         end=""
    #     )
    #     print(
    #         f"{tx['total_surplus']:<15.2f}" if tx['total_surplus'] is not None else f"{'None':<15}"
    #     )

    # Calculate optimal surplus from initial private values
    all_buyer_values = []
    all_seller_costs = []
    for key, values in initial_pv.items():
        if key.startswith('buy_'):
            all_buyer_values.extend(values.tolist() if hasattr(values, 'tolist') else values)
        elif key.startswith('sell_'):
            all_seller_costs.extend(values.tolist() if hasattr(values, 'tolist') else values)
    
    all_buyer_values.sort(reverse=True)
    all_seller_costs.sort()
    
    optimal_surplus = 0.0
    optimal_num_trades = 0
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        surplus = all_buyer_values[i] - all_seller_costs[i]
        if surplus > 0:
            optimal_surplus += surplus
            optimal_num_trades += 1
    
    # Print social welfare statistics
    print("\n\nSocial Welfare Statistics:")
    total_surpluses = [tx['total_surplus'] for tx in transactions if tx['total_surplus'] is not None]
    buyer_surpluses = [tx['buyer_surplus'] for tx in transactions if tx['buyer_surplus'] is not None]
    seller_surpluses = [tx['seller_surplus'] for tx in transactions if tx['seller_surplus'] is not None]

    if len(total_surpluses) > 0:
        total_surplus_sum = sum(total_surpluses)
        print(f"  Total surplus (sum of trades): {total_surplus_sum:.2f}")
        print(f"  Optimal potential surplus: {optimal_surplus:.2f}")
        pct_optimal = (total_surplus_sum / optimal_surplus * 100) if optimal_surplus > 0 else 0
        print(f"  % of optimal surplus: {pct_optimal:.2f}%")
        print(f"  Optimal number of trades: {optimal_num_trades}")
        print(f"  Average surplus per trade: {np.mean(total_surpluses):.2f}")
        print(f"  Buyer avg surplus: {np.mean(buyer_surpluses):.2f}")
        print(f"  Seller avg surplus: {np.mean(seller_surpluses):.2f}")
    else:
        print("  No valid transactions.")
        print(f"  Optimal potential surplus: {optimal_surplus:.2f}")
        print(f"  Optimal number of trades: {optimal_num_trades}")
    
    # Calculate agent surplus at end per paper formula:
    # Agent surplus = cash + value_of_holdings + r_T * H
    print("\n\nAgent Final Surplus (per paper formula):")
    final_fundamental = market.get_fundamental_value()
    total_agent_surplus = 0.0
    
    for i in range(num_buyers):
        agent = agents[f'hbl_buy_{i}']
        holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=True)
        agent_surplus = agent.cash + holdings_value + final_fundamental * agent.position
        total_agent_surplus += agent_surplus
        # print(f"  HBL Buyer {i}: pos={agent.position}, cash={agent.cash:.2f}, holdings_val={holdings_value:.2f}, surplus={agent_surplus:.2f}")
    
    for i in range(num_sellers):
        agent = agents[f'hbl_sell_{i}']
        holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=False)
        agent_surplus = agent.cash + holdings_value + final_fundamental * agent.position
        total_agent_surplus += agent_surplus
        # print(f"  HBL Seller {i}: pos={agent.position}, cash={agent.cash:.2f}, holdings_val={holdings_value:.2f}, surplus={agent_surplus:.2f}")
    
    print(f"  Total Agent Surplus (end-of-period): {total_agent_surplus:.2f}")

    # Print final agent states
    print(f"\n\nFinal Agent States:")
    print("\nHBL Buyers:")
    for i in range(num_buyers):
        agent = agents[f'hbl_buy_{i}']
        print(f"  HBL Buyer {i}: position={agent.position}, cash={agent.cash:.2f}, remaining PVs={len(agent.pv.buyer_values)}")

    print("\nHBL Sellers:")
    for i in range(num_sellers):
        agent = agents[f'hbl_sell_{i}']
        print(f"  HBL Seller {i}: position={agent.position}, cash={agent.cash:.2f}, remaining PVs={len(agent.pv.seller_costs)}")

    # Calculate equilibrium price from initial values (where supply meets demand)
    # This matches the comparison_test methodology
    eq_price = None
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        if all_seller_costs[i] >= all_buyer_values[i]:
            eq_price = (all_buyer_values[i] + all_seller_costs[i]) / 2.0
            break
    
    # PVs are always relative, so add fundamental value
    if eq_price is not None:
        eq_price += market.get_fundamental_value()
    
    print(f"\n\nEquilibrium Price: {eq_price:.2f}" if eq_price is not None else "\n\nEquilibrium Price: N/A")

    if debug:
        return agents, transactions, eq_price, initial_pv, debug_info

    return agents, transactions, eq_price, initial_pv


if __name__ == '__main__':
    # Test parameters
    num_buyers = 28
    num_sellers = 28
    q_max = 3
    pv_var = 1e5
    shade_range = [0, 100]
    timesteps = 200
    L = 6
    arrival_rate = 0.1
    fundamental_value = 100000.0

    agents, transactions, eq_price, initial_pv = run_hbl_test(
        num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps,
        L=L, arrival_rate=arrival_rate, fundamental_value=fundamental_value
    )
    
    # Plot transactions
    import matplotlib.pyplot as plt
    import os
    
    times = [tx['time'] for tx in transactions]
    prices = [tx['price'] for tx in transactions]
    
    os.makedirs('results', exist_ok=True)
    plt.figure(figsize=(12, 6))
    if len(times) > 0:
        plt.scatter(times, prices, s=50, alpha=0.7, label='Transactions', color='green')
    
    if eq_price is not None:
        plt.axhline(eq_price, color='red', linestyle='--', linewidth=2, label=f'Eq Price: {eq_price:.2f}')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Transaction Price', fontsize=12)
    plt.title('HBL Agent Transaction Prices over Time with Equilibrium Price', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"\nPlot saved to results/hbl_split_transactions.png")
    try:
        plt.show()
    except Exception:
        pass
