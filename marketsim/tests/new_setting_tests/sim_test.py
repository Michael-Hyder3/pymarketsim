"""
Simulator test with ZIAgentBuy and ZIAgentSell using absolute private values.
Tests the simulator.py with transaction logging and social welfare computation.
"""

import random
import numpy as np
from marketsim.market.market import Market
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.fourheap.constants import BUY, SELL
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fundamental.dummy_fundamental import DummyFundamental


# Use centralized DummyFundamental from marketsim.fundamental.dummy_fundamental


def run_simulator_test(num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_value=100000.0, shared_pv=None, fundamental=None, obs_noise_var=0.0, arrival_rate=None):
    """
    Run a simulation with ZIAgentBuy and ZIAgentSell using absolute private values.
    
    Args:
        num_buyers: number of buyer agents
        num_sellers: number of seller agents
        q_max: max quantity each agent can trade
        pv_var: variance of private values
        shade_range: [min_shade, max_shade] for agent price shading
        timesteps: number of simulation timesteps
        fundamental_value: base value for absolute private values
        shared_pv: optional dict of pre-generated private values (keys: 'buy_i', 'sell_i')
        fundamental: optional fundamental model (if None, uses DummyFundamental)
        obs_noise_var: observation noise variance for fundamental estimates (default 0.0)
        arrival_rate: Poisson arrival rate λ (if None, all agents act every timestep)
    
    Returns:
        agents: dict of all agents
        transactions: list of transaction records with prices and PVs
        eq_price: equilibrium price from remaining marginals
        initial_pv: dict of initial private values (before any consumption)
    """
    # Create market with fundamental
    if fundamental is None:
        dummy_fundamental = DummyFundamental(value=fundamental_value, final_time=timesteps)
        market = Market(fundamental=dummy_fundamental, time_steps=timesteps)
    else:
        market = Market(fundamental=fundamental, time_steps=timesteps)

    # Always use relative private values when agents use fundamental estimates
    pv_base_price = 0.0

    # Create agents with private values
    agents = {}
    id_to_agent = {}
    initial_pv = {} if shared_pv is None else shared_pv.copy()

    for i in range(num_buyers):
        agent = ZIAgentBuy(
            agent_id=i,
            market=market,
            q_max=q_max,
            shade=shade_range,
            pv_var=pv_var,
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
        
        agents[f'buy_{i}'] = agent
        id_to_agent[agent.get_id()] = agent

    for i in range(num_sellers):
        agent = ZIAgentSell(
            agent_id=num_buyers + i,
            market=market,
            q_max=q_max,
            shade=shade_range,
            pv_var=pv_var,
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
        
        agents[f'sell_{i}'] = agent
        id_to_agent[agent.get_id()] = agent

    # Print initial private values
    print(f"\n--- Simulator Test: {num_buyers} buyers, {num_sellers} sellers, {timesteps} timesteps ---\n")
    # Commented out for performance
    # print("Initial Private Values:")
    # for i in range(min(4, num_buyers)):
    #     agent = agents[f'buy_{i}']
    #     print(f"  Buyer {i}: values (desc) = {agent.pv.buyer_values.numpy()}")
    # for i in range(min(4, num_sellers)):
    #     agent = agents[f'sell_{i}']
    #     print(f"  Seller {i}: costs (asc) = {agent.pv.seller_costs.numpy()}")
    # print()

    transactions = []
    agent_pvs_before = {}  # store initial PVs for social welfare calculation

    # Record initial private values per agent
    for agent_name, agent in agents.items():
        try:
            if agent_name.startswith('buy_'):
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

        # Determine which agents act this timestep
        if arrival_rate is not None:
            # Poisson sampling: each agent has probability arrival_rate of acting
            active_agents = []
            for agent_name, agent in agents.items():
                if np.random.rand() < arrival_rate:
                    active_agents.append((agent_name, agent))
            random.shuffle(active_agents)
        else:
            # All agents act every timestep (original behavior)
            active_agents = list(agents.items())
            random.shuffle(active_agents)

        for agent_name, agent in active_agents:
            market.withdraw_all(agent.get_id())
            orders = agent.take_action()
            market.add_orders(orders)

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

    # Print social welfare statistics
    print("\n\nSocial Welfare Statistics:")
    total_surpluses = [tx['total_surplus'] for tx in transactions if tx['total_surplus'] is not None]
    buyer_surpluses = [tx['buyer_surplus'] for tx in transactions if tx['buyer_surplus'] is not None]
    seller_surpluses = [tx['seller_surplus'] for tx in transactions if tx['seller_surplus'] is not None]

    if len(total_surpluses) > 0:
        print(f"  Total surplus (sum of trades): {sum(total_surpluses):.2f}")
        print(f"  Average surplus per trade: {np.mean(total_surpluses):.2f}")
        print(f"  Buyer avg surplus: {np.mean(buyer_surpluses):.2f}")
        print(f"  Seller avg surplus: {np.mean(seller_surpluses):.2f}")
    else:
        print("  No valid transactions.")
    
    # Calculate agent surplus at end per paper formula:
    # Agent surplus = cash + value_of_holdings + r_T * H
    # where value_of_holdings = sum of θ^k for held units
    print("\n\nAgent Final Surplus (per paper formula):")
    final_fundamental = market.get_fundamental_value()
    total_agent_surplus = 0.0
    
    for i in range(num_buyers):
        agent = agents[f'buy_{i}']
        holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=True)
        agent_surplus = agent.cash + holdings_value + final_fundamental * agent.position
        total_agent_surplus += agent_surplus
        # print(f"  Buyer {i}: pos={agent.position}, cash={agent.cash:.2f}, holdings_val={holdings_value:.2f}, surplus={agent_surplus:.2f}")
    
    for i in range(num_sellers):
        agent = agents[f'sell_{i}']
        holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=False)
        agent_surplus = agent.cash + holdings_value + final_fundamental * agent.position
        total_agent_surplus += agent_surplus
        # print(f"  Seller {i}: pos={agent.position}, cash={agent.cash:.2f}, holdings_val={holdings_value:.2f}, surplus={agent_surplus:.2f}")
    
    print(f"  Total Agent Surplus (end-of-period): {total_agent_surplus:.2f}")

    # Print final agent states
    # Commented out for performance
    # print(f"\n\nFinal Agent States:")
    # print("\nBuyers:")
    # for i in range(num_buyers):
    #     agent = agents[f'buy_{i}']
    #     print(f"  Buyer {i}: position={agent.position}, cash={agent.cash:.2f}, remaining PVs={len(agent.pv.buyer_values)}")

    # print("\nSellers:")
    # for i in range(num_sellers):
    #     agent = agents[f'sell_{i}']
    #     print(f"  Seller {i}: position={agent.position}, cash={agent.cash:.2f}, remaining PVs={len(agent.pv.seller_costs)}")

    # Compute eq_price: average of largest remaining buy PV and smallest remaining sell PV
    # (these are the values that did NOT transact)
    remaining_buyer_vals = []
    remaining_seller_costs = []
    
    for i in range(num_buyers):
        agent = agents[f'buy_{i}']
        if len(agent.pv.buyer_values) > 0:
            remaining_buyer_vals.extend(agent.pv.buyer_values.numpy().tolist())
    
    for i in range(num_sellers):
        agent = agents[f'sell_{i}']
        if len(agent.pv.seller_costs) > 0:
            remaining_seller_costs.extend(agent.pv.seller_costs.numpy().tolist())
    
    eq_price = None
    if len(remaining_buyer_vals) > 0 and len(remaining_seller_costs) > 0:
        largest_remaining_buy = max(remaining_buyer_vals)
        smallest_remaining_sell = min(remaining_seller_costs)
        eq_price = (largest_remaining_buy + smallest_remaining_sell) / 2.0
        if getattr(agents[f'buy_0'].pv, 'base_price', None) == 0:
            eq_price += market.get_fundamental_value()
    
    print(f"\n\nEquilibrium Price (average of remaining marginals): {eq_price:.2f}" if eq_price else "\n\nNo equilibrium price (no remaining PVs)")

    return agents, transactions, eq_price, initial_pv


def save_llm_data(agents, transactions, eq_price, initial_pv, num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_value):
    """Save simulation data in LLM-friendly format."""
    import json
    import os
    
    os.makedirs('llm_calls/llm_as_policy', exist_ok=True)
    
    # Prepare data structure
    data = {
        'simulation_parameters': {
            'num_buyers': num_buyers,
            'num_sellers': num_sellers,
            'q_max': q_max,
            'pv_var': pv_var,
            'shade_range': shade_range,
            'timesteps': timesteps,
            'fundamental_value': fundamental_value
        },
        'private_values': {},
        'transactions': []
    }
    
    # Add INITIAL private values for each agent (before any consumption)
    for agent_name in initial_pv:
        agent_type = 'buyer' if 'buy' in agent_name else 'seller'
        agent = agents[agent_name]
        
        data['private_values'][agent_name] = {
            'type': agent_type,
            'agent_id': agent.get_id(),
            'initial_values': initial_pv[agent_name].tolist()
        }
    
    # Add transaction details
    for tx in transactions:
        data['transactions'].append({
            'time': int(tx['time']),
            'buyer_id': int(tx['buyer_id']),
            'seller_id': int(tx['seller_id']),
            'buyer_value': float(tx['buyer_value']) if tx['buyer_value'] is not None else None,
            'seller_cost': float(tx['seller_cost']) if tx['seller_cost'] is not None else None,
            'price': float(tx['price']),
            'price_setter_side': tx['price_setter_side'],
            'buyer_surplus': float(tx['buyer_surplus']) if tx['buyer_surplus'] is not None else None,
            'seller_surplus': float(tx['seller_surplus']) if tx['seller_surplus'] is not None else None,
            'total_surplus': float(tx['total_surplus']) if tx['total_surplus'] is not None else None
        })
    
    data['total_transactions'] = len(transactions)
    data['total_surplus'] = sum([tx['total_surplus'] for tx in transactions if tx['total_surplus'] is not None])
    
    # Save to JSON
    with open('llm_calls/llm_as_policy/sim_test_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"LLM data saved to llm_calls/llm_as_policy/sim_test_data.json")


if __name__ == '__main__':
    # Test parameters
    num_buyers = 4
    num_sellers = 4
    q_max = 6
    pv_var = 1e5
    shade_range = [0, 500]
    timesteps = 50
    fundamental_value = 100000.0

    agents, transactions, eq_price, initial_pv = run_simulator_test(
        num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_value
    )
    
    # Save data for LLM
    save_llm_data(agents, transactions, eq_price, initial_pv, num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_value)
