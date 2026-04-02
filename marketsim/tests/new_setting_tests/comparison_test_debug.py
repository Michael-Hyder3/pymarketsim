"""
Debug version of comparison test with HBL agent logging enabled.
"""

import random
import numpy as np
import torch
from marketsim.agent.hbl_agent_buy_sell import HBLAgentBuy, HBLAgentSell
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.market.market import Market
from marketsim.fundamental.dummy_fundamental import DummyFundamental
from marketsim.private_values.separated_private_values import SeparatedPrivateValues
from marketsim.fourheap.constants import BUY, SELL


def generate_shared_private_values(num_buyers, num_sellers, q_max, pv_var, base_price, fundamental_value=None):
    """Generate identical private values for both ZI and HBL agents."""
    initial_pv = {}
    all_buyer_values = []
    all_seller_costs = []
    
    # Generate per-agent private values
    for i in range(num_buyers):
        pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=base_price, role="buyer")
        initial_pv[f'buy_{i}'] = pv.buyer_values.numpy().copy()
        all_buyer_values.extend(pv.buyer_values.numpy().tolist())
    
    for i in range(num_sellers):
        pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=base_price, role="seller")
        initial_pv[f'sell_{i}'] = pv.seller_costs.numpy().copy()
        all_seller_costs.extend(pv.seller_costs.numpy().tolist())
    
    # Sort globally for optimal matching
    all_buyer_values.sort(reverse=True)
    all_seller_costs.sort()
    
    # Find equilibrium price
    equilibrium_price = base_price
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        if all_seller_costs[i] >= all_buyer_values[i]:
            equilibrium_price = (all_buyer_values[i] + all_seller_costs[i]) / 2.0
            break

    if fundamental_value is not None:
        equilibrium_price += fundamental_value
    
    # Calculate optimal surplus
    optimal_surplus = 0.0
    optimal_num_trades = 0
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        surplus = all_buyer_values[i] - all_seller_costs[i]
        if surplus > 0:
            optimal_surplus += surplus
            optimal_num_trades += 1
    
    return initial_pv, equilibrium_price, optimal_surplus, optimal_num_trades


def run_debug_test(agent_type='hbl', num_buyers=14, num_sellers=14, q_max=3, pv_var=1e5,
                    shade_range=None, timesteps=50, L=6, arrival_rate=0.1,
                    fundamental_value=100000.0):
    """Run test with debug logging enabled for HBL agents."""
    
    if shade_range is None:
        shade_range = [0, 100]
    
    print("=" * 100)
    print(f"DEBUG TEST: {agent_type.upper()} AGENTS")
    print("=" * 100)
    print(f"Parameters: buyers={num_buyers}, sellers={num_sellers}, q_max={q_max}, timesteps={timesteps}")
    print(f"Shade range: {shade_range}, L={L}, arrival_rate={arrival_rate}")
    print("=" * 100)
    
    # Generate private values
    pv_base_price = 0.0
    shared_pv, eq_price, optimal_surplus, optimal_trades = generate_shared_private_values(
        num_buyers, num_sellers, q_max, pv_var, pv_base_price, fundamental_value
    )
    
    print(f"\nOptimal Surplus: {optimal_surplus:.2f}, Optimal Trades: {optimal_trades}")
    print(f"Equilibrium Price: {eq_price:.2f}\n")
    
    # Set up market
    dummy_fundamental = DummyFundamental(value=fundamental_value, final_time=timesteps)
    market = Market(fundamental=dummy_fundamental, time_steps=timesteps)
    
    agents = []
    
    if agent_type == 'hbl':
        # Create HBL agents with debug=True
        for i in range(num_buyers):
            agent = HBLAgentBuy(
                agent_id=i,
                market=market,
                q_max=q_max,
                shade=shade_range,
                L=L,
                pv_var=pv_var,
                arrival_rate=arrival_rate,
                debug=True  # Enable debug logging
            )
            # Set shared private values
            pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="buyer")
            pv.buyer_values = torch.tensor(shared_pv[f'buy_{i}'])
            agent.pv = pv
            agents.append(agent)
        
        for i in range(num_sellers):
            agent = HBLAgentSell(
                agent_id=num_buyers + i,
                market=market,
                q_max=q_max,
                shade=shade_range,
                L=L,
                pv_var=pv_var,
                arrival_rate=arrival_rate,
                debug=True  # Enable debug logging
            )
            # Set shared private values
            pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="seller")
            pv.seller_costs = torch.tensor(shared_pv[f'sell_{i}'])
            agent.pv = pv
            agents.append(agent)
    
    else:  # ZI agents
        for i in range(num_buyers):
            agent = ZIAgentBuy(
                agent_id=i,
                market=market,
                q_max=q_max,
                shade=shade_range,
                pv_var=pv_var
            )
            pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="buyer")
            pv.buyer_values = torch.tensor(shared_pv[f'buy_{i}'])
            agent.pv = pv
            agents.append(agent)
        
        for i in range(num_sellers):
            agent = ZIAgentSell(
                agent_id=num_buyers + i,
                market=market,
                q_max=q_max,
                shade=shade_range,
                pv_var=pv_var
            )
            pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=0.0, role="seller")
            pv.seller_costs = torch.tensor(shared_pv[f'sell_{i}'])
            agent.pv = pv
            agents.append(agent)
    
    # Run simulation
    print("\n" + "=" * 100)
    print("STARTING SIMULATION")
    print("=" * 100 + "\n")
    
    transactions = []
    
    for t in range(timesteps):
        market.event_queue.set_time(t)
        
        # Pre-generate fundamental value
        _ = market.get_fundamental_value()
        
        # All agents act (shuffle for randomness)
        random.shuffle(agents)
        
        for agent in agents:
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
        
        # Process paired trades
        for trade in paired_trades.values():
            buyer_order = trade.get('buyer_order')
            seller_order = trade.get('seller_order')
            
            if buyer_order is not None and seller_order is not None:
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
                
                buyer_id = trade['buyer_id']
                seller_id = trade['seller_id']
                
                buyer_agent = next((a for a in agents if a.get_id() == buyer_id), None)
                seller_agent = next((a for a in agents if a.get_id() == seller_id), None)
                
                if buyer_agent and seller_agent:
                    # Get private values BEFORE updating positions
                    buyer_pos_before = buyer_agent.position
                    seller_pos_before = seller_agent.position
                    
                    # Bounds check to avoid index errors
                    if buyer_pos_before >= q_max or abs(seller_pos_before) >= q_max:
                        print(f"[t={t}] TRADE SKIPPED: position limits reached (buyer_pos={buyer_pos_before}, seller_pos={seller_pos_before}, q_max={q_max})")
                        continue
                    
                    buyer_pv = shared_pv[f'buy_{buyer_id}'][buyer_pos_before]
                    seller_pv = shared_pv[f'sell_{seller_id - num_buyers}'][abs(seller_pos_before)]
                    
                    buyer_surplus = buyer_pv - (trade_price - fundamental_value)
                    seller_surplus = (trade_price - fundamental_value) - seller_pv
                    total_surplus = buyer_surplus + seller_surplus
                    
                    print(f"[t={t}] TRADE: Buyer{buyer_id} ↔ Seller{seller_id} @ price={trade_price:.2f}, "
                          f"buyer_surplus={buyer_surplus:.2f}, seller_surplus={seller_surplus:.2f}, "
                          f"total_surplus={total_surplus:.2f}")
                    
                    transactions.append({
                        'time': t,
                        'price': trade_price,
                        'buyer_surplus': buyer_surplus,
                        'seller_surplus': seller_surplus,
                        'total_surplus': total_surplus
                    })
                    
                    # Update positions
                    buyer_agent.update_position(1, -trade_price)
                    seller_agent.update_position(-1, trade_price)
    
    # Summary
    print("\n" + "=" * 100)
    print("SIMULATION COMPLETE")
    print("=" * 100)
    
    total_surplus = sum(tx['total_surplus'] for tx in transactions)
    pct_optimal = (total_surplus / optimal_surplus * 100) if optimal_surplus > 0 else 0
    
    print(f"\nTotal Transactions: {len(transactions)}")
    print(f"Total Surplus: {total_surplus:.2f}")
    print(f"Optimal Surplus: {optimal_surplus:.2f}")
    print(f"% of Optimal: {pct_optimal:.2f}%")
    print(f"Avg Price: {np.mean([tx['price'] for tx in transactions]):.2f}" if transactions else "Avg Price: N/A")
    print("=" * 100)


if __name__ == '__main__':
    print("\n" + "#" * 100)
    print("# RUNNING HBL AGENTS WITH DEBUG LOGGING")
    print("#" * 100 + "\n")
    run_debug_test(agent_type='hbl', timesteps=50)
    
    # Uncomment to compare with ZI agents
    # print("\n\n" + "#" * 100)
    # print("# RUNNING ZI AGENTS FOR COMPARISON")
    # print("#" * 100 + "\n")
    # run_debug_test(agent_type='zi', timesteps=50)
