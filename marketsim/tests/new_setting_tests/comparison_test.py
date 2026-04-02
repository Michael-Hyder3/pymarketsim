"""
Comparison test: Run ZI agents vs HBL agents with identical private values and parameters.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.tests.sim_test import run_simulator_test
from marketsim.tests.hbl_tests.hbl_split_test import run_hbl_test
from marketsim.private_values.separated_private_values import SeparatedPrivateValues


def generate_shared_private_values(num_buyers, num_sellers, q_max, pv_var, base_price, fundamental_value=None):
    """
    Generate identical private values for both ZI and HBL agents.
    
    Each agent gets q_max independent draws from N(base_price, sqrt(pv_var)).
    Buyers' highest values are used first; sellers' lowest costs are used first.
    
    Returns:
        - initial_pv: dict of private values indexed by agent key ('buy_i', 'sell_i')
        - equilibrium_price: the calculated equilibrium price
        - optimal_surplus: the maximum possible total surplus from these private values
        - optimal_num_trades: number of trades to achieve optimal surplus
        - all_buyer_values: sorted buyer values (descending) - ALL q_max values across all agents
        - all_seller_costs: sorted seller costs (ascending) - ALL q_max values across all agents
    """
    initial_pv = {}
    all_buyer_values = []
    all_seller_costs = []
    
    # Generate per-agent private values and collect for global matching
    # BUYERS: Each buyer gets q_max values, stored descending (highest first)
    for i in range(num_buyers):
        pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=base_price, role="buyer")
        initial_pv[f'buy_{i}'] = pv.buyer_values.numpy().copy()
        all_buyer_values.extend(pv.buyer_values.numpy().tolist())
    
    # SELLERS: Each seller gets q_max costs, stored ascending (lowest first)
    for i in range(num_sellers):
        pv = SeparatedPrivateValues(q_max, val_var=pv_var, base_price=base_price, role="seller")
        initial_pv[f'sell_{i}'] = pv.seller_costs.numpy().copy()
        all_seller_costs.extend(pv.seller_costs.numpy().tolist())
    
    # Sort globally for optimal matching
    all_buyer_values.sort(reverse=True)
    all_seller_costs.sort()
    
    # Find equilibrium price: where supply = demand (seller cost <= buyer value)
    equilibrium_price = base_price
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        if all_seller_costs[i] >= all_buyer_values[i]:
            equilibrium_price = (all_buyer_values[i] + all_seller_costs[i]) / 2.0
            break

    if fundamental_value is not None:
        equilibrium_price += fundamental_value
    
    # Calculate optimal surplus: sum of positive surpluses when units are matched optimally
    # Unit i can trade if buyer_value[i] > seller_cost[i]
    # Also track which pairs would trade in optimal solution
    optimal_surplus = 0.0
    optimal_num_trades = 0
    optimal_trades = []  # List of (buyer_value, seller_cost) pairs that trade in optimal
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        surplus = all_buyer_values[i] - all_seller_costs[i]
        if surplus > 0:
            optimal_surplus += surplus
            optimal_num_trades += 1
            optimal_trades.append((all_buyer_values[i], all_seller_costs[i]))
        else:
            break
    
    return initial_pv, equilibrium_price, optimal_surplus, optimal_num_trades, all_buyer_values, all_seller_costs, optimal_trades


def compare_tests(num_buyers=6, num_sellers=6, q_max=10, pv_var=1e5, 
                  shade_range=None, timesteps=300, L=6, arrival_rate=0.1, 
                  fundamental_value=100000.0, use_time_varying=False,
                  mean_reversion_rate=0.05, shock_var=1e4,
                  obs_noise_var=1e3,
                  debug_surplus=False, debug_hbl_trades=False, show_plots=True):
    """Run both ZI and HBL tests with identical parameters and compare results."""
    
    if shade_range is None:
        shade_range = [0, 100]
    
    print("=" * 100)
    print("RUNNING COMPARISON: ZI AGENTS vs HBL AGENTS")
    print("=" * 100)
    print(f"\nCommon Parameters:")
    print(f"  Buyers: {num_buyers}, Sellers: {num_sellers}")
    print(f"  q_max: {q_max}, pv_var: {pv_var}")
    print(f"  Shade range: {shade_range}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Fundamental value: {fundamental_value}")
    if use_time_varying:
        print(f"  Mean Reversion Rate (r): {mean_reversion_rate:.4f}")
        print(f"  Shock Variance: {shock_var:.2f}")
    print(f"  Observation Noise Variance: {obs_noise_var:.2f}")
    print(f"\nHBL-specific Parameters:")
    print(f"  L: {L}, arrival_rate: {arrival_rate}")
    print("\n" + "=" * 100)
    
    # Generate shared private values before running experiments
    print("\nGenerating shared private values...")
    pv_base_price = 0.0
    shared_pv, equilibrium_price, optimal_surplus, optimal_num_trades, all_buyer_values, all_seller_costs, optimal_trades = generate_shared_private_values(
        num_buyers, num_sellers, q_max, pv_var, pv_base_price, fundamental_value
    )
    print(f"Equilibrium Price: {equilibrium_price:.2f}" if equilibrium_price is not None else "Equilibrium Price: N/A")
    print(f"Optimal Potential Surplus: {optimal_surplus:.2f}")
    print(f"Number of Trades for Optimal Surplus: {optimal_num_trades}")
    
    # Print sorted private values
    print("\n" + "-" * 100)
    print("SORTED INITIAL PRIVATE VALUES")
    print("-" * 100)
    
    print(f"\nBuyer Values (sorted descending - highest willingness-to-pay first):")
    print(f"  {[f'{v:.2f}' for v in all_buyer_values[:10]]}")
    if len(all_buyer_values) > 10:
        print(f"  ... ({len(all_buyer_values) - 10} more values) ...")
        print(f"  {[f'{v:.2f}' for v in all_buyer_values[-5:]]}")
    
    print(f"\nSeller Costs (sorted ascending - lowest cost first):")
    print(f"  {[f'{v:.2f}' for v in all_seller_costs[:10]]}")
    if len(all_seller_costs) > 10:
        print(f"  ... ({len(all_seller_costs) - 10} more values) ...")
        print(f"  {[f'{v:.2f}' for v in all_seller_costs[-5:]]}")
    if use_time_varying:
        fundamental_zi = LazyGaussianMeanReverting(
            final_time=timesteps,
            mean=fundamental_value,
            r=mean_reversion_rate,
            shock_var=shock_var,
        )
        fundamental_hbl = LazyGaussianMeanReverting(
            final_time=timesteps,
            mean=fundamental_value,
            r=mean_reversion_rate,
            shock_var=shock_var,
        )
    else:
        fundamental_zi = None
        fundamental_hbl = None

    # Run ZI test
    print("\n[1/2] Running ZI Agent Test...")
    zi_agents, zi_transactions, zi_eq_price, zi_initial_pv = run_simulator_test(
        num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_value,
        shared_pv=shared_pv,
        fundamental=fundamental_zi,
        arrival_rate=arrival_rate,
        obs_noise_var=obs_noise_var
    )
    
    # Run HBL test
    print("\n[2/2] Running HBL Agent Test...")
    hbl_agents, hbl_transactions, hbl_eq_price, hbl_initial_pv = run_hbl_test(
        num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps,
        L=L, arrival_rate=arrival_rate, fundamental_value=fundamental_value,
        shared_pv=shared_pv,
        fundamental=fundamental_hbl,
        obs_noise_var=obs_noise_var
    )
    
    # Calculate end-of-period agent surplus (paper's method)
    # Surplus = cash + value_of_holdings + r_T * H
    final_fundamental = fundamental_zi.get_value(timesteps) if fundamental_zi else fundamental_value
    
    def calculate_agent_surplus(agents, num_buyers, num_sellers, prefix=''):
        """
        Calculate total agent surplus at end of period (paper's formula).
        
        Args:
            agents: dict of agents
            num_buyers: number of buyer agents
            num_sellers: number of seller agents
            prefix: key prefix ('' for ZI, 'hbl_' for HBL)
        """
        total_surplus = 0.0
        positions_summary = []
        
        # Calculate buyer surplus
        for i in range(num_buyers):
            key = f'{prefix}buy_{i}'
            if key not in agents:
                continue
            agent = agents[key]
            holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=True)
            agent_surplus = agent.cash + holdings_value + final_fundamental * agent.position
            total_surplus += agent_surplus
            if agent.position != 0:
                positions_summary.append(f"{key}: pos={agent.position}, cash={agent.cash:.2f}, holdings_val={holdings_value:.2f}")
        
        # Calculate seller surplus
        for i in range(num_sellers):
            key = f'{prefix}sell_{i}'
            if key not in agents:
                continue
            agent = agents[key]
            holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=False)
            agent_surplus = agent.cash + holdings_value + final_fundamental * agent.position
            total_surplus += agent_surplus
            if agent.position != 0:
                positions_summary.append(f"{key}: pos={agent.position}, cash={agent.cash:.2f}, holdings_val={holdings_value:.2f}")
        
        if positions_summary:
            print(f"\n  DEBUG: Non-zero positions for {prefix or 'ZI'}:")
            for line in positions_summary[:10]:  # Show first 10
                print(f"    {line}")
            if len(positions_summary) > 10:
                print(f"    ... and {len(positions_summary) - 10} more")
        
        return total_surplus
    
    zi_agent_surplus = calculate_agent_surplus(zi_agents, num_buyers, num_sellers, prefix='')
    hbl_agent_surplus = calculate_agent_surplus(hbl_agents, num_buyers, num_sellers, prefix='hbl_')
    
    # Comparison summary
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Metric':<40} {'ZI Agents':<30} {'HBL Agents':<30}")
    print("-" * 100)
    print(f"{'Total Transactions':<40} {len(zi_transactions):<30} {len(hbl_transactions):<30}")
    
    # Calculate surpluses with NO artificial validation
    # (Realized >100% should NOT be possible - if it is, there's a bug)
    def calculate_surpluses(transactions, debug=False):
        """
        Calculate total surplus from all transactions.
        Each agent's private value is indexed by position and can only be used once.
        This naturally prevents double-counting.
        """
        total = 0.0
        count = 0
        used_positions = {}  # {(agent_id, side, position): count}
        
        if debug:
            print(f"\n  DEBUG: Analyzing {len(transactions)} transactions")
        
        for i, tx in enumerate(transactions):
            if tx['total_surplus'] is None:
                continue
            
            buyer_id = tx.get('buyer_id')
            seller_id = tx.get('seller_id')
            buyer_position = tx.get('buyer_position')
            seller_position = tx.get('seller_position')
            buyer_value = tx.get('buyer_value')
            seller_cost = tx.get('seller_cost')
            
            if debug:
                calc_surplus = buyer_value - seller_cost if (buyer_value is not None and seller_cost is not None) else None
                bv_str = f"{buyer_value:.2f}" if buyer_value is not None else "None"
                sc_str = f"{seller_cost:.2f}" if seller_cost is not None else "None"
                calc_str = f"{calc_surplus:.2f}" if calc_surplus is not None else "None"
                print(f"    TX {i}: BuyerID={buyer_id} pos{buyer_position}, SellerID={seller_id} pos{seller_position} | bv={bv_str}, sc={sc_str}, " 
                      f"surplus={tx['total_surplus']:.2f}")
            
            # Track agent positions to ensure no double-use (diagnostic only)
            buyer_key = (buyer_id, 'BUY', buyer_position)
            seller_key = (seller_id, 'SELL', seller_position)
            
            # In a correctly-implemented market, each position is used at most once
            if buyer_key in used_positions or seller_key in used_positions:
                # This indicates a bug in market implementation
                if debug:
                    print(f"    WARNING: Position reuse detected at TX {i}")
                continue
            
            used_positions[buyer_key] = True
            used_positions[seller_key] = True
            total += tx['total_surplus']
            count += 1
        
        if debug:
            print(f"  DEBUG: Total surplus = {total:.2f}, count = {count}")
        
        return total, count
    
    zi_debug = debug_surplus and not debug_hbl_trades
    hbl_debug = debug_surplus or debug_hbl_trades
    zi_realized_surplus, zi_counted = calculate_surpluses(zi_transactions, debug=zi_debug)
    hbl_realized_surplus, hbl_counted = calculate_surpluses(hbl_transactions, debug=hbl_debug)
    
    # Check if we exceeded optimal and re-run with debug if so
    hbl_pct = (hbl_realized_surplus / optimal_surplus * 100) if optimal_surplus > 0 else 0
    if hbl_pct > 100.5 and not debug_surplus:
        print(f"\n!!! DETECTED >100% SURPLUS ({hbl_pct:.2f}%) - Re-analyzing with debug !!!")
        print(f"Optimal surplus: {optimal_surplus:.2f} from {optimal_num_trades} trades")
        print(f"HBL realized: {hbl_realized_surplus:.2f} from {hbl_counted} trades")
        print(f"\nOptimal trades (all {len(optimal_trades)}):")
        opt_sum = 0
        for i, (bv, sc) in enumerate(optimal_trades):
            surplus = bv - sc
            opt_sum += surplus
            print(f"  Opt {i}: bv={bv:.2f}, sc={sc:.2f}, surplus={surplus:.2f}, running_total={opt_sum:.2f}")
        print(f"\nOptimal sum verification: {opt_sum:.2f} (should equal {optimal_surplus:.2f})")
        print(f"\nHBL transactions:")
        hbl_realized_surplus, hbl_counted = calculate_surpluses(hbl_transactions, debug=True)
    
    # Also calculate raw surplus for reference
    zi_total_surpluses = [tx['total_surplus'] for tx in zi_transactions if tx['total_surplus'] is not None]
    hbl_total_surpluses = [tx['total_surplus'] for tx in hbl_transactions if tx['total_surplus'] is not None]
    
    zi_buyer_surpluses = [tx['buyer_surplus'] for tx in zi_transactions if tx['buyer_surplus'] is not None]
    hbl_buyer_surpluses = [tx['buyer_surplus'] for tx in hbl_transactions if tx['buyer_surplus'] is not None]
    
    zi_seller_surpluses = [tx['seller_surplus'] for tx in zi_transactions if tx['seller_surplus'] is not None]
    hbl_seller_surpluses = [tx['seller_surplus'] for tx in hbl_transactions if tx['seller_surplus'] is not None]
    
    if zi_total_surpluses:
        zi_total_surplus_sum = sum(zi_total_surpluses)
        zi_avg_surplus = np.mean(zi_total_surpluses)
    else:
        zi_total_surplus_sum = 0
        zi_avg_surplus = 0
    
    if hbl_total_surpluses:
        hbl_total_surplus_sum = sum(hbl_total_surpluses)
        hbl_avg_surplus = np.mean(hbl_total_surpluses)
    else:
        hbl_total_surplus_sum = 0
        hbl_avg_surplus = 0
    
    # Report surpluses
    print(f"{'Transactions Executed':<40} {len(zi_transactions):<30} {len(hbl_transactions):<30}")
    print(f"{'Transactions Counted':<40} {zi_counted:<30} {hbl_counted:<30}")
    print(f"{'Total Realized Surplus (trade-based)':<40} {zi_realized_surplus:<30.2f} {hbl_realized_surplus:<30.2f}")
    print(f"{'Total Agent Surplus (end-of-period)':<40} {zi_agent_surplus:<30.2f} {hbl_agent_surplus:<30.2f}")
    print(f"{'Optimal Potential Surplus':<40} {optimal_surplus:<30.2f} {optimal_surplus:<30.2f}")
    print()
    
    # Calculate percentage using end-of-period agent surplus (paper's method)
    zi_pct_optimal = (zi_agent_surplus / optimal_surplus * 100) if optimal_surplus > 0 else 0
    hbl_pct_optimal = (hbl_agent_surplus / optimal_surplus * 100) if optimal_surplus > 0 else 0
    
    print(f"{'% of Sorted-Optimal (end-of-period)':<40} {zi_pct_optimal:<30.2f}% {hbl_pct_optimal:<30.2f}%")
    note_line1 = "(Note: >100% is possible in dynamic markets where agents"
    note_line2 = "don't trade in globally-optimal order)"
    print(f"{note_line1:<40}")
    print(f"{note_line2:<40}")
    
    print(f"{'Optimal Number of Trades':<40} {optimal_num_trades:<30} {optimal_num_trades:<30}")
    print(f"{'Avg Surplus per Trade':<40} {zi_avg_surplus:<30.2f} {hbl_avg_surplus:<30.2f}")
    
    if zi_buyer_surpluses:
        zi_buyer_avg = np.mean(zi_buyer_surpluses)
    else:
        zi_buyer_avg = 0
    
    if hbl_buyer_surpluses:
        hbl_buyer_avg = np.mean(hbl_buyer_surpluses)
    else:
        hbl_buyer_avg = 0
    
    print(f"{'Avg Buyer Surplus':<40} {zi_buyer_avg:<30.2f} {hbl_buyer_avg:<30.2f}")
    
    if zi_seller_surpluses:
        zi_seller_avg = np.mean(zi_seller_surpluses)
    else:
        zi_seller_avg = 0
    
    if hbl_seller_surpluses:
        hbl_seller_avg = np.mean(hbl_seller_surpluses)
    else:
        hbl_seller_avg = 0
    
    print(f"{'Avg Seller Surplus':<40} {zi_seller_avg:<30.2f} {hbl_seller_avg:<30.2f}")
    
    # Price statistics
    zi_prices = [tx['price'] for tx in zi_transactions]
    hbl_prices = [tx['price'] for tx in hbl_transactions]
    
    if zi_prices:
        zi_avg_price = np.mean(zi_prices)
        zi_price_std = np.std(zi_prices)
    else:
        zi_avg_price = 0
        zi_price_std = 0
    
    if hbl_prices:
        hbl_avg_price = np.mean(hbl_prices)
        hbl_price_std = np.std(hbl_prices)
    else:
        hbl_avg_price = 0
        hbl_price_std = 0
    
    # Calculate RMSD (Root Mean Square Deviation from equilibrium price)
    if equilibrium_price is not None:
        if zi_prices:
            zi_rmsd = np.sqrt(np.mean([(p - equilibrium_price)**2 for p in zi_prices]))
        else:
            zi_rmsd = 0
        if hbl_prices:
            hbl_rmsd = np.sqrt(np.mean([(p - equilibrium_price)**2 for p in hbl_prices]))
        else:
            hbl_rmsd = 0
    else:
        zi_rmsd = 0
        hbl_rmsd = 0
    
    print(f"{'Avg Transaction Price':<40} {zi_avg_price:<30.2f} {hbl_avg_price:<30.2f}")
    print(f"{'Price Std Dev':<40} {zi_price_std:<30.2f} {hbl_price_std:<30.2f}")
    
    # Both agents should have the same equilibrium price since they use shared private values
    if equilibrium_price is not None:
        print(f"{'Equilibrium Price':<40} {equilibrium_price:<30.2f} {equilibrium_price:<30.2f}")
        print(f"{'RMSD from Equilibrium':<40} {zi_rmsd:<30.2f} {hbl_rmsd:<30.2f}")
    else:
        print(f"{'Equilibrium Price':<40} {'N/A':<30} {'N/A':<30}")
    
    print("\n" + "=" * 100)
    
    if show_plots:
        # Create scatter plots with eq_price lines
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Transaction prices over time (ZI)
        ax = axes[0]
        zi_times = [tx['time'] for tx in zi_transactions]
        if zi_times:
            ax.scatter(zi_times, zi_prices, s=50, alpha=0.7, label='Transactions', color='blue')
        if equilibrium_price is not None:
            ax.axhline(equilibrium_price, color='red', linestyle='--', linewidth=2, label=f'Eq Price: {equilibrium_price:.2f}')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Transaction Price', fontsize=11)
        ax.set_title('ZI Agents: Transaction Prices over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Transaction prices over time (HBL)
        ax = axes[1]
        hbl_times = [tx['time'] for tx in hbl_transactions]
        if hbl_times:
            ax.scatter(hbl_times, hbl_prices, s=50, alpha=0.7, label='Transactions', color='green')
        if equilibrium_price is not None:
            ax.axhline(equilibrium_price, color='red', linestyle='--', linewidth=2, label=f'Eq Price: {equilibrium_price:.2f}')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Transaction Price', fontsize=11)
        ax.set_title('HBL Agents: Transaction Prices over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            pass
    
    # Return results for batched testing
    return {
        'zi_realized_surplus': zi_realized_surplus,
        'hbl_realized_surplus': hbl_realized_surplus,
        'zi_agent_surplus': zi_agent_surplus,
        'hbl_agent_surplus': hbl_agent_surplus,
        'optimal_surplus': optimal_surplus,
        'optimal_num_trades': optimal_num_trades,
        'equilibrium_price': equilibrium_price,
        'zi_num_transactions': len(zi_transactions),
        'hbl_num_transactions': len(hbl_transactions),
        'zi_avg_price': zi_avg_price,
        'hbl_avg_price': hbl_avg_price,
        'zi_rmsd': zi_rmsd,
        'hbl_rmsd': hbl_rmsd,
    }


if __name__ == '__main__':
    import sys

    # Configuration parameters - change these to adjust the simulation
    NUM_BUYERS = 6
    NUM_SELLERS = 6
    Q_MAX = 10
    TIMESTEPS = 300
    SHADE_RANGE = [0, 100]
    
    if len(sys.argv) > 1 and sys.argv[1] == 'loop':
        max_runs = None
        if len(sys.argv) > 2:
            try:
                max_runs = int(sys.argv[2])
            except ValueError:
                max_runs = None

        run_index = 0
        while True:
            run_index += 1
            if max_runs is not None and run_index > max_runs:
                print(f"Reached max_runs={max_runs} without exceeding 100%.")
                break

            print(f"\n========== RUN {run_index} ==========")
            results = compare_tests(
                num_buyers=NUM_BUYERS,
                num_sellers=NUM_SELLERS,
                q_max=Q_MAX,
                timesteps=TIMESTEPS,
                shade_range=SHADE_RANGE,
                debug_hbl_trades=True
            )
            optimal_surplus = results.get('optimal_surplus', 0)
            hbl_surplus = results.get('hbl_realized_surplus', 0)
            hbl_pct = (hbl_surplus / optimal_surplus * 100) if optimal_surplus > 0 else 0
            print(f"\nHBL: {hbl_pct:.2f}% of optimal")

            if hbl_pct > 100:
                print(f"\n!!! FOUND >100% CASE IN RUN {run_index} !!!")
                break
    else:
        compare_tests(
            num_buyers=NUM_BUYERS,
            num_sellers=NUM_SELLERS,
            q_max=Q_MAX,
            timesteps=TIMESTEPS,
            shade_range=SHADE_RANGE,
            debug_hbl_trades=True
        )

