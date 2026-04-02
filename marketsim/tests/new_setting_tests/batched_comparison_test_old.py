"""
Batched Comparison Test: Run the comparison test 25 times and aggregate statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from marketsim.tests.comparison_test import compare_tests, generate_shared_private_values
from marketsim.tests.sim_test import run_simulator_test
from marketsim.tests.hbl_tests.hbl_split_test import run_hbl_test
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
import sys
from io import StringIO


def analyze_fundamental_estimates(num_buyers=28, num_sellers=28, q_max=3, pv_var=1e5, 
                                  shade_range=None, timesteps=200, L=6, arrival_rate=0.1, 
                                  fundamental_value=100000.0, use_time_varying=True,
                                  mean_reversion_rate=0.05, shock_var=1e4):
    """
    Analyze estimated fundamental values for a single iteration to understand RMSD.
    """
    
    if shade_range is None:
        shade_range = [0, 100]
    
    print("=" * 100)
    print("ANALYZING FUNDAMENTAL VALUE ESTIMATES (SINGLE ITERATION)")
    print("=" * 100)
    
    # Generate shared private values
    pv_base_price = 0.0
    shared_pv, equilibrium_price, optimal_surplus, optimal_num_trades, all_buyer_values, all_seller_costs, optimal_trades = generate_shared_private_values(
        num_buyers, num_sellers, q_max, pv_var, pv_base_price, fundamental_value
    )
    
    print(f"\nMarket Parameters:")
    print(f"  Fundamental Type: {'Time-Varying (LazyGaussianMeanReverting)' if use_time_varying else 'Constant (DummyFundamental)'}")
    print(f"  Initial Fundamental Value: {fundamental_value:.2f}")
    if use_time_varying:
        print(f"  Mean Reversion Rate (r): {mean_reversion_rate:.4f}")
        print(f"  Shock Variance: {shock_var:.2f}")
    print(f"  Timesteps: {timesteps}")
    
    # Create fundamental model
    if use_time_varying:
        fundamental = LazyGaussianMeanReverting(
            final_time=timesteps,
            mean=fundamental_value,
            r=mean_reversion_rate,
            shock_var=shock_var
        )
    else:
        fundamental = None  # Use default DummyFundamental in test functions
    
    # Suppress output from simulator tests
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    # Run ZI test
    zi_agents, zi_transactions, zi_eq_price, zi_initial_pv = run_simulator_test(
        num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_value,
        shared_pv=shared_pv, fundamental=fundamental
    )
    
    # Run HBL test
    hbl_agents, hbl_transactions, hbl_eq_price, hbl_initial_pv = run_hbl_test(
        num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps,
        L=L, arrival_rate=arrival_rate, fundamental_value=fundamental_value,
        shared_pv=shared_pv, fundamental=fundamental
    )
    
    sys.stdout = old_stdout

    debug_info = None
    if len(hbl_transactions) == 0:
        debug_agents, debug_transactions, debug_eq_price, debug_initial_pv, debug_info = run_hbl_test(
            num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps,
            L=L, arrival_rate=arrival_rate, fundamental_value=fundamental_value,
            shared_pv=shared_pv, fundamental=fundamental, debug=True, debug_steps=10
        )
        print("\nHBL DEBUG (NO TRADES) - FIRST 10 TIMESTEPS:")
        print(f"{'Time':<6} {'Orders':<8} {'Buys':<6} {'Sells':<6} {'BestBid':<12} {'BestAsk':<12} {'NoOrders':<10} {'Crossing':<10}")
        print("-" * 80)
        for row in debug_info:
            best_bid_str = f"{row['best_bid']:<12.2f}" if row['best_bid'] is not None else f"{'None':<12}"
            best_ask_str = f"{row['best_ask']:<12.2f}" if row['best_ask'] is not None else f"{'None':<12}"
            print(
                f"{row['time']:<6} "
                f"{row['total_orders']:<8} "
                f"{row['num_buy_orders']:<6} "
                f"{row['num_sell_orders']:<6} "
                f"{best_bid_str} "
                f"{best_ask_str} "
                f"{row['agents_with_no_orders']:<10} "
                f"{str(row['crossing_possible']):<10}"
            )
    
    # Get market info and fundamental for estimation
    market = zi_agents[list(zi_agents.keys())[0]].market
    mean, r, T = market.get_info()
    fundamental_obj = market.fundamental
    
    print(f"\nFundamental Model Parameters:")
    print(f"  Mean (long-term average): {mean:.2f}")
    print(f"  Mean Reversion Rate (r): {r:.4f}")
    print(f"  Final Timestep (T): {T}")
    print(f"  Formula: estimate(t) = (1 - ρ) × mean + ρ × fundamental(t), where ρ = (1-r)^(T-t)")
    
    # Analyze fundamental estimates over time
    print(f"\n" + "-" * 100)
    print(f"ESTIMATED FUNDAMENTAL VALUES AT VARIOUS TIMEPOINTS:")
    print("-" * 100)
    
    timepoints = [0, timesteps // 4, timesteps // 2, 3 * timesteps // 4, timesteps - 1]
    print(f"\n{'Time':<10} {'Actual FV':<15} {'ρ (decay)':<15} {'Estimate':<15} {'Deviation':<20}")
    print("-" * 75)
    
    for t in timepoints:
        fv_at_t = fundamental_obj.get_value_at(t)
        rho = (1 - r) ** (T - t)
        estimate = (1 - rho) * mean + rho * fv_at_t
        deviation = estimate - fv_at_t
        print(f"{t:<10} {fv_at_t:<15.2f} {rho:<15.6f} {estimate:<15.2f} {deviation:<20.2f}")
    
    # Analyze transaction prices vs estimated fundamentals
    print(f"\n" + "-" * 100)
    print(f"TRANSACTION PRICE ANALYSIS:")
    print("-" * 100)
    
    print(f"\nZI AGENTS - First 10 transactions:")
    print(f"{'Time':<8} {'Price':<12} {'Actual FV':<12} {'Est. Fund':<12} {'Deviation':<12} {'% Deviation':<12}")
    print("-" * 68)
    for i, tx in enumerate(zi_transactions[:10]):
        if tx['price'] is not None and tx['time'] is not None:
            t = tx['time']
            fv_at_t = fundamental_obj.get_value_at(t)
            rho = (1 - r) ** (T - t)
            estimate = (1 - rho) * mean + rho * fv_at_t
            deviation = tx['price'] - estimate
            pct_deviation = (deviation / estimate) * 100 if estimate != 0 else 0
            print(f"{t:<8} {tx['price']:<12.2f} {fv_at_t:<12.2f} {estimate:<12.2f} {deviation:<12.2f} {pct_deviation:<12.4f}%")
    
    print(f"\nHBL AGENTS - First 10 transactions:")
    print(f"{'Time':<8} {'Price':<12} {'Actual FV':<12} {'Est. Fund':<12} {'Deviation':<12} {'% Deviation':<12}")
    print("-" * 68)
    for i, tx in enumerate(hbl_transactions[:10]):
        if tx['price'] is not None and tx['time'] is not None:
            t = tx['time']
            fv_at_t = fundamental_obj.get_value_at(t)
            rho = (1 - r) ** (T - t)
            estimate = (1 - rho) * mean + rho * fv_at_t
            deviation = tx['price'] - estimate
            pct_deviation = (deviation / estimate) * 100 if estimate != 0 else 0
            print(f"{t:<8} {tx['price']:<12.2f} {fv_at_t:<12.2f} {estimate:<12.2f} {deviation:<12.2f} {pct_deviation:<12.4f}%")
    
    # Calculate and display RMSD
    print(f"\n" + "-" * 100)
    print(f"RMSD CALCULATIONS:")
    print("-" * 100)
    
    zi_rmsd_values = []
    for tx in zi_transactions:
        if tx['price'] is not None and tx['time'] is not None:
            t = tx['time']
            fv_at_t = fundamental_obj.get_value_at(t)
            rho = (1 - r) ** (T - t)
            estimate = (1 - rho) * mean + rho * fv_at_t
            zi_rmsd_values.append((tx['price'] - estimate) ** 2)
    
    hbl_rmsd_values = []
    for tx in hbl_transactions:
        if tx['price'] is not None and tx['time'] is not None:
            t = tx['time']
            fv_at_t = fundamental_obj.get_value_at(t)
            rho = (1 - r) ** (T - t)
            estimate = (1 - rho) * mean + rho * fv_at_t
            hbl_rmsd_values.append((tx['price'] - estimate) ** 2)
    
    zi_rmsd = np.sqrt(np.mean(zi_rmsd_values)) if zi_rmsd_values else 0
    hbl_rmsd = np.sqrt(np.mean(hbl_rmsd_values)) if hbl_rmsd_values else 0
    
    print(f"\nZI Agents:")
    print(f"  Number of transactions: {len(zi_transactions)}")
    print(f"  RMSD from estimated fundamental: {zi_rmsd:.2f}")
    print(f"  % of fundamental value: {(zi_rmsd / fundamental_value) * 100:.4f}%")
    
    print(f"\nHBL Agents:")
    print(f"  Number of transactions: {len(hbl_transactions)}")
    print(f"  RMSD from estimated fundamental: {hbl_rmsd:.2f}")
    print(f"  % of fundamental value: {(hbl_rmsd / fundamental_value) * 100:.4f}%")
    
    print("\n" + "=" * 100)


def run_batched_comparison(num_iterations=25, num_buyers=28, num_sellers=28, q_max=3, pv_var=1e5, 
                           shade_range=None, timesteps=200, L=6, arrival_rate=0.1, 
                           fundamental_value=100000.0, use_time_varying=True,
                           mean_reversion_rate=0.05, shock_var=1e4):
    """
    Run the comparison test multiple times and aggregate statistics.
    
    Returns aggregated metrics across all iterations.
    """
    
    if shade_range is None:
        shade_range = [0, 100]
    
    print("=" * 100)
    print("RUNNING BATCHED COMPARISON: ZI AGENTS vs HBL AGENTS")
    print("=" * 100)
    print(f"\nCommon Parameters:")
    print(f"  Buyers: {num_buyers}, Sellers: {num_sellers}")
    print(f"  q_max: {q_max}, pv_var: {pv_var}")
    print(f"  Shade range: {shade_range}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Fundamental Type: {'Time-Varying (LazyGaussianMeanReverting)' if use_time_varying else 'Constant (DummyFundamental)'}")
    print(f"  Initial Fundamental value: {fundamental_value}")
    if use_time_varying:
        print(f"  Mean Reversion Rate (r): {mean_reversion_rate:.4f}")
        print(f"  Shock Variance: {shock_var:.2f}")
    print(f"  Number of iterations: {num_iterations}")
    print(f"\nHBL-specific Parameters:")
    print(f"  L: {L}, arrival_rate: {arrival_rate}")
    print("\n" + "=" * 100)
    
    # Storage for all metrics across iterations
    zi_total_surpluses_all = []
    hbl_total_surpluses_all = []
    zi_transactions_all = []
    hbl_transactions_all = []
    optimal_surpluses = []
    optimal_num_trades_all = []
    equilibrium_prices = []
    zi_rmsd_all = []
    hbl_rmsd_all = []
    
    # Run iterations
    for iteration in range(num_iterations):
        print(f"\n[{iteration + 1}/{num_iterations}] Running iteration...")
        
        # Generate shared private values
        pv_base_price = 0.0
        shared_pv, equilibrium_price, optimal_surplus, optimal_num_trades, all_buyer_values, all_seller_costs, optimal_trades = generate_shared_private_values(
            num_buyers, num_sellers, q_max, pv_var, pv_base_price, fundamental_value
        )
        
        # Create fundamental model for this iteration
        if use_time_varying:
            fundamental = LazyGaussianMeanReverting(
                final_time=timesteps,
                mean=fundamental_value,
                r=mean_reversion_rate,
                shock_var=shock_var
            )
        else:
            fundamental = None  # Use default DummyFundamental in test functions
        
        # Suppress output from simulator tests
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Run ZI test
        zi_agents, zi_transactions, zi_eq_price, zi_initial_pv = run_simulator_test(
            num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_value,
            shared_pv=shared_pv, fundamental=fundamental
        )
        
        # Run HBL test
        hbl_agents, hbl_transactions, hbl_eq_price, hbl_initial_pv = run_hbl_test(
            num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps,
            L=L, arrival_rate=arrival_rate, fundamental_value=fundamental_value,
            shared_pv=shared_pv, fundamental=fundamental
        )
        
        sys.stdout = old_stdout
        
        # Calculate surpluses with position tracking (no reuse validation)
        def calculate_surpluses(transactions):
            """Calculate total surplus from all transactions."""
            total = 0.0
            count = 0
            used_positions = {}  # {(agent_id, side, position): count}
            
            for tx in transactions:
                if tx['total_surplus'] is None:
                    continue
                
                buyer_id = tx.get('buyer_id')
                seller_id = tx.get('seller_id')
                buyer_position = tx.get('buyer_position')
                seller_position = tx.get('seller_position')
                
                # Track agent positions to ensure no double-use
                buyer_key = (buyer_id, 'BUY', buyer_position)
                seller_key = (seller_id, 'SELL', seller_position)
                
                if buyer_key in used_positions or seller_key in used_positions:
                    # This indicates a bug - skip
                    continue
                
                used_positions[buyer_key] = True
                used_positions[seller_key] = True
                total += tx['total_surplus']
                count += 1
            
            return total, count
        
        zi_total_surplus_sum, zi_tx_counted = calculate_surpluses(zi_transactions)
        hbl_total_surplus_sum, hbl_tx_counted = calculate_surpluses(hbl_transactions)
        
        # Get market info and fundamental for estimation
        market = zi_agents[list(zi_agents.keys())[0]].market
        mean, r, T = market.get_info()
        fundamental_obj = market.fundamental
        
        # Calculate RMSD for transaction prices compared to estimated fundamental value
        # For each transaction, calculate what the agent's estimate of fundamental would have been at that time
        zi_rmsd_values = []
        for tx in zi_transactions:
            if tx['price'] is not None and tx['time'] is not None:
                t = tx['time']
                fv_at_t = fundamental_obj.get_value_at(t)
                rho = (1 - r) ** (T - t)
                estimate = (1 - rho) * mean + rho * fv_at_t
                zi_rmsd_values.append((tx['price'] - estimate) ** 2)
        
        hbl_rmsd_values = []
        for tx in hbl_transactions:
            if tx['price'] is not None and tx['time'] is not None:
                t = tx['time']
                fv_at_t = fundamental_obj.get_value_at(t)
                rho = (1 - r) ** (T - t)
                estimate = (1 - rho) * mean + rho * fv_at_t
                hbl_rmsd_values.append((tx['price'] - estimate) ** 2)
        
        zi_rmsd = np.sqrt(np.mean(zi_rmsd_values)) if zi_rmsd_values else 0
        hbl_rmsd = np.sqrt(np.mean(hbl_rmsd_values)) if hbl_rmsd_values else 0
        
        zi_total_surpluses_all.append(zi_total_surplus_sum)
        hbl_total_surpluses_all.append(hbl_total_surplus_sum)
        zi_transactions_all.append(len(zi_transactions))
        hbl_transactions_all.append(len(hbl_transactions))
        optimal_surpluses.append(optimal_surplus)
        optimal_num_trades_all.append(optimal_num_trades)
        equilibrium_prices.append(equilibrium_price)
        zi_rmsd_all.append(zi_rmsd)
        hbl_rmsd_all.append(hbl_rmsd)
        
        print(f"  ZI Surplus: {zi_total_surplus_sum:.2f}, HBL Surplus: {hbl_total_surplus_sum:.2f}, Optimal: {optimal_surplus:.2f}")
    
    print("\n" + "=" * 100)
    print("BATCHED COMPARISON RESULTS (AVERAGES ACROSS {} ITERATIONS)".format(num_iterations))
    print("=" * 100)
    
    # Calculate averages
    avg_zi_total_surplus = np.mean(zi_total_surpluses_all)
    avg_hbl_total_surplus = np.mean(hbl_total_surpluses_all)
    avg_optimal_surplus = np.mean(optimal_surpluses)
    avg_optimal_num_trades = np.mean(optimal_num_trades_all)
    avg_equilibrium_price = np.mean([p for p in equilibrium_prices if p is not None])
    
    avg_zi_transactions = np.mean(zi_transactions_all)
    avg_hbl_transactions = np.mean(hbl_transactions_all)
    
    # Calculate percentages
    avg_zi_pct_optimal = (avg_zi_total_surplus / avg_optimal_surplus * 100) if avg_optimal_surplus > 0 else 0
    avg_hbl_pct_optimal = (avg_hbl_total_surplus / avg_optimal_surplus * 100) if avg_optimal_surplus > 0 else 0
    
    print(f"\n{'Metric':<40} {'ZI Agents':<30} {'HBL Agents':<30}")
    print("-" * 100)
    print(f"{'Avg Total Transactions':<40} {avg_zi_transactions:<30.2f} {avg_hbl_transactions:<30.2f}")
    print(f"{'Avg Total Surplus (sum)':<40} {avg_zi_total_surplus:<30.2f} {avg_hbl_total_surplus:<30.2f}")
    print(f"{'Avg Optimal Potential Surplus':<40} {avg_optimal_surplus:<30.2f} {avg_optimal_surplus:<30.2f}")
    print(f"{'Avg % of Optimal Surplus':<40} {avg_zi_pct_optimal:<30.2f} {avg_hbl_pct_optimal:<30.2f}")
    
    print(f"{'Avg Optimal Number of Trades':<40} {avg_optimal_num_trades:<30.2f} {avg_optimal_num_trades:<30.2f}")
    print(f"{'Avg Equilibrium Price':<40} {avg_equilibrium_price:<30.2f} {avg_equilibrium_price:<30.2f}")
    
    # Average RMSD across iterations
    avg_zi_rmsd = np.mean(zi_rmsd_all)
    avg_hbl_rmsd = np.mean(hbl_rmsd_all)
    print(f"\n{'Avg RMSD from Estimated Fundamental':<40} {avg_zi_rmsd:<30.2f} {avg_hbl_rmsd:<30.2f}")
    
    print("\n" + "=" * 100)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    iterations = np.arange(1, num_iterations + 1)
    
    # Plot 1: Total Surplus Comparison
    ax = axes[0]
    ax.scatter(iterations, zi_total_surpluses_all, s=100, alpha=0.7, label='ZI Agents', color='blue', marker='o')
    ax.scatter(iterations, hbl_total_surpluses_all, s=100, alpha=0.7, label='HBL Agents', color='green', marker='s')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Surplus', fontsize=12)
    ax.set_title(f'Total Surplus Comparison: ZI vs HBL Agents ({num_iterations} iterations)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RMSD from Estimated Fundamental Value
    ax = axes[1]
    ax.scatter(iterations, zi_rmsd_all, s=100, alpha=0.7, label='ZI Agents', color='blue', marker='o')
    ax.scatter(iterations, hbl_rmsd_all, s=100, alpha=0.7, label='HBL Agents', color='green', marker='s')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('RMSD from Estimated Fundamental', fontsize=12)
    ax.set_title(f'Price Efficiency: RMSD from Estimated Fundamental ({num_iterations} iterations)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    
    return {
        'zi_total_surpluses': zi_total_surpluses_all,
        'hbl_total_surpluses': hbl_total_surpluses_all,
        'avg_zi_total_surplus': avg_zi_total_surplus,
        'avg_hbl_total_surplus': avg_hbl_total_surplus,
        'avg_optimal_surplus': avg_optimal_surplus,
        'avg_zi_rmsd': avg_zi_rmsd,
        'avg_hbl_rmsd': avg_hbl_rmsd,
        'avg_zi_pct_optimal': avg_zi_pct_optimal,
        'avg_hbl_pct_optimal': avg_hbl_pct_optimal,
    }


if __name__ == '__main__':
    import sys
    
    # Parse command-line arguments
    num_iterations = 10  # Reduced default for faster testing (was 25)
    timesteps = 150  # Reduced default (was 200) - still allows HBL to converge
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            analyze_fundamental_estimates(timesteps=timesteps)
        else:
            # Allow passing num_iterations and timesteps as arguments
            if len(sys.argv) > 1:
                try:
                    num_iterations = int(sys.argv[1])
                except ValueError:
                    pass
            if len(sys.argv) > 2:
                try:
                    timesteps = int(sys.argv[2])
                except ValueError:
                    pass
            run_batched_comparison(num_iterations=num_iterations, timesteps=timesteps)
    else:
        run_batched_comparison(num_iterations=num_iterations, timesteps=timesteps)
