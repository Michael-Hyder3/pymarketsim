"""
Paper Comparison Test: Run the comparison test with specifications from the paper.

Paper specifications:
- Private value variance (σ²_s): 5 × 10^6
- Observation noise variance (σ²_n): 10^3
- Fundamental mean (r̄): 10^5
- Mean reversion rate: 0.05
- Shock variance (σ²_s): 10^5
- Arrival rate (λ_a): 0.005
"""

import numpy as np
import matplotlib.pyplot as plt
from marketsim.tests.comparison_test import generate_shared_private_values
from marketsim.tests.sim_test import run_simulator_test
from marketsim.tests.hbl_tests.hbl_split_test import run_hbl_test
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
import sys
from io import StringIO


def run_paper_batched_comparison(num_iterations=25, num_buyers=28, num_sellers=28, q_max=3,
                                  pv_var=5e6, obs_noise_var=1e3, shade_range=None, timesteps=200,
                                  L=6, arrival_rate=0.005, fundamental_mean=1e5,
                                  mean_reversion_rate=0.05, shock_var=1e5):
    """
    Run the comparison test multiple times with paper specifications.
    
    Paper Parameters:
    - pv_var: 5 × 10^6 (private value variance σ²_s)
    - obs_noise_var: 10^3 (observation noise variance σ²_n)
    - fundamental_mean: 10^5 (fundamental mean r̄)
    - mean_reversion_rate: 0.05 (mean reversion)
    - shock_var: 10^5 (shock variance σ²_s)
    - arrival_rate: 0.005 (Poisson arrival rate λ_a)
    
    Returns aggregated metrics across all iterations.
    """
    
    if shade_range is None:
        shade_range = [0, 100]
    
    print("=" * 100)
    print("RUNNING PAPER BATCHED COMPARISON: ZI AGENTS vs HBL AGENTS")
    print("=" * 100)
    print(f"\nPaper Specifications:")
    print(f"  Private Value Variance (σ²_s): {pv_var:.2e}")
    print(f"  Observation Noise Variance (σ²_n): {obs_noise_var:.2e}")
    print(f"  Fundamental Mean (r̄): {fundamental_mean:.2e}")
    print(f"  Mean Reversion Rate: {mean_reversion_rate:.4f}")
    print(f"  Shock Variance (σ²_s): {shock_var:.2e}")
    print(f"  Arrival Rate (λ_a): {arrival_rate:.4f}")
    print(f"\nMarket Setup:")
    print(f"  Buyers: {num_buyers}, Sellers: {num_sellers}")
    print(f"  q_max: {q_max}")
    print(f"  Shade range: {shade_range}")
    print(f"  Timesteps: {timesteps}")
    print(f"  Number of iterations: {num_iterations}")
    print(f"\nHBL-specific Parameters:")
    print(f"  L: {L}")
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
    zi_rmsd_vs_actual_all = []
    hbl_rmsd_vs_actual_all = []
    
    # Run iterations
    for iteration in range(num_iterations):
        print(f"\n[{iteration + 1}/{num_iterations}] Running iteration...")
        
        # Generate shared private values
        pv_base_price = 0.0
        shared_pv, equilibrium_price, optimal_surplus, optimal_num_trades, all_buyer_values, all_seller_costs = generate_shared_private_values(
            num_buyers, num_sellers, q_max, pv_var, pv_base_price, fundamental_mean
        )
        
        # Create time-varying fundamental model for this iteration
        fundamental = LazyGaussianMeanReverting(
            final_time=timesteps,
            mean=fundamental_mean,
            r=mean_reversion_rate,
            shock_var=shock_var
        )
        
        # Suppress output from simulator tests
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Run ZI test with observation noise
        zi_agents, zi_transactions, zi_eq_price, zi_initial_pv = run_simulator_test(
            num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_mean,
            shared_pv=shared_pv, fundamental=fundamental, obs_noise_var=obs_noise_var, 
            arrival_rate=arrival_rate
        )
        
        # Run HBL test with observation noise
        hbl_agents, hbl_transactions, hbl_eq_price, hbl_initial_pv = run_hbl_test(
            num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps,
            L=L, arrival_rate=arrival_rate, fundamental_value=fundamental_mean,
            shared_pv=shared_pv, fundamental=fundamental, obs_noise_var=obs_noise_var
        )
        
        sys.stdout = old_stdout
        
        # Calculate surpluses
        zi_total_surpluses = [tx['total_surplus'] for tx in zi_transactions if tx['total_surplus'] is not None]
        hbl_total_surpluses = [tx['total_surplus'] for tx in hbl_transactions if tx['total_surplus'] is not None]
        
        zi_total_surplus_sum = sum(zi_total_surpluses) if zi_total_surpluses else 0
        hbl_total_surplus_sum = sum(hbl_total_surpluses) if hbl_total_surpluses else 0
        
        # Get market info and fundamental for estimation
        market = zi_agents[list(zi_agents.keys())[0]].market
        mean, r, T = market.get_info()
        fundamental_obj = market.fundamental
        
        # Calculate RMSD for transaction prices compared to estimated fundamental value (with noise)
        # Note: We calculate what the estimate WOULD BE without noise for comparison
        zi_rmsd_values = []
        zi_rmsd_vs_actual_values = []
        for tx in zi_transactions:
            if tx['price'] is not None and tx['time'] is not None:
                t = tx['time']
                fv_at_t = fundamental_obj.get_value_at(t)
                rho = (1 - r) ** (T - t)
                estimate_no_noise = (1 - rho) * mean + rho * fv_at_t
                zi_rmsd_values.append((tx['price'] - estimate_no_noise) ** 2)
                zi_rmsd_vs_actual_values.append((tx['price'] - fv_at_t) ** 2)
        
        hbl_rmsd_values = []
        hbl_rmsd_vs_actual_values = []
        for tx in hbl_transactions:
            if tx['price'] is not None and tx['time'] is not None:
                t = tx['time']
                fv_at_t = fundamental_obj.get_value_at(t)
                rho = (1 - r) ** (T - t)
                estimate_no_noise = (1 - rho) * mean + rho * fv_at_t
                hbl_rmsd_values.append((tx['price'] - estimate_no_noise) ** 2)
                hbl_rmsd_vs_actual_values.append((tx['price'] - fv_at_t) ** 2)
        
        zi_rmsd = np.sqrt(np.mean(zi_rmsd_values)) if zi_rmsd_values else 0
        hbl_rmsd = np.sqrt(np.mean(hbl_rmsd_values)) if hbl_rmsd_values else 0
        zi_rmsd_vs_actual = np.sqrt(np.mean(zi_rmsd_vs_actual_values)) if zi_rmsd_vs_actual_values else 0
        hbl_rmsd_vs_actual = np.sqrt(np.mean(hbl_rmsd_vs_actual_values)) if hbl_rmsd_vs_actual_values else 0
        
        zi_total_surpluses_all.append(zi_total_surplus_sum)
        hbl_total_surpluses_all.append(hbl_total_surplus_sum)
        zi_transactions_all.append(len(zi_transactions))
        hbl_transactions_all.append(len(hbl_transactions))
        optimal_surpluses.append(optimal_surplus)
        optimal_num_trades_all.append(optimal_num_trades)
        equilibrium_prices.append(equilibrium_price)
        zi_rmsd_all.append(zi_rmsd)
        hbl_rmsd_all.append(hbl_rmsd)
        zi_rmsd_vs_actual_all.append(zi_rmsd_vs_actual)
        hbl_rmsd_vs_actual_all.append(hbl_rmsd_vs_actual)
        
        print(f"  ZI Surplus: {zi_total_surplus_sum:.2f}, HBL Surplus: {hbl_total_surplus_sum:.2f}, Optimal: {optimal_surplus:.2f}")
    
    print("\n" + "=" * 100)
    print("PAPER BATCHED COMPARISON RESULTS (AVERAGES ACROSS {} ITERATIONS)".format(num_iterations))
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
    avg_zi_rmsd_vs_actual = np.mean(zi_rmsd_vs_actual_all)
    avg_hbl_rmsd_vs_actual = np.mean(hbl_rmsd_vs_actual_all)
    
    print(f"\n{'Avg RMSD from Estimated Fundamental':<40} {avg_zi_rmsd:<30.2f} {avg_hbl_rmsd:<30.2f}")
    print(f"{'Avg RMSD from Actual Fundamental':<40} {avg_zi_rmsd_vs_actual:<30.2f} {avg_hbl_rmsd_vs_actual:<30.2f}")
    
    print("\n" + "=" * 100)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    iterations = np.arange(1, num_iterations + 1)
    
    # Plot 1: Total Surplus Comparison
    ax = axes[0, 0]
    ax.scatter(iterations, zi_total_surpluses_all, s=100, alpha=0.7, label='ZI Agents', color='blue', marker='o')
    ax.scatter(iterations, hbl_total_surpluses_all, s=100, alpha=0.7, label='HBL Agents', color='green', marker='s')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Surplus', fontsize=12)
    ax.set_title(f'Total Surplus Comparison: ZI vs HBL Agents ({num_iterations} iterations)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: RMSD from Estimated Fundamental Value
    ax = axes[0, 1]
    ax.scatter(iterations, zi_rmsd_all, s=100, alpha=0.7, label='ZI Agents', color='blue', marker='o')
    ax.scatter(iterations, hbl_rmsd_all, s=100, alpha=0.7, label='HBL Agents', color='green', marker='s')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('RMSD from Estimated Fundamental', fontsize=12)
    ax.set_title(f'Price Efficiency: RMSD from Estimated Fundamental ({num_iterations} iterations)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: RMSD from Actual Fundamental Value
    ax = axes[1, 0]
    ax.scatter(iterations, zi_rmsd_vs_actual_all, s=100, alpha=0.7, label='ZI Agents', color='blue', marker='o')
    ax.scatter(iterations, hbl_rmsd_vs_actual_all, s=100, alpha=0.7, label='HBL Agents', color='green', marker='s')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('RMSD from Actual Fundamental', fontsize=12)
    ax.set_title(f'Price Efficiency: RMSD from Actual Fundamental ({num_iterations} iterations)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Transaction Counts
    ax = axes[1, 1]
    ax.scatter(iterations, zi_transactions_all, s=100, alpha=0.7, label='ZI Agents', color='blue', marker='o')
    ax.scatter(iterations, hbl_transactions_all, s=100, alpha=0.7, label='HBL Agents', color='green', marker='s')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Transactions', fontsize=12)
    ax.set_title(f'Transaction Count Comparison ({num_iterations} iterations)', fontsize=14, fontweight='bold')
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
        'avg_zi_rmsd_vs_actual': avg_zi_rmsd_vs_actual,
        'avg_hbl_rmsd_vs_actual': avg_hbl_rmsd_vs_actual,
        'avg_zi_pct_optimal': avg_zi_pct_optimal,
        'avg_hbl_pct_optimal': avg_hbl_pct_optimal,
    }


if __name__ == '__main__':
    run_paper_batched_comparison(num_iterations=25)
