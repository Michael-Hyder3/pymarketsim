"""
Batched Comparison Test: Run the comparison test multiple times and aggregate statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from marketsim.tests.comparison_test import compare_tests
import sys
from io import StringIO


def run_batched_comparison(num_iterations=25, num_buyers=28, num_sellers=28, q_max=3, pv_var=1e5, 
                           shade_range=None, timesteps=200, L=6, arrival_rate=0.1, 
                           fundamental_value=100000.0, use_time_varying=True,
                           mean_reversion_rate=0.05, shock_var=1e4, obs_noise_var=1e3, show_plots=False):
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
    print(f"  Observation Noise Variance: {obs_noise_var:.2f}")
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
        
        # Suppress output from comparison test
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Run comparison test which handles everything internally
        results = compare_tests(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            q_max=q_max,
            pv_var=pv_var,
            shade_range=shade_range,
            timesteps=timesteps,
            L=L,
            arrival_rate=arrival_rate,
            fundamental_value=fundamental_value,
            use_time_varying=use_time_varying,
            mean_reversion_rate=mean_reversion_rate,
            shock_var=shock_var,
            obs_noise_var=obs_noise_var,
            debug_surplus=False,
            debug_hbl_trades=False,
            show_plots=False
        )
        
        sys.stdout = old_stdout
        
        # Extract results from comparison test
        zi_total_surplus_sum = results['zi_agent_surplus']  # Use end-of-period surplus
        hbl_total_surplus_sum = results['hbl_agent_surplus']  # Use end-of-period surplus
        optimal_surplus = results['optimal_surplus']
        optimal_num_trades = results['optimal_num_trades']
        equilibrium_price = results['equilibrium_price']
        
        zi_total_surpluses_all.append(zi_total_surplus_sum)
        hbl_total_surpluses_all.append(hbl_total_surplus_sum)
        zi_transactions_all.append(results['zi_num_transactions'])
        hbl_transactions_all.append(results['hbl_num_transactions'])
        optimal_surpluses.append(optimal_surplus)
        optimal_num_trades_all.append(optimal_num_trades)
        equilibrium_prices.append(equilibrium_price)
        zi_rmsd_all.append(results['zi_rmsd'])
        hbl_rmsd_all.append(results['hbl_rmsd'])
        
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
    
    # Calculate percentages and averages
    avg_zi_pct_optimal = (avg_zi_total_surplus / avg_optimal_surplus * 100) if avg_optimal_surplus > 0 else 0
    avg_hbl_pct_optimal = (avg_hbl_total_surplus / avg_optimal_surplus * 100) if avg_optimal_surplus > 0 else 0
    avg_zi_rmsd = np.mean(zi_rmsd_all)
    avg_hbl_rmsd = np.mean(hbl_rmsd_all)
    
    print(f"\n{'Metric':<40} {'ZI Agents':<30} {'HBL Agents':<30}")
    print("-" * 100)
    print(f"{'Avg Total Transactions':<40} {avg_zi_transactions:<30.2f} {avg_hbl_transactions:<30.2f}")
    print(f"{'Avg Total Surplus (sum)':<40} {avg_zi_total_surplus:<30.2f} {avg_hbl_total_surplus:<30.2f}")
    print(f"{'Avg Optimal Potential Surplus':<40} {avg_optimal_surplus:<30.2f} {avg_optimal_surplus:<30.2f}")
    print(f"{'Avg % of Optimal Surplus':<40} {avg_zi_pct_optimal:<30.2f} {avg_hbl_pct_optimal:<30.2f}")
    
    print(f"{'Avg Optimal Number of Trades':<40} {avg_optimal_num_trades:<30.2f} {avg_optimal_num_trades:<30.2f}")
    print(f"{'Avg Equilibrium Price':<40} {avg_equilibrium_price:<30.2f} {avg_equilibrium_price:<30.2f}")
    print(f"{'Avg RMSD from Equilibrium':<40} {avg_zi_rmsd:<30.2f} {avg_hbl_rmsd:<30.2f}")
    
    print("\n" + "=" * 100)
    
    if show_plots:
        # Create plots (only once, at the end)
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
        
        # Plot 2: % of Optimal
        ax = axes[1]
        zi_pct_all = [(zi / opt * 100) if opt > 0 else 0 for zi, opt in zip(zi_total_surpluses_all, optimal_surpluses)]
        hbl_pct_all = [(hbl / opt * 100) if opt > 0 else 0 for hbl, opt in zip(hbl_total_surpluses_all, optimal_surpluses)]
        ax.scatter(iterations, zi_pct_all, s=100, alpha=0.7, label='ZI Agents', color='blue', marker='o')
        ax.scatter(iterations, hbl_pct_all, s=100, alpha=0.7, label='HBL Agents', color='green', marker='s')
        ax.axhline(100, color='red', linestyle='--', linewidth=2, label='100% (Optimal)')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('% of Optimal Surplus', fontsize=12)
        ax.set_title(f'Efficiency: % of Optimal Surplus ({num_iterations} iterations)', fontsize=14, fontweight='bold')
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
        'avg_zi_pct_optimal': avg_zi_pct_optimal,
        'avg_hbl_pct_optimal': avg_hbl_pct_optimal,
        'avg_zi_rmsd': avg_zi_rmsd,
        'avg_hbl_rmsd': avg_hbl_rmsd,
    }


if __name__ == '__main__':
    # Parse command-line arguments
    num_iterations = 10
    timesteps = 200
    
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
    
    run_batched_comparison(num_iterations=num_iterations, timesteps=timesteps, show_plots=True)
