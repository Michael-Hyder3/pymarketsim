"""
Debug script to understand ZI agent pricing with paper specifications (10k timesteps).
"""

import numpy as np
import random
from marketsim.tests.comparison_test import generate_shared_private_values
from marketsim.tests.sim_test import run_simulator_test
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting

# Paper specifications
num_buyers = 28
num_sellers = 28
q_max = 3
pv_var = 5e6
obs_noise_var = 1e3
shade_range = [0, 100]
timesteps = 500  # Use shorter timesteps for faster debugging
fundamental_mean = 1e5
mean_reversion_rate = 0.05
shock_var = 1e5
arrival_rate = 0.005

print("=" * 80)
print("DEBUGGING ZI AGENTS WITH PAPER SPECIFICATIONS (LONGER TIMESTEPS)")
print("=" * 80)

# Generate shared private values
pv_base_price = 0.0
shared_pv, equilibrium_price, optimal_surplus, optimal_num_trades, all_buyer_values, all_seller_costs = generate_shared_private_values(
    num_buyers, num_sellers, q_max, pv_var, pv_base_price, fundamental_mean
)

print(f"\nOptimal trades: {optimal_num_trades}")
print(f"Expected agents per timestep (λ × num_agents): {arrival_rate * (num_buyers + num_sellers):.2f}")

# Analyze buyer and seller distributions
all_buyers = []
all_sellers = []
for i in range(num_buyers):
    all_buyers.extend(shared_pv[f'buy_{i}'])
for i in range(num_sellers):
    all_sellers.extend(shared_pv[f'sell_{i}'])

all_buyers = np.array(all_buyers)
all_sellers = np.array(all_sellers)

print(f"\nBuyer values: min={all_buyers.min():.2f}, max={all_buyers.max():.2f}, mean={all_buyers.mean():.2f}, std={all_buyers.std():.2f}")
print(f"Seller costs: min={all_sellers.min():.2f}, max={all_sellers.max():.2f}, mean={all_sellers.mean():.2f}, std={all_sellers.std():.2f}")

# Create time-varying fundamental
fundamental = LazyGaussianMeanReverting(
    final_time=timesteps,
    mean=fundamental_mean,
    r=mean_reversion_rate,
    shock_var=shock_var
)

print(f"\n" + "=" * 80)
print("RUNNING ZI SIMULATION")
print("=" * 80)

# Run ZI test with observation noise
zi_agents, zi_transactions, zi_eq_price, zi_initial_pv = run_simulator_test(
    num_buyers, num_sellers, q_max, pv_var, shade_range, timesteps, fundamental_mean,
    shared_pv=shared_pv, fundamental=fundamental, obs_noise_var=obs_noise_var,
    arrival_rate=arrival_rate
)

print(f"\nNumber of ZI transactions: {len(zi_transactions)}")
print(f"Optimal potential transactions: {optimal_num_trades}")

if len(zi_transactions) > 0:
    print(f"\nFirst 10 transactions:")
    for i, tx in enumerate(zi_transactions[:10]):
        print(f"  {i+1}. Time {tx['time']}, Price: {tx['price']:.2f}, Total Surplus: {tx.get('total_surplus', 'N/A'):.2f}")
else:
    print("\nNO TRANSACTIONS OCCURRED!")
    print("\nAnalyzing expected bid/ask spreads...")
    
    # Sample some expected prices from random agents
    print(f"\n--- Sample Expected Prices at Equilibrium ---")
    print(f"Equilibrium price: {equilibrium_price:.2f}")
    
    # Get fundamental estimate
    estimate_no_noise = fundamental_mean  # At t=0, estimate = mean since rho is tiny
    
    print(f"\nEstimated fundamental at t=0: {estimate_no_noise:.2f}")
    print(f"Observation noise std: {np.sqrt(obs_noise_var):.2f}")
    print(f"\nSample buyer bids (with shading down to [-100, 0]):")
    
    # Sample 5 random buyers
    np.random.seed(42)
    for trial in range(5):
        buyer_idx = np.random.randint(0, num_buyers)
        buyer_value = shared_pv[f'buy_{buyer_idx}'][0]
        obs_noise = np.random.normal(0, np.sqrt(obs_noise_var))
        shade = np.random.uniform(shade_range[0], 0)
        bid = estimate_no_noise + obs_noise + buyer_value + shade
        print(f"  Buyer {buyer_idx}: value={buyer_value:.2f}, noise={obs_noise:+.2f}, shade={shade:+.2f} -> bid={bid:.2f}")
    
    print(f"\nSample seller asks (with shading up to [0, 100]):")
    # Sample 5 random sellers
    np.random.seed(100)
    for trial in range(5):
        seller_idx = np.random.randint(0, num_sellers)
        seller_cost = shared_pv[f'sell_{seller_idx}'][0]
        obs_noise = np.random.normal(0, np.sqrt(obs_noise_var))
        shade = np.random.uniform(0, shade_range[1])
        ask = estimate_no_noise + obs_noise + seller_cost + shade
        print(f"  Seller {seller_idx}: cost={seller_cost:.2f}, noise={obs_noise:+.2f}, shade={shade:+.2f} -> ask={ask:.2f}")
    
    print(f"\n--- Analysis ---")
    print(f"Most optimistic buyer bid (largest PV + smallest noise + minimum shade):")
    best_buyer_value = all_buyers.max()
    best_buyer_bid = estimate_no_noise - np.sqrt(obs_noise_var) * 3 + best_buyer_value + shade_range[0]
    print(f"  Max buyer PV: {best_buyer_value:.2f}, Expected bid range: [{best_buyer_bid - 100:.2f}, {best_buyer_bid:.2f}]")
    
    print(f"\nMost pessimistic seller ask (smallest cost + largest noise + maximum shade):")
    best_seller_cost = all_sellers.max()
    best_seller_ask = estimate_no_noise + np.sqrt(obs_noise_var) * 3 + best_seller_cost + shade_range[1]
    print(f"  Max seller cost: {best_seller_cost:.2f}, Expected ask range: [{best_seller_ask:.2f}, {best_seller_ask + 100:.2f}]")
    
    print(f"\nWith sparse arrivals (λ={arrival_rate}), buyers and sellers rarely meet.")
    print(f"Even when they do, random shading makes price overlap unlikely!")
