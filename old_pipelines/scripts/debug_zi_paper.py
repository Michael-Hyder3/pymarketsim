"""
Debug script to understand why ZI agents aren't trading with paper specifications.
"""

import numpy as np
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
timesteps = 200
fundamental_mean = 1e5
mean_reversion_rate = 0.05
shock_var = 1e5

print("=" * 80)
print("DEBUGGING ZI AGENTS WITH PAPER SPECIFICATIONS")
print("=" * 80)

# Generate shared private values
pv_base_price = 0.0
shared_pv, equilibrium_price, optimal_surplus, optimal_num_trades, all_buyer_values, all_seller_costs = generate_shared_private_values(
    num_buyers, num_sellers, q_max, pv_var, pv_base_price, fundamental_mean
)

print(f"\nEquilibrium price: {equilibrium_price:.2f}")
print(f"Optimal surplus: {optimal_surplus:.2f}")
print(f"Optimal trades: {optimal_num_trades}")

print(f"\nBuyer values (first 5 buyers, all units):")
for i in range(min(5, num_buyers)):
    print(f"  Buyer {i}: {all_buyer_values[i]}")

print(f"\nSeller costs (first 5 sellers, all units):")
for i in range(min(5, num_sellers)):
    print(f"  Seller {i}: {all_seller_costs[i]}")

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
    shared_pv=shared_pv, fundamental=fundamental, obs_noise_var=obs_noise_var
)

print(f"\nNumber of ZI transactions: {len(zi_transactions)}")

if len(zi_transactions) > 0:
    print(f"\nFirst 10 transactions:")
    for i, tx in enumerate(zi_transactions[:10]):
        print(f"  {i+1}. Time {tx['time']}, Price: {tx['price']:.2f}, Buyer surplus: {tx.get('buyer_surplus', 'N/A')}, Seller surplus: {tx.get('seller_surplus', 'N/A')}")
else:
    print("\nNO TRANSACTIONS OCCURRED!")
    print("\nLet's check some sample orders from the simulation...")
    
    # Get a sample buyer and seller
    sample_buyer = zi_agents['buy_0']
    sample_seller = zi_agents['sell_0']
    
    print(f"\nSample Buyer 0:")
    print(f"  Private value for first unit: {sample_buyer.pv.buyer_values[0]:.2f}")
    print(f"  Position: {sample_buyer.position}")
    
    print(f"\nSample Seller 0:")
    print(f"  Private cost for first unit: {sample_seller.pv.seller_costs[0]:.2f}")
    print(f"  Position: {sample_seller.position}")
    
    # Simulate what prices they would generate at time 0
    market = sample_buyer.market
    print(f"\nFundamental at time 0: {fundamental.get_value_at(0):.2f}")
    
    # Check estimate at time 0
    mean, r, T = market.get_info()
    t = 0
    val = fundamental.get_value_at(t)
    rho = (1-r)**(T-t)
    estimate_no_noise = (1-rho)*mean + rho*val
    
    print(f"\nEstimated fundamental (no noise) at t=0: {estimate_no_noise:.2f}")
    print(f"  mean={mean:.2f}, r={r}, T={T}, t={t}")
    print(f"  rho={(1-r)**(T-t):.6f}")
    
    # Sample some buyer prices
    print(f"\nSample buyer bid prices (with noise and shading):")
    np.random.seed(42)
    for trial in range(5):
        buyer_value = sample_buyer.pv.buyer_values[0].item()
        obs_noise = np.random.normal(0, np.sqrt(obs_noise_var))
        shade_amount = np.random.uniform(shade_range[0], 0)  # Buyers shade down
        estimate_with_noise = estimate_no_noise + obs_noise
        price = estimate_with_noise + buyer_value + shade_amount
        print(f"  Trial {trial+1}: estimate={estimate_with_noise:.2f}, buyer_val={buyer_value:.2f}, shade={shade_amount:.2f} -> bid={price:.2f}")
    
    print(f"\nSample seller ask prices (with noise and shading):")
    np.random.seed(100)
    for trial in range(5):
        seller_cost = sample_seller.pv.seller_costs[0].item()
        obs_noise = np.random.normal(0, np.sqrt(obs_noise_var))
        shade_amount = np.random.uniform(0, shade_range[1])  # Sellers shade up
        estimate_with_noise = estimate_no_noise + obs_noise
        price = estimate_with_noise + seller_cost + shade_amount
        print(f"  Trial {trial+1}: estimate={estimate_with_noise:.2f}, seller_cost={seller_cost:.2f}, shade={shade_amount:.2f} -> ask={price:.2f}")
