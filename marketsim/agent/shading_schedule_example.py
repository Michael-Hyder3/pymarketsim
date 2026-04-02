"""
Example: Time-Based and Transaction-Based Shading Schedules for ZI Agents

This demonstrates how to use dynamic shading ranges that change over time
or based on the number of transactions that have occurred.
"""

from marketsim.tests.new_setting_tests.mixed_agent_simulation import run_mixed_agent_test

# ============================================================================
# Example 1: Static shading (traditional behavior)
# ============================================================================
# Agents always use the same shading range [0, 100]
static_example = run_mixed_agent_test(
    num_zi_buy=3,
    num_zi_sell=3,
    zi_buy_shade_ranges=[[0, 100]],  # Same range for all buyers
    zi_buy_shade_counts=[3],
    timesteps=100
)
print("Example 1 (Static): All buyers use [0, 100]")


# ============================================================================
# Example 2: Time-based shading schedule
# ============================================================================
# Buyers start aggressive [0, 100] and become less aggressive [0, 50] at t=50
# Schedule format: [(time_trigger, [shade_min, shade_max]), ...]
time_schedule = [
    (0, [0, 100]),   # At timestep 0, use [0, 100]
    (50, [0, 50]),   # At timestep 50, switch to [0, 50]
    (80, [0, 25]),   # At timestep 80, switch to [0, 25]
]

time_based_example = run_mixed_agent_test(
    num_zi_buy=3,
    num_zi_sell=3,
    zi_buy_shade_ranges=[time_schedule],  # Pass the schedule
    zi_buy_shade_counts=[3],
    shade_schedule_mode="time",  # Enable time-based scheduling
    timesteps=100
)
print("Example 2 (Time-based): Buyers use dynamic shading over time")


# ============================================================================
# Example 3: Transaction-based shading schedule
# ============================================================================
# Buyers become less aggressive as more transactions occur
# This allows adaptive behavior based on market activity
transaction_schedule = [
    (0, [0, 100]),     # Start aggressive: [0, 100]
    (10, [0, 75]),     # After 10 transactions: [0, 75]
    (25, [0, 50]),     # After 25 transactions: [0, 50]
    (50, [0, 25]),     # After 50 transactions: [0, 25]
]

transaction_based_example = run_mixed_agent_test(
    num_zi_buy=3,
    num_zi_sell=3,
    zi_buy_shade_ranges=[transaction_schedule],
    zi_buy_shade_counts=[3],
    shade_schedule_mode="transactions",  # Enable transaction-based scheduling
    timesteps=100
)
print("Example 3 (Transaction-based): Buyers adapt based on market activity")


# ============================================================================
# Example 4: Mix static and scheduled agents
# ============================================================================
# Some agents use static shading, others use schedules
mixed_example = run_mixed_agent_test(
    num_zi_buy=5,
    num_zi_sell=5,
    zi_buy_shade_ranges=[
        [0, 100],       # Buyers 0-1: static [0, 100]
        time_schedule,  # Buyers 2-3: time-based schedule
    ],
    zi_buy_shade_counts=[2, 2],  # 2 agents each
    shade_schedule_mode="time",
    timesteps=100
)
print("Example 4 (Mixed): Some agents static, others scheduled")


# ============================================================================
# Notes on Usage:
# ============================================================================
# 1. shade parameter can be:
#    - [min, max]  for static shading
#    - [(t1, [min1, max1]), (t2, [min2, max2]), ...] for schedules
#
# 2. Schedule mode ("time" or "transactions"):
#    - "time": Triggers based on current timestep
#    - "transactions": Triggers based on number of matched orders
#
# 3. Schedules are applied per-agent or per-group via shade_schedule_mode
#    in run_mixed_agent_test
#
# 4. The latest applicable trigger is used (schedule is monotonic)
#    Example: At t=75 with schedule [(0, [0,100]), (50, [0,50])],
#    the agent uses [0, 50]
