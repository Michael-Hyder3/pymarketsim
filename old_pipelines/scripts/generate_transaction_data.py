"""
Generate offline ZI trajectory data in the unified diagnostic JSON format.

Each output file contains one ZI agent's complete trajectory through a market
simulation, using the same schema that run_episode(capture_detail=True) produces.
This data grounds the LLM's initial strategy generation (episode 0 context).

Output schema per file:
{
  "agent_type": "buyer" | "seller",
  "total_transactions": int,       -- all trades in the market this episode
  "agent_transactions": int,       -- trades the ZI agent was involved in
  "agent_total_surplus": float,    -- surplus the ZI agent earned
  "timesteps": int,                -- simulation length
  "events": [                      -- sorted by time
    {
      "time": int,
      "event_type": "trade" | "agent_order",
      "agent_transacted": bool,         -- true if this agent executed a trade
      "agent_bid": bool,               -- true if this agent submitted an order
      "trade_price_relative": float,   -- present when event_type="trade"
      "agent_action": {                -- present when agent_bid=True
        "submitted_price_relative": float,
        "max_bid_relative": float,     -- buy agent only
        "shade_used": float,
        "surplus_earned": float        -- present on trade events only
      },
      "lob": {
        "best_bid_relative": float | null,
        "best_ask_relative": float | null,
        "bids_relative": [up to 4 prices],
        "asks_relative": [up to 4 prices]
      }
    },
    ...
  ]
}

All prices are RELATIVE to the fundamental value (~0, typical range ±500).
Opponent counts, pv_var, shade_range are NOT included — the LLM must infer
market structure from the price data alone.
"""

import json
import os
import random

import numpy as np

from marketsim.fourheap.constants import BUY, SELL
from marketsim.LLM.market_config import MarketConfig
from marketsim.LLM.episode_runner import run_episode
from marketsim.LLM.feedback_loop import DEFAULT_MARKET_CONFIGS


# ---------------------------------------------------------------------------
# ZI fixed-shade strategy — used as the "oracle" ZI agent in data generation
# ---------------------------------------------------------------------------

def zi_fixed_shade_strategy(private_values, market_history, market_state):
    """
    Zero-intelligence buyer: shade uniformly in [0, 500] from max_bid.
    This matches the ZI agent baseline the LLM must beat.
    """
    max_bid = market_state["max_bid"]
    shade = random.uniform(0, 500)
    return (max_bid - shade, 1)


def zi_fixed_shade_strategy_sell(private_values, market_history, market_state):
    """Zero-intelligence seller: shade uniformly in [0, 500] above min_ask."""
    min_ask = market_state["min_ask"]
    shade = random.uniform(0, 500)
    return (min_ask + shade, 1)


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_for_config(
    config: MarketConfig,
    seed: int,
    agent_type: str = "buyer",
    max_events: int = 60,
) -> dict:
    """
    Run one episode with a ZI strategy and return the clean diagnostic JSON.

    The ZI strategy is intentionally suboptimal — it represents the baseline
    the LLM is expected to improve upon.  The output shows:
      - How many trades happen and at what relative prices
      - The LOB state (best bid/ask, top-4 depth) at each event
      - When the agent submitted orders that went unexecuted (agent_order events)
      - How much surplus the ZI baseline earns

    No opponent internals (agent counts, pv_var, shade distribution) are included.

    max_events: cap on background (non-agent) events to keep JSON compact.
                All agent-involved events are always kept.
    """
    strategy = zi_fixed_shade_strategy if agent_type == "buyer" else zi_fixed_shade_strategy_sell
    order_type = BUY if agent_type == "buyer" else SELL

    result = run_episode(
        llm_strategy_func=strategy,
        llm_order_type=order_type,
        num_zi_buy=config.num_zi_buy,
        num_zi_sell=config.num_zi_sell,
        timesteps=config.timesteps,
        q_max=config.q_max,
        pv_var=config.pv_var,
        shade_range=config.shade_range,
        random_seed=seed,
        capture_detail=True,
    )

    diag = result["diagnostic_json"]
    # diagnostic_json already has the clean schema: no pv_var, shade_range, etc.

    all_events = diag.get("events", [])

    # Always keep agent trade events (agent_transacted=True).
    # Subsample agent_order events (agent_bid=True, agent_transacted=False)
    # and background trades (agent_transacted=False, agent_bid=False).
    agent_trade_events  = [e for e in all_events if e.get("agent_transacted")]
    agent_order_events  = [e for e in all_events if e.get("agent_bid") and not e.get("agent_transacted")]
    background_events   = [e for e in all_events if not e.get("agent_bid") and not e.get("agent_transacted")]

    # Budget for non-agent-trade events
    remaining = max(0, max_events - len(agent_trade_events))
    # Split remaining budget evenly between agent_order and background
    order_budget      = remaining // 2
    background_budget = remaining - order_budget

    if len(agent_order_events) > order_budget:
        step = len(agent_order_events) / order_budget
        agent_order_events = [agent_order_events[int(i * step)] for i in range(order_budget)]
    if len(background_events) > background_budget:
        step = len(background_events) / background_budget
        background_events = [background_events[int(i * step)] for i in range(background_budget)]

    sampled_events = sorted(agent_trade_events + agent_order_events + background_events,
                            key=lambda e: e["time"])
    diag["events"] = sampled_events

    return diag


def main():
    os.makedirs("results", exist_ok=True)

    configs = DEFAULT_MARKET_CONFIGS
    base_seed = 9999  # fixed seed for reproducibility

    print("=" * 65)
    print("Generating offline ZI trajectory data (unified format)")
    print("=" * 65)

    generated_files = []

    for i, cfg in enumerate(configs):
        seed = base_seed + i * 1000
        label = cfg.label or cfg.config_hash()
        fname = f"results/offline_zi_{label}.json"

        print(f"\n[{i+1}/{len(configs)}] Config: {cfg}  seed={seed}")
        random.seed(seed)
        np.random.seed(seed)

        data = generate_for_config(cfg, seed=seed, agent_type="buyer")

        with open(fname, "w") as f:
            json.dump(data, f, indent=2)

        n_txns = data["total_transactions"]
        n_agent = data["agent_transactions"]
        surplus = data["agent_total_surplus"]
        n_events = len(data["events"])
        n_agent_trades = sum(1 for e in data["events"] if e.get("agent_transacted"))
        n_agent_bids   = sum(1 for e in data["events"] if e.get("agent_bid") and not e.get("agent_transacted"))
        print(f"  Total market trades:   {n_txns}")
        print(f"  Agent trades:          {n_agent}")
        print(f"  Agent surplus:         ${surplus:.2f}")
        print(f"  Events in JSON:        {n_events} ({n_agent_trades} transacted, {n_agent_bids} bids-only)")
        print(f"  Saved → {fname}")
        generated_files.append(fname)

    # Combined summary file (for quick LLM reference)
    summary = {
        "description": (
            "Offline ZI buyer trajectories for each market configuration. "
            "Used as D_offline baseline for LLM strategy initialisation. "
            "All prices are RELATIVE to fundamental (~0, range approximately ±500). "
            "max_bid is the agent's relative valuation; shade = max_bid - bid_price. "
            "lob fields show the limit order book state at the time of each event."
        ),
        "configs": [],
    }
    for i, cfg in enumerate(configs):
        seed = base_seed + i * 1000
        label = cfg.label or cfg.config_hash()
        fname = f"results/offline_zi_{label}.json"
        with open(fname) as f:
            d = json.load(f)
        summary["configs"].append({
            "label": label,
            "agent_transactions": d["agent_transactions"],
            "agent_total_surplus": d["agent_total_surplus"],
            "sample_events": d["events"][:5],  # first 5 for overview
        })

    summary_path = "results/offline_zi_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Combined summary → {summary_path}")

    print("\n" + "=" * 65)
    print("Done. Files generated:")
    for p in generated_files:
        print(f"  {p}")
    print(f"  {summary_path}")
    print("=" * 65)
    print("\nTo use as offline_data_path in FeedbackLoopPipeline:")
    print("  FeedbackLoopPipeline(offline_data_path='results/offline_zi_baseline.json', ...)")


if __name__ == "__main__":
    main()
