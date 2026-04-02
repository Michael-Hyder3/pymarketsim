"""
Extended mixed agent test including LLMFirstZI agents for comparison.

This test extends the original mixed_agent_test to compare:
- ZI (Zero Intelligence) agents
- HBL (Human Behavior Learning) agents  
- LLMFirstZI (LLM-designed heuristic) agents
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.fundamental.dummy_fundamental import DummyFundamental
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.agent.hbl_agent_buy_sell import HBLAgentBuy, HBLAgentSell
from marketsim.agent.llm_first_zi_agent import LLMFirstZIAgentBuy, LLMFirstZIAgentSell


def _pair_matched_orders(matched_orders):
    """Pair buyer and seller orders from matched orders."""
    paired_trades = {}
    for mo in matched_orders:
        key = (mo.time, mo.order.quantity)
        if key not in paired_trades:
            paired_trades[key] = {
                "time": mo.time,
                "quantity": mo.order.quantity,
                "buyer_id": None,
                "seller_id": None,
                "buyer_order": None,
                "seller_order": None,
            }
        if mo.order.order_type == BUY:
            paired_trades[key]["buyer_id"] = mo.order.agent_id
            paired_trades[key]["buyer_order"] = mo.order
        else:
            paired_trades[key]["seller_id"] = mo.order.agent_id
            paired_trades[key]["seller_order"] = mo.order
    return paired_trades


def _trade_price_from_orders(buyer_order, seller_order):
    """Determine final trade price based on which order came first."""
    if buyer_order is not None and seller_order is not None:
        if buyer_order.time < seller_order.time:
            return buyer_order.price, "BUY"
        if buyer_order.time > seller_order.time:
            return seller_order.price, "SELL"
        b_seq = getattr(buyer_order, "_arrival_seq", 0)
        s_seq = getattr(seller_order, "_arrival_seq", 0)
        if b_seq <= s_seq:
            return buyer_order.price, "BUY"
        return seller_order.price, "SELL"
    if buyer_order is not None:
        return buyer_order.price, "BUY"
    if seller_order is not None:
        return seller_order.price, "SELL"
    return None, None


def run_extended_mixed_agent_test(
    num_zi_buy: int = 5,
    num_zi_sell: int = 5,
    num_hbl_buy: int = 0,
    num_hbl_sell: int = 0,
    num_llm_buy: int = 1,
    num_llm_sell: int = 1,
    timesteps: int = 500,
    q_max: int = 14,
    pv_var: float = 1e5,
    shade_range: List[float] = None,
    arrival_rate: float = 0.1,
    use_time_varying: bool = True,
    fundamental_value: float = 100000.0,
    mean_reversion_rate: float = 0.05,
    shock_var: float = 1e5,
    obs_noise_var: float = 1e3,
    zi_eta: float = 1.0,
    llm_adjustment_rate: float = 0.01,
    seed: int = 7,
):
    """
    Run a market simulation with ZI, HBL, and LLMFirstZI agents.
    
    Args:
        num_llm_buy: Number of LLMFirstZI buy agents
        num_llm_sell: Number of LLMFirstZI sell agents
        llm_adjustment_rate: Rate at which LLM agents adjust prices (default 1%)
        [other args match run_mixed_agent_test]
    """
    if shade_range is None:
        shade_range = [0, 500]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if use_time_varying:
        fundamental = LazyGaussianMeanReverting(
            final_time=timesteps,
            mean=fundamental_value,
            r=mean_reversion_rate,
            shock_var=shock_var,
        )
    else:
        fundamental = DummyFundamental(value=fundamental_value, final_time=timesteps)

    market = Market(fundamental=fundamental, time_steps=timesteps)

    agents: Dict[str, object] = {}
    id_to_agent: Dict[int, object] = {}
    agent_types: Dict[int, str] = {}
    next_id = 0

    # Create ZI agents
    for _ in range(num_zi_buy):
        agent = ZIAgentBuy(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            pv_var=pv_var,
            eta=zi_eta,
            obs_noise_var=obs_noise_var,
        )
        agents[f"zi_buy_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "zi_buy"
        next_id += 1

    for _ in range(num_zi_sell):
        agent = ZIAgentSell(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            pv_var=pv_var,
            eta=zi_eta,
            obs_noise_var=obs_noise_var,
        )
        agents[f"zi_sell_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "zi_sell"
        next_id += 1

    # Create HBL agents
    for _ in range(num_hbl_buy):
        agent = HBLAgentBuy(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            L=6,
            pv_var=pv_var,
            arrival_rate=arrival_rate,
            obs_noise_var=obs_noise_var,
        )
        agents[f"hbl_buy_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "hbl_buy"
        next_id += 1

    for _ in range(num_hbl_sell):
        agent = HBLAgentSell(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            L=6,
            pv_var=pv_var,
            arrival_rate=arrival_rate,
            obs_noise_var=obs_noise_var,
        )
        agents[f"hbl_sell_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "hbl_sell"
        next_id += 1

    # Create LLMFirstZI agents
    for _ in range(num_llm_buy):
        agent = LLMFirstZIAgentBuy(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            pv_var=pv_var,
            adjustment_rate=llm_adjustment_rate,
            obs_noise_var=obs_noise_var,
        )
        agents[f"llm_buy_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "llm_buy"
        next_id += 1

    for _ in range(num_llm_sell):
        agent = LLMFirstZIAgentSell(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade_range,
            pv_var=pv_var,
            adjustment_rate=llm_adjustment_rate,
            obs_noise_var=obs_noise_var,
        )
        agents[f"llm_sell_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "llm_sell"
        next_id += 1

    transactions = []
    
    # Run simulation
    for t in range(timesteps):
        market.event_queue.set_time(t)
        _ = market.get_fundamental_value()

        active_agents: List[Tuple[str, object]] = []
        for agent_name, agent in agents.items():
            if arrival_rate is None or np.random.rand() < arrival_rate:
                active_agents.append((agent_name, agent))
        random.shuffle(active_agents)

        for _, agent in active_agents:
            market.withdraw_all(agent.get_id())
            orders = agent.take_action()
            market.add_orders(orders)

        matched_orders = market.step()
        paired_trades = _pair_matched_orders(matched_orders)

        for trade in paired_trades.values():
            buyer_order = trade.get("buyer_order")
            seller_order = trade.get("seller_order")

            trade_price, price_setter_side = _trade_price_from_orders(buyer_order, seller_order)
            if trade_price is None:
                continue

            buyer_id = trade.get("buyer_id")
            seller_id = trade.get("seller_id")
            if buyer_id is None or seller_id is None:
                continue

            buyer = id_to_agent.get(buyer_id)
            seller = id_to_agent.get(seller_id)
            if buyer is None or seller is None:
                continue

            buyer_position_at_trade = buyer.position
            seller_position_at_trade = seller.position
            buyer_value = buyer.pv.consume_marginal(buyer.position, BUY)
            seller_cost = seller.pv.consume_marginal(seller.position, SELL)

            if hasattr(buyer, "consumed_buy_positions"):
                buyer.consumed_buy_positions.add(buyer_position_at_trade)
            if hasattr(seller, "consumed_sell_positions"):
                seller.consumed_sell_positions.add(abs(seller_position_at_trade))
            
            buyer.update_position(1, -trade_price)
            seller.update_position(-1, trade_price)

            transactions.append(
                {
                    "time": t,
                    "buyer_id": buyer_id,
                    "seller_id": seller_id,
                    "buyer_position": buyer_position_at_trade,
                    "seller_position": seller_position_at_trade,
                    "buyer_value": buyer_value,
                    "seller_cost": seller_cost,
                    "price": trade_price,
                    "price_setter_side": price_setter_side,
                }
            )

    # Calculate surplus by agent type
    zi_agent_ids = [agent_id for agent_id, a_type in agent_types.items() if a_type.startswith("zi_")]
    hbl_agent_ids = [agent_id for agent_id, a_type in agent_types.items() if a_type.startswith("hbl_")]
    llm_agent_ids = [agent_id for agent_id, a_type in agent_types.items() if a_type.startswith("llm_")]

    # Calculate optimal surplus
    all_buyer_values = []
    all_seller_costs = []
    for name, agent in agents.items():
        if "buy" in name:
            all_buyer_values.extend([float(v) for v in agent.pv.buyer_values])
        elif "sell" in name:
            all_seller_costs.extend([float(v) for v in agent.pv.seller_costs])

    all_buyer_values.sort(reverse=True)
    all_seller_costs.sort()

    optimal_surplus = 0.0
    eq_price_relative = None
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        surplus = all_buyer_values[i] - all_seller_costs[i]
        if surplus > 0:
            optimal_surplus += surplus
        if eq_price_relative is None and all_seller_costs[i] >= all_buyer_values[i]:
            eq_price_relative = (all_buyer_values[i] + all_seller_costs[i]) / 2.0

    final_fundamental = fundamental.get_value(timesteps) if hasattr(fundamental, "get_value") else fundamental_value
    eq_price = None
    if eq_price_relative is not None:
        eq_price = eq_price_relative + final_fundamental

    # Categorize prices by agent type participation
    zi_prices = []
    hbl_prices = []
    llm_prices = []
    for tx in transactions:
        buyer_type = agent_types.get(tx["buyer_id"], "")
        seller_type = agent_types.get(tx["seller_id"], "")
        price = tx["price"]
        
        if "llm" in buyer_type or "llm" in seller_type:
            llm_prices.append(price)
        elif "hbl" in buyer_type or "hbl" in seller_type:
            hbl_prices.append(price)
        else:
            zi_prices.append(price)

    def _rmsd(prices, reference):
        if reference is None or not prices:
            return 0.0
        return float(np.sqrt(np.mean([(p - reference) ** 2 for p in prices])))

    zi_rmsd = _rmsd(zi_prices, fundamental_value)
    hbl_rmsd = _rmsd(hbl_prices, fundamental_value)
    llm_rmsd = _rmsd(llm_prices, fundamental_value)

    # Calculate surplus per agent
    final_fundamental = fundamental.get_value(timesteps) if hasattr(fundamental, "get_value") else fundamental_value
    agent_surplus = {}
    for agent_id, agent in id_to_agent.items():
        agent_type = agent_types.get(agent_id, "")
        is_buyer = agent_type.endswith("buy")
        holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=is_buyer)
        agent_surplus[agent_id] = agent.cash + holdings_value + final_fundamental * agent.position

    zi_surpluses = [agent_surplus[agent_id] for agent_id in zi_agent_ids]
    hbl_surpluses = [agent_surplus[agent_id] for agent_id in hbl_agent_ids]
    llm_surpluses = [agent_surplus[agent_id] for agent_id in llm_agent_ids]
    
    zi_avg = float(np.mean(zi_surpluses)) if zi_surpluses else 0.0
    hbl_avg = float(np.mean(hbl_surpluses)) if hbl_surpluses else 0.0
    llm_avg = float(np.mean(llm_surpluses)) if llm_surpluses else 0.0

    results = {
        "transactions": transactions,
        "agent_surplus": agent_surplus,
        "agent_types": agent_types,
        "agents": agents,
        "fundamental": fundamental,
        "fundamental_value": fundamental_value,
        "timesteps": timesteps,
        "zi_avg_surplus": zi_avg,
        "hbl_avg_surplus": hbl_avg,
        "llm_avg_surplus": llm_avg,
        "zi_total_surplus": float(np.sum(zi_surpluses)) if zi_surpluses else 0.0,
        "hbl_total_surplus": float(np.sum(hbl_surpluses)) if hbl_surpluses else 0.0,
        "llm_total_surplus": float(np.sum(llm_surpluses)) if llm_surpluses else 0.0,
        "optimal_surplus": optimal_surplus,
        "eq_price": eq_price,
        "zi_rmsd": zi_rmsd,
        "hbl_rmsd": hbl_rmsd,
        "llm_rmsd": llm_rmsd,
        "total_transactions": len(transactions),
        "zi_transactions": len(zi_prices),
        "hbl_transactions": len(hbl_prices),
        "llm_transactions": len(llm_prices),
    }

    return results


def print_extended_summary(results):
    """Print summary of extended mixed agent test."""
    print("\nEXTENDED MIXED AGENT TEST SUMMARY")
    print("=" * 70)
    print(f"Total transactions: {results['total_transactions']}")
    print(f"  ZI-only trades: {results['zi_transactions']}")
    print(f"  HBL trades: {results['hbl_transactions']}")
    print(f"  LLM trades: {results['llm_transactions']}")
    
    print(f"\nTotal Surplus:")
    print(f"  ZI: {results['zi_total_surplus']:.2f}")
    print(f"  HBL: {results['hbl_total_surplus']:.2f}")
    print(f"  LLM: {results['llm_total_surplus']:.2f}")
    
    print(f"\nAverage Surplus per Agent:")
    print(f"  ZI: {results['zi_avg_surplus']:.2f}")
    print(f"  HBL: {results['hbl_avg_surplus']:.2f}")
    print(f"  LLM: {results['llm_avg_surplus']:.2f}")
    
    print(f"\nPrice Efficiency (RMSD from fundamental):")
    print(f"  ZI: {results['zi_rmsd']:.2f}")
    print(f"  HBL: {results['hbl_rmsd']:.2f}")
    print(f"  LLM: {results['llm_rmsd']:.2f}")
    
    # Calculate combined efficiency
    combined_total = (results['zi_total_surplus'] + results['hbl_total_surplus'] + 
                      results['llm_total_surplus'])
    efficiency = (combined_total / results['optimal_surplus'] * 100.0) if results['optimal_surplus'] > 0 else 0.0
    
    print(f"\nMarket Efficiency:")
    print(f"  Combined surplus: {combined_total:.2f}")
    print(f"  Optimal surplus: {results['optimal_surplus']:.2f}")
    print(f"  Efficiency: {efficiency:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    results = run_extended_mixed_agent_test(
        num_zi_buy=5,
        num_zi_sell=5,
        num_hbl_buy=0,
        num_hbl_sell=0,
        num_llm_buy=1,
        num_llm_sell=1,
        timesteps=300,
        q_max=14,
        seed=42,
    )
    print_extended_summary(results)
