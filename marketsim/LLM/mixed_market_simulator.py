import json
import inspect
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from marketsim.fourheap.constants import BUY, SELL
from marketsim.market.market import Market
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.fundamental.dummy_fundamental import DummyFundamental
from marketsim.agent.zi_agent_buy_sell import ZIAgentBuy, ZIAgentSell
from marketsim.agent.hbl_agent_buy_sell import HBLAgentBuy, HBLAgentSell
from marketsim.LLM.code_storage import CodeStorage


def _pair_matched_orders(matched_orders):
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


def _load_llm_shading_region(
    data_path: str = "results/zi_only_transactions.json",
    fallback: List[float] = None,
    function_name: str = "suggest_shading_region",
    storage_dir: str = "llm_calls",
) -> List[float]:
    if fallback is None:
        fallback = [0, 500]

    data_file = Path(data_path)
    if not data_file.exists():
        return fallback

    try:
        with data_file.open("r") as handle:
            data = json.load(handle)
    except Exception:
        return fallback

    participant = data.get("participant_private_values", {})
    private_values = participant.get("buyer_values") or participant.get("seller_costs") or []
    transactions = data.get("transactions", [])
    if not private_values or not transactions:
        return fallback

    storage = CodeStorage(storage_dir)
    latest = storage.get_latest_function(function_name, successful_only=True)
    if not latest:
        return fallback

    try:
        func = storage.import_function(latest["code_file"], function_name)
        try:
            params = inspect.signature(func).parameters
            if len(params) == 1:
                region = func(private_values)
            else:
                region = func(private_values, transactions)
        except Exception:
            region = func(private_values, transactions)
    except Exception:
        return fallback

    if not isinstance(region, (list, tuple)) or len(region) != 2:
        return fallback
    try:
        low = float(region[0])
        high = float(region[1])
    except Exception:
        return fallback

    low = max(0.0, min(low, high))
    high = max(0.0, max(low, high))
    return [low, high]


def _compute_llm_shading_from_agent(
    agent,
    function_name: str = "suggest_shading_region",
    storage_dir: str = "llm_calls",
    fallback: List[float] = None,
) -> List[float]:
    """
    Compute shading region for an agent using LLM function with agent's actual private values.
    This is called during simulation setup, using fresh values from each agent.
    
    Args:
        agent: ZIAgent with pv (private values) attribute
        function_name: Name of the LLM function
        storage_dir: Directory containing generated functions
        fallback: Default shading if LLM computation fails
    
    Returns:
        [min, max] shading region
    """
    if fallback is None:
        fallback = [0, 500]
    
    # Extract private values from agent
    try:
        if hasattr(agent.pv, 'buyer_values') and len(agent.pv.buyer_values) > 0:
            private_values = list(agent.pv.buyer_values)
        elif hasattr(agent.pv, 'seller_costs') and len(agent.pv.seller_costs) > 0:
            private_values = list(agent.pv.seller_costs)
        else:
            return fallback
    except Exception:
        return fallback
    
    # Load and execute LLM function
    storage = CodeStorage(storage_dir)
    latest = storage.get_latest_function(function_name, successful_only=True)
    if not latest:
        return fallback
    
    try:
        func = storage.import_function(latest["code_file"], function_name)
        try:
            params = inspect.signature(func).parameters
            if len(params) == 1:
                region = func(private_values)
            else:
                # If function expects 2 args, pass empty transactions list
                region = func(private_values, [])
        except Exception:
            region = func(private_values, [])
    except Exception:
        return fallback
    
    if not isinstance(region, (list, tuple)) or len(region) != 2:
        return fallback
    
    try:
        low = float(region[0])
        high = float(region[1])
    except Exception:
        return fallback
    
    low = max(0.0, min(low, high))
    high = max(0.0, max(low, high))
    return [low, high]


def _compute_llm_schedule_from_agent(
    agent,
    total_timesteps: int,
    function_name: str = "create_timestep_schedule",
    storage_dir: str = "llm_calls",
    fallback: List[Tuple[int, List[float]]] = None,
) -> List[Tuple[int, List[float]]]:
    if fallback is None:
        fallback = [(0, [0, 500])]

    try:
        if hasattr(agent.pv, "buyer_values") and len(agent.pv.buyer_values) > 0:
            private_values = list(agent.pv.buyer_values)
        elif hasattr(agent.pv, "seller_costs") and len(agent.pv.seller_costs) > 0:
            private_values = list(agent.pv.seller_costs)
        else:
            return fallback
    except Exception:
        return fallback

    storage = CodeStorage(storage_dir)
    latest = storage.get_latest_function(function_name, successful_only=True)
    if not latest:
        return fallback

    try:
        func = storage.import_function(latest["code_file"], function_name)
        schedule_result = func(private_values, total_timesteps)
    except Exception:
        return fallback

    if not isinstance(schedule_result, list) or len(schedule_result) == 0:
        return fallback

    normalized: List[Tuple[int, List[float]]] = []
    if isinstance(schedule_result[0], dict):
        for entry in schedule_result:
            if not isinstance(entry, dict):
                continue
            timestep = entry.get("timestep")
            region = entry.get("shading_region")
            if timestep is None or region is None:
                continue
            normalized.append((int(timestep), region))
    elif isinstance(schedule_result[0], (list, tuple)):
        for entry in schedule_result:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            normalized.append((int(entry[0]), entry[1]))
    else:
        return fallback

    if not normalized:
        return fallback

    normalized.sort(key=lambda x: x[0])

    compressed: List[Tuple[int, List[float]]] = []
    last_region = None
    for timestep, region in normalized:
        if not isinstance(region, (list, tuple)) or len(region) != 2:
            continue
        try:
            low = float(region[0])
            high = float(region[1])
        except Exception:
            continue
        low = max(0.0, min(low, high))
        high = max(0.0, max(low, high))
        region_list = [low, high]
        if last_region is None or region_list != last_region:
            compressed.append((timestep, region_list))
            last_region = region_list

    if not compressed:
        return fallback

    return compressed


def run_mixed_agent_test(
    num_zi_buy: int = 6,
    num_zi_sell: int = 6,
    num_hbl_buy: int = 0,
    num_hbl_sell: int = 0,
    timesteps: int = 1000,
    q_max: int = 14,
    pv_var: float = 1e5,
    shade_range: List[float] = None,
    zi_buy_shade_ranges: List[List[float]] = None,
    zi_buy_shade_counts: List[int] = None,
    zi_sell_shade_ranges: List[List[float]] = None,
    zi_sell_shade_counts: List[int] = None,
    arrival_rate: float = 0.1,
    use_time_varying: bool = True,
    fundamental_value: float = 100000.0,
    mean_reversion_rate: float = 0.05,
    shock_var: float = 1e5,
    obs_noise_var: float = 1e6,
    zi_eta: float = 1.0,
    shade_schedule_mode: str = "time",
    seed: int = None,
):
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
    agent_shades: Dict[int, List[float]] = {}
    next_id = 0

    def _build_shades(total_count: int, ranges: List[List[float]], counts: List[int]) -> List[List[float]]:
        if ranges is None:
            return [shade_range] * total_count
        if counts is None:
            base = total_count // len(ranges)
            remainder = total_count % len(ranges)
            counts = [base + (1 if i < remainder else 0) for i in range(len(ranges))]
        if len(counts) != len(ranges):
            raise ValueError("Shade counts must match number of shade ranges.")
        expanded = []
        for count, shade in zip(counts, ranges):
            expanded.extend([shade] * count)
        return expanded

    zi_buy_shades = _build_shades(num_zi_buy, zi_buy_shade_ranges, zi_buy_shade_counts)
    zi_sell_shades = _build_shades(num_zi_sell, zi_sell_shade_ranges, zi_sell_shade_counts)

    for shade in zi_buy_shades:
        # Handle "LLM" marker - will be replaced with dynamic shading later
        if shade == "LLM":
            shade = ["LLM_PLACEHOLDER"]  # Temporary marker
        if shade == "LLM_SCHEDULE":
            shade = ["LLM_SCHEDULE_PLACEHOLDER"]
        agent = ZIAgentBuy(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade,
            pv_var=pv_var,
            eta=zi_eta,
            obs_noise_var=obs_noise_var,
            shade_schedule_mode=shade_schedule_mode,
        )
        agents[f"zi_buy_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "zi_buy"
        agent_shades[next_id] = shade
        next_id += 1

    for shade in zi_sell_shades:
        # Handle "LLM" marker - will be replaced with dynamic shading later
        if shade == "LLM":
            shade = ["LLM_PLACEHOLDER"]  # Temporary marker
        if shade == "LLM_SCHEDULE":
            shade = ["LLM_SCHEDULE_PLACEHOLDER"]
        agent = ZIAgentSell(
            agent_id=next_id,
            market=market,
            q_max=q_max,
            shade=shade,
            pv_var=pv_var,
            eta=zi_eta,
            obs_noise_var=obs_noise_var,
            shade_schedule_mode=shade_schedule_mode,
        )
        agents[f"zi_sell_{next_id}"] = agent
        id_to_agent[next_id] = agent
        agent_types[next_id] = "zi_sell"
        agent_shades[next_id] = shade
        next_id += 1

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
    
    # Replace "LLM_PLACEHOLDER" markers with dynamic LLM-computed shading from agent's private values
    for agent_id, agent in id_to_agent.items():
        shade = agent_shades.get(agent_id)
        if isinstance(shade, list) and len(shade) == 1 and shade[0] == "LLM_PLACEHOLDER":
            # Compute LLM shading from this agent's private values
            llm_shade = _compute_llm_shading_from_agent(agent)
            agent.shade = llm_shade
            agent_shades[agent_id] = llm_shade
        if isinstance(shade, list) and len(shade) == 1 and shade[0] == "LLM_SCHEDULE_PLACEHOLDER":
            llm_schedule = _compute_llm_schedule_from_agent(agent, total_timesteps=timesteps)
            agent.shade = llm_schedule
            agent_shades[agent_id] = llm_schedule
    
    transactions = []
    agent_surplus = {agent_id: 0.0 for agent_id in id_to_agent}
    
    # Track when each agent last submitted orders for automatic expiration
    agent_last_order_time = {agent_id: -1 for agent_id in id_to_agent}
    order_lifetime = int(1.0 / arrival_rate) if arrival_rate is not None and arrival_rate > 0 else None

    for t in range(timesteps):
        market.event_queue.set_time(t)
        _ = market.get_fundamental_value()
        
        # Withdraw expired orders (orders older than 1/arrival_rate timesteps)
        if order_lifetime is not None:
            for agent_id in id_to_agent:
                last_order_time = agent_last_order_time[agent_id]
                if last_order_time >= 0 and (t - last_order_time) >= order_lifetime:
                    market.withdraw_all(agent_id)
                    agent_last_order_time[agent_id] = -1

        active_agents: List[Tuple[str, object]] = []
        for agent_name, agent in agents.items():
            if arrival_rate is None or np.random.rand() < arrival_rate:
                active_agents.append((agent_name, agent))
        random.shuffle(active_agents)

        for _, agent in active_agents:
            agent_id = agent.get_id()
            # Withdraw previous orders when agent enters market
            market.withdraw_all(agent_id)
            orders = agent.take_action()
            market.add_orders(orders)
            # Track when this agent submitted orders
            agent_last_order_time[agent_id] = t

        matched_orders = market.step()
        paired_trades = _pair_matched_orders(matched_orders)

        for trade in paired_trades.values():
            buyer_order = trade.get("buyer_order")
            seller_order = trade.get("seller_order")

            trade_price, price_setter_side = _trade_price_from_orders(buyer_order, seller_order)
            if trade_price is None:
                continue

            # Shift price by fundamental_value
            trade_price_shifted = trade_price + fundamental_value

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
            # Shift private values by fundamental_value
            buyer_value = buyer.pv.consume_marginal(buyer.position, BUY) + fundamental_value
            seller_cost = seller.pv.consume_marginal(seller.position, SELL) + fundamental_value

            # Transaction-level surplus calculation
            if agent_types.get(buyer_id, '').startswith('zi_'):
                agent_surplus[buyer_id] += buyer_value - trade_price_shifted
            if agent_types.get(seller_id, '').startswith('zi_'):
                agent_surplus[seller_id] += trade_price_shifted - seller_cost

            if hasattr(buyer, "consumed_buy_positions"):
                buyer.consumed_buy_positions.add(buyer_position_at_trade)
            if hasattr(seller, "consumed_sell_positions"):
                seller.consumed_sell_positions.add(abs(seller_position_at_trade))
            buyer.update_position(1, -trade_price_shifted)
            seller.update_position(-1, trade_price_shifted)

            transactions.append(
                {
                    "time": t,
                    "buyer_id": buyer_id,
                    "seller_id": seller_id,
                    "buyer_position": buyer_position_at_trade,
                    "seller_position": seller_position_at_trade,
                    "buyer_value": buyer_value,
                    "seller_cost": seller_cost,
                    "price": trade_price_shifted,
                    "price_setter_side": price_setter_side,
                }
            )

    zi_agent_ids = [agent_id for agent_id, a_type in agent_types.items() if a_type.startswith("zi_")]
    hbl_agent_ids = [agent_id for agent_id, a_type in agent_types.items() if a_type.startswith("hbl_")]
    hbl_buy_agent_ids = [agent_id for agent_id, a_type in agent_types.items() if a_type == "hbl_buy"]
    hbl_sell_agent_ids = [agent_id for agent_id, a_type in agent_types.items() if a_type == "hbl_sell"]

    # Optimal surplus from sorted buyer values (desc) and seller costs (asc)
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

    zi_prices = []
    hbl_prices = []
    for tx in transactions:
        buyer_type = agent_types.get(tx["buyer_id"], "")
        seller_type = agent_types.get(tx["seller_id"], "")
        if "hbl" in buyer_type or "hbl" in seller_type:
            hbl_prices.append(tx["price"])
        else:
            zi_prices.append(tx["price"])

    def _rmsd(prices, reference):
        if reference is None or not prices:
            return 0.0
        return float(np.sqrt(np.mean([(p - reference) ** 2 for p in prices])))

    zi_rmsd = _rmsd(zi_prices, fundamental_value)
    hbl_rmsd = _rmsd(hbl_prices, fundamental_value)

    # For HBL agents, compute mark-to-market surplus at end
    final_fundamental = fundamental.get_value(timesteps) if hasattr(fundamental, "get_value") else fundamental_value
    for agent_id, agent in id_to_agent.items():
        agent_type = agent_types.get(agent_id, "")
        if agent_type.startswith('hbl_'):
            is_buyer = agent_type.endswith("buy")
            holdings_value = agent.pv.value_of_holdings(agent.position, is_buyer=is_buyer)
            agent_surplus[agent_id] = agent.cash + holdings_value + final_fundamental * agent.position

    zi_surpluses = [agent_surplus[agent_id] for agent_id in zi_agent_ids]
    hbl_surpluses = [agent_surplus[agent_id] for agent_id in hbl_agent_ids]
    hbl_buy_surpluses = [agent_surplus[agent_id] for agent_id in hbl_buy_agent_ids]
    hbl_sell_surpluses = [agent_surplus[agent_id] for agent_id in hbl_sell_agent_ids]
    
    zi_avg = float(np.mean(zi_surpluses)) if zi_surpluses else 0.0
    hbl_avg = float(np.mean(hbl_surpluses)) if hbl_surpluses else 0.0
    hbl_buy_avg = float(np.mean(hbl_buy_surpluses)) if hbl_buy_surpluses else 0.0
    hbl_sell_avg = float(np.mean(hbl_sell_surpluses)) if hbl_sell_surpluses else 0.0

    trade_counts = {agent_id: 0 for agent_id in id_to_agent}
    buy_trade_counts = {agent_id: 0 for agent_id in id_to_agent}
    sell_trade_counts = {agent_id: 0 for agent_id in id_to_agent}
    for tx in transactions:
        buyer_id = tx.get("buyer_id")
        seller_id = tx.get("seller_id")
        if buyer_id in trade_counts:
            trade_counts[buyer_id] += 1
            buy_trade_counts[buyer_id] += 1
        if seller_id in trade_counts:
            trade_counts[seller_id] += 1
            sell_trade_counts[seller_id] += 1

    zi_agent_stats = []
    for agent_id in zi_agent_ids:
        zi_agent_stats.append(
            {
                "agent_id": agent_id,
                "agent_type": agent_types.get(agent_id),
                "shade_range": agent_shades.get(agent_id),
                "surplus": agent_surplus.get(agent_id, 0.0),
                "trades": trade_counts.get(agent_id, 0),
                "buy_trades": buy_trade_counts.get(agent_id, 0),
                "sell_trades": sell_trade_counts.get(agent_id, 0),
            }
        )

    results = {
        "transactions": transactions,
        "agent_surplus": agent_surplus,
        "agent_types": agent_types,
        "agent_shades": agent_shades,
        "zi_agent_stats": zi_agent_stats,
        "agents": agents,
        "fundamental": fundamental,
        "fundamental_value": fundamental_value,
        "timesteps": timesteps,
        "zi_avg_surplus": zi_avg,
        "hbl_avg_surplus": hbl_avg,
        "hbl_buy_avg_surplus": hbl_buy_avg,
        "hbl_sell_avg_surplus": hbl_sell_avg,
        "zi_total_surplus": float(np.sum(zi_surpluses)) if zi_surpluses else 0.0,
        "hbl_total_surplus": float(np.sum(hbl_surpluses)) if hbl_surpluses else 0.0,
        "optimal_surplus": optimal_surplus,
        "eq_price": eq_price,
        "zi_rmsd": zi_rmsd,
        "hbl_rmsd": hbl_rmsd,
        "total_transactions": len(transactions),
    }

    return results


def _print_summary(results):
    print("\nMIXED AGENT TEST SUMMARY")
    print("=" * 60)
    print(f"Total transactions: {results['total_transactions']}")
    print(f"ZI total surplus: {results['zi_total_surplus']:.2f}")
    print(f"HBL total surplus: {results['hbl_total_surplus']:.2f}")
    print(f"ZI avg surplus per agent: {results['zi_avg_surplus']:.2f}")
    print(f"HBL avg surplus per agent: {results['hbl_avg_surplus']:.2f}")
    print(f"  HBL Buy avg surplus: {results['hbl_buy_avg_surplus']:.2f}")
    print(f"  HBL Sell avg surplus: {results['hbl_sell_avg_surplus']:.2f}")
    
    # Calculate efficiency metrics
    combined_total = results['zi_total_surplus'] + results['hbl_total_surplus']
    efficiency = (combined_total / results['optimal_surplus'] * 100.0) if results['optimal_surplus'] > 0 else 0.0
    print(f"\nEfficiency Metric:")
    print(f"Combined total surplus (ZI + HBL): {combined_total:.2f}")
    print(f"Optimal total surplus: {results['optimal_surplus']:.2f}")
    print(f"Efficiency: {efficiency:.2f}%")

    zi_agent_stats = results.get("zi_agent_stats", [])
    if zi_agent_stats:
        print("\nZI Agent Stats:")
        for stat in sorted(zi_agent_stats, key=lambda s: s["agent_id"]):
            shade = stat.get("shade_range")
            shade_str = f"{shade}" if shade is not None else "None"
            print(
                "  "
                f"id={stat['agent_id']} type={stat['agent_type']} shade={shade_str} "
                f"surplus={stat['surplus']:.2f} trades={stat['trades']} "
                f"buy={stat['buy_trades']} sell={stat['sell_trades']}"
            )

    agents = results.get("agents", {})
    if agents:
        print("\nAgent Private Values:")
        for name in sorted(agents.keys()):
            agent = agents[name]
            if "buy" in name:
                vals = agent.pv.buyer_values
                print(f"  {name} buyer_values: {[float(v) for v in vals]}")
            elif "sell" in name:
                costs = agent.pv.seller_costs
                print(f"  {name} seller_costs: {[float(v) for v in costs]}")


def _compute_equilibrium_price(results):
    """Compute equilibrium price from sorted buyer values and seller costs."""
    agents = results.get("agents", {})
    if not agents:
        return None

    all_buyer_values = []
    all_seller_costs = []
    for name, agent in agents.items():
        if "buy" in name:
            all_buyer_values.extend([float(v) for v in agent.pv.buyer_values])
        elif "sell" in name:
            all_seller_costs.extend([float(v) for v in agent.pv.seller_costs])

    if not all_buyer_values or not all_seller_costs:
        return None

    all_buyer_values.sort(reverse=True)
    all_seller_costs.sort()

    eq_price = None
    for i in range(min(len(all_buyer_values), len(all_seller_costs))):
        if all_seller_costs[i] >= all_buyer_values[i]:
            eq_price = (all_buyer_values[i] + all_seller_costs[i]) / 2.0
            break

    if eq_price is None:
        return None

    fundamental = results.get("fundamental")
    timesteps = results.get("timesteps")
    base_fundamental = results.get("fundamental_value")
    if fundamental is not None and timesteps is not None and hasattr(fundamental, "get_value"):
        return eq_price + fundamental.get_value(timesteps)
    if base_fundamental is not None:
        return eq_price + base_fundamental
    return eq_price


def _plot_transactions(results):
    """Plot transactions over time, color-coded by agent type (ZI vs HBL)."""
    transactions = results["transactions"]
    agent_types = results["agent_types"]
    
    zi_times = []
    zi_prices = []
    hbl_times = []
    hbl_prices = []
    
    for tx in transactions:
        time = tx["time"]
        price = tx["price"]
        buyer_id = tx["buyer_id"]
        seller_id = tx["seller_id"]
        
        # Check if either buyer or seller is HBL
        buyer_type = agent_types.get(buyer_id, "")
        seller_type = agent_types.get(seller_id, "")
        
        if "hbl" in buyer_type or "hbl" in seller_type:
            hbl_times.append(time)
            hbl_prices.append(price)
        else:
            zi_times.append(time)
            zi_prices.append(price)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(zi_times, zi_prices, c='blue', alpha=0.6, label='ZI-only trades', s=50)
    plt.scatter(hbl_times, hbl_prices, c='red', alpha=0.8, label='HBL trades', s=80, marker='x')
    eq_price = _compute_equilibrium_price(results)
    if eq_price is not None:
        plt.axhline(eq_price, color='black', linestyle='--', linewidth=1.5, label=f'Eq Price: {eq_price:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Transaction Price')
    plt.title('Transactions: ZI vs HBL Participation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_mixed_agent_batch(num_iterations: int = 25, **kwargs):
    """Run mixed-agent test multiple times and print a summary at the end."""
    summaries = {
        "total_transactions": [],
        "zi_total_surplus": [],
        "hbl_total_surplus": [],
        "hbl_buy_avg_surplus": [],
        "hbl_sell_avg_surplus": [],
        "zi_avg_surplus": [],
        "hbl_avg_surplus": [],
        "optimal_surplus": [],
        "zi_rmsd": [],
        "hbl_rmsd": [],
        "eq_price": [],
    }

    zi_agent_aggregates = {}

    base_seed = kwargs.get("seed")
    for i in range(num_iterations):
        if base_seed is not None:
            kwargs["seed"] = base_seed + i
        results = run_mixed_agent_test(**kwargs)
        summaries["total_transactions"].append(results["total_transactions"])
        summaries["zi_total_surplus"].append(results["zi_total_surplus"])
        summaries["hbl_total_surplus"].append(results["hbl_total_surplus"])
        summaries["hbl_buy_avg_surplus"].append(results["hbl_buy_avg_surplus"])
        summaries["hbl_sell_avg_surplus"].append(results["hbl_sell_avg_surplus"])
        summaries["zi_avg_surplus"].append(results["zi_avg_surplus"])
        summaries["hbl_avg_surplus"].append(results["hbl_avg_surplus"])
        summaries["optimal_surplus"].append(results["optimal_surplus"])
        summaries["zi_rmsd"].append(results["zi_rmsd"])
        summaries["hbl_rmsd"].append(results["hbl_rmsd"])
        eq_price = results.get("eq_price")
        if eq_price is not None:
            summaries["eq_price"].append(eq_price)

        for stat in results.get("zi_agent_stats", []):
            agent_id = stat["agent_id"]
            aggregate = zi_agent_aggregates.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "agent_type": stat.get("agent_type"),
                    "shade_range": stat.get("shade_range"),
                    "surplus_sum": 0.0,
                    "trades_sum": 0,
                    "buy_trades_sum": 0,
                    "sell_trades_sum": 0,
                    "count": 0,
                },
            )
            aggregate["surplus_sum"] += stat.get("surplus", 0.0)
            aggregate["trades_sum"] += stat.get("trades", 0)
            aggregate["buy_trades_sum"] += stat.get("buy_trades", 0)
            aggregate["sell_trades_sum"] += stat.get("sell_trades", 0)
            aggregate["count"] += 1

    print("\nMIXED AGENT BATCH SUMMARY")
    print("=" * 60)
    print(f"Runs: {num_iterations}")
    print(f"Avg transactions: {np.mean(summaries['total_transactions']):.2f}")
    print(f"Avg ZI total surplus: {np.mean(summaries['zi_total_surplus']):.2f}")
    print(f"Avg HBL total surplus: {np.mean(summaries['hbl_total_surplus']):.2f}")
    print(f"Avg ZI avg surplus: {np.mean(summaries['zi_avg_surplus']):.2f}")
    print(f"Avg HBL avg surplus: {np.mean(summaries['hbl_avg_surplus']):.2f}")
    print(f"Avg max total surplus (optimal): {np.mean(summaries['optimal_surplus']):.2f}")
    print(f"Avg ZI RMSD: {np.mean(summaries['zi_rmsd']):.2f}")
    print(f"Avg HBL RMSD: {np.mean(summaries['hbl_rmsd']):.2f}")
    if summaries["eq_price"]:
        print(f"Avg equilibrium price: {np.mean(summaries['eq_price']):.2f}")
    
    # Calculate average efficiency metric
    avg_combined_total = np.mean(summaries['zi_total_surplus']) + np.mean(summaries['hbl_total_surplus'])
    avg_optimal = np.mean(summaries['optimal_surplus'])
    avg_efficiency = (avg_combined_total / avg_optimal * 100.0) if avg_optimal > 0 else 0.0
    print(f"\nAverage Efficiency Metric:")
    print(f"Avg combined total surplus (ZI + HBL): {avg_combined_total:.2f}")
    print(f"Avg optimal total surplus: {avg_optimal:.2f}")
    print(f"Average Efficiency: {avg_efficiency:.2f}%")

    if zi_agent_aggregates:
        print("\nAverage ZI Agent Stats:")
        for agent_id in sorted(zi_agent_aggregates.keys()):
            aggregate = zi_agent_aggregates[agent_id]
            count = aggregate["count"] or 1
            shade = aggregate.get("shade_range")
            shade_str = f"{shade}" if shade is not None else "None"
            print(
                "  "
                f"id={agent_id} type={aggregate.get('agent_type')} shade={shade_str} "
                f"surplus_avg={aggregate['surplus_sum'] / count:.2f} "
                f"trades_avg={aggregate['trades_sum'] / count:.2f}"
            )

    summaries["zi_agent_avg_stats"] = [
        {
            "agent_id": aggregate["agent_id"],
            "agent_type": aggregate.get("agent_type"),
            "shade_range": aggregate.get("shade_range"),
            "surplus_avg": aggregate["surplus_sum"] / (aggregate["count"] or 1),
            "trades_avg": aggregate["trades_sum"] / (aggregate["count"] or 1),
            "buy_trades_avg": aggregate["buy_trades_sum"] / (aggregate["count"] or 1),
            "sell_trades_avg": aggregate["sell_trades_sum"] / (aggregate["count"] or 1),
        }
        for aggregate in sorted(zi_agent_aggregates.values(), key=lambda a: a["agent_id"])
    ]

    return summaries


if __name__ == "__main__":
    BATCH_RUNS = 1  # Run 10 iterations to see variation in dynamic LLM shading
    USE_MULTI_SHADE_EXAMPLE = True
    USE_DYNAMIC_LLM_SHADING = False  # Compute LLM shading from agent's private values
    USE_SHADING_SCHEDULE = False  # Change to True to test schedule
    USE_LLM_TIMESTEP_SCHEDULE = False  # Use LLM-generated timestep schedule for one agent
    example_kwargs = {}
    if USE_MULTI_SHADE_EXAMPLE:
        if USE_DYNAMIC_LLM_SHADING:
            # Use special marker "LLM" to compute shading dynamically from each agent's private values
            example_kwargs = dict(
                num_zi_buy=6,
                num_zi_sell=6,
                zi_buy_shade_ranges=[[0, 500], "LLM"],  # "LLM" will be replaced with dynamic shading
                zi_buy_shade_counts=[5, 1],
                zi_sell_shade_ranges=[[0, 500], [150, 400]],
                zi_sell_shade_counts=[6, 0],
            )
            print("Using dynamic LLM shading (computed from each agent's private values)")
            print("  Buyers 0-4: static [0, 500]")
            print("  Buyer 5: LLM-generated shading (fresh for each run)")
        else:
            # Use pre-computed LLM shading from historical data
            llm_shade = _load_llm_shading_region()
            print(f"Using LLM shading region from historical data: {llm_shade}")
            example_kwargs = dict(
                num_zi_buy=6,
                num_zi_sell=6,
                zi_buy_shade_ranges=[[0, 500], llm_shade],
                zi_buy_shade_counts=[5, 1],
                zi_sell_shade_ranges=[[0, 500], [150, 400]],
                zi_sell_shade_counts=[6, 0],
            )
        
        if USE_SHADING_SCHEDULE:
            # Test with time-based shading schedule
            # Buyers start aggressive, become conservative over time
            schedule = [
                (0, [0, 500]),      # t=0-2000: aggressive [0, 500]
                (2000, [0, 250]),   # t=2000-5000: moderate [0, 250]
                (5000, [0, 100]),   # t=5000+: conservative [0, 100]
            ]
            if USE_LLM_TIMESTEP_SCHEDULE:
                example_kwargs["zi_buy_shade_ranges"] = [schedule, "LLM_SCHEDULE"]
                example_kwargs["zi_buy_shade_counts"] = [5, 1]
                print("Using LLM-generated timestep schedule for Buyer 5")
            else:
                example_kwargs["zi_buy_shade_ranges"] = [schedule, "LLM" if USE_DYNAMIC_LLM_SHADING else _load_llm_shading_region()]
            example_kwargs["shade_schedule_mode"] = "time"
            example_kwargs["timesteps"] = 10000
            print("Using time-based shading schedule:")
            print("  Buyers 0-4: [0,500] → [0,250] → [0,100] over time")
            if USE_LLM_TIMESTEP_SCHEDULE:
                print("  Buyer 5: LLM-generated timestep schedule")
            else:
                print("  Buyer 5: LLM-generated shading")

    if BATCH_RUNS > 1:
        run_mixed_agent_batch(num_iterations=BATCH_RUNS, **example_kwargs)
    else:
        results = run_mixed_agent_test(**example_kwargs) if example_kwargs else run_mixed_agent_test()
        print(f"\nTotal transactions: {results['total_transactions']}")
        print(f"ZI avg surplus: {results['zi_avg_surplus']:.2f}")
        _print_summary(results)
        _plot_transactions(results)
