def agent_strategy(market_history):
    best_bid = market_history.get("best_bid")
    best_ask = market_history.get("best_ask")
    current_pv = float(market_history.get("current_pv") or 0.0)
    fundamental_estimate = float(market_history.get("fundamental_estimate") or 0.0)
    bids = list(market_history.get("bids") or [])
    asks = list(market_history.get("asks") or [])
    orders = list(market_history.get("orders") or [])

    if not bids and not asks:
        return 0.2  # changed default shade to 0.2

    imbalance = sum([bid["price_rel"] for bid in bids]) - sum([ask["price_rel"] for ask in asks])

    recent_buys = sum([1 for order in orders if order["side"] == 1 and order["accepted"]])
    recent_sells = sum([1 for order in orders if order["side"] == -1 and order["accepted"]])
    recent_pressure = (recent_buys - recent_sells) / len(orders)

    shade = (current_pv / fundamental_estimate) if fundamental_estimate != 0 else 0.5

    if imbalance > 0:
        shade *= (1 + recent_pressure / 3)  # changed recent_pressure to recent_pressure / 3
    elif imbalance < 0:
        shade *= (1 - recent_pressure / 3)  # changed recent_pressure to recent_pressure / 3

    if imbalance > 0:
        shade *= 0.6
    elif imbalance < 0:
        shade *= 0.9

    shade *= (1 + recent_pressure / 3)  # changed recent_pressure to recent_pressure / 3

    shade = min(max(shade, 0.0), 1.0)

    return shade
