def agent_strategy(private_values, market_history, market_state):
    # Extract relevant market state values
    best_bid = market_state.get('best_bid')
    best_ask = market_state.get('best_ask')
    units_available = market_state.get('units_available')
    q_max = market_state.get('q_max')
    fundamental_estimate = market_state.get('fundamental_estimate')

    # Compute trade-price stats from market history
    trades = market_history.get('market_trades', [])
    trade_prices = [trade['price'] for trade in trades]
    if trade_prices:
        mean_price = sum(trade_prices) / len(trade_prices)
    else:
        mean_price = fundamental_estimate or 0

    # Compute shading level
    if best_bid is None or best_ask is None:
        shading_level = 0
    elif best_ask <= best_bid:
        shading_level = 0
    else:
        shading_level = min(1, (best_ask - mean_price) / (best_ask - best_bid))

    # Apply private value shading
    if q_max > 0:
        shading_level *= min(1, private_values / q_max)
    else:
        shading_level *= 0

    return shading_level