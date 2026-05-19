def agent_strategy(private_values, market_history, market_state):
    if not market_history['market_trades']:
        trade_price_stats = {'mean': market_state['fundamental_estimate'], 'std': 0}
    else:
        trade_prices = [trade['price'] for trade in market_history['market_trades']]
        trade_price_stats = {'mean': sum(trade_prices) / len(trade_prices), 'std': 0}

    best_bid = market_state['best_bid']
    best_ask = market_state['best_ask']
    units_available = market_state['units_available']
    q_max = market_state['q_max']

    shading_level = 0
    if private_values > best_ask:
        shading_level = min(1, (private_values - best_ask) / (q_max * best_ask))
    elif private_values < best_bid:
        shading_level = min(1, (best_bid - private_values) / (q_max * best_bid))

    # Check if shading_level is within the range [0, 1]
    if shading_level < 0:
        shading_level = 0
    elif shading_level > 1:
        shading_level = 1

    return float(shading_level)