def agent_strategy(private_values, market_history, market_state):
    if market_state['best_ask'] is None or market_state['best_bid'] is None:
        return 0.0  # Avoid division by zero

    if market_state['best_ask'] <= market_state['best_bid']:
        return 0.0  # Spread is zero, no shading

    trade_price_stats = compute_trade_price_stats(market_history)
    best_bid = market_state['best_bid']
    best_ask = market_state['best_ask']
    units_available = market_state['units_available']

    if private_values is not None:
        private_values_relative = [pv - best_bid for pv in private_values]
        private_values_relative.sort()
        shade = compute_shade(private_values_relative, best_bid, best_ask)
    else:
        shade = 0.0

    return min(shade, 1.0)


def compute_trade_price_stats(market_history):
    if market_history['market_trades']:
        trade_prices = [trade['trade_price_relative'] for trade in market_history['market_trades']]
        mean_trade_price = sum(trade_prices) / len(trade_prices)
        return mean_trade_price
    else:
        return market_state['fundamental_estimate']