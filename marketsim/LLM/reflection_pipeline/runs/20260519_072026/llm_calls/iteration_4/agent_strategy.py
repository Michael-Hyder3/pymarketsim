def agent_strategy(private_values, market_history, market_state):
    # Calculate trade-price stats
    trades = market_history.get('market_trades', [])
    if trades:
        trade_prices = [trade['price'] for trade in trades]
        mean_price = sum(trade_prices) / len(trades) if trade_prices else market_state.get('fundamental_estimate', 0)
    else:
        mean_price = market_state.get('fundamental_estimate', 0)

    # Calculate best bid and ask
    best_bid = market_state.get('best_bid')
    best_ask = market_state.get('best_ask')
    if best_bid is None or best_ask is None:
        best_bid = mean_price
        best_ask = mean_price

    # Calculate spread
    spread = best_ask - best_bid
    if spread <= 0:
        spread = 0

    # Calculate shading level
    shading_level = (private_values - best_bid) / spread if spread > 0 else 0

    # Ensure shading level is within [0, 1]
    shading_level = min(max(shading_level, 0), 1)

    return shading_level