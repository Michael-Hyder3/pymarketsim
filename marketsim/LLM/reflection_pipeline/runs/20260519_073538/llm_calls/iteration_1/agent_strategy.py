def agent_strategy(private_values, market_history, market_state):
    if not market_history['market_trades']:
        trade_price_stats = [market_state['fundamental_estimate']]
    else:
        trade_price_stats = [trade['trade_price_relative'] for trade in market_history['market_trades']]
    
    best_bid = market_state['best_bid']
    best_ask = market_state['best_ask']
    
    if best_bid is None or best_ask is None:
        return 0.5  # Avoid division by zero
    
    if best_ask <= best_bid:
        return 0.5  # Avoid division
    
    shade = 0
    for value in private_values:
        shade += (value - best_bid) / (best_ask - best_bid)
    
    return min(max(shade, 0), 1)