def agent_strategy(private_values, market_history, market_state):
    if not market_history['market_trades']:
        trade_price_stats = [market_state['fundamental_estimate']]
    else:
        trade_price_stats = [trade['trade_price_relative'] for trade in market_history['market_trades']]
    
    best_bid = market_state['best_bid']
    best_ask = market_state['best_ask']
    
    if best_bid is None or best_ask is None:
        best_bid = market_state['fundamental_estimate']
        best_ask = market_state['fundamental_estimate']
    
    if best_ask <= best_bid:
        return 0
    
    shade = 0
    for private_value in private_values:
        if private_value > best_bid:
            shade += private_value - best_bid
        elif private_value < best_ask:
            shade += best_ask - private_value
    
    x_cap = 1
    x = min(shade / (best_ask - best_bid), x_cap)
    return x