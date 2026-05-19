def agent_strategy(private_values, market_history, market_state):
    if not market_history['market_trades']:
        trade_price = market_state['fundamental_estimate'] or 0
    else:
        trade_price = sum([trade['price'] for trade in market_history['market_trades']]) / len(market_history['market_trades'])
    
    best_bid = market_state['best_bid']
    best_ask = market_state['best_ask']
    
    if trade_price > best_bid:
        shading_level = (trade_price - best_bid) / (best_ask - best_bid)
    elif trade_price < best_ask:
        shading_level = (best_ask - trade_price) / (best_ask - best_bid)
    else:
        shading_level = 0
    
    return min(shading_level, 1)