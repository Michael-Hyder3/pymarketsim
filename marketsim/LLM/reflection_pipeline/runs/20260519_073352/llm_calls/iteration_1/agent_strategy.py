def agent_strategy(private_values, market_history, market_state):
    if not market_history['market_trades']:
        trade_price_stats = [market_state['fundamental_estimate']]
    else:
        trade_price_stats = [trade['price'] for trade in market_history['market_trades']]
    best_bid = market_state['best_bid']
    best_ask = market_state['best_ask']
    if best_bid is None or best_ask is None:
        best_bid = market_state['fundamental_estimate']
        best_ask = market_state['fundamental_estimate']
    if best_ask <= best_bid:
        best_bid = best_ask
    shade = 0
    for i, value in enumerate(private_values):
        if i % 2 == 0:
            shade += value
        else:
            shade -= value
    shade /= len(private_values)
    x_cap = 1
    if best_bid > best_ask:
        x_cap = (best_bid - best_ask) / best_bid
    return min(shade, x_cap)