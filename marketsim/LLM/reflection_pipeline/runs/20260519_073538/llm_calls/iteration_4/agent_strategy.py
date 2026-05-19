def agent_strategy(private_values, market_history, market_state):
    if not market_history['market_trades']:
        return 0.5  # fallback to a neutral shade

    best_bid = market_state['best_bid']
    best_ask = market_state['best_ask']
    if best_bid is None or best_ask is None:
        return 0.5  # avoid division by zero

    if best_ask <= best_bid:
        return 0.5  # treat spread as zero and avoid division

    trade_price_stats = compute_trade_price_stats(market_history['market_trades'])
    if trade_price_stats['mean'] is None:
        return 0.5  # fallback to a neutral shade

    shade = (best_ask - best_bid) / trade_price_stats['mean']
    return min(max(shade, 0), 1)  # ensure shade is in [0,1]


def compute_trade_price_stats(trades):
    if not trades:
        return {'mean': None}

    prices = [trade['trade_price_relative'] for trade in trades]
    return {'mean': sum(prices) / len(prices)}