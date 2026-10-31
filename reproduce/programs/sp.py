def agent_strategy(market_state):
    best_bid = market_state.get('best_bid')
    best_ask = market_state.get('best_ask')
    bids = list(market_state.get('bids') or [])
    asks = list(market_state.get('asks') or [])
    units_available = float(market_state.get('units_available') or 0.0)
    q_max = float(market_state.get('q_max') or 0.0)
    current_pv = float(market_state.get('current_pv') or 0.0)
    private_values = list(market_state.get('private_values') or [])
    prev_event = market_state.get('prev_event')
    if isinstance(prev_event, dict):
        price_abs = float(prev_event.get('price_abs') or 0.0)
        price_rel = float(prev_event.get('price_rel') or 0.0)
        side = float(prev_event.get('side') or 0.0)
        accepted = 1.0 if prev_event.get('accepted') else 0.0
    else:
        price_abs = 0.0
        price_rel = 0.0
        side = 0.0
        accepted = 0.0

    mid_quote = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else 0.0
    inventory_remaining = units_available / q_max

    shade = 0.0
    if best_bid is not None:
        shade += (inventory_remaining - 0.5) * (1 - shade) * (1 + (best_bid - current_pv) / q_max) * 0.8
    if best_ask is not None:
        shade -= (inventory_remaining - 0.5) * (1 - shade) * (1 - (best_ask - current_pv) / q_max) * 0.8

    shade_error_direction = 1.0 if inventory_remaining > 0.5 else -1.0

    shade = shade + shade_error_direction * (inventory_remaining - 0.5) * 0.05

    shade = 1 - (1 - shade) ** 2

    shade = max(0.0, min(1.0, shade)) + 0.001

    # Introduce a piecewise_by_sign form class to refine the shade map
    if mid_quote < market_state['fundamental_estimate']:
        shade += 0.1 * (1 - shade)
    else:
        shade -= 0.1 * (1 - shade)

    shade = min(1.0, max(0.0, shade))

    # Introduce a new form class to refine the shade map
    shade = shade * (inventory_remaining - 0.5) * 0.1

    return shade
