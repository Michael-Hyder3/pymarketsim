"""LOB methodology seed (book_orders / o_LOB foothold).

Binds only the observation fields from the paper LOB family
(BID, ASK, v, F̃, resting books, submit tape), unpacks order-record fields
like the Spread prev_event foothold, then returns a fixed mild shade.
"""


def agent_strategy(market_history):
    # o_LOB: best quotes, open unit, pricing estimate, resting books, submit tape.
    best_bid = market_history.get("best_bid")
    best_ask = market_history.get("best_ask")
    current_pv = float(market_history.get("current_pv") or 0.0)
    fundamental_estimate = float(
        market_history.get("fundamental_estimate") or 0.0
    )
    bids = list(market_history.get("bids") or [])
    asks = list(market_history.get("asks") or [])
    orders = list(market_history.get("orders") or [])

    if orders:
        o0 = orders[0]
        time = int(o0.get("time") or 0)
        side = int(o0.get("side") or 0)
        price_rel = float(o0.get("price_rel") or 0.0)
        quantity = float(o0.get("quantity") or 0.0)
        accepted = 1.0 if o0.get("accepted") else 0.0
        match_price_rel = o0.get("match_price_rel")
        if match_price_rel is None:
            match_price_rel = 0.0
        else:
            match_price_rel = float(match_price_rel)
        agent_id = int(o0.get("agent_id") or 0)
        order_id = int(o0.get("order_id") or 0)
    else:
        time = 0
        side = 0
        price_rel = 0.0
        quantity = 0.0
        accepted = 0.0
        match_price_rel = 0.0
        agent_id = 0
        order_id = 0

    if best_bid is None or best_ask is None:
        return 1.0
    return 0.01
