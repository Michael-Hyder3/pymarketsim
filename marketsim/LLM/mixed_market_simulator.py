from marketsim.fourheap.constants import BUY


def _pair_matched_orders(matched_orders):
    paired_trades = {}
    for mo in matched_orders:
        key = (mo.time, mo.order.quantity)
        if key not in paired_trades:
            paired_trades[key] = {
                "time": mo.time,
                "quantity": mo.order.quantity,
                "buyer_id": None,
                "seller_id": None,
                "buyer_order": None,
                "seller_order": None,
            }
        if mo.order.order_type == BUY:
            paired_trades[key]["buyer_id"] = mo.order.agent_id
            paired_trades[key]["buyer_order"] = mo.order
        else:
            paired_trades[key]["seller_id"] = mo.order.agent_id
            paired_trades[key]["seller_order"] = mo.order
    return paired_trades


def _trade_price_from_orders(buyer_order, seller_order):
    if buyer_order is not None and seller_order is not None:
        if buyer_order.time < seller_order.time:
            return buyer_order.price, "BUY"
        if buyer_order.time > seller_order.time:
            return seller_order.price, "SELL"
        b_seq = getattr(buyer_order, "_arrival_seq", 0)
        s_seq = getattr(seller_order, "_arrival_seq", 0)
        if b_seq <= s_seq:
            return buyer_order.price, "BUY"
        return seller_order.price, "SELL"
    if buyer_order is not None:
        return buyer_order.price, "BUY"
    if seller_order is not None:
        return seller_order.price, "SELL"
    return None, None
