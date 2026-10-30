"""Compact HBL-like LOB shade seed (single function; LLM-editable).

Belief on the hbl_parity L-window + coarse grid surplus max, then invert the
chosen relative limit into shade via pricing_fundamental. Same idea as full
HBL distill, without multi-helper AST / withdrawal scans.

Source of truth for ``--lob-seed-scale hbl``.
"""


def agent_strategy(market_history):
    orders = list(market_history.get("orders") or [])
    v = float(market_history.get("current_pv") or 0.0)
    bb = market_history.get("best_bid_relative")
    ba = market_history.get("best_ask_relative")
    side = int(market_history.get("side") or 1)
    L = int(market_history.get("L") or 6)
    txn = int(market_history.get("transaction_count") or 0)
    t_now = int(market_history.get("timestep") or 0)
    grace = float(market_history.get("grace_period") or 10.0) or 1.0
    F = float(market_history.get("pricing_fundamental") or 0.0)
    if bb is None or ba is None:
        return 0.01
    bb, ba = float(bb), float(ba)

    def clip01(x):
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

    def to_shade(p):
        d = F + v
        if d <= 0.0:
            return 0.0
        return clip01((v - p) / d if side >= 0 else (p - v) / d)

    # Early / empty memory: mild shade; take opposite quote if profitable.
    if txn < L or not orders:
        q = ba if side >= 0 else bb
        if (side >= 0 and q <= v) or (side < 0 and q >= v):
            return to_shade(q)
        return 0.01

    def belief(p):
        good = bad = 0.0
        for o in orders:
            qty = float(o.get("quantity") or 1)
            s = int(o.get("side") or 0)
            pr = float(o.get("price_rel") or 0.0)
            alive = max(0.0, float(t_now - int(o.get("time") or 0)))
            w = 1.0 if alive >= grace else alive / grace
            if side >= 0:
                if s < 0 and pr <= p:
                    good += qty
                if o.get("accepted"):
                    mp = o.get("match_price_rel")
                    if mp is None:
                        mp = pr
                    if s > 0 and float(mp) <= p:
                        good += qty
                elif s > 0 and pr >= p:
                    bad += qty * w
            else:
                if s > 0 and pr >= p:
                    good += qty
                if o.get("accepted"):
                    mp = o.get("match_price_rel")
                    if mp is None:
                        mp = pr
                    if s < 0 and float(mp) >= p:
                        good += qty
                elif s < 0 and pr <= p:
                    bad += qty * w
        return 0.0 if good <= 0.0 else good / (good + bad)

    side_sign = 1 if side >= 0 else -1
    prices = [
        float(o.get("price_rel") or 0.0)
        for o in orders
        if int(o.get("side") or 0) * side_sign > 0
    ]
    if side >= 0:
        lb = min(prices) if prices else bb
        ub = ba
    else:
        lb = min(prices) if prices else bb
        ub = max(prices) if prices else ba
    lb = min(lb, bb, ba)
    ub = max(ub, bb, ba)
    best_p = min(v, ba) if side >= 0 else max(v, bb)
    best_s = -1e300
    n_grid = 24
    if ub > lb:
        for i in range(n_grid + 1):
            p = lb + (ub - lb) * (i / float(n_grid))
            if side >= 0 and p > v:
                continue
            if side < 0 and p < v:
                continue
            surp = ((v - p) if side >= 0 else (p - v)) * belief(p)
            if surp > best_s:
                best_s, best_p = surp, p
    if side >= 0:
        best_p = min(best_p, v)
    else:
        best_p = max(best_p, v)
    return to_shade(best_p)
