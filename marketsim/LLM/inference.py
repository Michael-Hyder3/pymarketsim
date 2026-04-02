"""
Inference function μ : A → Δ(Θ) for the LLM agent.

Game-theoretic framing
──────────────────────
  A  = observable actions in the LOB
         • ask prices resting on the sell side  (submitted by sellers)
         • bid prices resting on the buy side   (submitted by competing buyers)
         • trade prices                         (crossing of a resting order)
  Θ  = private type space (marginal costs for sellers, valuations for buyers)

  μ_s : ask_history ─→ Δ(Θ_s)   posterior belief over seller costs
  μ_b : bid_history ─→ Δ(Θ_b)   posterior belief over buyer valuations

What the LLM agent IS and IS NOT allowed to know
─────────────────────────────────────────────────
  ALLOWED (directly observable from the LOB):
    • best_ask, best_bid at each timestep
    • prices of recent trades
    • its OWN private values

  NOT ALLOWED (opponent-private, must be inferred):
    • opponent shade distribution / shade range
    • opponent private value distribution (pv_var)
    • number of opponent agents
    • which side set the price in any given trade

This module therefore uses NO assumed likelihood model and NO knowledge of
opponent parameters.  It accumulates empirical statistics over the sequence
of observed LOB prices and summarises them as actionable signals.  The
inference is fully non-parametric: the agent learns from what it sees, nothing more.
"""

import math
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Running statistics accumulator (no assumed distribution)
# ---------------------------------------------------------------------------

class PriceHistory:
    """
    Accumulates a stream of observed prices and computes order-statistics-based
    summaries without assuming any parametric form.

    The agent sees bid/ask/trade prices; these are the raw actions in A.
    Summaries (min, percentiles, mean, trend) describe the *empirical* price
    distribution the agent has encountered — nothing about opponent internals.
    """

    def __init__(self, maxlen: int = 500):
        self._prices: List[float] = []
        self.maxlen = maxlen
        self.n_total: int = 0         # cumulative count (including evicted)

    def observe(self, price: float) -> None:
        self._prices.append(price)
        if len(self._prices) > self.maxlen:
            self._prices.pop(0)
        self.n_total += 1

    def observe_batch(self, prices: List[float]) -> None:
        for p in prices:
            self.observe(p)

    @property
    def n_obs(self) -> int:
        return len(self._prices)

    def mean(self) -> Optional[float]:
        if not self._prices:
            return None
        return sum(self._prices) / len(self._prices)

    def std(self) -> Optional[float]:
        if len(self._prices) < 2:
            return None
        mu = self.mean()
        var = sum((p - mu) ** 2 for p in self._prices) / len(self._prices)
        return math.sqrt(var)

    def minimum(self) -> Optional[float]:
        return min(self._prices) if self._prices else None

    def maximum(self) -> Optional[float]:
        return max(self._prices) if self._prices else None

    def percentile(self, q: float) -> Optional[float]:
        """
        q-th percentile (0 <= q <= 100) of the observed price distribution.
        Uses linear interpolation between sorted observations.
        """
        if not self._prices:
            return None
        sorted_p = sorted(self._prices)
        n = len(sorted_p)
        idx = (q / 100.0) * (n - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= n:
            return sorted_p[-1]
        frac = idx - lo
        return sorted_p[lo] + frac * (sorted_p[hi] - sorted_p[lo])

    def recent_trend(self, window: int = 20) -> Optional[float]:
        """
        Slope (price change per step) of the most recent `window` prices,
        estimated by least-squares linear regression on (i, price[i]).
        Positive = rising, negative = falling.
        Returns None if fewer than 2 observations.
        """
        pts = self._prices[-window:]
        n = len(pts)
        if n < 2:
            return None
        xm = (n - 1) / 2.0
        ym = sum(pts) / n
        num = sum((i - xm) * (pts[i] - ym) for i in range(n))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    def summarise(self, label: str) -> Dict:
        """Return a flat dict of empirical statistics, prefixed by label."""
        return {
            f"{label}_n_obs":  self.n_total,
            f"{label}_mean":   round(self.mean(), 2)         if self.mean()         is not None else None,
            f"{label}_std":    round(self.std(), 2)          if self.std()          is not None else None,
            f"{label}_min":    round(self.minimum(), 2)      if self.minimum()      is not None else None,
            f"{label}_max":    round(self.maximum(), 2)      if self.maximum()      is not None else None,
            f"{label}_p10":    round(self.percentile(10), 2) if self.percentile(10) is not None else None,
            f"{label}_p25":    round(self.percentile(25), 2) if self.percentile(25) is not None else None,
            f"{label}_p50":    round(self.percentile(50), 2) if self.percentile(50) is not None else None,
            f"{label}_p75":    round(self.percentile(75), 2) if self.percentile(75) is not None else None,
            f"{label}_p90":    round(self.percentile(90), 2) if self.percentile(90) is not None else None,
            f"{label}_trend":  round(self.recent_trend(), 4) if self.recent_trend() is not None else None,
        }

    def reset(self) -> None:
        self._prices = []
        self.n_total = 0


# ---------------------------------------------------------------------------
# Market-level inference: wraps ask side, bid side, and trade prices
# ---------------------------------------------------------------------------

class MarketInference:
    """
    Observes LOB prices and maintains empirical summaries for the strategy.

    Three independent price streams are tracked:
      ask_prices   — resting asks seen in best_ask each timestep
      bid_prices   — resting bids seen in best_bid each timestep
      trade_prices — prices at which trades executed

    No parametric model is assumed.  The agent learns the empirical
    distribution of market prices entirely from what it observes.

    This is the inference function  mu : A -> Delta(Theta)  in the
    game-theoretic sense — expressed as order statistics of observed prices
    because the agent does not know opponent shade ranges or pv distributions.
    """

    def __init__(self, maxlen: int = 500):
        self._seen_trade_times: List[int] = []

        self.ask_prices   = PriceHistory(maxlen=maxlen)
        self.bid_prices   = PriceHistory(maxlen=maxlen)
        self.trade_prices = PriceHistory(maxlen=maxlen)

    # ── Core update — call every timestep before strategy_func ────────── #

    def update(
        self,
        market_history: Dict,
        market_state: Dict,
    ) -> None:
        """
        Absorb new observable information from the current timestep.

        Observable signals (all legitimately visible to the agent)
        ----------------------------------------------------------
        market_state['best_ask'] : float | None
            The lowest resting sell order currently in the book.
            Directly observable; the agent does not know the seller's cost
            or how much shade produced this ask.

        market_state['best_bid'] : float | None
            The highest resting buy order from a competing buyer.
            Directly observable; the agent does not know the buyer's value.

        market_history['market_trades'] : List[{time, price}]
            Trade prices. The agent cannot observe which side was passive
            (price-setting), nor the private values of either party.
        """
        trades = market_history.get("market_trades", [])
        for tr in trades:
            if tr["time"] not in self._seen_trade_times:
                self._seen_trade_times.append(tr["time"])
                self.trade_prices.observe(tr["price"])

        best_ask = market_state.get("best_ask")
        if best_ask is not None:
            self.ask_prices.observe(best_ask)

        best_bid = market_state.get("best_bid")
        if best_bid is not None:
            self.bid_prices.observe(best_bid)

    # ── Summary for strategy consumption ─────────────────────────────── #

    def summarise(self) -> Dict:
        """
        Return empirical price statistics for all three observed streams.

        All statistics are computed purely from prices the agent has seen.
        No opponent parameters (shade range, pv_var, agent count) are used
        or exposed.

        Keys in the returned dict
        ─────────────────────────
        ask_n_obs   : int    how many best_ask snapshots collected
        ask_mean    : float  mean of observed ask prices  (None until first observation)
        ask_std     : float  std  of observed ask prices
        ask_min     : float  lowest ask ever seen in this episode
        ask_p10     : float  10th percentile of observed asks
        ask_p25/p50/p75/p90 : further percentiles
        ask_trend   : float  recent slope of ask prices (positive=rising, negative=falling)

        bid_*       : same statistics for observed best_bid prices

        trade_*     : same statistics for executed trade prices

        spread_mean : float | None   ask_mean - bid_mean  (None until both sides seen)

        Typical strategy usage
        ──────────────────────
        inf = market_state['inference']

        # Is the current ask unusually cheap? (below the 10th percentile of what we've seen)
        cheap_ask = ms['best_ask'] is not None and inf['ask_p10'] is not None
                    and ms['best_ask'] < inf['ask_p10']

        # Is the ask market trending down? (sellers getting more aggressive)
        falling_asks = inf['ask_trend'] is not None and inf['ask_trend'] < 0

        # Is current ask above the historical median? (potentially expensive, wait)
        expensive_ask = ms['best_ask'] is not None and inf['ask_p50'] is not None
                        and ms['best_ask'] > inf['ask_p50']
        """
        result = {}
        result.update(self.ask_prices.summarise("ask"))
        result.update(self.bid_prices.summarise("bid"))
        result.update(self.trade_prices.summarise("trade"))

        ask_mean = result.get("ask_mean")
        bid_mean = result.get("bid_mean")
        if ask_mean is not None and bid_mean is not None:
            result["spread_mean"] = round(ask_mean - bid_mean, 2)
        else:
            result["spread_mean"] = None

        return result

    def reset(self) -> None:
        self.ask_prices.reset()
        self.bid_prices.reset()
        self.trade_prices.reset()
        self._seen_trade_times = []


# ---------------------------------------------------------------------------
# Convenience: one-shot inference from a trade list (stateless)
# ---------------------------------------------------------------------------

def infer_from_trades(trades: List[Dict]) -> Dict:
    """
    One-shot inference from a list of trade dicts [{time, price}, ...].
    Returns the same summary dict as MarketInference.summarise().
    Useful for offline analysis or cold-start summaries inside a strategy.
    """
    inf = MarketInference()
    dummy_state = {"best_ask": None, "best_bid": None}
    dummy_hist  = {"market_trades": trades, "timestep": 0, "total_trades": len(trades)}
    inf.update(dummy_hist, dummy_state)
    return inf.summarise()
