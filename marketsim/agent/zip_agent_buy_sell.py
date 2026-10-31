from __future__ import annotations

import random
from typing import List, Optional, Tuple, Union

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.agent.shading import (
    maybe_take_best_quote_buy,
    maybe_take_best_quote_sell,
    zi_limit_price_buy,
    zi_limit_price_sell,
)
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.market.market import Market
from marketsim.private_values.separated_private_values import SeparatedPrivateValues

_DEFAULT_ZI_SHADE = [0.0, 0.015]


_MU_BUY_INIT = (-0.015, -0.001)
_MU_SELL_INIT = (0.001, 0.015)
_MU_BUY_BOUNDS = (-0.05, 0.0)
_MU_SELL_BOUNDS = (0.0, 0.05)
_BETA_RANGE = (0.1, 0.5)
_GAMMA_RANGE = (0.0, 0.1)
_R_UP = (1.0, 1.002)
_R_DOWN = (0.998, 1.0)
_A_UP = (0.0, 5.0)
_A_DOWN = (-5.0, 0.0)
_SHADE_EPS = 1e-4


def shade_range_to_zip_mu_init(
    shade_range: Optional[List[float]],
    *,
    side: int,
) -> Tuple[float, float]:
    shade = list(shade_range or [0.0, 0.015])
    s_lo = float(shade[0] if shade else 0.0)
    s_hi = float(shade[-1] if shade else 0.015)
    if s_hi < s_lo:
        s_lo, s_hi = s_hi, s_lo
    s_hi = max(s_hi, _SHADE_EPS)
    inner = max(s_lo, _SHADE_EPS)
    if inner >= s_hi:
        inner = min(s_hi * 0.25, s_hi)
        inner = max(inner, min(_SHADE_EPS, s_hi))
    if int(side) == BUY:
        return (-s_hi, -inner)
    return (inner, s_hi)


def shade_hi_to_fixed_zip_mu(shade_hi: float, *, side: int) -> float:
    s = max(float(shade_hi), 0.0)
    if int(side) == BUY:
        return -s
    return s


def _peek_best(side_heap) -> Optional[float]:
    try:
        if side_heap.peek_order() is None:
            return None
        return float(side_heap.peek())
    except Exception:
        return None


class _ZIPAgentBase(Agent):
    ROLE: str = "buyer"
    ORDER_TYPE: int = BUY

    def __init__(
        self,
        agent_id: int,
        market: Market,
        q_max: int,
        pv_var: float,
        obs_noise_var: float = 0.0,
        generator=None,
        episode_seed: int = 0,
        *,
        beta: Optional[float] = None,
        momentum: Optional[float] = None,
        initial_mu: Optional[float] = None,
        mu_init_range: Optional[Tuple[float, float]] = None,
        r_up: Optional[Tuple[float, float]] = None,
        r_down: Optional[Tuple[float, float]] = None,
        a_up: Optional[Tuple[float, float]] = None,
        a_down: Optional[Tuple[float, float]] = None,
        freeze_margin: bool = False,
        zi_warmup_steps: int = 0,
        shade: Optional[Union[List[float], List[Tuple[float, float]]]] = None,
        eta: float = 1.0,
        shade_schedule_mode: str = "time",
        shade_sampling_mode: str = "range",
    ):
        self.agent_id = int(agent_id)
        self.market = market
        self.q_max = int(q_max)
        self.pv_var = float(pv_var)
        self.obs_noise_var = float(obs_noise_var)
        self._generator = generator
        self.position = 0
        self.cash = 0.0
        self.freeze_margin = bool(freeze_margin)
        self.zi_warmup_steps = max(0, int(zi_warmup_steps))
        self.shade = list(shade) if shade is not None else list(_DEFAULT_ZI_SHADE)
        self.eta = float(eta)
        self.shade_schedule_mode = str(shade_schedule_mode or "time")
        self.shade_sampling_mode = str(shade_sampling_mode or "range")
        self.pv = SeparatedPrivateValues(
            self.q_max,
            val_var=self.pv_var,
            base_price=0.0,
            role=self.ROLE,
            generator=generator,
        )

        agent_seed = (int(episode_seed) * 10_000 + int(self.agent_id)) & 0xFFFFFFFF
        self._rng = random.Random(agent_seed)
        self._np_rng = np.random.default_rng(agent_seed)

        self._r_up = tuple(r_up) if r_up is not None else _R_UP
        self._r_down = tuple(r_down) if r_down is not None else _R_DOWN
        self._a_up = tuple(a_up) if a_up is not None else _A_UP
        self._a_down = tuple(a_down) if a_down is not None else _A_DOWN

        if beta is None:
            self.beta = float(self._rng.uniform(*_BETA_RANGE))
        else:
            self.beta = float(beta)
        if momentum is None:
            self.gamma = float(self._rng.uniform(*_GAMMA_RANGE))
        else:
            self.gamma = float(momentum)
        self.Gamma = 0.0

        self._explicit_initial_mu = initial_mu is not None
        if initial_mu is None:
            if mu_init_range is not None:
                lo, hi = float(mu_init_range[0]), float(mu_init_range[1])
            elif self.ORDER_TYPE == BUY:
                lo, hi = _MU_BUY_INIT
            else:
                lo, hi = _MU_SELL_INIT
            self.mu = float(self._rng.uniform(min(lo, hi), max(lo, hi)))
        else:
            self.mu = float(initial_mu)
        self._clamp_mu()
        self._mu_after_warmup = float(self.mu) if self._explicit_initial_mu else None
        self._warmup_mu_snapped = self.zi_warmup_steps <= 0

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self) -> float:
        from marketsim.agent.fundamental_estimate import absolute_fundamental_estimate

        return absolute_fundamental_estimate(
            self.market,
            obs_noise_var=self.obs_noise_var,
            np_rng=self._np_rng,
        )

    def is_active(self) -> bool:
        if self.ORDER_TYPE == BUY:
            return int(self.position) < self.q_max
        return int(self.position) > -self.q_max

    def _fundamental_now(self) -> float:
        return float(self.estimate_fundamental())

    def _limit_rel(self) -> Optional[float]:
        try:
            return float(self.pv.value_for_exchange(self.position, self.ORDER_TYPE))
        except Exception:
            return None

    def _limit_abs(self) -> Optional[float]:
        lim_rel = self._limit_rel()
        if lim_rel is None:
            return None
        return float(lim_rel) + self._fundamental_now()

    def _shout_abs(self, limit_abs: Optional[float] = None) -> Optional[float]:
        lam = self._limit_abs() if limit_abs is None else float(limit_abs)
        if lam is None or lam <= 0.0:
            return None
        return float(lam) * (1.0 + float(self.mu))

    def _clamp_mu(self) -> None:
        if self.ORDER_TYPE == BUY:
            lo, hi = _MU_BUY_BOUNDS
            self.mu = float(min(hi, max(lo, self.mu)))
        else:
            lo, hi = _MU_SELL_BOUNDS
            self.mu = float(min(hi, max(lo, self.mu)))

    def _target_price(self, q: float, *, increase_price: bool) -> float:
        if increase_price:
            R = self._rng.uniform(*self._r_up)
            A = self._rng.uniform(*self._a_up)
        else:
            R = self._rng.uniform(*self._r_down)
            A = self._rng.uniform(*self._a_down)
        return float(R) * float(q) + float(A)

    def _update_margin(self, *, raise_margin: bool, q: float) -> None:
        lam = self._limit_abs()
        if lam is None or lam <= 0.0:
            return
        p = self._shout_abs(lam)
        if p is None:
            return

        if self.ORDER_TYPE == BUY:
            increase_price = not raise_margin
        else:
            increase_price = raise_margin

        tau = self._target_price(q, increase_price=increase_price)
        delta = float(self.beta) * (float(tau) - float(p))
        self.Gamma = float(self.gamma) * float(self.Gamma) + (1.0 - float(self.gamma)) * delta
        p_new = float(p) + float(self.Gamma)
        self.mu = p_new / float(lam) - 1.0
        self._clamp_mu()

    def _in_zi_warmup(self) -> bool:
        if self.zi_warmup_steps <= 0:
            return False
        return int(self.market.get_time()) < int(self.zi_warmup_steps)

    def _maybe_snap_mu_after_warmup(self) -> None:
        if self._warmup_mu_snapped:
            return
        if self._in_zi_warmup():
            return
        self._warmup_mu_snapped = True
        if self._mu_after_warmup is not None:
            self.mu = float(self._mu_after_warmup)
            self._clamp_mu()
            self.Gamma = 0.0

    def observe_zip_shout(
        self,
        *,
        price_abs: float,
        side: int,
        accepted: bool,
        price_rel: Optional[float] = None,
    ) -> None:
        self._maybe_snap_mu_after_warmup()
        if self._in_zi_warmup() and self._mu_after_warmup is not None:
            return
        if self.freeze_margin and not self._in_zi_warmup():
            return
        if price_rel is not None:
            price_abs = float(price_rel) + self._fundamental_now()
        lam = self._limit_abs()
        p = self._shout_abs(lam)
        if p is None:
            return
        q = float(price_abs)
        active = self.is_active()
        is_bid = int(side) == BUY
        is_ask = int(side) == SELL

        if self.ORDER_TYPE == BUY:
            if accepted:
                if p >= q:
                    self._update_margin(raise_margin=True, q=q)
                if is_ask and active and p <= q:
                    self._update_margin(raise_margin=False, q=q)
            else:
                if is_bid and active and p <= q:
                    self._update_margin(raise_margin=False, q=q)
        else:
            if accepted:
                if p <= q:
                    self._update_margin(raise_margin=True, q=q)
                if is_bid and active and p >= q:
                    self._update_margin(raise_margin=False, q=q)
            else:
                if is_ask and active and p >= q:
                    self._update_margin(raise_margin=False, q=q)

    def _maybe_take_crossing_quote(self, shout_rel: float, limit_rel: float, t: int) -> Optional[Order]:
        ob = self.market.order_book
        if self.ORDER_TYPE == BUY:
            best_ask = _peek_best(ob.sell_unmatched)
            if best_ask is None:
                return None
            if best_ask <= shout_rel and best_ask <= limit_rel:
                return Order(
                    price=float(best_ask),
                    quantity=1,
                    agent_id=self.get_id(),
                    time=t,
                    order_type=BUY,
                    order_id=self._rng.randint(1, 10_000_000),
                )
            return None

        best_bid = _peek_best(ob.buy_unmatched)
        if best_bid is None:
            return None
        if best_bid >= shout_rel and best_bid >= limit_rel:
            return Order(
                price=float(best_bid),
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=SELL,
                order_id=self._rng.randint(1, 10_000_000),
            )
        return None

    def _zi_warmup_take_action(self, t: int) -> List[Order]:
        limit_rel = self._limit_rel()
        if limit_rel is None:
            return []
        self._rng.seed(int(t) + int(self.agent_id))
        if self.ORDER_TYPE == BUY:
            lifted = maybe_take_best_quote_buy(
                market=self.market,
                buyer_value=float(limit_rel),
                eta=self.eta,
                agent_id=self.get_id(),
                time=int(t),
                rng=self._rng,
            )
            if lifted is not None:
                return [lifted]
            price, _ = zi_limit_price_buy(
                buyer_value=float(limit_rel),
                fundamental_now=self._fundamental_now(),
                shade_spec=self.shade,
                current_time=int(t),
                num_transactions=len(self.market.matched_orders),
                schedule_mode=self.shade_schedule_mode,
                rng=self._rng,
                sampling_mode=self.shade_sampling_mode,
            )
        else:
            lifted = maybe_take_best_quote_sell(
                market=self.market,
                seller_cost=float(limit_rel),
                eta=self.eta,
                agent_id=self.get_id(),
                time=int(t),
                rng=self._rng,
            )
            if lifted is not None:
                return [lifted]
            price, _ = zi_limit_price_sell(
                seller_cost=float(limit_rel),
                fundamental_now=self._fundamental_now(),
                shade_spec=self.shade,
                current_time=int(t),
                num_transactions=len(self.market.matched_orders),
                schedule_mode=self.shade_schedule_mode,
                rng=self._rng,
                sampling_mode=self.shade_sampling_mode,
            )
        return [
            Order(
                price=float(price),
                quantity=1,
                agent_id=self.get_id(),
                time=int(t),
                order_type=self.ORDER_TYPE,
                order_id=self._rng.randint(1, 10_000_000),
            )
        ]

    def take_action(self) -> List[Order]:
        if not self.is_active():
            return []
        t = self.market.get_time()
        if self._in_zi_warmup():
            return self._zi_warmup_take_action(int(t))
        self._maybe_snap_mu_after_warmup()

        limit_rel = self._limit_rel()
        if limit_rel is None:
            return []
        F = self._fundamental_now()
        limit_abs = float(limit_rel) + F
        if limit_abs <= 0.0:
            return []
        shout_abs = float(limit_abs) * (1.0 + float(self.mu))
        shout_rel = shout_abs - F

        lifted = self._maybe_take_crossing_quote(shout_rel, float(limit_rel), t)
        if lifted is not None:
            return [lifted]

        return [
            Order(
                price=float(shout_rel),
                quantity=1,
                agent_id=self.get_id(),
                time=t,
                order_type=self.ORDER_TYPE,
                order_id=self._rng.randint(1, 10_000_000),
            )
        ]

    def update_position(self, q, p):
        self.position += q
        self.cash += p




class ZIPAgentBuy(_ZIPAgentBase):
    ROLE = "buyer"
    ORDER_TYPE = BUY



class ZIPAgentSell(_ZIPAgentBase):
    ROLE = "seller"
    ORDER_TYPE = SELL



class _ZIPAgentNoisyFundamentalBase(_ZIPAgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._f_hat: Optional[float] = None
        self._f_hat_time: Optional[int] = None

    def _fundamental_now(self) -> float:
        t = int(self.market.get_time())
        if self._f_hat is None or self._f_hat_time != t:
            self._f_hat = float(self.estimate_fundamental())
            self._f_hat_time = t
        return float(self._f_hat)

    def observe_zip_shout(
        self,
        *,
        price_abs: float,
        side: int,
        accepted: bool,
        price_rel: Optional[float] = None,
    ) -> None:
        if price_rel is not None:
            price_abs = float(price_rel) + self._fundamental_now()
        super().observe_zip_shout(
            price_abs=float(price_abs),
            side=int(side),
            accepted=bool(accepted),
            price_rel=None,
        )



class ZIPAgentBuyNoisy(_ZIPAgentNoisyFundamentalBase):
    ROLE = "buyer"
    ORDER_TYPE = BUY



class ZIPAgentSellNoisy(_ZIPAgentNoisyFundamentalBase):
    ROLE = "seller"
    ORDER_TYPE = SELL

