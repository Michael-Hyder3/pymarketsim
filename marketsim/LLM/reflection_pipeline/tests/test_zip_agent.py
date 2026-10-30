"""Unit tests for Cliff & Bruten ZIP margin heuristics and buy/sell agents."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from marketsim.agent.zip_agent_buy_sell import ZIPAgentBuy, ZIPAgentSell
from marketsim.fourheap.constants import BUY, SELL


class _FakeOrderBook:
    def __init__(self):
        self.buy_unmatched = SimpleNamespace(peek_order=lambda: None, peek=lambda: None)
        self.sell_unmatched = SimpleNamespace(peek_order=lambda: None, peek=lambda: None)


class _FakeMarket:
    def __init__(self, fundamental: float = 100_000.0):
        self._f = float(fundamental)
        self._t = 0
        self.order_book = _FakeOrderBook()
        self.matched_orders = []

    def get_fundamental_value(self):
        return self._f

    def get_time(self):
        return self._t

    def get_info(self):
        return self._f, 0.0, 100


def _buyer(**kwargs) -> ZIPAgentBuy:
    gen = torch.Generator()
    gen.manual_seed(7)
    return ZIPAgentBuy(
        agent_id=0,
        market=_FakeMarket(),
        q_max=3,
        pv_var=1e4,
        generator=gen,
        episode_seed=1,
        beta=0.5,
        momentum=0.0,
        initial_mu=-0.01,
        **kwargs,
    )


def _seller(**kwargs) -> ZIPAgentSell:
    gen = torch.Generator()
    gen.manual_seed(11)
    return ZIPAgentSell(
        agent_id=1,
        market=_FakeMarket(),
        q_max=3,
        pv_var=1e4,
        generator=gen,
        episode_seed=1,
        beta=0.5,
        momentum=0.0,
        initial_mu=0.01,
        **kwargs,
    )


def test_buyer_raises_margin_on_cheap_accepted_trade():
    ag = _buyer()
    p0 = ag._shout_abs()
    assert p0 is not None
    # Accepted trade well below current bid ⇒ raise μ (lower bid).
    # Buyer raise ⇒ μ becomes more negative.
    q = p0 - 500.0
    mu_before = ag.mu
    ag.observe_zip_shout(price_abs=q, side=BUY, accepted=True)
    assert ag.mu < mu_before
    assert ag._shout_abs() < p0


def test_buyer_lowers_margin_on_rejected_higher_bid():
    ag = _buyer()
    p0 = ag._shout_abs()
    assert p0 is not None
    # Rejected bid above current shout ⇒ active buyer with p ≤ q lowers μ (bids up).
    # Buyer lower ⇒ μ rises toward 0.
    q = p0 + 500.0
    mu_before = ag.mu
    ag.observe_zip_shout(price_abs=q, side=BUY, accepted=False)
    assert ag.mu > mu_before
    assert ag._shout_abs() > p0


def test_seller_raises_margin_on_expensive_accepted_trade():
    ag = _seller()
    p0 = ag._shout_abs()
    assert p0 is not None
    q = p0 + 500.0
    mu_before = ag.mu
    ag.observe_zip_shout(price_abs=q, side=SELL, accepted=True)
    assert ag.mu > mu_before
    assert ag._shout_abs() > p0


def test_seller_lowers_margin_on_rejected_cheaper_ask():
    ag = _seller()
    p0 = ag._shout_abs()
    assert p0 is not None
    q = p0 - 500.0
    mu_before = ag.mu
    ag.observe_zip_shout(price_abs=q, side=SELL, accepted=False)
    assert ag.mu < mu_before
    assert ag._shout_abs() < p0


def test_inactive_buyer_does_not_lower_margin():
    ag = _buyer()
    ag.position = ag.q_max  # inactive
    p0 = ag._shout_abs()
    assert p0 is not None
    mu_before = ag.mu
    ag.observe_zip_shout(price_abs=p0 + 100.0, side=BUY, accepted=False)
    assert ag.mu == mu_before


def test_take_action_buy_only_and_sell_only():
    buy = _buyer()
    sell = _seller()
    buy_orders = buy.take_action()
    sell_orders = sell.take_action()
    assert len(buy_orders) == 1
    assert len(sell_orders) == 1
    assert buy_orders[0].order_type == BUY
    assert sell_orders[0].order_type == SELL
    # Relative book prices should be shout_abs - F.
    F = 100_000.0
    assert abs(buy_orders[0].price - (buy._shout_abs() - F)) < 1e-6
    assert abs(sell_orders[0].price - (sell._shout_abs() - F)) < 1e-6


def test_zip_episode_smoke():
    from marketsim.LLM.episode_runner import run_episode

    def _dummy(_obs):
        return 0.0

    result = run_episode(
        llm_strategy_func=_dummy,
        llm_order_type=BUY,
        num_zi_buy=2,
        num_zi_sell=2,
        timesteps=40,
        q_max=2,
        pv_var=1e5,
        shade_range=[0.0, 0.1],
        arrival_rate=0.2,
        random_seed=42,
        special_agent_kind="zip",
        use_dummy_fundamental=True,
    )
    assert "llm_surplus" in result
    assert result["llm_units_available"] == 2


def test_noisy_zip_matches_true_f_when_obs_noise_zero():
    from marketsim.agent.zip_agent_buy_sell import ZIPAgentBuyNoisy

    gen = torch.Generator()
    gen.manual_seed(7)
    true = _buyer(obs_noise_var=0.0)
    noisy = ZIPAgentBuyNoisy(
        agent_id=0,
        market=_FakeMarket(),
        q_max=3,
        pv_var=1e4,
        generator=gen,
        episode_seed=1,
        beta=0.5,
        momentum=0.0,
        initial_mu=-0.01,
        obs_noise_var=0.0,
    )
    # Same PV draw path: re-seed and copy PV from true agent for a fair compare.
    noisy.pv = true.pv
    noisy.mu = true.mu
    assert abs(noisy._fundamental_now() - true._fundamental_now()) < 1e-9
    assert abs(noisy._limit_abs() - true._limit_abs()) < 1e-6
    o_true = true.take_action()[0]
    o_noisy = noisy.take_action()[0]
    assert abs(o_true.price - o_noisy.price) < 1e-6


def test_noisy_zip_limit_uses_estimate_not_true_f():
    from marketsim.agent.zip_agent_buy_sell import ZIPAgentBuyNoisy

    gen = torch.Generator()
    gen.manual_seed(99)
    mkt = _FakeMarket(fundamental=100_000.0)
    ag = ZIPAgentBuyNoisy(
        agent_id=0,
        market=mkt,
        q_max=3,
        pv_var=1e4,
        generator=gen,
        episode_seed=1,
        beta=0.5,
        momentum=0.0,
        initial_mu=-0.01,
        obs_noise_var=1e6,  # σ=1000
    )
    f_hat = ag._fundamental_now()
    assert abs(f_hat - 100_000.0) > 1.0  # noise almost surely moves estimate
    lim = ag._limit_abs()
    assert lim is not None
    # λ = θ + F̂, not θ + F_true
    assert abs(lim - (ag._limit_rel() + f_hat)) < 1e-9
    assert abs(lim - (ag._limit_rel() + 100_000.0)) > 1.0


def test_noisy_zip_observe_rebuilds_abs_from_rel():
    from marketsim.agent.zip_agent_buy_sell import ZIPAgentBuyNoisy

    gen = torch.Generator()
    gen.manual_seed(3)
    ag = ZIPAgentBuyNoisy(
        agent_id=0,
        market=_FakeMarket(fundamental=100_000.0),
        q_max=3,
        pv_var=1e4,
        generator=gen,
        episode_seed=1,
        beta=0.5,
        momentum=0.0,
        initial_mu=-0.01,
        obs_noise_var=1e6,
    )
    f_hat = ag._fundamental_now()
    p0 = ag._shout_abs()
    assert p0 is not None
    # Rejected bid above current shout on the agent's F̂ scale ⇒ lower margin.
    price_rel = (p0 + 500.0) - f_hat
    mu_before = ag.mu
    ag.observe_zip_shout(
        price_abs=p0 + 500.0 + 50_000.0,  # deliberately wrong abs (true-F scale junk)
        price_rel=price_rel,
        side=BUY,
        accepted=False,
    )
    assert ag.mu > mu_before


def test_zip_noisy_episode_smoke():
    from marketsim.LLM.episode_runner import run_episode

    def _dummy(_obs):
        return 0.05

    result = run_episode(
        llm_strategy_func=_dummy,
        llm_order_type=BUY,
        num_zi_buy=2,
        num_zi_sell=2,
        timesteps=40,
        q_max=2,
        pv_var=1e5,
        shade_range=[0.0, 0.1],
        arrival_rate=0.2,
        random_seed=42,
        special_agent_kind="zip_noisy",
        background_agent_kind="zip_noisy",
        strategy_type="spread",
        spread_input_mode="classic",
        obs_noise_var=1e6,
        use_dummy_fundamental=True,
    )
    assert "llm_surplus" in result
    assert result["market_config"]["background_agent_kind"] == "zip_noisy"


class _LiftSafeBook:
    """Minimal book for ZI warm-start take_action (needs get_best_*)."""

    def __init__(self):
        self.buy_unmatched = SimpleNamespace(
            peek_order=lambda: None, peek=lambda: None
        )
        self.sell_unmatched = SimpleNamespace(
            peek_order=lambda: None, peek=lambda: None
        )

    def get_best_ask(self):
        return float("inf")

    def get_best_bid(self):
        return float("-inf")


def test_zi_warmup_gates_shout_and_allows_learning_under_freeze():
    """First N steps post ZI; freeze_margin still learns μ until N, then holds."""
    ag = _buyer(zi_warmup_steps=25, shade=[0.0, 0.01], freeze_margin=True)
    ag.market.order_book = _LiftSafeBook()
    ag.market._t = 0
    assert ag._in_zi_warmup()
    assert ag.take_action()
    mu0 = ag.mu
    p0 = ag._shout_abs()
    assert p0 is not None
    ag.observe_zip_shout(price_abs=p0 - 500.0, side=BUY, accepted=True)
    assert ag.mu != mu0
    mu1 = ag.mu
    ag.market._t = 25
    assert not ag._in_zi_warmup()
    ag.observe_zip_shout(price_abs=p0 - 500.0, side=BUY, accepted=True)
    assert ag.mu == mu1


def test_zip_episode_reports_zi_warmup_steps():
    from marketsim.LLM.episode_runner import run_episode

    def _dummy(_obs):
        return 0.05

    result = run_episode(
        llm_strategy_func=_dummy,
        llm_order_type=BUY,
        num_zi_buy=2,
        num_zi_sell=2,
        timesteps=60,
        q_max=2,
        pv_var=1e5,
        shade_range=[0.0, 0.015],
        arrival_rate=0.2,
        random_seed=3,
        special_agent_kind="llm",
        background_agent_kind="zip_fixed",
        background_zip_fixed_mu=0.0015,
        background_zip_zi_warmup_steps=25,
        strategy_type="spread",
        spread_input_mode="book",
        obs_noise_var=1e6,
        use_dummy_fundamental=True,
    )
    assert result["market_config"]["background_zip_zi_warmup_steps"] == 25

