from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from marketsim.fourheap.constants import BUY


HOLDOUT_OOD_TIERS: Tuple[str, ...] = ("near", "info", "compound")

OOD_AXES: Tuple[str, ...] = (
    "num_zi_buy",
    "num_zi_sell",
    "timesteps",
    "q_max",
    "shade_hi",
    "arrival_rate",
    "shock_var",
    "obs_noise_var",
)


@dataclass
class ReflectionMarketConfig:
    label: str
    num_zi_buy: int = 5
    num_zi_sell: int = 5
    timesteps: int = 500
    q_max: int = 10
    pv_var: float = 1e5
    shade_range: List[float] = field(default_factory=lambda: [0.0, 0.015])
    fundamental_value: float = 100000.0
    mean_reversion_rate: float = 0.05
    shock_var: float = 1e5
    obs_noise_var: float = 1e6
    arrival_rate: float = 0.1
    llm_order_type: int = BUY
    fixed_seed: Optional[int] = None
    fixed_private_values_relative: Optional[List[float]] = None
    ood_tier: Optional[str] = None

    def shade_hi(self) -> float:
        shade = list(self.shade_range or [0.0, 0.0])
        return float(shade[1] if len(shade) > 1 else shade[0])

    def observable_state(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "num_zi_buy": self.num_zi_buy,
            "num_zi_sell": self.num_zi_sell,
            "timesteps": self.timesteps,
            "q_max": self.q_max,
            "shade_range": list(self.shade_range),
            "arrival_rate": self.arrival_rate,
            "fixed_private_values_relative": self.fixed_private_values_relative,
        }

    def to_run_episode_kwargs(self) -> Dict[str, Any]:
        return {
            "llm_order_type": self.llm_order_type,
            "llm_private_values": self.fixed_private_values_relative,
            "num_zi_buy": self.num_zi_buy,
            "num_zi_sell": self.num_zi_sell,
            "timesteps": self.timesteps,
            "q_max": self.q_max,
            "pv_var": self.pv_var,
            "shade_range": self.shade_range,
            "fundamental_value": self.fundamental_value,
            "mean_reversion_rate": self.mean_reversion_rate,
            "shock_var": self.shock_var,
            "obs_noise_var": self.obs_noise_var,
            "arrival_rate": self.arrival_rate,
        }

    def seed_for(self, *, base_seed: int, step: int, config_index: int, episode_index: int) -> int:
        if self.fixed_seed is not None:
            return int(self.fixed_seed) + int(episode_index)
        return int(base_seed + (step + 1) * 10_000 + config_index * 1_000 + episode_index)


    def ood_cell(
        self,
    ) -> Tuple[int, int, int, int, float, float, float, float]:
        return (
            int(self.num_zi_buy),
            int(self.num_zi_sell),
            int(self.timesteps),
            int(self.q_max),
            round(self.shade_hi(), 6),
            round(float(self.arrival_rate), 6),
            round(float(self.shock_var), 6),
            round(float(self.obs_noise_var), 6),
        )


@dataclass
class ReflectionConfigSpace:
    num_zi_buy: List[int] = field(default_factory=lambda: [5, 10])
    num_zi_sell: List[int] = field(default_factory=lambda: [5, 10])
    timesteps: List[int] = field(default_factory=lambda: [300, 400, 600])
    q_max: List[int] = field(default_factory=lambda: [10])
    shade_ranges: List[List[float]] = field(
        default_factory=lambda: zip_market_shade_ranges()
    )
    arrival_rates: List[float] = field(default_factory=lambda: [0.1, 0.15])
    pv_var: float = 1e5
    fundamental_value: float = 100000.0
    mean_reversion_rate: float = 0.05
    shock_var: float = 1e5
    obs_noise_var: float = 1e6
    shock_vars: Optional[List[float]] = None
    obs_noise_vars: Optional[List[float]] = None
    pv_vars: Optional[List[float]] = None
    llm_order_type: int = BUY
    sample_label_prefix: str = "market"
    sample_ood_tier: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_zi_buy": list(self.num_zi_buy),
            "num_zi_sell": list(self.num_zi_sell),
            "timesteps": list(self.timesteps),
            "q_max": list(self.q_max),
            "shade_ranges": [list(r) for r in self.shade_ranges],
            "arrival_rates": list(self.arrival_rates),
            "pv_var": self.pv_var,
            "fundamental_value": self.fundamental_value,
            "mean_reversion_rate": self.mean_reversion_rate,
            "shock_var": self.shock_var,
            "obs_noise_var": self.obs_noise_var,
            "shock_vars": (
                list(self.shock_vars) if self.shock_vars is not None else None
            ),
            "obs_noise_vars": (
                list(self.obs_noise_vars) if self.obs_noise_vars is not None else None
            ),
            "pv_vars": list(self.pv_vars) if self.pv_vars is not None else None,
            "llm_order_type": self.llm_order_type,
            "sample_label_prefix": self.sample_label_prefix,
            "sample_ood_tier": self.sample_ood_tier,
        }

    def sample(self, rng: random.Random, *, batch_index: int = 0) -> ReflectionMarketConfig:
        num_zi_buy = rng.choice(self.num_zi_buy)
        num_zi_sell = rng.choice(self.num_zi_sell)
        timesteps = rng.choice(self.timesteps)
        q_max = rng.choice(self.q_max)
        shade_range = list(rng.choice(self.shade_ranges))
        arrival_rate = rng.choice(self.arrival_rates)
        if self.shock_vars:
            shock = float(rng.choice(self.shock_vars))
        else:
            shock = float(self.shock_var)
        if self.obs_noise_vars:
            obs = float(rng.choice(self.obs_noise_vars))
        else:
            obs = float(self.obs_noise_var)
        if self.pv_vars:
            pv = float(rng.choice(self.pv_vars))
        else:
            pv = float(self.pv_var)
        prefix = str(self.sample_label_prefix or "market")
        label = f"{prefix}_{int(batch_index):04d}"
        return ReflectionMarketConfig(
            label=label,
            num_zi_buy=num_zi_buy,
            num_zi_sell=num_zi_sell,
            timesteps=timesteps,
            q_max=q_max,
            pv_var=pv,
            shade_range=shade_range,
            fundamental_value=self.fundamental_value,
            mean_reversion_rate=self.mean_reversion_rate,
            shock_var=shock,
            obs_noise_var=obs,
            arrival_rate=arrival_rate,
            llm_order_type=self.llm_order_type,
            ood_tier=self.sample_ood_tier,
        )

    def sample_batch(self, rng: random.Random, batch_size: int) -> List[ReflectionMarketConfig]:
        size = max(1, int(batch_size))
        return [self.sample(rng, batch_index=i) for i in range(size)]


HOME_SET_IMBALANCE_STRATA: Tuple[str, ...] = (
    "buyer_heavy",
    "seller_heavy",
    "balanced",
)
HOME_SET_STRATA: Tuple[Tuple[str, str], ...] = (
    ("buyer_heavy", "tight"),
    ("seller_heavy", "wide"),
    ("balanced", "mid"),
)


def _pick_imbalance_counts(
    space: ReflectionConfigSpace,
    kind: str,
    rng: random.Random,
) -> Tuple[int, int]:
    buys = [int(x) for x in space.num_zi_buy]
    sells = [int(x) for x in space.num_zi_sell]
    if kind == "buyer_heavy":
        buy = max(buys)
        lighter = [s for s in sells if s < buy]
        sell = min(lighter) if lighter else min(sells)
        return buy, sell
    if kind == "seller_heavy":
        sell = max(sells)
        lighter = [b for b in buys if b < sell]
        buy = min(lighter) if lighter else min(buys)
        return buy, sell
    common = sorted(set(buys) & set(sells))
    if common:
        n = int(rng.choice(common))
        return n, n
    buy, sell = min(
        ((b, s) for b in buys for s in sells),
        key=lambda pair: abs(pair[0] - pair[1]),
    )
    return int(buy), int(sell)


def _pick_shade_range(
    space: ReflectionConfigSpace,
    kind: Optional[str] = None,
    *,
    rng: Optional[random.Random] = None,
    avoid_his: Optional[Set[float]] = None,
) -> List[float]:
    ranges = sorted(
        (list(r) for r in space.shade_ranges),
        key=lambda r: float(r[-1] if r else 0.0),
    )
    if not ranges:
        return [0.0, 0.015]
    if kind in {"tight", "wide", "mid"}:
        if kind == "tight":
            chosen = ranges[0]
        elif kind == "wide":
            chosen = ranges[-1]
        else:
            chosen = ranges[len(ranges) // 2]
        return list(chosen)
    sampler = rng if rng is not None else random.Random()
    blocked = set(avoid_his or ())
    preferred = [r for r in ranges if float(r[-1]) not in blocked]
    pool = preferred if preferred else ranges
    return list(sampler.choice(pool))


def sample_stratified_home_set(
    space: ReflectionConfigSpace,
    rng: random.Random,
    *,
    size: int,
    exclude: Optional[Sequence[ReflectionMarketConfig]] = None,
    label_offset: int = 0,
    strata_offset: int = 0,
) -> List[ReflectionMarketConfig]:
    n = max(1, int(size))
    offset = max(0, int(label_offset))
    s_off = int(strata_offset) % len(HOME_SET_IMBALANCE_STRATA)
    blocked: Set[Tuple[int, int, int, int, float, float, float, float]] = set()
    for cfg in exclude or []:
        blocked.add(cfg.ood_cell())
    chosen: List[ReflectionMarketConfig] = []
    used_shade_his: Set[float] = set()
    for i in range(n):
        imbalance = HOME_SET_IMBALANCE_STRATA[
            (i + s_off) % len(HOME_SET_IMBALANCE_STRATA)
        ]
        buy, sell = _pick_imbalance_counts(space, imbalance, rng)
        shade_range = _pick_shade_range(space, rng=rng, avoid_his=used_shade_his)
        cfg = None
        for _ in range(40):
            candidate = ReflectionMarketConfig(
                label=f"market_{i + offset:04d}",
                num_zi_buy=buy,
                num_zi_sell=sell,
                timesteps=int(rng.choice(space.timesteps)),
                q_max=int(rng.choice(space.q_max)),
                pv_var=space.pv_var,
                shade_range=list(shade_range),
                fundamental_value=space.fundamental_value,
                mean_reversion_rate=space.mean_reversion_rate,
                shock_var=space.shock_var,
                obs_noise_var=space.obs_noise_var,
                arrival_rate=float(rng.choice(space.arrival_rates)),
                llm_order_type=space.llm_order_type,
            )
            cell = candidate.ood_cell()
            if cell in blocked:
                shade_range = _pick_shade_range(
                    space, rng=rng, avoid_his=used_shade_his
                )
                continue
            cfg = candidate
            blocked.add(cell)
            used_shade_his.add(float(shade_range[-1]))
            break
        if cfg is None:
            cfg = ReflectionMarketConfig(
                label=f"market_{i + offset:04d}",
                num_zi_buy=buy,
                num_zi_sell=sell,
                timesteps=int(rng.choice(space.timesteps)),
                q_max=int(rng.choice(space.q_max)),
                pv_var=space.pv_var,
                shade_range=list(shade_range),
                fundamental_value=space.fundamental_value,
                mean_reversion_rate=space.mean_reversion_rate,
                shock_var=space.shock_var,
                obs_noise_var=space.obs_noise_var,
                arrival_rate=float(rng.choice(space.arrival_rates)),
                llm_order_type=space.llm_order_type,
            )
            used_shade_his.add(float(cfg.shade_range[-1]))
        chosen.append(cfg)
    return chosen


def build_default_config_space() -> ReflectionConfigSpace:
    return ReflectionConfigSpace(
        num_zi_buy=[5, 10, 12, 15],
        num_zi_sell=[5, 10, 12, 15],
        timesteps=[300, 400, 600],
        q_max=[8, 10, 12],
        shade_ranges=zip_market_shade_ranges(),
        arrival_rates=[0.1, 0.15],
    )


DEFAULT_Q_EVAL_MODE = "incremental"
ZI_Q_REWARD_MODES = frozenset({"zi_beat_fraction", "zi_advantage"})
DEFAULT_Q_REWARD_MODE = "advantage"
DEFAULT_Q_UPDATE_MODE = "replace"
FULL_REFRESH_PARALLEL_WORKERS = 60
Q_REFRESH_TOP_K = 30
Q_REFRESH_ALWAYS_RECENT = 3
Q_REFRESH_EPISODE_WORKERS = 2
DEFAULT_PARALLEL_EPISODE_WORKERS = 60
DEFAULT_ZI_BENCHMARK_SHADE_RANGES: List[List[float]] = [
    [0.0, 0.001],
    [0.0, 0.0015],
    [0.0, 0.002],
    [0.0, 0.003],
    [0.0, 0.004],
    [0.0, 0.006],
    [0.0, 0.008],
    [0.0, 0.012],
    [0.0, 0.016],
    [0.0, 0.024],
    [0.0, 0.032],
    [0.0, 0.048],
    [0.0, 0.064],
    [0.0, 0.096],
    [0.0, 0.128],
    [0.0, 0.192],
    [0.0, 0.256],
    [0.0, 0.384],
    [0.0, 0.512],
    [0.0, 0.756],
    [0.0, 1.0],
]
DEFAULT_ZI_BENCHMARK_COUNT = len(DEFAULT_ZI_BENCHMARK_SHADE_RANGES)
DEFAULT_ZIP_BENCHMARK_SHADE_RANGES: List[List[float]] = [
    [0.0, 0.001],
    [0.0, 0.0015],
    [0.0, 0.002],
    [0.0, 0.003],
    [0.0, 0.004],
    [0.0, 0.006],
    [0.0, 0.008],
    [0.0, 0.012],
]
DEFAULT_ZIP_BENCHMARK_COUNT = len(DEFAULT_ZIP_BENCHMARK_SHADE_RANGES)
DEFAULT_ZIP_LADDER_COUNT = DEFAULT_ZIP_BENCHMARK_COUNT + 1
ZIP_MARKET_SHADE_HI_RANKED: Tuple[float, ...] = (
    0.0015,
    0.002,
    0.003,
    0.001,
    0.004,
    0.006,
    0.008,
    0.012,
)
DEFAULT_ZIP_MARKET_SHADE_TOP_K = 5


def zip_market_shade_ranges(*, top_k: int = DEFAULT_ZIP_MARKET_SHADE_TOP_K) -> List[List[float]]:
    k = max(1, min(int(top_k), len(ZIP_MARKET_SHADE_HI_RANKED)))
    allowed = {float(r[-1]) for r in DEFAULT_ZIP_BENCHMARK_SHADE_RANGES}
    chosen = [h for h in ZIP_MARKET_SHADE_HI_RANKED[:k] if float(h) in allowed]
    if not chosen:
        chosen = [float(r[-1]) for r in DEFAULT_ZIP_BENCHMARK_SHADE_RANGES[:k]]
    return [[0.0, float(h)] for h in sorted(chosen)]


RESAMPLE_MARKET_SHADE_HIS: Tuple[float, ...] = (
    0.001,
    0.002,
    0.003,
    0.004,
    0.006,
    0.008,
    0.012,
    0.015,
    0.02,
    0.025,
)


def resample_market_shade_ranges() -> List[List[float]]:
    return [[0.0, float(h)] for h in RESAMPLE_MARKET_SHADE_HIS]


def train_support_spec() -> Dict[str, Any]:
    space = build_default_config_space()
    homeset_his = sorted({float(r[-1]) for r in zip_market_shade_ranges()})
    resample_his = sorted({float(h) for h in RESAMPLE_MARKET_SHADE_HIS})
    return {
        "axes": list(OOD_AXES),
        "num_zi_buy": list(space.num_zi_buy),
        "num_zi_sell": list(space.num_zi_sell),
        "timesteps": list(space.timesteps),
        "q_max": list(space.q_max),
        "arrival_rates": list(space.arrival_rates),
        "shade_hi_homeset": homeset_his,
        "shade_hi_resample": resample_his,
        "shade_hi_union": sorted(set(homeset_his) | set(resample_his)),
        "shock_var": [float(space.shock_var)],
        "obs_noise_var": [float(space.obs_noise_var)],
        "pv_var": [float(space.pv_var)],
        "fundamental_value": [float(space.fundamental_value)],
        "mean_reversion_rate": [float(space.mean_reversion_rate)],
    }


def outside_train_support_axes(cfg: ReflectionMarketConfig) -> List[str]:
    spec = train_support_spec()
    shade_ok = set(float(x) for x in spec["shade_hi_union"])
    hits: List[str] = []
    if int(cfg.num_zi_buy) not in {int(x) for x in spec["num_zi_buy"]}:
        hits.append("num_zi_buy")
    if int(cfg.num_zi_sell) not in {int(x) for x in spec["num_zi_sell"]}:
        hits.append("num_zi_sell")
    if int(cfg.timesteps) not in {int(x) for x in spec["timesteps"]}:
        hits.append("timesteps")
    if int(cfg.q_max) not in {int(x) for x in spec["q_max"]}:
        hits.append("q_max")
    if round(cfg.shade_hi(), 6) not in {round(float(x), 6) for x in shade_ok}:
        hits.append("shade_hi")
    if round(float(cfg.arrival_rate), 6) not in {
        round(float(x), 6) for x in spec["arrival_rates"]
    }:
        hits.append("arrival_rate")
    if round(float(cfg.shock_var), 6) not in {
        round(float(x), 6) for x in spec["shock_var"]
    }:
        hits.append("shock_var")
    if round(float(cfg.obs_noise_var), 6) not in {
        round(float(x), 6) for x in spec["obs_noise_var"]
    }:
        hits.append("obs_noise_var")
    return hits


def is_outside_train_support(cfg: ReflectionMarketConfig) -> bool:
    return bool(outside_train_support_axes(cfg))


def build_ood_holdout_config_space() -> ReflectionConfigSpace:
    spec = train_support_spec()
    train_buys = {int(x) for x in spec["num_zi_buy"]}
    train_sells = {int(x) for x in spec["num_zi_sell"]}
    train_t = {int(x) for x in spec["timesteps"]}
    train_q = {int(x) for x in spec["q_max"]}
    train_lam = {round(float(x), 6) for x in spec["arrival_rates"]}
    train_shade = {round(float(x), 6) for x in spec["shade_hi_union"]}
    train_shock = {round(float(x), 6) for x in spec["shock_var"]}
    train_obs = {round(float(x), 6) for x in spec["obs_noise_var"]}

    buys = [3, 4, 18, 20, 24]
    sells = [3, 4, 18, 20, 24]
    timesteps = [150, 200, 250, 700, 800, 1000]
    q_max = [4, 6, 14, 16, 20]
    arrival_rates = [0.05, 0.08, 0.2, 0.25, 0.3]
    shade_his = [0.005, 0.007, 0.01, 0.03, 0.04, 0.05]
    shock_vars = [5e5, 1e6]
    obs_noise_vars = [1e3, 1e9]

    def _assert_disjoint(name: str, values, blocked) -> None:
        overlap = [v for v in values if v in blocked]
        if overlap:
            raise ValueError(
                f"OOD holdout space {name} overlaps P_train: {overlap}"
            )

    _assert_disjoint("num_zi_buy", buys, train_buys)
    _assert_disjoint("num_zi_sell", sells, train_sells)
    _assert_disjoint("timesteps", timesteps, train_t)
    _assert_disjoint("q_max", q_max, train_q)
    _assert_disjoint(
        "arrival_rates",
        [round(float(x), 6) for x in arrival_rates],
        train_lam,
    )
    _assert_disjoint(
        "shade_hi",
        [round(float(x), 6) for x in shade_his],
        train_shade,
    )
    _assert_disjoint(
        "shock_vars",
        [round(float(x), 6) for x in shock_vars],
        train_shock,
    )
    _assert_disjoint(
        "obs_noise_vars",
        [round(float(x), 6) for x in obs_noise_vars],
        train_obs,
    )

    return ReflectionConfigSpace(
        num_zi_buy=buys,
        num_zi_sell=sells,
        timesteps=timesteps,
        q_max=q_max,
        shade_ranges=[[0.0, float(h)] for h in shade_his],
        arrival_rates=arrival_rates,
        shock_vars=shock_vars,
        obs_noise_vars=obs_noise_vars,
        shock_var=1e5,
        obs_noise_var=1e6,
        pv_var=1e5,
        sample_label_prefix="ood_ep",
        sample_ood_tier="sampled",
    )


def sample_ood_holdout_episode_configs(
    n: int,
    rng: random.Random,
    *,
    space: Optional[ReflectionConfigSpace] = None,
) -> List[ReflectionMarketConfig]:
    space = space or build_ood_holdout_config_space()
    n = max(1, int(n))
    out: List[ReflectionMarketConfig] = []
    for i in range(n):
        cfg = None
        for _ in range(40):
            candidate = space.sample(rng, batch_index=i)
            if is_outside_train_support(candidate):
                cfg = candidate
                break
        if cfg is None:
            raise RuntimeError(
                f"failed to sample support-OOD holdout config at index {i}"
            )
        out.append(cfg)
    return out


def build_id_holdout_config_space() -> ReflectionConfigSpace:
    space = build_default_config_space()
    spec = train_support_spec()
    return ReflectionConfigSpace(
        num_zi_buy=list(space.num_zi_buy),
        num_zi_sell=list(space.num_zi_sell),
        timesteps=list(space.timesteps),
        q_max=list(space.q_max),
        shade_ranges=[[0.0, float(h)] for h in spec["shade_hi_union"]],
        arrival_rates=list(space.arrival_rates),
        shock_var=float(space.shock_var),
        obs_noise_var=float(space.obs_noise_var),
        pv_var=float(space.pv_var),
        sample_label_prefix="id_ep",
        sample_ood_tier=None,
    )


def sample_id_holdout_episode_configs(
    n: int,
    rng: random.Random,
    *,
    space: Optional[ReflectionConfigSpace] = None,
) -> List[ReflectionMarketConfig]:
    space = space or build_id_holdout_config_space()
    n = max(1, int(n))
    out: List[ReflectionMarketConfig] = []
    for i in range(n):
        cfg = None
        for _ in range(40):
            candidate = space.sample(rng, batch_index=i)
            if not is_outside_train_support(candidate):
                cfg = candidate
                break
        if cfg is None:
            raise RuntimeError(
                f"failed to sample in-distribution holdout config at index {i}"
            )
        out.append(cfg)
    return out


DEFAULT_HBL_BENCHMARK_L_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
DEFAULT_HBL_BENCHMARK_COUNT = len(DEFAULT_HBL_BENCHMARK_L_VALUES)
DEFAULT_HBL_LADDER_COUNT = DEFAULT_HBL_BENCHMARK_COUNT


THOMPSON_PRIOR_ALPHA = 1.0
THOMPSON_PRIOR_BETA = 1.0
DEFAULT_LADDER_PARALLEL_WORKERS = 56
ZI_BENCHMARK_PARALLEL_WORKERS = max(
    DEFAULT_ZI_BENCHMARK_COUNT,
    DEFAULT_ZIP_LADDER_COUNT,
    DEFAULT_HBL_LADDER_COUNT,
    DEFAULT_LADDER_PARALLEL_WORKERS,
)

CURRICULUM_ZI_BENCHMARK_SHADE_RANGES: List[List[float]] = [
    [0.0, 0.005],
    [0.0, 0.01],
    [0.0, 0.02],
    [0.0, 0.04],
    [0.0, 0.08],
]
CURRICULUM_ZI_BENCHMARK_COUNT = len(CURRICULUM_ZI_BENCHMARK_SHADE_RANGES)

CRITIC_MODES = frozenset(
    {
        "default",
        "structured_zi_curriculum",
        "structured_principle_form",
        "structured_hypothesis_loop",
    }
)
DEFAULT_CRITIC_MODE = "default"

PARENT_SELECTION_MODES = frozenset({"auto", "ucb", "newest"})
DEFAULT_PARENT_SELECTION_MODE = "ucb"

INACTIVE_GATE_MIN_NODES = 10
INACTIVE_SURPLUS_GAP = 300.0
INACTIVE_BEAT_GAP = 3




def build_default_config_pool() -> List[ReflectionMarketConfig]:
    train_shock = 1e5
    train_obs = 1e6
    high_shock = 1e6
    low_obs = 1e3
    high_obs = 1e9
    pool = [
        ReflectionMarketConfig(
            label="ood_near_thick_slow",
            ood_tier="near",
            num_zi_buy=20,
            num_zi_sell=20,
            timesteps=800,
            q_max=16,
            pv_var=1e5,
            shade_range=[0.0, 0.03],
            arrival_rate=0.05,
            shock_var=train_shock,
            obs_noise_var=train_obs,
        ),
        ReflectionMarketConfig(
            label="ood_near_thin_fast",
            ood_tier="near",
            num_zi_buy=3,
            num_zi_sell=3,
            timesteps=200,
            q_max=6,
            pv_var=1e5,
            shade_range=[0.0, 0.005],
            arrival_rate=0.25,
            shock_var=train_shock,
            obs_noise_var=train_obs,
        ),
        ReflectionMarketConfig(
            label="ood_info_highshock_lownoise",
            ood_tier="info",
            num_zi_buy=10,
            num_zi_sell=10,
            timesteps=400,
            q_max=10,
            pv_var=1e5,
            shade_range=[0.0, 0.003],
            arrival_rate=0.1,
            shock_var=high_shock,
            obs_noise_var=low_obs,
        ),
        ReflectionMarketConfig(
            label="ood_info_lowshock_highnoise",
            ood_tier="info",
            num_zi_buy=10,
            num_zi_sell=10,
            timesteps=400,
            q_max=10,
            pv_var=1e5,
            shade_range=[0.0, 0.003],
            arrival_rate=0.1,
            shock_var=train_shock,
            obs_noise_var=high_obs,
        ),
        ReflectionMarketConfig(
            label="ood_compound_thick_highshock",
            ood_tier="compound",
            num_zi_buy=20,
            num_zi_sell=20,
            timesteps=800,
            q_max=16,
            pv_var=1e5,
            shade_range=[0.0, 0.03],
            arrival_rate=0.05,
            shock_var=high_shock,
            obs_noise_var=low_obs,
        ),
        ReflectionMarketConfig(
            label="ood_compound_thin_highnoise",
            ood_tier="compound",
            num_zi_buy=3,
            num_zi_sell=3,
            timesteps=200,
            q_max=6,
            pv_var=1e5,
            shade_range=[0.0, 0.005],
            arrival_rate=0.25,
            shock_var=train_shock,
            obs_noise_var=high_obs,
        ),
    ]
    _validate_ood_holdout_pool(pool)
    return pool


def _validate_ood_holdout_pool(pool: Sequence[ReflectionMarketConfig]) -> None:
    if len(pool) != 6:
        raise ValueError(f"holdout pool must have 6 markets, got {len(pool)}")
    tiers = [str(c.ood_tier or "") for c in pool]
    for tier in HOLDOUT_OOD_TIERS:
        if tiers.count(tier) != 2:
            raise ValueError(
                f"holdout pool needs exactly 2 markets with ood_tier={tier!r}, "
                f"got {tiers.count(tier)}"
            )
    cells = [c.ood_cell() for c in pool]
    if len(set(cells)) != len(cells):
        raise ValueError("holdout pool ood_cells must be unique")
    for cfg in pool:
        axes = outside_train_support_axes(cfg)
        if not axes:
            raise ValueError(
                f"holdout {cfg.label!r} is inside P_train support on all A_ood axes"
            )
        if cfg.ood_tier == "near":
            info_hits = [a for a in axes if a in ("shock_var", "obs_noise_var")]
            if info_hits:
                raise ValueError(
                    f"near-OOD {cfg.label!r} must keep train shock/obs; "
                    f"shifted {info_hits}"
                )
            if not any(
                a in axes
                for a in (
                    "num_zi_buy",
                    "num_zi_sell",
                    "timesteps",
                    "q_max",
                    "shade_hi",
                    "arrival_rate",
                )
            ):
                raise ValueError(
                    f"near-OOD {cfg.label!r} must shift a structural axis"
                )
        if cfg.ood_tier == "info":
            if not any(a in axes for a in ("shock_var", "obs_noise_var")):
                raise ValueError(
                    f"info-OOD {cfg.label!r} must shift shock_var or obs_noise_var"
                )
        if cfg.ood_tier == "compound":
            struct = any(
                a in axes
                for a in (
                    "num_zi_buy",
                    "num_zi_sell",
                    "timesteps",
                    "q_max",
                    "shade_hi",
                    "arrival_rate",
                )
            )
            info = any(a in axes for a in ("shock_var", "obs_noise_var"))
            if not (struct and info):
                raise ValueError(
                    f"compound-OOD {cfg.label!r} must shift both structural "
                    f"and info axes; got {axes}"
                )


