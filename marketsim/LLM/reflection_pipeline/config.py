from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from marketsim.fourheap.constants import BUY


@dataclass
class ReflectionMarketConfig:
    label: str
    num_zi_buy: int = 5
    num_zi_sell: int = 5
    timesteps: int = 500
    q_max: int = 10
    pv_var: float = 1e5
    shade_range: List[float] = field(default_factory=lambda: [0.0, 1.0])
    fundamental_value: float = 1000.0
    mean_reversion_rate: float = 0.05
    shock_var: float = 1e5
    obs_noise_var: float = 1e6
    arrival_rate: float = 0.1
    llm_order_type: int = BUY
    fixed_seed: Optional[int] = None
    fixed_private_values_relative: Optional[List[float]] = None

    def observable_state(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "timesteps": self.timesteps,
            "q_max": self.q_max,
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


def build_default_config_pool() -> List[ReflectionMarketConfig]:
    return [
        ReflectionMarketConfig(
            label="reflection_baseline",
            timesteps=400,
            q_max=10,
            pv_var=1e5,
            num_zi_buy=5,
            num_zi_sell=5,
            shade_range=[0.0, 1.0],
            arrival_rate=0.1,
        ),
        ReflectionMarketConfig(
            label="reflection_dense",
            timesteps=600,
            q_max=10,
            pv_var=1e5,
            num_zi_buy=10,
            num_zi_sell=10,
            shade_range=[0.0, 1.0],
            arrival_rate=0.1,
        ),
        ReflectionMarketConfig(
            label="reflection_short",
            timesteps=300,
            q_max=10,
            pv_var=1e5,
            num_zi_buy=5,
            num_zi_sell=5,
            shade_range=[0.0, 0.6],
            arrival_rate=0.15,
        ),
    ]
