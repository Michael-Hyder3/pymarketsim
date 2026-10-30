from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from marketsim.LLM.reflection_pipeline.config import ReflectionMarketConfig


def _config_to_dict(config: ReflectionMarketConfig) -> Dict[str, Any]:
    return asdict(config)


def _config_from_dict(payload: Dict[str, Any]) -> ReflectionMarketConfig:
    return ReflectionMarketConfig(**payload)


@dataclass
class FixedEvalSuite:
    configs: List[ReflectionMarketConfig]
    episode_seeds: List[List[int]]
    suite_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "configs": [_config_to_dict(cfg) for cfg in self.configs],
            "episode_seeds": self.episode_seeds,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FixedEvalSuite":
        configs = [_config_from_dict(cfg) for cfg in payload["configs"]]
        return cls(
            configs=configs,
            episode_seeds=payload["episode_seeds"],
            suite_id=str(payload["suite_id"]),
        )


def _suite_id(configs: List[ReflectionMarketConfig], episode_seeds: List[List[int]]) -> str:
    payload = {
        "configs": [cfg.observable_state() for cfg in configs],
        "episode_seeds": episode_seeds,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def build_fixed_eval_suite(
    *,
    configs: List[ReflectionMarketConfig],
    episodes_per_config: int,
    base_seed: int,
    step_index: int,
) -> FixedEvalSuite:
    episode_seeds: List[List[int]] = []

    for config_index, config in enumerate(configs):
        config_seeds: List[int] = []
        for episode_index in range(episodes_per_config):
            config_seeds.append(
                config.seed_for(
                    base_seed=base_seed,
                    step=step_index,
                    config_index=config_index,
                    episode_index=episode_index,
                )
            )
        episode_seeds.append(config_seeds)

    return FixedEvalSuite(
        configs=list(configs),
        episode_seeds=episode_seeds,
        suite_id=_suite_id(configs, episode_seeds),
    )
