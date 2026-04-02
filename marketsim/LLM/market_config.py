"""
MarketConfig: defines a single RL "state" for the multi-state feedback loop.

In the RL framing:
  State  s  = a MarketConfig (market parameters)
  Action a  = a strategy (Python function / code string)
  Reward R  = surplus earned by that strategy in the diagnostic run for state s

The LLM acts as the policy π(a | s, context) — given the state description and
the diagnostic trajectory JSON, it proposes a new strategy (action) to try next.

The value table V[s] approximates V(s) = best surplus seen so far in state s,
allowing cross-state comparison and meta-learning ("which strategy families
work in small vs large markets?").
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class MarketConfig:
    """
    A market configuration — one "state" in the multi-state RL problem.

    Fields match the keyword arguments of episode_runner.run_episode() so a
    config can be unpacked directly into a run_episode call.

    Args:
        num_zi_buy:   Number of ZI buy agents competing alongside the LLM agent.
        num_zi_sell:  Number of ZI sell agents.
        timesteps:    Simulation length (number of discrete time steps).
        q_max:        Maximum units any single agent can trade.
        pv_var:       Private value variance (controls how spread out valuations are).
        shade_range:  [lo, hi] — ZI agents shade uniformly in this range (RELATIVE units).
            fixed_private_values_relative:
                     Optional fixed private-value schedule for the LLM agent.
                     When provided, this schedule is part of the RL state.
        label:        Human-readable name for this config (used in logs and prompts).
    """
    num_zi_buy: int = 5
    num_zi_sell: int = 5
    timesteps: int = 1000
    q_max: int = 10
    pv_var: float = 1e5
    shade_range: List[float] = field(default_factory=lambda: [0.0, 500.0])
    fixed_private_values_relative: Optional[List[float]] = None
    label: str = ""
    fixed_seed: Optional[int] = None  # if set, ALL episodes of this config use this seed

    # ------------------------------------------------------------------ #
    #  Identity / hashing                                                   #
    # ------------------------------------------------------------------ #

    def config_hash(self) -> str:
        """
        Deterministic 8-character hex hash of the config parameters (excluding label).
        Used as the key in the value table.
        """
        key = json.dumps({
            "num_zi_buy": self.num_zi_buy,
            "num_zi_sell": self.num_zi_sell,
            "timesteps": self.timesteps,
            "q_max": self.q_max,
            "pv_var": self.pv_var,
            "shade_range": self.shade_range,
            "fixed_private_values_relative": self.fixed_private_values_relative,
        }, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:8]

    def diagnostic_seed(self, episode: int) -> int:
        """
        Seed for the diagnostic run of this config at a given episode.

        If fixed_seed is set, that value is returned for every episode —
        meaning all runs of this config face the exact same market draw,
        making surplus directly comparable across episodes.

        Otherwise, a per-episode seed is derived from the episode number
        and a config-specific offset, so different episodes see different
        market draws (but are still fully reproducible).

        Formula (default): (episode * 10_000) + (int(config_hash, 16) % 10_000)
        """
        if self.fixed_seed is not None:
            return self.fixed_seed
        offset = int(self.config_hash(), 16) % 10_000
        return episode * 10_000 + offset

    def to_dict(self) -> dict:
        """Serialise to plain dict (JSON-safe) — passed to LLM as raw JSON.

        Only includes observable market structure: the LLM should not see
        opponent private value distributions, shade ranges, or agent counts.
        """
        return {
            "label": self.label or self.config_hash(),
            "timesteps": self.timesteps,
            "q_max": self.q_max,
            "agent_private_values_relative": self.fixed_private_values_relative,
        }

    def __str__(self) -> str:
        lbl = self.label or self.config_hash()
        return (
            f"MarketConfig({lbl}: zi={self.num_zi_buy}B/{self.num_zi_sell}S, "
            f"T={self.timesteps}, q={self.q_max}, pv_var={self.pv_var:.0e}, "
            f"shade=[{self.shade_range[0]:.0f},{self.shade_range[1]:.0f}], "
            f"pv_set={'fixed' if self.fixed_private_values_relative else 'drawn'})"
        )


# ------------------------------------------------------------------ #
#  Value table                                                          #
# ------------------------------------------------------------------ #

class ValueTable:
    """
    V(s) approximation: tracks the best surplus seen per market config.

    This is the "value function" in our RL framing:
        V(s) ≈ max_{a seen so far} R(s, a)

    Also stores the best strategy code and its episode number per config,
    so the LLM can be shown what worked best in each state.

    The value table is passed as raw JSON to the LLM so it can reason
    across states: "In large markets (20 agents) shade=200 worked best;
    in small markets (5 agents) shade=80 worked best."
    """

    def __init__(self):
        # config_hash -> {"best_surplus": float, "best_code": str, "best_episode": int,
        #                  "history": [{"episode": int, "surplus": float, "code_snippet": str}]}
        self._table: dict = {}

    def update(self, config: MarketConfig, episode: int, surplus: float, code: str) -> bool:
        """
        Update the value for this config. Returns True if this is a new best.
        """
        h = config.config_hash()
        is_new_best = False
        if h not in self._table:
            self._table[h] = {
                "config": config.to_dict(),
                "best_surplus": surplus,
                "best_code": code,
                "best_episode": episode,
                "history": [],
            }
            is_new_best = True
        else:
            if surplus > self._table[h]["best_surplus"]:
                self._table[h]["best_surplus"] = surplus
                self._table[h]["best_code"] = code
                self._table[h]["best_episode"] = episode
                is_new_best = True

        snippet = "\n".join(code.strip().splitlines()[:6])
        self._table[h]["history"].append({
            "episode": episode,
            "surplus": round(surplus, 4),
            "code_snippet": snippet,
        })
        return is_new_best

    def best_for_config(self, config: MarketConfig) -> Optional[dict]:
        """Return best entry for a config, or None if unseen."""
        return self._table.get(config.config_hash())

    def to_json(self, max_history: int = 3) -> str:
        """
        Serialise to compact JSON string for the LLM prompt.
        Trims history to the last max_history entries per config.
        Only exposes observable market structure — no ZI internals.
        """
        trimmed = {}
        for h, entry in self._table.items():
            cfg = entry["config"]
            observable_cfg = {
                k: cfg[k] for k in ("label", "timesteps", "q_max")
                if k in cfg
            }
            trimmed[h] = {
                "config": observable_cfg,
                "best_surplus": entry["best_surplus"],
                "history": entry["history"][-max_history:],
            }
        return json.dumps(trimmed, separators=(",", ":"))

    def summary_table(self) -> str:
        """Human-readable table for terminal logging."""
        if not self._table:
            return "  (empty)"
        lines = [f"  {'Config':>10}  {'BestSurplus':>12}  {'BestEp':>7}  Label"]
        for h, e in self._table.items():
            lbl = e["config"].get("label", h)
            lines.append(
                f"  {h:>10}  {e['best_surplus']:>12.2f}  {e['best_episode']:>7}  {lbl}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._table)
