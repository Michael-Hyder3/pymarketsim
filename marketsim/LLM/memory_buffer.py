"""
Rolling memory buffer for meta-learning in feedback loop.

Maintains B^(k) = {(σ_i^(j), U_i^(j))}_{j=k-N}^{k}

Stores recent strategies and their utilities, enabling the LLM to learn
patterns in high-performing strategies.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class EpisodeRecord:
    """Single episode record: strategy code and utility."""
    episode_number: int
    strategy_code: str
    utility: float  # Realized surplus
    units_executed: int
    execution_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationEntry:
    """An entry in the per-config scored population (FunSearch island)."""
    code: str
    avg_surplus: float
    family_tag: str   # which exploration family produced this
    episode: int


class ScoredPopulation:
    """
    Per-config top-K program population (FunSearch-style island).

    Maintains the best `max_size` strategies ever seen for a given config,
    ranked by avg_surplus.  Supports diverse sampling so recombination
    prompts draw from different strategy families.
    """

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._entries: List[PopulationEntry] = []

    def add(self, code: str, avg_surplus: float, family_tag: str, episode: int) -> None:
        """Add a strategy; drop the worst if over capacity."""
        self._entries.append(PopulationEntry(code, avg_surplus, family_tag, episode))
        self._entries.sort(key=lambda e: e.avg_surplus, reverse=True)
        if len(self._entries) > self.max_size:
            self._entries = self._entries[: self.max_size]

    def top_k(self, k: int = 3) -> List[PopulationEntry]:
        """Return the top k entries by surplus."""
        return self._entries[:min(k, len(self._entries))]

    def diverse_sample(self, k: int = 3) -> List[PopulationEntry]:
        """
        Return up to k entries chosen to maximise family diversity.
        Always includes the #1 entry; fills remaining slots by picking
        the best entry from each unseen family in turn.
        """
        if not self._entries:
            return []
        selected: List[PopulationEntry] = [self._entries[0]]
        seen_families = {self._entries[0].family_tag}
        for entry in self._entries[1:]:
            if len(selected) >= k:
                break
            if entry.family_tag not in seen_families:
                selected.append(entry)
                seen_families.add(entry.family_tag)
        # pad with top entries if not enough diverse ones
        for entry in self._entries:
            if len(selected) >= k:
                break
            if entry not in selected:
                selected.append(entry)
        return selected

    def surplus_history_ascending(self) -> List[Tuple[float, str]]:
        """Return (avg_surplus, family_tag) pairs sorted ascending (for OPRO prompt)."""
        return [(e.avg_surplus, e.family_tag) for e in sorted(self._entries, key=lambda e: e.avg_surplus)]

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)


class FamilyBandit:
    """
    UCB1 multi-armed bandit over strategy exploration families.

    Each family is an arm.  After each episode the winning family's arm
    is updated with the observed surplus.  UCB1 ranks arms for the next
    episode's exploration slots, balancing exploitation of high-reward
    families with exploration of under-tried ones.

    UCB score:  Q(f) + c * sqrt(ln N / n_f)
    where Q(f) = avg surplus for family f, N = total pulls, n_f = pulls of f.
    """

    def __init__(self, families: List[str], c: float = 1.0):
        self.families = families
        self.c = c
        self._counts: Dict[str, int] = {f: 0 for f in families}
        self._totals: Dict[str, float] = {f: 0.0 for f in families}
        self._total_pulls: int = 0

    def update(self, family_tag: str, surplus: float) -> None:
        """Record a reward for the given family arm."""
        if family_tag not in self._counts:
            self._counts[family_tag] = 0
            self._totals[family_tag] = 0.0
        self._counts[family_tag] += 1
        self._totals[family_tag] += surplus
        self._total_pulls += 1

    def ranked_families(self) -> List[str]:
        """
        Return families ranked by UCB1 score (descending).
        Untried families always score +inf so they get priority first.
        """
        scores: List[Tuple[float, str]] = []
        for f in self.families:
            n = self._counts.get(f, 0)
            if n == 0 or self._total_pulls == 0:
                scores.append((float('inf'), f))
            else:
                q = self._totals[f] / n
                bonus = self.c * math.sqrt(math.log(self._total_pulls) / n)
                scores.append((q + bonus, f))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scores]

    def summary(self) -> Dict[str, Any]:
        """Return per-family stats for logging."""
        return {
            f: {"pulls": self._counts.get(f, 0),
                "avg_surplus": self._totals.get(f, 0.0) / max(1, self._counts.get(f, 0))}
            for f in self.families
        }


class RollingMemoryBuffer:
    """
    Maintains a rolling buffer of recent episodes for meta-learning.
    
    Stores up to N episodes with their strategies and utilities.
    Provides methods to:
    - Add new episodes
    - Query high-performing strategies
    - Generate meta-learning prompts
    """
    
    def __init__(self, buffer_size: int = 5):
        """
        Args:
            buffer_size: Maximum number of episodes to retain (N in paper)
        """
        self.buffer_size = buffer_size
        self.buffer: deque = deque(maxlen=buffer_size)
        self.total_episodes = 0
    
    def add_episode(
        self,
        episode_number: int,
        strategy_code: str,
        utility: float,
        units_executed: int,
        execution_ratio: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a new episode record to the buffer.
        
        Args:
            episode_number: Episode index k
            strategy_code: The strategy function source code
            utility: Realized surplus from this episode
            units_executed: Number of units traded
            execution_ratio: Units executed / units available
            metadata: Optional additional data
        """
        record = EpisodeRecord(
            episode_number=episode_number,
            strategy_code=strategy_code,
            utility=utility,
            units_executed=units_executed,
            execution_ratio=execution_ratio,
            metadata=metadata or {},
        )
        self.buffer.append(record)
        self.total_episodes += 1
    
    def get_buffer(self) -> List[EpisodeRecord]:
        """Return all records currently in buffer."""
        return list(self.buffer)
    
    def get_top_k_strategies(self, k: int = 3) -> List[EpisodeRecord]:
        """
        Return top k strategies by utility.
        
        Args:
            k: Number of top strategies to return
        
        Returns:
            List of top k records sorted by utility (descending)
        """
        sorted_buffer = sorted(self.buffer, key=lambda x: x.utility, reverse=True)
        return sorted_buffer[:min(k, len(sorted_buffer))]
    
    def get_utility_trend(self) -> Tuple[List[int], List[float]]:
        """
        Get utility trend over time in buffer.
        
        Returns:
            (episode_numbers, utilities) for plotting/analysis
        """
        records = sorted(self.buffer, key=lambda x: x.episode_number)
        episodes = [r.episode_number for r in records]
        utilities = [r.utility for r in records]
        return episodes, utilities
    
    def get_average_utility(self) -> float:
        """Return average utility across buffer."""
        if not self.buffer:
            return 0.0
        return sum(r.utility for r in self.buffer) / len(self.buffer)
    
    def get_best_utility(self) -> float:
        """Return best utility seen in buffer."""
        if not self.buffer:
            return 0.0
        return max(r.utility for r in self.buffer)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


def generate_meta_learning_prompt(
    buffer: RollingMemoryBuffer,
    offline_data: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
) -> str:
    """
    Generate a meta-learning prompt for next strategy generation.
    
    This prompt:
    1. Shows the agent recent episode results
    2. Highlights high-performing strategy patterns
    3. Identifies execution gaps or inefficiencies
    4. Instructs LLM to synthesize improved strategy
    
    Args:
        buffer: Rolling memory with recent episodes
        offline_data: Optional offline market trajectory data
        top_k: Number of top strategies to highlight
    
    Returns:
        Prompt string for LLM code generation
    """
    
    # Get analytics
    avg_utility = buffer.get_average_utility()
    best_utility = buffer.get_best_utility()
    episodes, utilities = buffer.get_utility_trend()
    top_strategies = buffer.get_top_k_strategies(k=top_k)
    
    prompt = f"""You are an AI trading agent optimizing your bidding strategy through empirical learning.

## Recent Performance (Last {len(buffer)} Episodes)

- Average Surplus: ${avg_utility:.2f}
- Best Surplus: ${best_utility:.2f}
- Utility Trend: {utilities}

## Top Performing Strategies

"""
    
    for i, record in enumerate(top_strategies, 1):
        execution_pct = record.execution_ratio * 100
        prompt += f"""
### Strategy #{i} (Episode {record.episode_number})
**Realized Surplus: ${record.utility:.2f}**
**Execution Rate: {execution_pct:.1f}% ({record.units_executed} units)**

```python
{record.strategy_code}
```

"""
    
    # Add analysis
    if buffer.get_utility_trend()[0]:  # If we have history
        trend = utilities[-1] - utilities[0] if len(utilities) > 1 else 0
        if trend > 0:
            trend_str = "IMPROVING - continue refining successful patterns"
        elif trend < 0:
            trend_str = "DECLINING - revisit strategies from earlier episodes"
        else:
            trend_str = "STABLE - explore new approaches"
        
        prompt += f"""
## Performance Analysis
- Trend: {trend_str}
- Best Strategy: Episode {top_strategies[0].episode_number if top_strategies else 'N/A'}

"""
    
    # Offline data context (if provided)
    if offline_data:
        num_transactions = offline_data.get("total_transactions", 0)
        num_timesteps = offline_data.get("timesteps", 0)
        prompt += f"""
## Market Context (Offline Data)
- Historical Transactions: {num_transactions}
- Duration: {num_timesteps} timesteps
"""
    
    prompt += """
## Task for Next Episode

Based on the patterns observed above, generate an improved Python function for the next episode:

```python
def agent_strategy(private_values, market_history, market_state):
    \"\"\"
    Generate optimal bid/ask prices given market conditions.
    
    Args:
        private_values: List[float] - your valuations (highest to lowest)
        market_history: Dict - recent market state
        market_state: Dict - current best_bid, best_ask, units_available
    
    Returns:
        (price: float, quantity: int) or None to skip
    \"\"\"
    # Your implementation here
    pass
```

Key considerations:
1. Maximize execution (sell as many units as possible)
2. Maximize surplus (achieve best prices)
3. Adapt to market conditions shown in market_history
4. Respect constraints: buyer bid <= valuation, seller ask >= cost
"""
    
    return prompt


def analyze_strategy_patterns(
    buffer: RollingMemoryBuffer,
) -> Dict[str, Any]:
    """
    Analyze patterns in high-performing strategies.
    
    Returns:
        Dict with analysis of common patterns, techniques, execution strategies
    """
    top_strategies = buffer.get_top_k_strategies(k=3)
    
    analysis = {
        "num_strategies_analyzed": len(top_strategies),
        "avg_utility_top_k": sum(s.utility for s in top_strategies) / max(1, len(top_strategies)),
        "execution_rates": [s.execution_ratio for s in top_strategies],
        "avg_execution_rate": sum(s.execution_ratio for s in top_strategies) / max(1, len(top_strategies)),
        "utility_range": (buffer.get_best_utility() - min((s.utility for s in buffer.get_buffer()), default=0),),
        "buffer_capacity_used": len(buffer) / buffer.buffer_size,
    }
    
    return analysis
