from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PipelineRecipe:
    recipe_id: str
    description: str
    init_types: List[str]
    q_reward_mode: str
    parent_selection_mode: str
    episodes_per_config: int = 50
    batch_size: int = 3
    steps: int = 125
    critic_enabled: bool = True
    critic_include_data: bool = True
    critic_per_config_metrics: bool = False
    generator_include_critic_data: Optional[bool] = None
    generator_include_sibling: Optional[bool] = None
    q_eval_mode: str = "incremental"
    q_update_mode: str = "replace"
    strategy_output_mode: str = "x_cap"
    reward_mode: str = "advantage"
    ucb_constant: float = 3.0
    inactive_gate_enabled: bool = True
    critic_mode: str = "default"
    forbid_numpy_random: bool = False



SIMPLE_PV_ZI_ADVANTAGE_UCB_PRINCIPLE = PipelineRecipe(
    recipe_id="simple_pv_zi_advantage_ucb_principle",
    description=(
        "Same as simple_pv_zi_advantage_ucb with structured_principle_form critic: "
        "DIAGNOSIS picks principle + form_class for diverse PV→x_cap maps; "
        "advantage is Q; ZI beats stay soft diagnostics only. "
        "Default pipeline plateau escape + elite parent bias enabled."
    ),
    init_types=["pv"],
    q_reward_mode="zi_advantage",
    parent_selection_mode="ucb",
    generator_include_critic_data=True,
    generator_include_sibling=False,
    q_eval_mode="full_refresh",
    q_update_mode="average",
    inactive_gate_enabled=False,
    critic_mode="structured_principle_form",
)

SIMPLE_LOB_ZI_ADVANTAGE_UCB_PRINCIPLE = PipelineRecipe(
    recipe_id="simple_lob_zi_advantage_ucb_principle",
    description=(
        "LOB-only structured_principle_form critic: DIAGNOSIS picks LOB principle + "
        "form_class for diverse market_history→shade maps; advantage is Q; "
        "ZI beats stay soft diagnostics only. Same UCB / full-refresh / plateau "
        "escape + elite parent bias defaults as the PV principle recipe."
    ),
    init_types=["lob"],
    q_reward_mode="zi_advantage",
    parent_selection_mode="ucb",
    generator_include_critic_data=True,
    generator_include_sibling=False,
    q_eval_mode="full_refresh",
    q_update_mode="average",
    inactive_gate_enabled=False,
    critic_mode="structured_principle_form",
)

SIMPLE_SPREAD_ZI_ADVANTAGE_UCB_PRINCIPLE = PipelineRecipe(
    recipe_id="simple_spread_zi_advantage_ucb_principle",
    description=(
        "Spread-only structured_principle_form critic: DIAGNOSIS picks Spread "
        "principle + form_class for diverse market_state→shade maps; advantage is Q; "
        "ZI beats stay soft diagnostics only. Same UCB / full-refresh / plateau "
        "escape + elite parent bias defaults as the PV / LOB principle recipes. "
        "Use with --spread-input-mode book for the gold o_Sp observation."
    ),
    init_types=["spread"],
    q_reward_mode="zi_advantage",
    parent_selection_mode="ucb",
    generator_include_critic_data=True,
    generator_include_sibling=False,
    q_eval_mode="full_refresh",
    q_update_mode="average",
    inactive_gate_enabled=False,
    critic_mode="structured_principle_form",
)


DEFAULT_SIMPLE_RECIPE = SIMPLE_PV_ZI_ADVANTAGE_UCB_PRINCIPLE

RECIPE_REGISTRY: Dict[str, PipelineRecipe] = {
    recipe.recipe_id: recipe
    for recipe in (
        SIMPLE_PV_ZI_ADVANTAGE_UCB_PRINCIPLE,
        SIMPLE_LOB_ZI_ADVANTAGE_UCB_PRINCIPLE,
        SIMPLE_SPREAD_ZI_ADVANTAGE_UCB_PRINCIPLE,
    )
}


def list_recipes() -> List[str]:
    return sorted(RECIPE_REGISTRY.keys())


def get_recipe(recipe_id: str) -> PipelineRecipe:
    key = str(recipe_id).strip()
    if key not in RECIPE_REGISTRY:
        known = ", ".join(list_recipes())
        raise KeyError(f"Unknown recipe '{key}'. Known: {known}")
    return RECIPE_REGISTRY[key]
