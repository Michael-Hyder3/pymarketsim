from __future__ import annotations

from typing import Optional

PROMPT_PROFILES = ("standard",)
DEFAULT_PROMPT_PROFILE = "standard"

_ACTIVE_PROFILE = DEFAULT_PROMPT_PROFILE


def activate_prompt_profile(profile: Optional[str] = None) -> str:
    """Activate the prompt stack."""
    global _ACTIVE_PROFILE
    name = (profile or DEFAULT_PROMPT_PROFILE).strip().lower()
    if name not in PROMPT_PROFILES:
        raise ValueError(
            f"Unknown prompt profile {profile!r}; expected one of {PROMPT_PROFILES}"
        )

    import marketsim.LLM.reflection_pipeline.pipeline as pipeline_mod

    pipeline_mod._SLIM_PROMPT_PROFILE_ACTIVE = False
    _ACTIVE_PROFILE = "standard"
    return _ACTIVE_PROFILE


def slim_prompt_profile_active() -> bool:
    return _ACTIVE_PROFILE == "slim"
