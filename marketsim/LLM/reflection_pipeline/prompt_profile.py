from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Optional

PROMPT_PROFILES = ("standard", "slim")
DEFAULT_PROMPT_PROFILE = "standard"

_PROFILES_DIR = Path(__file__).resolve().parent / "prompt_profiles"
_ACTIVE_PROFILE = DEFAULT_PROMPT_PROFILE

_SYSTEM_PROMPT_ATTRS = (
    "CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM",
    "CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_LOB",
    "CRITIC_SYSTEM_PROMPT_STRUCTURED_PRINCIPLE_FORM_SPREAD",
)








def activate_prompt_profile(profile: str) -> str:
    global _ACTIVE_PROFILE
    name = (profile or DEFAULT_PROMPT_PROFILE).strip().lower()
    if name != "standard":
        raise ValueError(
            f"Unknown prompt profile {profile!r}; expected one of {PROMPT_PROFILES}"
        )

    import marketsim.LLM.reflection_pipeline.pipeline as pipeline_mod

    pipeline_mod._SLIM_PROMPT_PROFILE_ACTIVE = False
    _ACTIVE_PROFILE = "standard"
    return _ACTIVE_PROFILE


def slim_prompt_profile_active() -> bool:
    return _ACTIVE_PROFILE == "slim"
