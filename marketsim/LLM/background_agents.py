from __future__ import annotations

from typing import Optional

BACKGROUND_AGENT_KINDS = (
    "zi",
    "zip",
    "zip_noisy",
    "zip_fixed",
    "hbl",
    "hbl_fast",
    "llm",
)
DEFAULT_BACKGROUND_AGENT_KIND = "zi"
DEFAULT_ZIP_FIXED_COMPETITIVE_MU = 0.002


def normalize_background_agent_kind(kind: Optional[str]) -> str:
    resolved = str(kind or DEFAULT_BACKGROUND_AGENT_KIND).strip().lower()
    if resolved not in BACKGROUND_AGENT_KINDS:
        known = ", ".join(BACKGROUND_AGENT_KINDS)
        raise ValueError(
            f"Unsupported background_agent_kind {kind!r}. Known: {known}"
        )
    return resolved










