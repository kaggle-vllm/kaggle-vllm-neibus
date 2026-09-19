from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    nebius_api_key: str | None
    nebius_base_url: str
    nebius_model: str
    max_agent_turns: int


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        nebius_api_key=os.getenv("NEBIUS_API_KEY") or None,
        nebius_base_url=os.getenv(
            "NEBIUS_BASE_URL",
            "https://api.tokenfactory.nebius.com/v1/",
        ),
        nebius_model=os.getenv(
            "NEBIUS_MODEL",
            "nvidia/Nemotron-3_5-Lightning",
        ),
        max_agent_turns=int(os.getenv("NEBIUS_MAX_AGENT_TURNS", "8")),
    )
