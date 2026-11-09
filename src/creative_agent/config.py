"""Application-wide configuration.

Settings are read from environment variables (and a local ``.env`` file) once at import time and exposed as the
module-level :data:`settings` singleton, which the rest of the package imports.
"""
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Populate os.environ from .env so that third-party SDKs (e.g. the Anthropic/OpenAI clients used through LangChain)
# can also see their API keys, not just the fields declared on Settings below.
load_dotenv()


class Settings(BaseSettings):
    """Typed runtime settings. Field names map to upper-case env vars (``model`` -> ``MODEL``)."""

    # extra="ignore": the .env file also holds provider API keys that are not declared as fields here.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model: str                                   # "provider:model", e.g. "anthropic:claude-sonnet-5-5"
    model_vision: str | None = None              # defaults to MODEL
    max_attempts: int = 3                        # build -> QC attempts per size before it is marked failed
    max_concurrent_jobs: int = 2                 # jobs the agent server runs at once
    max_open_pages: int = 4                      # headless-browser pages open at once during QC
    data_dir: Path = Path("data")                # where job folders, caches and traces are written
    brands_dir: Path = Path("config/brands")     # one sub-folder per brand_id (brand_kit.json, fonts, logos)
    pricing_file: Path = Path("config/pricing.yaml")  # per-model token prices used for cost reporting
    agent_url: str = "http://127.0.0.1:8000"     # base URL of the agent server (used by client and MCP server)
    agent_token: str | None = None               # optional bearer token; auth is disabled when unset

    @property
    def vision_model(self) -> str:
        """Model used for image/video understanding; falls back to the main model if none is configured."""
        return self.model_vision or self.model

    @property
    def agent_host_port(self) -> tuple[str, int]:
        """``(host, port)`` parsed from ``agent_url``, with local defaults for any missing part."""
        u = urlparse(self.agent_url)
        return u.hostname or "127.0.0.1", u.port or 8000


# Shared singleton; import this rather than constructing Settings elsewhere.
settings = Settings()
