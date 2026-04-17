"""
aria/config.py — Pydantic configuration loader for ARIA.

Loads configuration from aria.yaml in the current directory,
falling back to ~/.aria/aria.yaml. All paths and settings are
validated through Pydantic models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Load .env if present
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Config search paths (in priority order)
# ---------------------------------------------------------------------------
_CONFIG_SEARCH_PATHS: list[Path] = [
    Path.cwd() / "aria.yaml",
    Path.home() / ".aria" / "aria.yaml",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Configuration for the LLM provider (Ollama)."""

    provider: str = Field(default="ollama", description="LLM provider name.")
    model: str = Field(default="llama3", description="Model identifier to use.")
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for the Ollama REST API.",
    )
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="Sampling temperature."
    )
    max_tokens: int = Field(
        default=4096, gt=0, description="Maximum tokens in a completion."
    )
    timeout: int = Field(
        default=60, gt=0, description="Request timeout in seconds."
    )


class AriaConfig(BaseModel):
    """Core ARIA runtime settings."""

    name: str = Field(default="ARIA", description="Agent display name.")
    version: str = Field(default="0.1.0", description="Agent version string.")
    log_level: str = Field(
        default="INFO",
        description="Logging level: TRACE | DEBUG | INFO | WARNING | ERROR.",
    )
    log_format: Literal["json", "pretty"] = Field(
        default="json",
        description="Log output format: 'json' for structured, 'pretty' for dev.",
    )
    data_dir: str = Field(
        default="~/.aria/data", description="Directory for persistent data."
    )
    memory_dir: str = Field(
        default="~/.aria/memory", description="Directory for memory storage."
    )


class InterfacesConfig(BaseModel):
    """Feature flags for available ARIA interfaces."""

    cli: bool = Field(default=True, description="Enable CLI interface.")
    voice: bool = Field(default=False, description="Enable voice interface.")
    api: bool = Field(default=False, description="Enable REST API interface.")
    desktop: bool = Field(default=False, description="Enable desktop GUI.")


class GoogleDriveIntegration(BaseModel):
    """Google Drive integration settings."""

    enabled: bool = Field(default=False, description="Enable Google Drive.")
    credentials_path: str = Field(
        default="~/.aria/google_credentials.json",
        description="Path to Google service-account credentials.",
    )


class TelegramIntegration(BaseModel):
    """Telegram bot integration settings."""

    enabled: bool = Field(default=False, description="Enable Telegram bot.")
    bot_token: str = Field(default="", description="Telegram bot token.")


class IntegrationsConfig(BaseModel):
    """Third-party integration settings."""

    google_drive: GoogleDriveIntegration = Field(
        default_factory=GoogleDriveIntegration
    )
    telegram: TelegramIntegration = Field(default_factory=TelegramIntegration)


class Settings(BaseModel):
    """Top-level ARIA settings — the root config object."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    aria: AriaConfig = Field(default_factory=AriaConfig)
    interfaces: InterfacesConfig = Field(default_factory=InterfacesConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _find_config_file() -> Path | None:
    """Locate the first existing config file from the search paths.

    Returns:
        The resolved Path to the config file, or None if not found.
    """
    for candidate in _CONFIG_SEARCH_PATHS:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    return None


def load_settings() -> Settings:
    """Load and validate ARIA settings from YAML + env overrides.

    Search order:
        1. ./aria.yaml  (project-local)
        2. ~/.aria/aria.yaml  (user-global)

    Environment variable overrides (applied after YAML):
        - OLLAMA_BASE_URL  → llm.base_url
        - OLLAMA_MODEL     → llm.model
        - ARIA_LOG_LEVEL   → aria.log_level

    Returns:
        A fully validated Settings instance.
    """
    config_path = _find_config_file()
    raw: dict = {}

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    settings = Settings(**raw)

    # --- env overrides ---
    if env_url := os.getenv("OLLAMA_BASE_URL"):
        settings.llm.base_url = env_url
    if env_model := os.getenv("OLLAMA_MODEL"):
        settings.llm.model = env_model
    if env_log := os.getenv("ARIA_LOG_LEVEL"):
        settings.aria.log_level = env_log

    return settings


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached global Settings singleton, loading on first call.

    Returns:
        The global Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings
