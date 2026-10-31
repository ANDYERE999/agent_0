from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    api_key_env: str
    base_url: str | None
    request_timeout_seconds: float
    max_retries: int
    retry_backoff_base_seconds: float
    retry_backoff_max_seconds: float
    max_output_tokens: int
    stream_include_usage: bool

    @property
    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        value = os.getenv(self.api_key_env)
        if not value:
            raise ConfigError(
                f"Set the {self.api_key_env} environment variable before starting the Agent."
            )
        return value


def load_settings(
    config_path: str | Path,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> ModelSettings:
    path = Path(config_path).expanduser().resolve(strict=True)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"Cannot read configuration {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML object.")

    provider = provider_override or raw.get("default_provider", "")
    aliases = {
        "OpenAI Compatible": "openai_compatible",
        "OpenAI Responses": "openai_responses",
        "Anthropic Messages": "anthropic_messages",
        "anthropic": "anthropic_messages",
    }
    provider = aliases.get(provider, provider)
    if provider not in {"openai_compatible", "openai_responses", "anthropic_messages"}:
        raise ConfigError(
            "provider must be openai_compatible, openai_responses, or anthropic_messages."
        )

    provider_configs = raw.get("providers", {})
    if not isinstance(provider_configs, dict):
        raise ConfigError("providers must be a YAML object.")
    provider_config = provider_configs.get(provider, {})
    if not isinstance(provider_config, dict):
        raise ConfigError(f"providers.{provider} must be a YAML object.")
    model = model_override or provider_config.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError(f"Set providers.{provider}.model in {path} or use --model.")

    retry = raw.get("retry", {})
    if not isinstance(retry, dict):
        raise ConfigError("retry must be a YAML object.")
    settings = ModelSettings(
        provider=provider,
        model=model.strip(),
        api_key_env=str(provider_config.get("api_key_env", "")),
        base_url=provider_config.get("base_url") or None,
        request_timeout_seconds=float(raw.get("request_timeout_seconds", 300)),
        max_retries=int(retry.get("max_retries", 3)),
        retry_backoff_base_seconds=float(retry.get("backoff_base_seconds", 1)),
        retry_backoff_max_seconds=float(retry.get("backoff_max_seconds", 60)),
        max_output_tokens=int(raw.get("max_output_tokens", 4096)),
        stream_include_usage=bool(provider_config.get("stream_include_usage", False)),
    )
    if settings.request_timeout_seconds <= 0:
        raise ConfigError("request_timeout_seconds must be positive.")
    if settings.max_retries < 0:
        raise ConfigError("retry.max_retries cannot be negative.")
    if settings.retry_backoff_base_seconds < 0 or settings.retry_backoff_max_seconds < 0:
        raise ConfigError("Retry backoff values cannot be negative.")
    if settings.max_output_tokens <= 0:
        raise ConfigError("max_output_tokens must be positive.")
    if provider == "openai_compatible" and not settings.base_url:
        raise ConfigError("providers.openai_compatible.base_url is required.")
    if settings.api_key_env:
        settings.api_key
    elif provider != "openai_compatible":
        raise ConfigError(f"providers.{provider}.api_key_env is required.")
    return settings


def load_raw_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML object.")
    return raw
