import pytest
import yaml

from utilities.config import ConfigError, load_settings
from utilities.model_client import ModelClient


def write_config(tmp_path, provider, **values):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "default_provider": provider,
                "providers": {
                    provider: {
                        "model": "test-model",
                        "base_url": "http://localhost:8787/v1",
                        **values,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize(
    "provider", ["openai_compatible", "openai_responses", "anthropic_messages"]
)
def test_configured_key_reaches_sdk_without_environment(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("OPENAI_API_KEY", "different-environment-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "different-environment-key")
    config_path = write_config(tmp_path, provider, api_key="configured-test-key")
    settings = load_settings(config_path)
    captured = {}

    def capture_client(**options):
        captured.update(options)
        return object()

    sdk_class = (
        "utilities.model_client.anthropic.Anthropic"
        if provider == "anthropic_messages"
        else "utilities.model_client.openai.OpenAI"
    )
    monkeypatch.setattr(sdk_class, capture_client)
    ModelClient(settings)

    assert captured["api_key"] == "configured-test-key"
    assert "configured-test-key" not in repr(settings)


def test_compatible_endpoint_allows_empty_key(tmp_path):
    settings = load_settings(write_config(tmp_path, "openai_compatible", api_key=""))
    assert settings.api_key == ""


@pytest.mark.parametrize("provider", ["openai_responses", "anthropic_messages"])
def test_authenticated_provider_requires_configured_key(tmp_path, provider):
    with pytest.raises(ConfigError, match=r"\.api_key is required"):
        load_settings(write_config(tmp_path, provider))


@pytest.mark.parametrize("invalid_key", [None, 123, ["key"]])
def test_key_must_be_a_string(tmp_path, invalid_key):
    with pytest.raises(ConfigError, match=r"\.api_key must be a string"):
        load_settings(write_config(tmp_path, "openai_compatible", api_key=invalid_key))
