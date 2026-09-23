"""Tests for YAML configuration loading and validation."""

import pytest
from pathlib import Path

from prompt_engine import load_yaml_config


class TestLoadYamlConfig:

    def test_loads_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
defaults:
  max_retries: 3
  retry_base_seconds: 30

providers:
  chatgpt:
    type: openai
    api_key_env_var: OPENAI_API_KEY
    models:
      low:
        model: gpt-5.4-mini
        temperature: 0.2
        max_tokens: 4000
      middle:
        model: gpt-5.4
        max_tokens: 8000
      high:
        model: gpt-5.5
        max_completion_tokens: 12000
""")
        config = load_yaml_config(config_file)
        assert "providers" in config
        assert "chatgpt" in config["providers"]
        assert config["providers"]["chatgpt"]["models"]["low"]["model"] == "gpt-5.4-mini"

    def test_loads_example_config_file(self):
        # The committed example, not the live ai-mcp-server.yaml: the live file
        # is gitignored and machine-specific, so it must not decide test results.
        config = load_yaml_config(Path("ai-mcp-server.yaml.example"))
        assert "chatgpt" in config["providers"]
        assert "gemini" in config["providers"]
        assert "claude" in config["providers"]

    def test_fails_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_yaml_config(tmp_path / "nonexistent.yaml")

    def test_fails_if_providers_missing(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("defaults:\n  temperature: 0.2\n")
        with pytest.raises(ValueError, match="non-empty 'providers' mapping"):
            load_yaml_config(config_file)

    def test_fails_if_providers_empty(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("providers: {}\n")
        with pytest.raises(ValueError, match="non-empty 'providers' mapping"):
            load_yaml_config(config_file)

    def test_fails_if_provider_has_no_models(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
providers:
  chatgpt:
    type: openai
""")
        with pytest.raises(ValueError, match="must have a non-empty 'models' mapping"):
            load_yaml_config(config_file)

    def test_fails_if_model_tier_missing_model_key(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
providers:
  chatgpt:
    type: openai
    models:
      low:
        temperature: 0.2
""")
        with pytest.raises(ValueError, match="missing 'model' key"):
            load_yaml_config(config_file)

    def test_defaults_key_is_optional(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
providers:
  chatgpt:
    type: openai
    models:
      low:
        model: gpt-5.4-mini
""")
        config = load_yaml_config(config_file)
        assert config.get("defaults") is None
        assert config["providers"]["chatgpt"]["models"]["low"]["model"] == "gpt-5.4-mini"

    def test_does_not_require_api_keys(self, tmp_path):
        """API keys are resolved lazily at provider creation, not config load time."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
providers:
  chatgpt:
    type: openai
    api_key_env_var: TOTALLY_FAKE_KEY_12345
    models:
      low:
        model: gpt-5.4-mini
""")
        config = load_yaml_config(config_file)
        assert config["providers"]["chatgpt"]["api_key_env_var"] == "TOTALLY_FAKE_KEY_12345"
