"""Tests for LLMRouter — provider/intelligence resolution with fake providers."""

import pytest

from prompt_engine import LLMRouter
from llm_providers import LLMOptions, LLMResult, LLMProvider


class FakeProvider(LLMProvider):
    """A fake provider that records calls and returns canned responses."""

    def __init__(self, provider_config):
        super().__init__(provider_config)
        self.calls = []

    @property
    def name(self) -> str:
        return "Fake"

    def generate(self, system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "options": options,
        })
        return LLMResult(text=f"response for {options.model}", model=options.model)


@pytest.fixture
def config():
    return {
        "defaults": {
            "max_retries": 3,
            "retry_base_seconds": 30,
        },
        "providers": {
            "chatgpt": {
                "type": "openai",
                "api_key_env_var": "OPENAI_API_KEY",
                "models": {
                    "low": {"model": "gpt-4.1-mini", "temperature": 0.3, "max_tokens": 4000},
                    "middle": {"model": "gpt-4.1", "max_tokens": 8000},
                    "high": {
                        "model": "gpt-5.5",
                        "temperature": 0.1,
                        "max_completion_tokens": 12000,
                        "reasoning_effort": "medium",
                    },
                },
            },
            "gemini": {
                "type": "gemini",
                "api_key_env_var": "GEMINI_API_KEY",
                "models": {
                    "low": {"model": "gemini-2.0-flash", "max_output_tokens": 4000},
                    "middle": {"model": "gemini-2.5-flash", "max_output_tokens": 8000},
                    "high": {"model": "gemini-2.5-pro", "max_output_tokens": 12000},
                },
            },
            "claude": {
                "type": "anthropic",
                "api_key_env_var": "ANTHROPIC_API_KEY",
                "models": {
                    "low": {"model": "claude-haiku-4", "max_tokens": 4000},
                    "middle": {"model": "claude-sonnet-4", "max_tokens": 8000},
                    "high": {"model": "claude-opus-4", "max_tokens": 12000},
                },
            },
        },
    }


@pytest.fixture
def router_with_fake(config):
    """Create a router and inject a FakeProvider for chatgpt."""
    router = LLMRouter(config)
    fake = FakeProvider(config["providers"]["chatgpt"])
    router._providers["chatgpt"] = fake
    return router, fake


class TestLLMRouterResolution:

    def test_chatgpt_low_resolves_model(self, router_with_fake):
        router, fake = router_with_fake
        result = router.query("chatgpt", "low", "sys", "usr")
        assert result.model == "gpt-4.1-mini"
        assert fake.calls[0]["options"].model == "gpt-4.1-mini"

    def test_chatgpt_middle_resolves_model(self, router_with_fake):
        router, fake = router_with_fake
        result = router.query("chatgpt", "middle", "sys", "usr")
        assert result.model == "gpt-4.1"

    def test_chatgpt_high_resolves_model(self, router_with_fake):
        router, fake = router_with_fake
        result = router.query("chatgpt", "high", "sys", "usr")
        assert result.model == "gpt-5.5"

    def test_model_tier_temperature_overrides_default(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "low", "sys", "usr")
        assert fake.calls[0]["options"].parameters["temperature"] == 0.3

    def test_tier_without_temperature_does_not_invent_one(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "middle", "sys", "usr")
        assert "temperature" not in fake.calls[0]["options"].parameters

    def test_model_tier_parameters_are_passed_through(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "high", "sys", "usr")
        assert fake.calls[0]["options"].parameters["max_completion_tokens"] == 12000
        assert fake.calls[0]["options"].parameters["reasoning_effort"] == "medium"

    def test_max_retries_from_defaults(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "low", "sys", "usr")
        assert fake.calls[0]["options"].max_retries == 3

    def test_system_prompt_passed_separately(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "low", "my system prompt", "my user prompt")
        assert fake.calls[0]["system_prompt"] == "my system prompt"

    def test_user_prompt_passed_separately(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "low", "my system prompt", "my user prompt")
        assert fake.calls[0]["user_prompt"] == "my user prompt"

    def test_unknown_provider_raises(self, config):
        router = LLMRouter(config)
        with pytest.raises(ValueError, match="Unknown provider 'invalid'"):
            router.query("invalid", "low", "sys", "usr")

    def test_unknown_intelligence_raises(self, router_with_fake):
        router, _ = router_with_fake
        with pytest.raises(ValueError, match="Unknown intelligence 'ultra'"):
            router.query("chatgpt", "ultra", "sys", "usr")

    def test_gemini_middle_resolves(self, config):
        router = LLMRouter(config)
        fake = FakeProvider(config["providers"]["gemini"])
        router._providers["gemini"] = fake
        result = router.query("gemini", "middle", "sys", "usr")
        assert result.model == "gemini-2.5-flash"

    def test_claude_high_resolves(self, config):
        router = LLMRouter(config)
        fake = FakeProvider(config["providers"]["claude"])
        router._providers["claude"] = fake
        result = router.query("claude", "high", "sys", "usr")
        assert result.model == "claude-opus-4"

    def test_provider_is_cached(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "low", "sys", "usr")
        router.query("chatgpt", "high", "sys", "usr")
        assert len(fake.calls) == 2
        assert router._providers["chatgpt"] is fake
