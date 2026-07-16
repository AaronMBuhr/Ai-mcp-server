# Phase 5: Tests — Discrete Implementation Steps

Create a `tests/` directory with focused pytest test files. All tests run without real API keys by using fake/mock providers.

**Files created:**
- `tests/__init__.py`
- `tests/test_config_loading.py`
- `tests/test_prompt_rendering.py`
- `tests/test_llm_routing.py`
- `tests/test_query_ai.py`

**Prerequisites:**
- `pytest` must be installed in the environment
- All tests run from the project root: `pytest tests/`

---

## Step 1: Create `tests/__init__.py`

Create an empty `tests/__init__.py` so pytest recognizes the directory as a package.

```python
# (empty file)
```

---

## Step 2: Create `tests/test_config_loading.py`

Tests for `prompt_engine.load_yaml_config()`:

```python
"""Tests for YAML configuration loading and validation."""

import pytest
from pathlib import Path
import tempfile

from prompt_engine import load_yaml_config


class TestLoadYamlConfig:

    def test_loads_valid_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
defaults:
  temperature: 0.2
  max_output_tokens: 8000
  max_retries: 3
  retry_base_seconds: 30

providers:
  chatgpt:
    type: openai
    api_key_env_var: OPENAI_API_KEY
    models:
      low:
        model: gpt-4.1-mini
        temperature: 0.2
        max_output_tokens: 4000
      middle:
        model: gpt-4.1
        max_output_tokens: 8000
      high:
        model: o3
        max_output_tokens: 12000
""")
        config = load_yaml_config(config_file)
        assert "providers" in config
        assert "chatgpt" in config["providers"]
        assert config["providers"]["chatgpt"]["models"]["low"]["model"] == "gpt-4.1-mini"

    def test_loads_real_config_file(self):
        config = load_yaml_config(Path("ai-mcp-server.yaml"))
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
        model: gpt-4.1-mini
""")
        config = load_yaml_config(config_file)
        assert config.get("defaults") is None
        assert config["providers"]["chatgpt"]["models"]["low"]["model"] == "gpt-4.1-mini"

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
        model: gpt-4.1-mini
""")
        config = load_yaml_config(config_file)
        assert config["providers"]["chatgpt"]["api_key_env_var"] == "TOTALLY_FAKE_KEY_12345"
```

---

## Step 3: Create `tests/test_prompt_rendering.py`

Tests for `resolve_prompt_path()`, `render_template()`, `combine_user_prompt()`, and `sha256_text()`:

```python
"""Tests for prompt path resolution, template rendering, and utility functions."""

import pytest
from pathlib import Path

from prompt_engine import (
    resolve_prompt_path,
    render_template,
    combine_user_prompt,
    sha256_text,
)


class TestResolvePromptPath:

    @pytest.fixture
    def prompt_dir(self, tmp_path):
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "test-system.md").write_text("system prompt")
        (d / "test-user.md").write_text("hello {name}")
        nested = d / "resume"
        nested.mkdir()
        (nested / "draft-system.md").write_text("nested system")
        return d

    def test_loads_simple_template(self, prompt_dir):
        path = resolve_prompt_path(prompt_dir, "test-system")
        assert path.exists()
        assert path.name == "test-system.md"

    def test_loads_nested_template(self, prompt_dir):
        path = resolve_prompt_path(prompt_dir, "resume/draft-system")
        assert path.exists()
        assert path.name == "draft-system.md"

    def test_rejects_dotdot(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "../secret")

    def test_rejects_absolute_path(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "/etc/passwd")

    def test_rejects_windows_drive_path(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "C:/Windows/file")

    def test_rejects_backslash(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "foo\\bar")

    def test_rejects_invalid_chars(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "test system")  # space

    def test_fails_on_missing_template(self, prompt_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_prompt_path(prompt_dir, "nonexistent")

    def test_loads_from_real_prompts_dir(self):
        prompt_dir = Path("prompts").resolve()
        path = resolve_prompt_path(prompt_dir, "test-system")
        assert path.exists()


class TestRenderTemplate:

    def test_replaces_single_placeholder(self):
        rendered, warnings = render_template("Hello {name}!", {"name": "Aaron"})
        assert rendered == "Hello Aaron!"
        assert warnings == []

    def test_replaces_multiple_occurrences(self):
        rendered, _ = render_template("{name} says hi to {name}", {"name": "Aaron"})
        assert rendered == "Aaron says hi to Aaron"

    def test_replaces_multiple_keys(self):
        template = "{greeting}, {name}!"
        rendered, _ = render_template(template, {"greeting": "Hello", "name": "World"})
        assert rendered == "Hello, World!"

    def test_renders_dict_as_json(self):
        template = "Data: {payload}"
        rendered, _ = render_template(template, {"payload": {"key": "value"}})
        assert '"key"' in rendered
        assert '"value"' in rendered

    def test_renders_list_as_json(self):
        template = "Items: {items}"
        rendered, _ = render_template(template, {"items": [1, 2, 3]})
        assert "1" in rendered
        assert "2" in rendered
        assert "3" in rendered

    def test_errors_on_missing_placeholder(self):
        with pytest.raises(ValueError, match="Missing replacement values"):
            render_template("Hello {name}!", {})

    def test_warns_on_unused_key(self):
        rendered, warnings = render_template("Hello!", {"unused": "value"})
        assert len(warnings) == 1
        assert "unused" in warnings[0]

    def test_does_not_treat_json_braces_as_placeholders(self):
        template = 'Example: {"key": "value"} and {name}'
        rendered, warnings = render_template(template, {"name": "Aaron"})
        assert '{"key": "value"}' in rendered
        assert "Aaron" in rendered
        assert len(warnings) == 0

    def test_does_not_use_str_format_behavior(self):
        """Ensure {0}, {1}, etc. are not treated as placeholders."""
        template = "Index {0} and name {name}"
        rendered, warnings = render_template(template, {"name": "Aaron"})
        assert "{0}" in rendered
        assert "Aaron" in rendered

    def test_rejects_invalid_replacement_key(self):
        with pytest.raises(ValueError, match="Invalid replacement key"):
            render_template("Hi", {"bad-key": "value"})

    def test_rejects_key_starting_with_digit(self):
        with pytest.raises(ValueError, match="Invalid replacement key"):
            render_template("Hi", {"0key": "value"})


class TestCombineUserPrompt:

    def test_appends_additional_with_separator(self):
        result = combine_user_prompt("Main prompt.", "Be brief.")
        assert "---" in result
        assert "Additional user instructions:" in result
        assert "Be brief." in result
        assert result.startswith("Main prompt.")

    def test_returns_plain_when_additional_empty(self):
        result = combine_user_prompt("Main prompt.", "")
        assert "---" not in result
        assert result == "Main prompt."

    def test_returns_plain_when_additional_whitespace(self):
        result = combine_user_prompt("Main prompt.", "   \n  ")
        assert "---" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = combine_user_prompt("  Main prompt.  ", "")
        assert result == "Main prompt."


class TestSha256Text:

    def test_returns_64_char_hex(self):
        h = sha256_text("hello")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert sha256_text("test") == sha256_text("test")

    def test_different_inputs_different_hashes(self):
        assert sha256_text("a") != sha256_text("b")
```

---

## Step 4: Create `tests/test_llm_routing.py`

Tests for `LLMRouter` using a fake provider (no API keys needed):

```python
"""Tests for LLMRouter — provider/intelligence resolution with fake providers."""

import pytest
from unittest.mock import patch, MagicMock

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
            "temperature": 0.2,
            "max_output_tokens": 8000,
            "max_retries": 3,
            "retry_base_seconds": 30,
        },
        "providers": {
            "chatgpt": {
                "type": "openai",
                "api_key_env_var": "OPENAI_API_KEY",
                "models": {
                    "low": {"model": "gpt-4.1-mini", "temperature": 0.3, "max_output_tokens": 4000},
                    "middle": {"model": "gpt-4.1", "max_output_tokens": 8000},
                    "high": {"model": "o3", "temperature": 0.1, "max_output_tokens": 12000},
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
                    "low": {"model": "claude-haiku-4", "max_output_tokens": 4000},
                    "middle": {"model": "claude-sonnet-4", "max_output_tokens": 8000},
                    "high": {"model": "claude-opus-4", "max_output_tokens": 12000},
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
        assert result.model == "o3"

    def test_model_tier_temperature_overrides_default(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "low", "sys", "usr")
        assert fake.calls[0]["options"].temperature == 0.3

    def test_default_temperature_used_when_tier_omits(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "middle", "sys", "usr")
        assert fake.calls[0]["options"].temperature == 0.2

    def test_max_output_tokens_from_tier(self, router_with_fake):
        router, fake = router_with_fake
        router.query("chatgpt", "high", "sys", "usr")
        assert fake.calls[0]["options"].max_output_tokens == 12000

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
```

---

## Step 5: Create `tests/test_query_ai.py`

Tests for the `query_ai` MCP tool function using a mocked router:

```python
"""Tests for the query_ai MCP tool function with mocked router."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

from llm_providers import LLMResult
from prompt_engine import LLMRouter


def _get_server_and_tools():
    """Create MCPServer and return tool functions by name."""
    exec_globals = {}
    code = Path("ai-mcp-server.py").read_text(encoding="utf-8")
    code = code.split("if __name__")[0]
    exec(code, exec_globals)
    ServerSettings = exec_globals["ServerSettings"]
    MCPServer = exec_globals["MCPServer"]
    s = ServerSettings()
    m = MCPServer(s)
    ps = m.create_public_server()
    tools = {}
    for tool in ps._tool_manager.list_tools():
        fn = getattr(tool, "function", None) or getattr(tool, "fn", None)
        tools[tool.name] = fn
    return m, tools


class TestQueryAiTool:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.server, self.tools = _get_server_and_tools()
        self.query_ai = self.tools["query_ai"]

    def test_missing_template_returns_error(self):
        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="nonexistent",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert r["error_type"] == "FileNotFoundError"
        assert "not found" in r["message"]

    def test_missing_replacement_returns_error(self):
        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={},
        )
        assert r["status"] == "error"
        assert "Missing replacement values" in r["message"]

    def test_unsafe_path_returns_error(self):
        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="../etc/passwd",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert r["error_type"] == "ValueError"

    def test_unknown_provider_returns_error(self):
        r = self.query_ai(
            provider="invalid", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert "Unknown provider" in r["message"]

    def test_unknown_intelligence_returns_error(self):
        r = self.query_ai(
            provider="chatgpt", intelligence="ultra",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert "Unknown intelligence" in r["message"]

    def test_successful_query_with_mocked_router(self):
        fake_result = LLMResult(text="Hello Aaron!", model="gpt-4.1-mini")
        self.server._router.query = MagicMock(return_value=fake_result)

        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "Aaron"},
        )
        assert r["status"] == "success"
        assert r["response_text"] == "Hello Aaron!"
        assert r["model"] == "gpt-4.1-mini"
        assert r["provider"] == "chatgpt"
        assert r["intelligence"] == "low"
        assert "elapsed_ms" in r
        assert isinstance(r["elapsed_ms"], int)
        assert r["system_prompt_template"] == "test-system"
        assert r["user_prompt_template"] == "test-user"
        assert len(r["system_prompt_sha256"]) == 64
        assert len(r["user_prompt_sha256"]) == 64
        assert r["warnings"] == []

    def test_warnings_included_in_response(self):
        fake_result = LLMResult(text="Hi", model="gpt-4.1-mini")
        self.server._router.query = MagicMock(return_value=fake_result)

        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "Aaron", "unused_key": "value"},
        )
        assert r["status"] == "success"
        assert len(r["warnings"]) > 0
        assert "unused_key" in r["warnings"][0]

    def test_additional_user_prompt_appended(self):
        fake_result = LLMResult(text="response", model="gpt-4.1-mini")
        self.server._router.query = MagicMock(return_value=fake_result)

        self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            additional_user_prompt="Be extra concise.",
            replacement_values={"name": "Aaron"},
        )

        call_args = self.server._router.query.call_args
        user_prompt_sent = call_args[0][3]
        assert "Be extra concise." in user_prompt_sent
        assert "---" in user_prompt_sent
```

---

## Step 6: Run all tests

```bash
pytest tests/ -v
```

Expected: All tests pass. No API keys needed.

---

## Summary of test coverage

| File | Tests | What it covers |
|------|-------|----------------|
| `test_config_loading.py` | 9 tests | YAML loading, validation, missing files, bad structure |
| `test_prompt_rendering.py` | 21 tests | Path safety, template rendering, combine, sha256 |
| `test_llm_routing.py` | 15 tests | Router resolution, config merging, caching, error handling |
| `test_query_ai.py` | 8 tests | MCP tool function end-to-end with mocked router |

**Total: ~53 tests, all runnable without API keys.**

---

## Notes

- `test_query_ai.py` uses `exec()` to load `ai-mcp-server.py` without running `main()`. This avoids import issues with the hyphenated filename. An alternative would be renaming the file, but we preserve the existing naming convention.
- `test_llm_routing.py` uses a `FakeProvider` subclass that records calls, letting us verify exactly what model/options/prompts are sent.
- The tests do NOT cover transport (stdio/http) behavior — that's Phase 6 manual smoke testing.
