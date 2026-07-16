# Phase 2: LLM Providers Module — Discrete Implementation Steps

Create a new file `llm_providers.py` by extracting and revising the provider abstraction from `llm.py`. The existing `llm.py` remains untouched (it's still used by `find-jobs.py`).

**File being created:** `llm_providers.py`

**Key changes from existing `llm.py`:**
- New `LLMOptions` dataclass controls model, temperature, tokens, retries per-call
- New `LLMResult` dataclass returns text + metadata
- `generate()` signature changes from `generate(prompt: str) -> str` to `generate(system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult`
- Providers no longer store `self._model` — model comes from `LLMOptions` per call
- Each provider sends system prompt separately (Claude native `system=`, OpenAI `role: "system"`, Gemini `system_instruction`)
- Retry logic uses `options.max_retries` and `options.retry_base_seconds` instead of hardcoded values
- Replace `verbose()` / `log_debug()` with standard `logging` module
- No job/resume/batch code whatsoever

---

## Step 1: Create the file scaffold with imports and logging

Create `llm_providers.py` with:

```python
"""LLM provider abstraction for the AI MCP Server.

Provides a unified interface across ChatGPT (OpenAI), Claude (Anthropic),
and Gemini (Google GenAI) with per-call model selection, separate system/user
prompts, and configurable retry logic.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("ai-mcp-server")
```

**Verify:** `python -c "import llm_providers"` succeeds (no import errors from these standard libs).

---

## Step 2: Add the LLMOptions dataclass

Append below the imports:

```python
@dataclass
class LLMOptions:
    """Per-call options controlling model selection and behavior."""
    model: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_base_seconds: int = 30
```

**Fields:**
- `model` — required; the concrete model name (e.g., `gpt-4.1-mini`)
- `temperature` — optional; provider default if None
- `max_output_tokens` — optional; provider default if None
- `timeout_seconds` — optional; not enforced in first revision (future use)
- `max_retries` — how many attempts on rate-limit (default 3)
- `retry_base_seconds` — base wait between retries, multiplied by attempt (default 30)

**Verify:** `from llm_providers import LLMOptions; LLMOptions(model="test")` works.

---

## Step 3: Add the LLMResult dataclass

Append below `LLMOptions`:

```python
@dataclass
class LLMResult:
    """Result returned from a provider generate() call."""
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
```

**Fields:**
- `text` — the LLM's response text
- `model` — the model that was actually used (echoed back)
- `usage` — optional token usage dict (provider-specific shape)
- `finish_reason` — optional stop reason string

**Verify:** `from llm_providers import LLMResult; LLMResult(text="hi", model="x")` works.

---

## Step 4: Add the LLMProvider abstract base class

Append below `LLMResult`:

```python
class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, provider_config: Dict[str, Any]):
        self.config = provider_config

    def _resolve_api_key(self) -> str:
        """Resolve API key from environment variable or direct config.

        Priority:
        1. Environment variable named in api_key_env_var
        2. Direct api_key value in config

        Raises:
            ValueError: If no API key can be resolved.
        """
        env_var_name = self.config.get("api_key_env_var")
        if env_var_name:
            env_key = os.environ.get(env_var_name)
            if env_key:
                return env_key

        api_key = self.config.get("api_key")
        if api_key:
            return api_key

        if env_var_name:
            raise ValueError(
                f"{self.name} provider requires an API key. "
                f"Set the '{env_var_name}' environment variable or provide 'api_key' in configuration."
            )
        else:
            raise ValueError(
                f"{self.name} provider requires 'api_key' in configuration, "
                f"or set 'api_key_env_var' to specify an environment variable name."
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the display name of this provider."""
        ...

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult:
        """Generate a response from the LLM.

        Args:
            system_prompt: The system instructions.
            user_prompt: The user message.
            options: Per-call options (model, temperature, tokens, retries).

        Returns:
            LLMResult with the response text and metadata.
        """
        ...
```

**Key difference from existing:** No `model` property on the ABC. No `jobs_per_llm_batch`. The model is passed via `options` each call.

**Verify:** Cannot instantiate (it's abstract), but file still imports cleanly.

---

## Step 5: Implement ChatGPTProvider

Append below `LLMProvider`:

```python
class ChatGPTProvider(LLMProvider):
    """OpenAI ChatGPT provider."""

    def __init__(self, provider_config: Dict[str, Any]):
        super().__init__(provider_config)
        import openai
        self._openai = openai
        api_key = self._resolve_api_key()
        self._client = openai.OpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "ChatGPT"

    def generate(self, system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(options.max_retries):
            try:
                result_text = ""
                stream = self._client.chat.completions.create(
                    model=options.model,
                    messages=messages,
                    temperature=options.temperature,
                    max_tokens=options.max_output_tokens,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        result_text += chunk.choices[0].delta.content

                return LLMResult(text=result_text, model=options.model)

            except self._openai.AuthenticationError as e:
                raise ValueError(
                    f"OpenAI authentication failed. Check your API key. Error: {e}"
                ) from e
            except self._openai.PermissionDeniedError as e:
                raise ValueError(
                    f"OpenAI permission denied. Your account may lack access to model '{options.model}'. Error: {e}"
                ) from e
            except self._openai.NotFoundError as e:
                raise ValueError(
                    f"OpenAI model '{options.model}' not found. Check the model name. Error: {e}"
                ) from e
            except self._openai.RateLimitError as e:
                error_str = str(e).lower()
                if "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
                    raise ValueError(
                        f"OpenAI quota exceeded or billing issue. Error: {e}"
                    ) from e
                if attempt < options.max_retries - 1:
                    wait_time = options.retry_base_seconds * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 2}/{options.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) and "rate" in error_str and "quota" not in error_str:
                    if attempt < options.max_retries - 1:
                        wait_time = options.retry_base_seconds * (attempt + 1)
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 2}/{options.max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

        return LLMResult(text="", model=options.model)
```

**Key differences from existing `llm.py`:**
- `system_prompt` is sent as a separate `{"role": "system", ...}` message
- `options.model` used instead of `self._model`
- `options.max_output_tokens` used instead of `self._max_tokens`
- `options.temperature` passed to the API
- Retry uses `options.max_retries` and `options.retry_base_seconds`
- Returns `LLMResult` instead of bare `str`
- Uses `logger.warning()` instead of `verbose()`

**Verify:** File imports. Cannot test generate() without API key, but structure is correct.

---

## Step 6: Implement ClaudeProvider

Append below `ChatGPTProvider`:

```python
class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, provider_config: Dict[str, Any]):
        super().__init__(provider_config)
        import anthropic
        self._anthropic = anthropic
        api_key = self._resolve_api_key()
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "Claude"

    def generate(self, system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult:
        max_tokens = options.max_output_tokens or 8192

        for attempt in range(options.max_retries):
            try:
                result_text = ""
                with self._client.messages.stream(
                    model=options.model,
                    max_tokens=max_tokens,
                    temperature=options.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        result_text += text

                return LLMResult(text=result_text, model=options.model)

            except self._anthropic.RateLimitError:
                if attempt < options.max_retries - 1:
                    wait_time = options.retry_base_seconds * (attempt + 1)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 2}/{options.max_retries}...")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    if attempt < options.max_retries - 1:
                        wait_time = options.retry_base_seconds * (attempt + 1)
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 2}/{options.max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

        return LLMResult(text="", model=options.model)
```

**Key difference from existing:** `system=system_prompt` is passed separately to the Anthropic API (not crammed into the user message). This is Claude's native system prompt support.

**Verify:** File imports cleanly.

---

## Step 7: Implement GeminiProvider

Append below `ClaudeProvider`:

```python
class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, provider_config: Dict[str, Any]):
        super().__init__(provider_config)
        from google import genai
        self._genai = genai
        api_key = self._resolve_api_key()
        self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "Gemini"

    def generate(self, system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult:
        config = {}
        if options.temperature is not None:
            config["temperature"] = options.temperature
        if options.max_output_tokens is not None:
            config["max_output_tokens"] = options.max_output_tokens
        config["system_instruction"] = system_prompt

        for attempt in range(options.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=options.model,
                    contents=user_prompt,
                    config=config,
                )
                return LLMResult(text=response.text, model=options.model)

            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < options.max_retries - 1:
                        wait_time = options.retry_base_seconds * (attempt + 1)
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 2}/{options.max_retries}...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

        return LLMResult(text="", model=options.model)
```

**Key difference from existing:** `system_instruction` is passed in the config dict so Gemini treats it as a system prompt rather than prepending it to the user message.

**Note:** The `google-genai` SDK's config parameter may expect a typed object depending on installed version. If the installed version uses typed config, adapt accordingly. The dict form works with recent versions of the `google-genai` package.

**Verify:** File imports cleanly (assuming `google-genai` is installed).

---

## Step 8: Implement create_llm_provider() factory

Append at the bottom of the file:

```python
def create_llm_provider(provider_name: str, provider_config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create an LLM provider instance.

    Args:
        provider_name: Name of the provider ("chatgpt", "claude", or "gemini").
        provider_config: Provider-specific configuration dict from YAML.

    Returns:
        An initialized LLMProvider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    providers = {
        "chatgpt": ChatGPTProvider,
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        valid_providers = ", ".join(sorted(providers.keys()))
        raise ValueError(f"Unknown LLM provider '{provider_name}'. Valid options: {valid_providers}")

    return provider_class(provider_config)
```

**Verify:** `from llm_providers import create_llm_provider` works.

---

## Step 9: Verify the complete file imports without errors

Run:
```
python -c "import llm_providers; print('OK')"
```

This should succeed without attempting to create any provider instances (which would require API keys).

Also verify:
```
python -c "from llm_providers import LLMOptions, LLMResult, LLMProvider, ChatGPTProvider, ClaudeProvider, GeminiProvider, create_llm_provider; print('all exports OK')"
```

---

## Step 10: Verify structural correctness with a mock test

Run a quick inline check that the interface works without real API keys:

```python
python -c "
from llm_providers import LLMOptions, LLMResult, LLMProvider, create_llm_provider
import os

# Verify LLMOptions
opts = LLMOptions(model='gpt-4.1-mini', temperature=0.2, max_output_tokens=4000)
assert opts.model == 'gpt-4.1-mini'
assert opts.max_retries == 3
assert opts.retry_base_seconds == 30

# Verify LLMResult
result = LLMResult(text='hello', model='gpt-4.1-mini')
assert result.text == 'hello'
assert result.usage is None

# Verify factory rejects unknown provider
try:
    create_llm_provider('unknown', {})
    assert False, 'Should have raised'
except ValueError as e:
    assert 'Unknown LLM provider' in str(e)

# Verify factory rejects missing API key
try:
    create_llm_provider('chatgpt', {'api_key_env_var': 'NONEXISTENT_KEY_12345'})
    assert False, 'Should have raised'
except ValueError as e:
    assert 'requires an API key' in str(e)

print('All structural checks passed')
"
```

**Verify:** Prints "All structural checks passed".

---

## Summary of what `llm_providers.py` looks like after Phase 2

```
llm_providers.py
├── Module docstring
├── Imports (logging, os, time, abc, dataclasses, typing)
├── logger
├── LLMOptions (dataclass)
├── LLMResult (dataclass)
├── LLMProvider (ABC)
│   ├── __init__(provider_config)
│   ├── _resolve_api_key()
│   ├── name (abstract property)
│   └── generate(system_prompt, user_prompt, options) → LLMResult (abstract)
├── ChatGPTProvider
│   ├── __init__ — creates OpenAI client
│   ├── name → "ChatGPT"
│   └── generate — streaming, system message role, retry with options
├── ClaudeProvider
│   ├── __init__ — creates Anthropic client
│   ├── name → "Claude"
│   └── generate — streaming, native system= param, retry with options
├── GeminiProvider
│   ├── __init__ — creates GenAI client
│   ├── name → "Gemini"
│   └── generate — system_instruction in config, retry with options
└── create_llm_provider(name, config) → LLMProvider
```

**What was NOT carried over from `llm.py`:**
- `DEFAULT_JOBS_PER_LLM_BATCH`, `MAX_LLM_JOBS_PER_ROUND`
- `jobs_per_llm_batch` property
- `model` property on the ABC
- `load_resume_text`, resume cache, compression, verification
- `build_exclusion_rules`, `build_scoring_guidance`
- `extract_whitelisted_job_fields`, `build_jobs_listings_text`
- `build_prompt`, `build_batch_prompt`, `build_final_prompt`
- `batch_jobs`, `analyze_jobs`, `analyze_jobs_batched`
- `parse_llm_job_scores`, `normalize_scored_jobs`
- `merge_scores_with_jobs`, markdown formatting
- `clean_duplicate_output`, `validate_results`
- All `from docx import Document` and `from log_util import ...`
