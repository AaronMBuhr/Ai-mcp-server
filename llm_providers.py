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


@dataclass
class LLMOptions:
    """Per-call options controlling model selection and behavior."""
    model: str
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_base_seconds: int = 30
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    """Result returned from a provider generate() call."""
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


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
            options: Per-call options (model, retry controls, provider parameters).

        Returns:
            LLMResult with the response text and metadata.
        """
        ...


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
                request_kwargs: Dict[str, Any] = {
                    "model": options.model,
                    "messages": messages,
                    "stream": True,
                }
                request_kwargs.update(
                    {key: value for key, value in options.parameters.items() if value is not None}
                )

                stream = self._client.chat.completions.create(**request_kwargs)
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
                    logger.warning(
                        f"Rate limited. Waiting {wait_time}s before retry "
                        f"{attempt + 2}/{options.max_retries}..."
                    )
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                error_str = str(e).lower()
                if "429" in str(e) and "rate" in error_str and "quota" not in error_str:
                    if attempt < options.max_retries - 1:
                        wait_time = options.retry_base_seconds * (attempt + 1)
                        logger.warning(
                            f"Rate limited. Waiting {wait_time}s before retry "
                            f"{attempt + 2}/{options.max_retries}..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

        return LLMResult(text="", model=options.model)


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
        request_kwargs: Dict[str, Any] = {
            "model": options.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        request_kwargs.update(
            {key: value for key, value in options.parameters.items() if value is not None}
        )
        for attempt in range(options.max_retries):
            try:
                result_text = ""
                with self._client.messages.stream(**request_kwargs) as stream:
                    for text in stream.text_stream:
                        result_text += text

                return LLMResult(text=result_text, model=options.model)

            except self._anthropic.RateLimitError:
                if attempt < options.max_retries - 1:
                    wait_time = options.retry_base_seconds * (attempt + 1)
                    logger.warning(
                        f"Rate limited. Waiting {wait_time}s before retry "
                        f"{attempt + 2}/{options.max_retries}..."
                    )
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    if attempt < options.max_retries - 1:
                        wait_time = options.retry_base_seconds * (attempt + 1)
                        logger.warning(
                            f"Rate limited. Waiting {wait_time}s before retry "
                            f"{attempt + 2}/{options.max_retries}..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

        return LLMResult(text="", model=options.model)


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
        config: Dict[str, Any] = {"system_instruction": system_prompt}
        config.update(
            {key: value for key, value in options.parameters.items() if value is not None}
        )

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
                        logger.warning(
                            f"Rate limited. Waiting {wait_time}s before retry "
                            f"{attempt + 2}/{options.max_retries}..."
                        )
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

        return LLMResult(text="", model=options.model)


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
