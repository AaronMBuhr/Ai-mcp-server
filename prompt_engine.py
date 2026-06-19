"""Config loading, prompt template rendering, and utility functions for the AI MCP Server."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from llm_providers import LLMOptions, LLMResult, LLMProvider, create_llm_provider


def load_yaml_config(path: Path) -> dict:
    """Load and validate the YAML configuration file.

    Validates:
    - File exists
    - YAML parses
    - 'providers' key exists and is a non-empty dict
    - Each provider has a 'models' dict with at least one tier

    Args:
        path: Path to the YAML config file.

    Returns:
        The parsed config dict.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(config).__name__}")

    providers = config.get("providers")
    if not providers or not isinstance(providers, dict):
        raise ValueError("Config must have a non-empty 'providers' mapping")

    for provider_name, provider_conf in providers.items():
        if not isinstance(provider_conf, dict):
            raise ValueError(f"Provider '{provider_name}' must be a mapping")
        models = provider_conf.get("models")
        if not models or not isinstance(models, dict):
            raise ValueError(
                f"Provider '{provider_name}' must have a non-empty 'models' mapping"
            )
        for tier_name, tier_conf in models.items():
            if not isinstance(tier_conf, dict):
                raise ValueError(
                    f"Provider '{provider_name}' model tier '{tier_name}' must be a mapping"
                )
            if "model" not in tier_conf:
                raise ValueError(
                    f"Provider '{provider_name}' model tier '{tier_name}' is missing 'model' key"
                )

    return config


_TEMPLATE_NAME_RE = re.compile(r"^[A-Za-z0-9_/-]+$")


def resolve_prompt_path(prompt_dir: Path, template_name: str) -> Path:
    """Resolve a template name to a safe path inside the prompt directory.

    Security rules:
    - Template name must match ^[A-Za-z0-9_/-]+$
    - No '..' path segments
    - No backslashes
    - Resolved path must be inside prompt_dir
    - File must exist (with .md extension appended)

    Args:
        prompt_dir: The base prompts directory (resolved).
        template_name: Relative template name (e.g., "test-system" or "resume/draft-user").

    Returns:
        The resolved Path to the .md file.

    Raises:
        ValueError: If template name is unsafe or resolves outside prompt_dir.
        FileNotFoundError: If the template file doesn't exist.
    """
    if not _TEMPLATE_NAME_RE.match(template_name):
        raise ValueError(f"Unsafe prompt template name: '{template_name}'")

    if "\\" in template_name:
        raise ValueError("Prompt template may not contain backslashes")

    if ".." in template_name.split("/"):
        raise ValueError("Prompt template may not contain '..'")

    root = prompt_dir.resolve()
    path = (root / f"{template_name}.md").resolve()

    if root not in path.parents and path != root:
        raise ValueError("Prompt template resolved outside prompt directory")

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_name}.md")

    return path


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_REPLACEMENT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _render_value(value: Any) -> str:
    """Convert a replacement value to its string representation."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_template(template_text: str, replacement_values: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Render a prompt template by replacing {placeholder} tokens.

    Uses explicit string replacement, NOT str.format(), so that literal braces
    in JSON/code examples within templates are safe.

    Args:
        template_text: The raw template text with {placeholder} tokens.
        replacement_values: Dict mapping placeholder names to values.

    Returns:
        Tuple of (rendered_text, warnings).
        Warnings list unused replacement keys.

    Raises:
        ValueError: If replacement keys are invalid or required placeholders are missing.
    """
    warnings: List[str] = []

    for key in replacement_values:
        if not _REPLACEMENT_KEY_RE.match(key):
            raise ValueError(f"Invalid replacement key: '{key}'")

    placeholders = set(_PLACEHOLDER_RE.findall(template_text))

    missing = sorted(placeholders - set(replacement_values.keys()))
    if missing:
        raise ValueError(f"Missing replacement values: {', '.join(missing)}")

    unused = sorted(set(replacement_values.keys()) - placeholders)
    for key in unused:
        warnings.append(f"Replacement value '{key}' was supplied but not used by this template.")

    rendered = template_text
    for key in placeholders:
        rendered = rendered.replace("{" + key + "}", _render_value(replacement_values[key]))

    return rendered, warnings


def combine_user_prompt(rendered_user_prompt: str, additional_user_prompt: str) -> str:
    """Combine the rendered user template with optional additional instructions.

    If additional_user_prompt is blank, returns just the rendered template.
    Otherwise appends it after a separator.

    Args:
        rendered_user_prompt: The already-rendered user template.
        additional_user_prompt: Extra user instructions (may be empty).

    Returns:
        The combined user prompt string.
    """
    extra = (additional_user_prompt or "").strip()
    if not extra:
        return rendered_user_prompt.strip()

    return (
        rendered_user_prompt.rstrip()
        + "\n\n---\n\nAdditional user instructions:\n\n"
        + extra
    )


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest of a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_debug_prompt(
    debug_dir: str,
    provider: str,
    intelligence: str,
    system_prompt_template: str,
    user_prompt_template: str,
    replacement_keys: List[str],
    system_prompt: str,
    user_prompt: str,
    warnings: List[str],
) -> Path:
    """Save rendered prompts to a debug JSON file.

    Creates the debug directory if it doesn't exist.

    Args:
        debug_dir: Directory to write debug files into.
        provider: Provider name.
        intelligence: Intelligence tier.
        system_prompt_template: Template name used for system prompt.
        user_prompt_template: Template name used for user prompt.
        replacement_keys: List of replacement key names that were provided.
        system_prompt: The fully rendered system prompt.
        user_prompt: The fully rendered user prompt.
        warnings: Any warnings from template rendering.

    Returns:
        Path to the written debug file.
    """
    dir_path = Path(debug_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"query-ai-{timestamp}-{provider}-{intelligence}.json"
    file_path = dir_path / filename

    debug_data = {
        "timestamp": timestamp,
        "provider": provider,
        "intelligence": intelligence,
        "system_prompt_template": system_prompt_template,
        "user_prompt_template": user_prompt_template,
        "replacement_keys": replacement_keys,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "warnings": warnings,
    }

    file_path.write_text(json.dumps(debug_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return file_path


class LLMRouter:
    """Routes provider + intelligence requests to the correct LLM provider.

    Responsibilities:
    - Validate provider and intelligence tier exist in config
    - Merge defaults → provider config → model tier config into LLMOptions
    - Cache provider client instances (one per provider name)
    - Call provider.generate() with the merged options
    """

    def __init__(self, config: dict):
        self.config = config
        self._providers: Dict[str, LLMProvider] = {}

    def query(
        self, provider_name: str, intelligence: str, system_prompt: str, user_prompt: str
    ) -> LLMResult:
        """Route a query to the specified provider at the given intelligence tier.

        Args:
            provider_name: Provider key from config (e.g., "chatgpt", "gemini", "claude").
            intelligence: Tier key (e.g., "low", "middle", "high").
            system_prompt: Rendered system prompt text.
            user_prompt: Rendered user prompt text.

        Returns:
            LLMResult with response text and metadata.

        Raises:
            ValueError: If provider or intelligence is not found in config.
        """
        providers = self.config.get("providers") or {}
        if provider_name not in providers:
            valid = ", ".join(sorted(providers.keys()))
            raise ValueError(f"Unknown provider '{provider_name}'. Valid providers: {valid}")

        provider_config = providers[provider_name]
        models = provider_config.get("models") or {}
        if intelligence not in models:
            valid = ", ".join(sorted(models.keys()))
            raise ValueError(
                f"Unknown intelligence '{intelligence}' for provider '{provider_name}'. "
                f"Valid values: {valid}"
            )

        defaults = self.config.get("defaults") or {}
        model_config = models[intelligence]

        merged = {
            **defaults,
            **provider_config,
            **model_config,
        }

        options = LLMOptions(
            model=merged["model"],
            temperature=merged.get("temperature"),
            max_output_tokens=merged.get("max_output_tokens") or merged.get("max_tokens"),
            timeout_seconds=merged.get("timeout_seconds"),
            max_retries=int(merged.get("max_retries", 3)),
            retry_base_seconds=int(merged.get("retry_base_seconds", 30)),
        )

        provider = self._get_provider(provider_name, provider_config)
        return provider.generate(system_prompt, user_prompt, options)

    def _get_provider(self, provider_name: str, provider_config: dict) -> LLMProvider:
        """Get or create a cached provider instance."""
        if provider_name not in self._providers:
            self._providers[provider_name] = create_llm_provider(provider_name, provider_config)
        return self._providers[provider_name]
