# Phase 3: Config Loading + Prompt Engine — Discrete Implementation Steps

This phase creates:
1. A new utility module `prompt_engine.py` with all pure-function helpers
2. The `ai-mcp-server.yaml` configuration file
3. Test prompt templates under `prompts/`

These have zero provider dependencies — they can be fully tested without API keys.

---

## Step 1: Create `prompt_engine.py` with imports and constants

Create `prompt_engine.py`:

```python
"""Config loading, prompt template rendering, and utility functions for the AI MCP Server."""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
```

**Verify:** `python -c "import prompt_engine"` succeeds.

---

## Step 2: Add `load_yaml_config()`

Append to `prompt_engine.py`:

```python
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
```

**What this validates:**
- File exists and is valid YAML
- Top-level is a dict
- `providers` is a non-empty dict
- Each provider has a `models` dict
- Each tier in `models` is a dict with a `model` key

**What this does NOT validate (by design):**
- Doesn't require all three tiers (low/middle/high) — callers may only configure what they use
- Doesn't check API keys — those are resolved lazily when a provider is first instantiated
- Doesn't require `defaults` key — it's optional

**Verify:** After Step 5 (when the YAML file exists), test this loads it without error.

---

## Step 3: Add `resolve_prompt_path()`

Append to `prompt_engine.py`:

```python
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
```

**Verify:** Can test in isolation with a temp directory (Step 8).

---

## Step 4: Add `render_template()` and `combine_user_prompt()`

Append to `prompt_engine.py`:

```python
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
```

**Key design decisions:**
- `_PLACEHOLDER_RE` only matches `{identifier_style_names}` — arbitrary braces like `{"key": "val"}` in JSON examples are left untouched
- Missing placeholders → error (caller forgot a replacement value)
- Unused replacement values → warning (caller passed extra data, not fatal)
- Non-string values are JSON-serialized with pretty printing

**Verify:** Can test inline (Step 8).

---

## Step 5: Add `sha256_text()` and `save_debug_prompt()`

Append to `prompt_engine.py`:

```python
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
```

**Verify:** `sha256_text("hello")` returns a 64-char hex string.

---

## Step 6: Create `ai-mcp-server.yaml`

Create `ai-mcp-server.yaml` in the project root:

```yaml
defaults:
  temperature: 0.2
  max_output_tokens: 8000
  timeout_seconds: 180
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
        temperature: 0.2
        max_output_tokens: 8000
      high:
        model: o3
        temperature: 0.1
        max_output_tokens: 12000

  gemini:
    type: gemini
    api_key_env_var: GEMINI_API_KEY
    models:
      low:
        model: gemini-2.0-flash
        temperature: 0.2
        max_output_tokens: 4000
      middle:
        model: gemini-2.5-flash
        temperature: 0.2
        max_output_tokens: 8000
      high:
        model: gemini-2.5-pro
        temperature: 0.1
        max_output_tokens: 12000

  claude:
    type: anthropic
    api_key_env_var: ANTHROPIC_API_KEY
    models:
      low:
        model: claude-haiku-4-20250414
        temperature: 0.2
        max_output_tokens: 4000
      middle:
        model: claude-sonnet-4-20250514
        temperature: 0.2
        max_output_tokens: 8000
      high:
        model: claude-opus-4-20250514
        temperature: 0.1
        max_output_tokens: 12000
```

**Note:** Model names should be reviewed and updated to current available models. The ones above are reasonable defaults as of mid-2025. Replace any that aren't valid for your account.

**Verify:** `python -c "import yaml, pathlib; d=yaml.safe_load(pathlib.Path('ai-mcp-server.yaml').read_text()); print(list(d['providers'].keys()))"` prints `['chatgpt', 'gemini', 'claude']`.

---

## Step 7: Create test prompt templates

Create directory `prompts/` and two files:

**`prompts/test-system.md`:**
```markdown
You are a concise test assistant.
Return exactly what the user asks for, without extra commentary.
```

**`prompts/test-user.md`:**
```markdown
Say hello to {name}.
```

These provide a cheap smoke test for the full pipeline.

**Verify:** Both files exist and contain the expected content.

---

## Step 8: Verify all functions work end-to-end

Run this validation script:

```python
python -c "
from pathlib import Path
from prompt_engine import (
    load_yaml_config,
    resolve_prompt_path,
    render_template,
    combine_user_prompt,
    sha256_text,
)

# 1. Config loading
config = load_yaml_config(Path('ai-mcp-server.yaml'))
assert 'providers' in config
assert 'chatgpt' in config['providers']
assert config['providers']['chatgpt']['models']['low']['model'] == 'gpt-4.1-mini'
print('config loading: OK')

# 2. Prompt path resolution
prompt_dir = Path('prompts').resolve()
path = resolve_prompt_path(prompt_dir, 'test-system')
assert path.exists()
assert path.name == 'test-system.md'
print('prompt path resolve: OK')

# 3. Reject unsafe paths
import traceback
for bad in ['../secret', '/etc/passwd', 'C:/Windows/file', 'foo\\\\bar', 'a/../b']:
    try:
        resolve_prompt_path(prompt_dir, bad)
        assert False, f'Should have rejected: {bad}'
    except (ValueError, FileNotFoundError):
        pass
print('path safety checks: OK')

# 4. Template rendering
text = Path('prompts/test-user.md').read_text(encoding='utf-8')
rendered, warnings = render_template(text, {'name': 'Aaron'})
assert 'Aaron' in rendered
assert '{name}' not in rendered
assert len(warnings) == 0
print('render_template: OK')

# 5. Missing placeholder → error
try:
    render_template(text, {})
    assert False
except ValueError as e:
    assert 'Missing replacement values' in str(e)
print('missing placeholder error: OK')

# 6. Unused key → warning
rendered, warnings = render_template(text, {'name': 'Aaron', 'extra': 'unused'})
assert len(warnings) == 1
assert 'extra' in warnings[0]
print('unused key warning: OK')

# 7. Non-string value → JSON
template = 'Data: {payload}'
rendered, _ = render_template(template, {'payload': {'key': 'value'}})
assert '\"key\"' in rendered
print('JSON rendering: OK')

# 8. combine_user_prompt
combined = combine_user_prompt('Hello world.', 'Be brief.')
assert '---' in combined
assert 'Be brief.' in combined
plain = combine_user_prompt('Hello world.', '')
assert '---' not in plain
print('combine_user_prompt: OK')

# 9. sha256
h = sha256_text('hello')
assert len(h) == 64
assert h == sha256_text('hello')  # deterministic
print('sha256_text: OK')

print()
print('=== All Phase 3 checks passed ===')
"
```

**Verify:** Prints "All Phase 3 checks passed".

---

## Summary of what Phase 3 produces

```
prompt_engine.py
├── load_yaml_config(path) → dict
├── resolve_prompt_path(prompt_dir, template_name) → Path
├── render_template(template_text, replacement_values) → (str, list[str])
├── combine_user_prompt(rendered, additional) → str
├── sha256_text(text) → str
└── save_debug_prompt(...) → Path

ai-mcp-server.yaml
├── defaults (temperature, max_output_tokens, timeout, retries)
└── providers
    ├── chatgpt (low/middle/high models)
    ├── gemini (low/middle/high models)
    └── claude (low/middle/high models)

prompts/
├── test-system.md
└── test-user.md
```

**What this does NOT include (saved for Phase 4):**
- `LLMRouter` class
- `query_ai` tool integration
- Wiring these utilities into `ai-mcp-server.py`
