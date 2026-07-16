# Phase 4: Router + Tool Integration — Discrete Implementation Steps

This phase wires Phases 1–3 together by:
1. Adding `LLMRouter` to `prompt_engine.py`
2. Revising `create_public_server()` in `ai-mcp-server.py` to load config, create the router, and register both the `query_ai` and revised `health_check` tools
3. Adding the necessary imports to `ai-mcp-server.py`

After this phase, calling `query_ai` from an MCP client will load templates, render them, route to a provider, and return a normalized response.

**Files modified:**
- `prompt_engine.py` — add `LLMRouter` class
- `ai-mcp-server.py` — add imports, revise `create_public_server()`

---

## Step 1: Add `LLMRouter` to `prompt_engine.py`

Append to the bottom of `prompt_engine.py`:

```python
from llm_providers import LLMOptions, LLMResult, LLMProvider, create_llm_provider


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
```

**Key design decisions:**
- Provider instances are cached — one client per provider name, reused across intelligence tiers
- Model comes from merged config → `LLMOptions`, NOT stored on the provider instance
- Merge priority: `model_config` overrides `provider_config` overrides `defaults`
- The `merged` dict will contain extra keys like `type`, `api_key_env_var`, `models` — those are harmless because `LLMOptions` only picks the keys it cares about

**Verify:** `from prompt_engine import LLMRouter` succeeds.

---

## Step 2: Add imports to `ai-mcp-server.py`

Add these imports after the existing `import yaml` line (around line 28):

```python
import time
import traceback

from prompt_engine import (
    load_yaml_config,
    resolve_prompt_path,
    render_template,
    combine_user_prompt,
    sha256_text,
    save_debug_prompt,
    LLMRouter,
)
```

**What changed:** Added `time` (for elapsed_ms measurement), `traceback` (for logging unexpected exceptions), and all `prompt_engine` utilities.

**Verify:** File still parses.

---

## Step 3: Update `MCPServer._router` type annotation

In `MCPServer.__init__`, change:

```python
self._router: Optional[Any] = None
```

to:

```python
self._router: Optional[LLMRouter] = None
```

**Verify:** File still parses. The type is now properly annotated.

---

## Step 4: Rewrite `create_public_server()` — initialization block

Replace the try/except block at the top of `create_public_server()` (where `self._prompt_dir` is set and `mcp` is created) with:

```python
def create_public_server(self) -> Optional[FastMCP]:
    """Create the AI MCP server with tool registration."""
    try:
        self._config = load_yaml_config(Path(self._settings.config_path))
        self._prompt_dir = Path(self._settings.prompt_dir).resolve()
        self._router = LLMRouter(self._config)

        mcp = FastMCP(
            name=self._settings.server_name,
            instructions=(
                "Generic AI provider routing server. "
                "Renders prompt templates and queries configured LLM providers."
            ),
            debug=True,
            stateless_http=True,
        )

        self._is_ready = True

    except Exception as e:
        self._initialization_error = str(e)
        logger.error(f"Failed to initialize AI MCP server: {e}")
        return None
```

**What changed:** Added `load_yaml_config()` and `LLMRouter()` initialization before creating the FastMCP instance.

---

## Step 5: Rewrite `health_check` tool

Replace the existing `health_check` tool definition with:

```python
    @mcp.tool()
    def health_check() -> Dict[str, Any]:
        """Check the health status of the AI MCP server."""
        if not self._is_ready:
            return {
                "status": "error",
                "message": self._initialization_error or "Server not initialized",
            }

        providers = sorted((self._config.get("providers") or {}).keys())

        return {
            "status": "ready",
            "server_name": self._settings.server_name,
            "providers": providers,
            "prompt_dir": str(self._prompt_dir),
            "config_path": self._settings.config_path,
        }
```

**What changed:** Now includes `providers` list in the response.

---

## Step 6: Add the `query_ai` tool

After `health_check`, before the tool compatibility patch, add the `query_ai` tool:

```python
    @mcp.tool(
        name="query_ai",
        description="Render system/user prompt templates and query a configured LLM provider."
    )
    def query_ai(
        provider: Annotated[str, Field(description="Provider name from YAML config (e.g., chatgpt, gemini, claude)")],
        intelligence: Annotated[str, Field(description="Model tier: low, middle, or high")],
        system_prompt_template: Annotated[str, Field(description="System prompt template path under prompts/, without .md extension")],
        user_prompt_template: Annotated[str, Field(description="User prompt template path under prompts/, without .md extension")],
        additional_user_prompt: Annotated[str, Field(description="Extra user instructions appended after the rendered user template")] = "",
        replacement_values: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        """Render prompt templates, query the configured LLM, return normalized response."""
        if not self._is_ready or self._router is None or self._prompt_dir is None:
            return {
                "status": "error",
                "error_type": "ServerNotReady",
                "message": self._initialization_error or "Server not initialized",
            }

        try:
            # Load templates
            sys_path = resolve_prompt_path(self._prompt_dir, system_prompt_template)
            usr_path = resolve_prompt_path(self._prompt_dir, user_prompt_template)

            sys_template_text = sys_path.read_text(encoding="utf-8")
            usr_template_text = usr_path.read_text(encoding="utf-8")

            # Render templates
            system_prompt, sys_warnings = render_template(sys_template_text, replacement_values)
            user_prompt_rendered, usr_warnings = render_template(usr_template_text, replacement_values)
            warnings = sys_warnings + usr_warnings

            # Combine user prompt with additional instructions
            user_prompt = combine_user_prompt(user_prompt_rendered, additional_user_prompt)

            # Compute hashes
            system_prompt_hash = sha256_text(system_prompt)
            user_prompt_hash = sha256_text(user_prompt)

            # Optionally save debug prompts
            if self._settings.save_debug_prompts:
                save_debug_prompt(
                    debug_dir=self._settings.debug_dir,
                    provider=provider,
                    intelligence=intelligence,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                    replacement_keys=sorted(replacement_values.keys()),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    warnings=warnings,
                )

            # Call the LLM
            start_time = time.time()
            result = self._router.query(provider, intelligence, system_prompt, user_prompt)
            elapsed_ms = int((time.time() - start_time) * 1000)

            return {
                "status": "success",
                "provider": provider,
                "intelligence": intelligence,
                "model": result.model,
                "response_text": result.text,
                "warnings": warnings,
                "usage": result.usage,
                "elapsed_ms": elapsed_ms,
                "system_prompt_template": system_prompt_template,
                "user_prompt_template": user_prompt_template,
                "system_prompt_sha256": system_prompt_hash,
                "user_prompt_sha256": user_prompt_hash,
            }

        except (ValueError, FileNotFoundError) as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
            }
        except Exception as e:
            logger.error(f"Unexpected error in query_ai: {traceback.format_exc()}")
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
            }
```

**Implementation flow:**
1. Check server is ready
2. Load system + user template files via safe path resolution
3. Render both with replacement_values (errors if placeholders missing)
4. Combine user prompt with additional instructions
5. Compute SHA-256 hashes of final prompts
6. Optionally save debug files
7. Call `self._router.query()` (which resolves model, creates/caches provider, calls generate)
8. Return normalized success response with all metadata
9. Catch expected errors (ValueError, FileNotFoundError) → clean error response
10. Catch unexpected errors → log traceback, return generic error

---

## Step 7: Keep the tool compatibility patch

The compatibility patch that already exists at the bottom of `create_public_server()` stays in place, unchanged:

```python
    # Tool compatibility patch
    try:
        for tool in mcp._tool_manager.list_tools():
            if not hasattr(tool, "function"):
                func = getattr(tool, "_callback", None) or getattr(tool, "callback", None) or getattr(tool, "call", None)
                if func is not None:
                    setattr(tool, "function", func)
    except Exception:
        pass

    return mcp
```

No changes needed here.

---

## Step 8: Verify the file parses and tools register

Run:
```
python -c "import ast; ast.parse(open('ai-mcp-server.py', encoding='utf-8').read()); print('parse ok')"
```

Then verify initialization and tool registration:
```python
python -c "
import sys; sys.path.insert(0, '.')
exec(open('ai-mcp-server.py', encoding='utf-8').read().split('if __name__')[0])
s = ServerSettings()
m = MCPServer(s)
ps = m.create_public_server()
assert ps is not None, f'create_public_server failed: {m._initialization_error}'
assert m._is_ready
assert m._router is not None
assert m._config is not None
tools = [t.name for t in ps._tool_manager.list_tools()]
assert 'health_check' in tools
assert 'query_ai' in tools
print(f'Tools registered: {tools}')
print('Phase 4 init check: OK')
"
```

---

## Step 9: Verify health_check returns providers list

```python
python -c "
import sys; sys.path.insert(0, '.')
exec(open('ai-mcp-server.py', encoding='utf-8').read().split('if __name__')[0])
s = ServerSettings()
m = MCPServer(s)
ps = m.create_public_server()
# Find and call health_check
for tool in ps._tool_manager.list_tools():
    if tool.name == 'health_check':
        fn = getattr(tool, 'function', None) or getattr(tool, '_callback', None) or getattr(tool, 'callback', None)
        if fn:
            result = fn()
            assert result['status'] == 'ready'
            assert 'chatgpt' in result['providers']
            assert 'gemini' in result['providers']
            assert 'claude' in result['providers']
            print(f'health_check result: {result}')
            break
print('Phase 4 health_check: OK')
"
```

---

## Step 10: Verify query_ai returns validation errors correctly

Test without a real API key — just verify that template loading and rendering work up to the point of provider instantiation:

```python
python -c "
import sys; sys.path.insert(0, '.')
exec(open('ai-mcp-server.py', encoding='utf-8').read().split('if __name__')[0])
s = ServerSettings()
m = MCPServer(s)
ps = m.create_public_server()

# Find query_ai function
query_fn = None
for tool in ps._tool_manager.list_tools():
    if tool.name == 'query_ai':
        query_fn = getattr(tool, 'function', None) or getattr(tool, '_callback', None) or getattr(tool, 'callback', None)
        break

# Test 1: Bad provider
r = query_fn(provider='invalid', intelligence='low', system_prompt_template='test-system', user_prompt_template='test-user', replacement_values={'name': 'test'})
assert r['status'] == 'error'
assert 'Unknown provider' in r['message']
print(f'bad provider: {r[\"message\"]}')

# Test 2: Bad intelligence
r = query_fn(provider='chatgpt', intelligence='ultra', system_prompt_template='test-system', user_prompt_template='test-user', replacement_values={'name': 'test'})
assert r['status'] == 'error'
assert 'Unknown intelligence' in r['message']
print(f'bad intelligence: {r[\"message\"]}')

# Test 3: Missing template
r = query_fn(provider='chatgpt', intelligence='low', system_prompt_template='nonexistent', user_prompt_template='test-user', replacement_values={'name': 'test'})
assert r['status'] == 'error'
assert 'not found' in r['message']
print(f'missing template: {r[\"message\"]}')

# Test 4: Missing replacement value
r = query_fn(provider='chatgpt', intelligence='low', system_prompt_template='test-system', user_prompt_template='test-user', replacement_values={})
assert r['status'] == 'error'
assert 'Missing replacement values' in r['message']
print(f'missing replacement: {r[\"message\"]}')

# Test 5: Unsafe template path
r = query_fn(provider='chatgpt', intelligence='low', system_prompt_template='../etc/passwd', user_prompt_template='test-user', replacement_values={'name': 'test'})
assert r['status'] == 'error'
print(f'unsafe path: {r[\"message\"]}')

print()
print('Phase 4 query_ai validation: OK')
"
```

---

## Summary of what the file looks like after Phase 4

```
ai-mcp-server.py
├── Imports (now includes time, traceback, prompt_engine utilities, LLMRouter)
├── setup_logger()
├── ServerSettings
├── FastAppSettings
├── FastApp
├── MCPServer
│   ├── __init__(settings) — _router typed as Optional[LLMRouter]
│   ├── create_public_server()
│   │   ├── load_yaml_config() → self._config
│   │   ├── resolve prompt_dir → self._prompt_dir
│   │   ├── LLMRouter(config) → self._router
│   │   ├── FastMCP creation
│   │   ├── health_check tool (returns providers list)
│   │   ├── query_ai tool (full pipeline)
│   │   └── tool compatibility patch
│   ├── run_stdio_server()
│   ├── run_http_server_with_mcp()
│   └── run_http_server()
└── main()
```

**What Phase 4 achieves:**
- The server loads config and creates a router on startup
- `health_check` reports available providers
- `query_ai` performs the full pipeline: path validation → template loading → rendering → provider routing → normalized response
- Errors are returned as structured objects, never raw tracebacks
- Real API calls only happen when `query_ai` is called with a valid provider that has its API key set
