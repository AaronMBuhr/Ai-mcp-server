
# Implementation Plan: Revise Existing Python MCP Shell into Generic AI MCP Server

## 1. Objective

Revise the existing working MCP server shell into a generic AI routing MCP server.

The new server will expose one primary MCP tool:

```text
query_ai
```

The tool will accept:

```text
provider
intelligence: low | middle | high
system_prompt_template
user_prompt_template
additional_user_prompt
replacement_values
```

The server will:

```text
1. Load ai-mcp-server.yaml.
2. Resolve provider + intelligence to a concrete model.
3. Load the requested system/user prompt template Markdown files from prompts/.
4. Replace placeholders like {job_description} using replacement_values.
5. Append additional_user_prompt to the rendered user prompt.
6. Send system + user prompts to the selected LLM provider.
7. Return a normalized response to Claude Code.
```

The MCP server must remain generic. It must not contain resume workflow logic, job analysis logic, DOCX generation logic, or Claude Code orchestration logic.

Those belong in:

```text
Claude Code SKILL.md
prompt templates
caller-side workflow
```

## 2. Major Design Decision

Do not rewrite the MCP server shell from scratch.

Use the existing working MCP server file as the base because it already has:

```text
working FastMCP setup
working stdio mode
working HTTP/HTTPS mode
working FastAPI mounting
working CLI/env argument pattern
working MCPServer class wrapper
working health_check-style utility tool
working logging-to-stderr setup
```

Replace the documentation-search internals with AI-routing internals.

## 3. What to Keep from the Current MCP Server Shell

Keep and revise these pieces:

```text
setup_logger()
ServerSettings
FastAppSettings
FastApp
MCPServer
MCPServer.create_public_server()
MCPServer.run_stdio_server()
MCPServer.run_http_server_with_mcp()
MCPServer.run_http_server()
main()
```

Keep the basic structure:

```text
load env
parse args
create settings
create MCPServer(settings)
if stdio: run stdio
if http/https: run HTTP/HTTPS
```

Keep the FastMCP registration pattern:

```python
@mcp.tool(...)
def query_ai(...):
    ...
```

Keep the HTTP mounting behavior:

```python
app.mount("/mcp", server.streamable_http_app())
app.mount(f"/{server.name}", server.streamable_http_app())
```

Keep logging to stderr. This matters for stdio MCP because stdout must remain clean for protocol messages.

## 4. What to Remove from the Current MCP Server Shell

Remove all documentation/search-specific imports:

```python
from search_engine_faiss import FaissSearchEngine
from models import Document, DocumentChunk, SearchEngine
```

Remove documentation/search settings:

```text
model_name
embedding_path
embedding_name
transformer_device
device argument
DEFAULTS.MODEL_NAME
DEFAULTS.EMBEDDING_PATH
```

Remove search engine fields from `MCPServer`:

```python
self._search_engine
self._device
```

Remove these MCP tools:

```text
search_for_chunks
search_for_documents
```

Remove FAISS/search initialization from `create_public_server()`.

Remove any error help text that talks about FAISS index files or embeddings.

The new server has nothing to do with embeddings or documentation search.

## 5. What to Keep from the Existing LLM Library

Keep and revise:

```text
LLMProvider
GeminiProvider
ClaudeProvider
ChatGPTProvider
create_llm_provider
API key resolution from env vars
provider-specific authentication errors
provider-specific model-not-found errors
rate-limit retry behavior
streaming response accumulation
```

The current library already has the right broad provider split:

```text
gemini
claude
chatgpt
```

It already requires explicit model configuration, which is good because API calls cost money.

## 6. What to Remove from the Existing LLM Library

Remove or move out of the generic AI MCP server:

```text
DEFAULT_JOBS_PER_LLM_BATCH
MAX_LLM_JOBS_PER_ROUND
jobs_per_llm_batch
load_resume_text
resume cache
resume compression prompts
resume verification prompts
get_resume_for_matching
build_exclusion_rules
build_scoring_guidance
extract_whitelisted_job_fields
build_jobs_listings_text
job prompt builders
batch job analysis
final round job analysis
job score parsing
job score normalization
merge_scores_with_jobs
markdown job formatting
duplicate-output cleanup
job result validation
```

These may remain in `find-jobs.py` or a separate job-matching library, but they should not be reachable from this generic MCP server.

## 7. Recommended Final File Layout

Do not over-engineer the first revision. Keep it understandable for a junior developer.

Recommended layout:

```text
ai-mcp-server/
  ai-mcp-server.py
  llm_providers.py
  ai-mcp-server.yaml
  prompts/
    test-system.md
    test-user.md
    job/
      analyze-system.md
      analyze-user.md
      synthesize-system.md
      synthesize-user.md
    resume/
      draft-system.md
      draft-user.md
      docx-script-system.md
      docx-script-user.md
  tests/
    test_prompt_rendering.py
    test_config_loading.py
    test_llm_routing.py
```

Explanation:

```text
ai-mcp-server.py
  Holds the existing revised MCP shell, settings, tool registration, and transport handling.

llm_providers.py
  Holds the revised provider abstraction and Gemini/Claude/ChatGPT provider implementations.

ai-mcp-server.yaml
  Holds model/provider/prompt directory configuration.

prompts/
  Holds Markdown system/user prompt templates.

tests/
  Holds focused tests for config, prompt rendering, and routing.
```

A package layout can come later. The immediate goal is a working, obvious, maintainable script.

## 8. New ServerSettings Design

Revise `ServerSettings` to remove documentation fields and add AI server fields.

Use:

```python
class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_", extra="allow")

    transport: str = Field(default="stdio")
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    server_url: AnyHttpUrl = Field(default="http://localhost:8000")

    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    server_name: str = Field(default="ai-mcp-server")
    description: str = Field(default="generic AI provider routing server")

    config_path: str = Field(default="ai-mcp-server.yaml")
    prompt_dir: str = Field(default="prompts")

    save_debug_prompts: bool = Field(default=False)
    debug_dir: str = Field(default=".debug/ai-mcp-server")
```

Environment variables should continue to use the `MCP_` prefix:

```text
MCP_TRANSPORT
MCP_HOST
MCP_PORT
MCP_SERVER_NAME
MCP_DESCRIPTION
MCP_CONFIG_PATH
MCP_PROMPT_DIR
MCP_SAVE_DEBUG_PROMPTS
MCP_DEBUG_DIR
```

## 9. Revised CLI Arguments

Keep the current CLI style, but remove embedding/search options.

Keep:

```text
--debug
--transport
--host
--port
--ssl-cert
--ssl-key
--server-name
--description
```

Add:

```text
--config
--prompt-dir
--save-debug-prompts
--debug-dir
```

Remove:

```text
--device
--model
--embedding-path
--embedding-name
```

Revised parser arguments:

```python
parser.add_argument("--config", type=str,
    default=os.getenv("MCP_CONFIG_PATH", "ai-mcp-server.yaml"),
    help="Path to ai-mcp-server YAML config.")

parser.add_argument("--prompt-dir", type=str,
    default=os.getenv("MCP_PROMPT_DIR", "prompts"),
    help="Directory containing prompt templates.")

parser.add_argument("--save-debug-prompts", action="store_true",
    default=os.getenv("MCP_SAVE_DEBUG_PROMPTS", "").lower() in ("1", "true", "yes"),
    help="Save rendered prompts to debug files.")

parser.add_argument("--debug-dir", type=str,
    default=os.getenv("MCP_DEBUG_DIR", ".debug/ai-mcp-server"),
    help="Directory for debug prompt files.")
```

## 10. YAML Configuration

Create `ai-mcp-server.yaml`.

Shape:

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
        model: REPLACE_WITH_PREFERRED_HIGH_OPENAI_MODEL
        temperature: 0.1
        max_output_tokens: 12000

  gemini:
    type: gemini
    api_key_env_var: GEMINI_API_KEY
    models:
      low:
        model: REPLACE_WITH_PREFERRED_LOW_GEMINI_MODEL
        temperature: 0.2
        max_output_tokens: 4000
      middle:
        model: REPLACE_WITH_PREFERRED_MIDDLE_GEMINI_MODEL
        temperature: 0.2
        max_output_tokens: 8000
      high:
        model: REPLACE_WITH_PREFERRED_HIGH_GEMINI_MODEL
        temperature: 0.1
        max_output_tokens: 12000

  claude:
    type: anthropic
    api_key_env_var: ANTHROPIC_API_KEY
    models:
      low:
        model: REPLACE_WITH_PREFERRED_LOW_CLAUDE_MODEL
        temperature: 0.2
        max_output_tokens: 4000
      middle:
        model: REPLACE_WITH_PREFERRED_MIDDLE_CLAUDE_MODEL
        temperature: 0.2
        max_output_tokens: 8000
      high:
        model: REPLACE_WITH_PREFERRED_HIGH_CLAUDE_MODEL
        temperature: 0.1
        max_output_tokens: 12000
```

Decision:

```text
Claude Code and callers pass provider + intelligence.
They do not pass raw model names.
Raw model names live only in YAML.
```

## 11. MCP Tool Contract

Expose one main tool:

```text
query_ai
```

Tool parameters:

```python
provider: Annotated[str, Field(description="Provider name: chatgpt, gemini, or claude")]
intelligence: Annotated[str, Field(description="Model tier: low, middle, or high")]
system_prompt_template: Annotated[str, Field(description="System prompt template name under prompts/, without .md")]
user_prompt_template: Annotated[str, Field(description="User prompt template name under prompts/, without .md")]
additional_user_prompt: Annotated[str, Field(description="Extra user instructions appended after the rendered user template")] = ""
replacement_values: Dict[str, Any] = {}
```

The request will look like:

```json
{
  "provider": "chatgpt",
  "intelligence": "high",
  "system_prompt_template": "resume/draft-system",
  "user_prompt_template": "resume/draft-user",
  "additional_user_prompt": "Emphasize .NET/backend work and avoid frontend-heavy positioning.",
  "replacement_values": {
    "job_description": "...",
    "master_resume": "...",
    "analysis_gemini": "...",
    "analysis_chatgpt": "..."
  }
}
```

Tool response:

```json
{
  "status": "success",
  "provider": "chatgpt",
  "intelligence": "high",
  "model": "configured-model-name",
  "response_text": "...",
  "warnings": [],
  "usage": null,
  "elapsed_ms": 12345,
  "system_prompt_template": "resume/draft-system",
  "user_prompt_template": "resume/draft-user",
  "system_prompt_sha256": "...",
  "user_prompt_sha256": "..."
}
```

On validation or provider failure:

```json
{
  "status": "error",
  "error_type": "ValidationError",
  "message": "Unknown provider 'x'. Valid providers: chatgpt, gemini, claude"
}
```

Use returned error objects rather than throwing raw tracebacks from tool functions. Log the traceback server-side if debug logging is enabled.

## 12. Prompt Template Loading

Templates live under `prompts/`.

If the caller passes:

```text
test-system
```

load:

```text
prompts/test-system.md
```

If the caller passes:

```text
resume/draft-user
```

load:

```text
prompts/resume/draft-user.md
```

Rules:

```text
1. Template names are relative.
2. Do not allow absolute paths.
3. Do not allow '..'.
4. Do not allow backslashes.
5. Do not allow drive-letter paths.
6. Automatically append .md.
7. Resolve the final path and confirm it is inside the prompt directory.
8. Return validation error if missing.
```

Allowed template name regex:

```regex
^[A-Za-z0-9_/-]+$
```

Path safety helper:

```python
def resolve_prompt_path(prompt_dir: Path, template_name: str) -> Path:
    if not re.match(r"^[A-Za-z0-9_/-]+$", template_name):
        raise ValueError("Unsafe prompt template name")

    if ".." in template_name.split("/"):
        raise ValueError("Prompt template may not contain '..'")

    if "\\" in template_name:
        raise ValueError("Prompt template may not contain backslashes")

    path = (prompt_dir / f"{template_name}.md").resolve()
    root = prompt_dir.resolve()

    if root not in path.parents and path != root:
        raise ValueError("Prompt template resolved outside prompt directory")

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_name}.md")

    return path
```

## 13. Replacement Rules

Placeholders use this format:

```text
{key}
```

Example template:

```markdown
Analyze this job for Aaron.

MASTER RESUME:
{master_resume}

JOB DESCRIPTION:
{job_description}
```

`replacement_values` is a JSON object/dict.

Rules:

```text
1. Replacement keys must match ^[A-Za-z_][A-Za-z0-9_]*$.
2. String values are inserted exactly.
3. int/float/bool/null/list/dict values are converted to pretty JSON.
4. Replace exact occurrences of {key}.
5. Missing placeholders are errors.
6. Supplied-but-unused replacement values are warnings.
7. Do not use Python str.format().
```

Why not `str.format()`:

```text
Prompt templates may contain JSON examples, Markdown examples, code, or literal braces.
Using str.format() makes those brittle.
Use explicit placeholder replacement instead.
```

Implementation:

```python
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

def render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)

def render_template(template_text: str, replacement_values: Dict[str, Any]) -> tuple[str, list[str]]:
    warnings = []

    for key in replacement_values:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise ValueError(f"Invalid replacement key: {key}")

    placeholders = set(PLACEHOLDER_RE.findall(template_text))

    missing = sorted(placeholders - set(replacement_values.keys()))
    if missing:
        raise ValueError(f"Missing replacement values: {', '.join(missing)}")

    unused = sorted(set(replacement_values.keys()) - placeholders)
    for key in unused:
        warnings.append(f"Replacement value '{key}' was supplied but not used by this template.")

    rendered = template_text
    for key in placeholders:
        rendered = rendered.replace("{" + key + "}", render_value(replacement_values[key]))

    return rendered, warnings
```

## 14. Combining Additional User Prompt

Render the user template first.

Then append `additional_user_prompt` only if non-blank:

```markdown
---

Additional user instructions:

<additional_user_prompt>
```

Do not append an empty section.

Implementation:

```python
def combine_user_prompt(rendered_user_prompt: str, additional_user_prompt: str) -> str:
    extra = (additional_user_prompt or "").strip()
    if not extra:
        return rendered_user_prompt.strip()

    return (
        rendered_user_prompt.rstrip()
        + "\n\n---\n\nAdditional user instructions:\n\n"
        + extra
    )
```

## 15. Revised LLM Provider Interface

The current provider method accepts one prompt:

```python
generate(prompt: str) -> str
```

Revise it to accept separate system and user prompts:

```python
generate(system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult
```

Add:

```python
@dataclass
class LLMOptions:
    model: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_base_seconds: int = 30

@dataclass
class LLMResult:
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
```

The provider object should still own:

```text
API key resolution
client construction
provider-specific call syntax
provider-specific error handling
```

The router owns:

```text
provider + intelligence → model/options
```

## 16. Revised llm_providers.py

Create `llm_providers.py` from the useful top portion of the existing LLM library.

Keep:

```text
LLMProvider base class
GeminiProvider
ClaudeProvider
ChatGPTProvider
create_llm_provider
```

Remove the job/resume functions.

Suggested file structure:

```python
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMOptions:
    model: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    timeout_seconds: Optional[int] = None
    max_retries: int = 3
    retry_base_seconds: int = 30


@dataclass
class LLMResult:
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class LLMProvider(ABC):
    def __init__(self, provider_config: dict[str, Any]):
        self.config = provider_config

    def _resolve_api_key(self) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, options: LLMOptions) -> LLMResult:
        ...


class GeminiProvider(LLMProvider):
    ...


class ClaudeProvider(LLMProvider):
    ...


class ChatGPTProvider(LLMProvider):
    ...


def create_llm_provider(provider_name: str, provider_config: dict[str, Any]) -> LLMProvider:
    ...
```

## 17. Provider Call Details

### 17.1 ChatGPTProvider

Use OpenAI Python client.

First acceptable implementation can use `chat.completions.create`, because the existing code already works that way.

Messages:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
```

Call:

```python
stream = self._client.chat.completions.create(
    model=options.model,
    max_tokens=options.max_output_tokens,
    temperature=options.temperature,
    messages=messages,
    stream=True,
)
```

Accumulate streamed text.

Handle:

```text
AuthenticationError
PermissionDeniedError
NotFoundError
RateLimitError
quota/billing errors
```

Use existing retry logic, but replace hardcoded retry count and wait times with `options.max_retries` and `options.retry_base_seconds`.

### 17.2 ClaudeProvider

Use Anthropic Python client.

Call with system prompt separated:

```python
with self._client.messages.stream(
    model=options.model,
    max_tokens=options.max_output_tokens or 8192,
    temperature=options.temperature,
    system=system_prompt,
    messages=[
        {"role": "user", "content": user_prompt}
    ],
) as stream:
    ...
```

Keep streaming text accumulation.

Use existing Anthropic rate-limit handling.

### 17.3 GeminiProvider

Use Google GenAI client.

Call with system prompt as system instruction if supported by your installed SDK version.

Target shape:

```python
response = self._client.models.generate_content(
    model=options.model,
    contents=user_prompt,
    config={
        "system_instruction": system_prompt,
        "temperature": options.temperature,
        "max_output_tokens": options.max_output_tokens,
    },
)
```

If the local SDK version uses typed config objects, adapt to that SDK. The junior developer should check the installed `google-genai` version and use the matching syntax.

Retry on:

```text
429
RESOURCE_EXHAUSTED
rate limit
```

## 18. LLM Router

Add router code to `ai-mcp-server.py` or a small helper module.

Responsibilities:

```text
load config
validate provider exists
validate intelligence exists
merge defaults with model config
cache provider clients
call provider.generate()
return LLMResult
```

Pseudo-code:

```python
class LLMRouter:
    def __init__(self, config: dict):
        self.config = config
        self._providers: Dict[str, LLMProvider] = {}

    def query(self, provider_name: str, intelligence: str, system_prompt: str, user_prompt: str) -> LLMResult:
        providers = self.config.get("providers") or {}
        if provider_name not in providers:
            raise ValueError(f"Unknown provider '{provider_name}'. Valid providers: {', '.join(providers)}")

        provider_config = providers[provider_name]
        models = provider_config.get("models") or {}
        if intelligence not in models:
            raise ValueError(
                f"Unknown intelligence '{intelligence}' for provider '{provider_name}'. "
                f"Valid values: {', '.join(models)}"
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
        if provider_name not in self._providers:
            self._providers[provider_name] = create_llm_provider(provider_name, provider_config)
        return self._providers[provider_name]
```

Important detail:

```text
Do not put the selected model into the cached provider instance.
The model comes from LLMOptions per call.
```

This is a change from the existing LLM provider code, where each provider stores `self._model`.

## 19. Revised MCPServer Class

Revise `MCPServer.__init__`:

```python
class MCPServer:
    def __init__(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._logger = logger
        self._is_ready = False
        self._initialization_error: Optional[str] = None
        self._config: Optional[dict] = None
        self._router: Optional[LLMRouter] = None
        self._prompt_dir: Optional[Path] = None
```

Remove:

```python
self._device
self._search_engine
```

## 20. Revised create_public_server()

Replace FAISS initialization with config/router initialization.

Conceptual code:

```python
def create_public_server(self) -> Optional[FastMCP]:
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

Then register tools:

```python
@mcp.tool(name="query_ai", description="Render prompt templates and query a configured LLM provider.")
def query_ai(...):
    ...
```

Keep the `health_check()` tool but revise it.

## 21. query_ai Implementation

Inside `create_public_server()`, register:

```python
@mcp.tool(
    name="query_ai",
    description="Render system/user prompt templates and query a configured LLM provider."
)
def query_ai(
    provider: Annotated[str, Field(description="Provider name from YAML config")],
    intelligence: Annotated[str, Field(description="low, middle, or high")],
    system_prompt_template: Annotated[str, Field(description="System prompt template under prompts/, without .md")],
    user_prompt_template: Annotated[str, Field(description="User prompt template under prompts/, without .md")],
    additional_user_prompt: Annotated[str, Field(description="Additional user instructions appended to user prompt")] = "",
    replacement_values: Dict[str, Any] = {},
) -> Dict[str, Any]:
    ...
```

Implementation flow:

```text
1. If server not ready, return status=error.
2. Validate provider/intelligence strings.
3. Load system template.
4. Load user template.
5. Render system template with replacement_values.
6. Render user template with replacement_values.
7. Combine user prompt with additional_user_prompt.
8. Compute SHA256 hashes of rendered prompts.
9. Optionally save debug prompt file.
10. Call self._router.query(...).
11. Return normalized success object.
12. Catch expected errors and return normalized error object.
13. Log unexpected exceptions with traceback, return generic error object.
```

Return success:

```python
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
    "system_prompt_sha256": sha256_text(system_prompt),
    "user_prompt_sha256": sha256_text(user_prompt),
}
```

Return error:

```python
return {
    "status": "error",
    "error_type": type(e).__name__,
    "message": str(e),
}
```

## 22. Revised health_check Tool

Keep `health_check`, but make it AI-specific:

```python
@mcp.tool()
def health_check() -> Dict[str, Any]:
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

Do not test real API keys in health check. Health check should be cheap and non-billable.

## 23. Debug Prompt Files

If `save_debug_prompts` is enabled, save rendered prompts to:

```text
.debug/ai-mcp-server/
```

Filename:

```text
query-ai-YYYYMMDD-HHMMSS-provider-intelligence.json
```

Contents:

```json
{
  "timestamp": "...",
  "provider": "chatgpt",
  "intelligence": "high",
  "system_prompt_template": "resume/draft-system",
  "user_prompt_template": "resume/draft-user",
  "replacement_keys": ["job_description", "master_resume"],
  "system_prompt": "...",
  "user_prompt": "...",
  "warnings": []
}
```

Never save API keys.

Debug prompt saving is off by default.

## 24. Revised main()

Keep current `main()` shape, but revise labels and args.

New initialization log:

```python
logger.info("🚀 AI MCP Server initializing...")
```

Remove FAISS-specific help.

Transport switch remains:

```python
mcp_server = MCPServer(settings)

if settings.transport == "stdio":
    mcp_server.run_stdio_server()
elif settings.transport in ["http", "https"]:
    return mcp_server.run_http_server_with_mcp()
else:
    ...
```

## 25. FastApp Help Page

Keep the existing help page idea, but revise the text.

Change title:

```text
AI MCP Server
```

Description:

```text
This server provides generic LLM routing through the Model Context Protocol.
```

Keep listing tools. That existing behavior is useful.

## 26. Tool Compatibility Patch

Keep this section unless it causes problems:

```python
for tool in mcp._tool_manager.list_tools():
    if not hasattr(tool, "function"):
        func = getattr(tool, "_callback", None) or getattr(tool, "callback", None) or getattr(tool, "call", None)
        if func is not None:
            setattr(tool, "function", func)
```

Reason:

```text
It appears to be a compatibility workaround for FastMCP tool objects in the current shell.
Do not remove it in the first revision unless tests prove it is unnecessary.
```

## 27. First Prompt Files

Create `prompts/test-system.md`:

```markdown
You are a concise test assistant.
Return exactly what the user asks for, without extra commentary.
```

Create `prompts/test-user.md`:

```markdown
Say hello to {name}.
```

These allow a cheap smoke test.

## 28. Example MCP Request

```json
{
  "provider": "chatgpt",
  "intelligence": "low",
  "system_prompt_template": "test-system",
  "user_prompt_template": "test-user",
  "additional_user_prompt": "Add one short sentence saying the AI MCP server worked.",
  "replacement_values": {
    "name": "Aaron"
  }
}
```

Expected behavior:

```text
Loads prompts/test-system.md
Loads prompts/test-user.md
Replaces {name} with Aaron
Appends additional user prompt
Calls configured chatgpt low model
Returns response_text
```

## 29. Example Resume Workflow Calls

The server does not know this is a resume workflow. Claude Code SKILL.md will orchestrate calls like:

```json
{
  "provider": "gemini",
  "intelligence": "middle",
  "system_prompt_template": "job/analyze-system",
  "user_prompt_template": "job/analyze-user",
  "additional_user_prompt": "",
  "replacement_values": {
    "job_description": "...",
    "master_resume": "..."
  }
}
```

Then:

```json
{
  "provider": "chatgpt",
  "intelligence": "middle",
  "system_prompt_template": "job/analyze-system",
  "user_prompt_template": "job/analyze-user",
  "additional_user_prompt": "",
  "replacement_values": {
    "job_description": "...",
    "master_resume": "..."
  }
}
```

Then:

```json
{
  "provider": "chatgpt",
  "intelligence": "high",
  "system_prompt_template": "job/synthesize-system",
  "user_prompt_template": "job/synthesize-user",
  "additional_user_prompt": "",
  "replacement_values": {
    "job_description": "...",
    "master_resume": "...",
    "gemini_analysis": "...",
    "chatgpt_analysis": "..."
  }
}
```

Then:

```json
{
  "provider": "chatgpt",
  "intelligence": "high",
  "system_prompt_template": "resume/draft-system",
  "user_prompt_template": "resume/draft-user",
  "additional_user_prompt": "",
  "replacement_values": {
    "job_description": "...",
    "master_resume": "...",
    "consensus_analysis": "..."
  }
}
```

Then:

```json
{
  "provider": "chatgpt",
  "intelligence": "middle",
  "system_prompt_template": "resume/docx-script-system",
  "user_prompt_template": "resume/docx-script-user",
  "additional_user_prompt": "",
  "replacement_values": {
    "final_resume": "...",
    "style_notes": "Produce a polished ATS-safe Node.js docx renderer script."
  }
}
```

## 30. Tests

### 30.1 Config Tests

Test:

```text
loads valid yaml
fails if config file missing
fails if providers missing
fails if provider has no models
fails if provider missing low/middle/high
merges defaults + provider + model options correctly
does not require API keys until provider client is actually created
```

### 30.2 Prompt Path Tests

Test:

```text
loads test-system
loads nested resume/draft-system
rejects ../secret
rejects /absolute/path
rejects C:/Windows/file
rejects backslash path
rejects template with invalid chars
fails cleanly on missing template
```

### 30.3 Prompt Rendering Tests

Test:

```text
replaces {name}
replaces multiple occurrences of same key
renders dict as pretty JSON
renders list as pretty JSON
errors on missing placeholder
warns on unused replacement value
does not treat arbitrary JSON braces as placeholders unless they match {identifier}
does not use str.format behavior
```

### 30.4 Router Tests

Use fake provider classes.

Test:

```text
chatgpt + low resolves configured low model
chatgpt + middle resolves configured middle model
chatgpt + high resolves configured high model
gemini + middle resolves correctly
claude + high resolves correctly
unknown provider returns validation error
unknown intelligence returns validation error
system prompt is passed separately
user prompt is passed separately
```

### 30.5 MCP Tool Tests

Call the `query_ai` function directly with a fake router.

Test:

```text
successful query returns status=success
response_text is included
warnings are included
elapsed_ms is included
prompt hashes are included
provider/model are included
missing template returns status=error
missing replacement returns status=error
unsafe path returns status=error
```

### 30.6 Transport Smoke Tests

Test stdio first, because Claude Code will use stdio.

Then test HTTP only if needed.

## 31. Manual Smoke Test

Set API key:

PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

Run:

```powershell
python ai-mcp-server.py --debug --transport stdio
```

From Claude Code, configure MCP server and call `health_check`.

Then call `query_ai` with the test prompt.

Expected:

```text
health_check returns ready
query_ai returns status=success
response_text contains greeting
no debug prompt file unless --save-debug-prompts was enabled
stderr contains logs
stdout is reserved for MCP protocol
```

## 32. Claude Code MCP Configuration

Example:

```json
{
  "mcpServers": {
    "ai": {
      "command": "python",
      "args": [
        "S:/SourceMine/ai-mcp-server/ai-mcp-server.py",
        "--transport",
        "stdio"
      ],
      "env": {
        "OPENAI_API_KEY": "...",
        "GEMINI_API_KEY": "...",
        "ANTHROPIC_API_KEY": "..."
      }
    }
  }
}
```

If environment variables are already set globally, omit `env`.

## 33. Definition of Done

This task is done when:

```text
1. Existing MCP shell still runs in stdio mode.
2. Existing HTTP/HTTPS mode still works or fails gracefully if FastAPI is not installed.
3. Documentation search imports are gone.
4. FAISS/search initialization is gone.
5. search_for_chunks and search_for_documents are gone.
6. query_ai MCP tool exists.
7. health_check MCP tool exists and is AI-specific.
8. ai-mcp-server.yaml controls providers and model tiers.
9. provider + intelligence resolves to the configured model.
10. prompt files are loaded from prompts/.
11. template replacements work.
12. missing replacements error.
13. unused replacement values warn.
14. system and user prompts are sent separately to providers.
15. ChatGPT provider works.
16. Gemini provider is wired.
17. Claude provider is wired.
18. No job/resume workflow logic exists inside the generic MCP server.
19. A real low-cost ChatGPT smoke test succeeds.
20. Claude Code can call the server through MCP stdio.
```

## 34. Suggested Junior Developer Task Order

### Task 1: Copy the working MCP shell

Start from the existing MCP documentation server file.

Rename or copy it to:

```text
ai-mcp-server.py
```

Run it once before editing to confirm baseline behavior.

### Task 2: Remove documentation/search dependencies

Delete FAISS/model imports, embedding settings, and search tools.

Leave `health_check`.

### Task 3: Add AI settings

Revise `ServerSettings` and CLI arguments.

Add:

```text
config_path
prompt_dir
save_debug_prompts
debug_dir
```

### Task 4: Create llm_providers.py

Copy only the useful provider abstraction from the existing LLM library.

Revise `generate(prompt)` to `generate(system_prompt, user_prompt, options)`.

Remove all resume/job code.

### Task 5: Add YAML loader

Implement:

```python
def load_yaml_config(path: Path) -> dict:
    ...
```

Validate minimal required structure.

### Task 6: Add prompt renderer

Implement:

```python
resolve_prompt_path()
render_template()
combine_user_prompt()
sha256_text()
```

### Task 7: Add LLMRouter

Implement provider/intelligence resolution.

Use fake provider tests before real API calls.

### Task 8: Replace create_public_server internals

Initialize config, prompt dir, router, FastMCP.

Register `query_ai` and revised `health_check`.

### Task 9: Add test prompts

Create:

```text
prompts/test-system.md
prompts/test-user.md
```

### Task 10: Smoke test with ChatGPT low

Set `OPENAI_API_KEY`.

Call `query_ai`.

### Task 11: Wire Claude Code

Add MCP config.

Call `health_check` and `query_ai` from Claude Code.

## 35. Non-Goals

Do not implement these in this ticket:

```text
resume orchestration
job workflow orchestration
multi-model consensus workflow
DOCX rendering
Node script execution
file artifact management
automatic template discovery
prompt versioning
parallel LLM calls
request persistence
usage/cost database
web UI
```