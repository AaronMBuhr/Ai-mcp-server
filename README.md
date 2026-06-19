# AI MCP Server

A **use-case-agnostic** [Model Context Protocol](https://modelcontextprotocol.io/) server. Callers supply prompt templates and replacement values; the server renders them and routes the query to a configured LLM provider (ChatGPT, Gemini, or Claude). It contains no domain workflow logic (no job matching, resume handling, document search, etc.).

## Features

- **`query_ai`** — load Markdown prompt templates from `prompts/`, render placeholders, call a provider at a chosen intelligence tier
- **`health_check`** — report server readiness and configured providers
- Transports: **stdio** (MCP clients), **HTTP**, **HTTPS**

## Quick start

### 1. Install dependencies

Use the `mypython312` conda environment (or any Python 3.12+ env with the same packages):

```text
mcp
fastapi
uvicorn
pydantic
pydantic-settings
python-dotenv
pyyaml
openai
anthropic
google-genai
pytest
```

### 2. Configure providers

Edit `ai-mcp-server.yaml` for model names and tiers. Set API keys via environment variables (see `.env.example`):

```text
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### 3. Run the server

**Stdio (for Cursor / Claude Code):**

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 ai-mcp-server.py --transport stdio
```

**HTTP:**

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 ai-mcp-server.py --transport http --port 8000
```

Help page: `http://localhost:8000/help`

### 4. Cursor MCP configuration

This repo includes [`.cursor/mcp.json`](.cursor/mcp.json). Enable the **ai** server in Cursor MCP settings. API keys are read from your environment (or add an `env` block to `mcp.json` locally — do not commit secrets).

Restart Cursor after changing MCP config, then invoke `health_check` or `query_ai`.

### 5. Example `query_ai` call

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

Template paths are relative to `prompts/` without the `.md` extension.

## Testing

**Unit tests (no API keys):**

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 -m pytest tests/ -v
```

**Phase 6 smoke tests:**

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 scripts/smoke_test.py
```

Live provider tests run only when the corresponding API key is set. Debug prompt files are written to `.debug/ai-mcp-server/` when `--save-debug-prompts` is enabled.

## CLI options

| Flag | Env var | Default |
|------|---------|---------|
| `--transport` | `MCP_TRANSPORT` | `stdio` |
| `--config` | `MCP_CONFIG_PATH` | `ai-mcp-server.yaml` |
| `--prompt-dir` | `MCP_PROMPT_DIR` | `prompts` |
| `--save-debug-prompts` | `MCP_SAVE_DEBUG_PROMPTS` | off |
| `--debug-dir` | `MCP_DEBUG_DIR` | `.debug/ai-mcp-server` |

Run `python ai-mcp-server.py --help` for the full list.

## Project layout

```text
ai-mcp-server.py      MCP server entrypoint
ai-mcp-server.yaml    Provider and model tier config
prompt_engine.py      Config loading, template rendering, LLMRouter
llm_providers.py      ChatGPT / Claude / Gemini providers
prompts/              Markdown prompt templates
tests/                Pytest suite
scripts/smoke_test.py Phase 6 acceptance smoke tests
```

## Specification

The MCP server exposes two tools:

| Tool | Purpose |
|------|---------|
| `query_ai` | Load `prompts/*.md` templates, render `{placeholders}`, call provider at `intelligence` tier |
| `health_check` | Report readiness, configured providers, config and prompt paths |

`query_ai` parameters:

```text
provider                  # chatgpt | gemini | claude (from ai-mcp-server.yaml)
intelligence              # low | middle | high
system_prompt_template    # path under prompts/ without .md
user_prompt_template      # path under prompts/ without .md
additional_user_prompt    # optional extra user text
replacement_values        # dict of placeholder values
```

Provider models, temperature, and retries are controlled by `ai-mcp-server.yaml`. API keys are resolved from environment variables named in that file.

## MCP-only project files

```text
ai-mcp-server.py
ai-mcp-server.yaml
prompt_engine.py
llm_providers.py
prompts/
tests/
scripts/smoke_test.py
.cursor/mcp.json
```

Other files in this directory (e.g. `find-jobs.py`, `llm.py`) are unrelated reference code from another project and are **not** imported or used by the MCP server.
