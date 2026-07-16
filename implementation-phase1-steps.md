# Phase 1: Strip and Scaffold — Discrete Implementation Steps

Each step below is a single, verifiable edit. Complete them in order. After each step, the file should still be valid Python (no syntax errors). After all steps, the server should start without crashing in stdio mode.

**File being edited:** `ai-mcp-server.py`

---

## Step 1: Replace the module docstring

Replace the entire docstring at the top of the file (lines 2–16) with:

```python
"""
AI MCP Server
=============

A generic Model Context Protocol (MCP) server that routes prompt templates
to configured LLM providers (ChatGPT, Gemini, Claude).

Features:
- Prompt template loading and placeholder rendering
- Provider/intelligence-tier routing via YAML config
- Support for stdio, HTTP, and HTTPS transports
"""
```

**Verify:** File parses without errors.

---

## Step 2: Remove the search-specific imports

Delete these two lines:

```python
from search_engine_faiss import FaissSearchEngine
from models import Document, DocumentChunk, SearchEngine
```

**Verify:** No `ImportError` from these modules at startup.

---

## Step 3: Remove the DEFAULTS class

Delete the entire `DEFAULTS` class:

```python
class DEFAULTS:
    """Default configuration values with type annotations."""
    MODEL_NAME: str = "thenlper/gte-large"
    EMBEDDING_PATH: str = "./embeddings"
    DEVICE: str = "auto"
```

**Verify:** File parses. Nothing else references `DEFAULTS` after later steps are completed.

---

## Step 4: Update the logger name

In `setup_logger()`, change the logger name from `"mcp-docs-server"` to `"ai-mcp-server"`:

```python
logger = logging.getLogger("ai-mcp-server")
```

**Verify:** Logger still prints to stderr.

---

## Step 5: Rewrite ServerSettings

Replace the entire `ServerSettings` class with:

```python
class ServerSettings(BaseSettings):
    """Settings for the AI MCP Server."""

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="allow")

    # Transport settings
    transport: str = Field(default="stdio", description="Transport mode: stdio, http, or https")
    host: str = Field(default="localhost", description="Server host address")
    port: int = Field(default=8000, description="Server port number")
    server_url: AnyHttpUrl = Field(default="http://localhost:8000", description="Full server URL")

    # SSL settings
    ssl_cert_path: Optional[str] = Field(default=None, description="Path to SSL certificate file")
    ssl_key_path: Optional[str] = Field(default=None, description="Path to SSL private key file")

    # Server identity
    server_name: str = Field(default="ai-mcp-server", description="Name of the MCP server")
    description: str = Field(default="generic AI provider routing server", description="Server description")

    # AI-specific settings
    config_path: str = Field(default="ai-mcp-server.yaml", description="Path to YAML config file")
    prompt_dir: str = Field(default="prompts", description="Directory containing prompt templates")
    save_debug_prompts: bool = Field(default=False, description="Save rendered prompts to debug files")
    debug_dir: str = Field(default=".debug/ai-mcp-server", description="Directory for debug prompt files")
```

**What was removed:** `transformer_device`, `model_name`, `embedding_path`, `embedding_name`.

**What was added:** `config_path`, `prompt_dir`, `save_debug_prompts`, `debug_dir`.

**What was changed:** `server_name` default from `"docs-server"` to `"ai-mcp-server"`. `description` default from `"company documentation"` to `"generic AI provider routing server"`.

**Verify:** `ServerSettings()` can be instantiated without errors.

---

## Step 6: Update the help page in FastApp

Inside the `FastApp.create_app()` method, in the `help()` function, replace the HTML title and description text:

Change:
```python
<head><title>MCP Documentation Server</title></head>
```
To:
```python
<head><title>AI MCP Server</title></head>
```

Change:
```python
<h1>MCP Documentation Server</h1>
<p>This server provides access to documentation through the Model Context Protocol (MCP).</p>
```
To:
```python
<h1>AI MCP Server</h1>
<p>This server provides generic LLM routing through the Model Context Protocol.</p>
```

**Verify:** If you start in HTTP mode, the `/help` page shows the new text.

---

## Step 7: Rewrite MCPServer.__init__

Replace the `MCPServer.__init__` method with:

```python
def __init__(self, settings: ServerSettings) -> None:
    self._settings: ServerSettings = settings
    self._logger: logging.Logger = logger
    self._is_ready: bool = False
    self._initialization_error: Optional[str] = None
    self._config: Optional[dict] = None
    self._router: Optional[Any] = None
    self._prompt_dir: Optional[Path] = None
```

**What was removed:** `device` parameter, `self._device`, `self._search_engine`.

**What was added:** `self._config`, `self._router`, `self._prompt_dir`.

**Verify:** File parses. The `Optional[Any]` for `_router` is a placeholder — it will be typed properly in Phase 4.

---

## Step 8: Rewrite create_public_server()

Replace the entire `create_public_server()` method body with:

```python
def create_public_server(self) -> Optional[FastMCP]:
    """Create the AI MCP server with tool registration."""
    try:
        self._prompt_dir = Path(self._settings.prompt_dir).resolve()

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

    # --- Tools ---

    @mcp.tool()
    def health_check() -> Dict[str, Any]:
        """Check the health status of the AI MCP server."""
        if not self._is_ready:
            return {
                "status": "error",
                "message": self._initialization_error or "Server not initialized",
            }

        return {
            "status": "ready",
            "server_name": self._settings.server_name,
            "prompt_dir": str(self._prompt_dir),
            "config_path": self._settings.config_path,
        }

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

**What was removed:** FAISS initialization, `search_for_chunks` tool, `search_for_documents` tool, the comment about document resources.

**What was kept:** `health_check` tool (revised), tool compatibility patch.

**Verify:** Server starts in stdio mode without crashing. `health_check` returns `status: "ready"`.

---

## Step 9: Rewrite the CLI argument parser in main()

Replace the entire argument parser block (from `parser = argparse.ArgumentParser(...)` through `args = parser.parse_args()`) with:

```python
parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="AI MCP Server - A generic Model Context Protocol server for LLM routing",
    epilog="""
Environment Variables:
  All command line arguments can be set via environment variables with MCP_ prefix.
  Command line arguments override environment variables.

  Examples:
    export MCP_TRANSPORT=http
    export MCP_HOST=127.0.0.1
    export MCP_PORT=8008
    export MCP_SERVER_NAME=ai-mcp-server
    export MCP_CONFIG_PATH=ai-mcp-server.yaml
    export MCP_PROMPT_DIR=prompts

    """,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("-d", "--debug", action="store_true",
                   help="Enable debug logging output.")
parser.add_argument("--transport", choices=["stdio", "http", "https"],
                   default=os.getenv("MCP_TRANSPORT", "stdio"),
                   help="Transport protocol: 'stdio' for MCP clients, 'http'/'https' for web servers. Default: stdio.")
parser.add_argument("--host", type=str,
                   default=os.getenv("MCP_HOST", "localhost"),
                   help="Host address to bind server to (only for http/https transport). Default: localhost.")
parser.add_argument("--port", type=int,
                   default=int(os.getenv("MCP_PORT", "8000")),
                   help="Port number to bind server to (only for http/https transport). Default: 8000.")
parser.add_argument("--ssl-cert", type=str,
                   default=os.getenv("MCP_SSL_CERT_PATH"),
                   help="Path to SSL certificate file for HTTPS transport.")
parser.add_argument("--ssl-key", type=str,
                   default=os.getenv("MCP_SSL_KEY_PATH"),
                   help="Path to SSL private key file for HTTPS transport.")
parser.add_argument("-n", "--server-name", type=str,
                   default=os.getenv("MCP_SERVER_NAME", "ai-mcp-server"),
                   help="Name identifier for the MCP server instance. Default: ai-mcp-server.")
parser.add_argument("--description", type=str,
                   default=os.getenv("MCP_DESCRIPTION", "generic AI provider routing server"),
                   help="Human-readable description of the MCP server.")
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
args: argparse.Namespace = parser.parse_args()
```

**What was removed:** `--device`, `--model` / `-m`, `--embedding-path` / `-p`, `--embedding-name` / `-e`.

**What was added:** `--config`, `--prompt-dir`, `--save-debug-prompts`, `--debug-dir`.

**Verify:** `python ai-mcp-server.py --help` shows the new arguments.

---

## Step 10: Update the initialization log message in main()

Change:
```python
logger.info("🚀 MCP Documentation Server initializing...")
logger.debug("=== BUILD v3.0 - Non-blocking MCP Server with MCPServer class ===")
```

To:
```python
logger.info("🚀 AI MCP Server initializing...")
```

(Delete the BUILD debug line entirely.)

**Verify:** Startup log says "AI MCP Server".

---

## Step 11: Rewrite the settings construction in main()

Replace the `ServerSettings(...)` call and surrounding comments with:

```python
settings: ServerSettings = ServerSettings(
    transport=args.transport,
    host=args.host,
    port=args.port,
    server_url=f"http{'s' if args.transport == 'https' else ''}://{args.host}:{args.port}",
    ssl_cert_path=args.ssl_cert,
    ssl_key_path=args.ssl_key,
    server_name=args.server_name,
    description=args.description,
    config_path=args.config,
    prompt_dir=args.prompt_dir,
    save_debug_prompts=args.save_debug_prompts,
    debug_dir=args.debug_dir,
)
```

**What was removed:** `transformer_device`, `model_name`, `embedding_path`, `embedding_name`.

**What was added:** `config_path`, `prompt_dir`, `save_debug_prompts`, `debug_dir`.

**Verify:** No error on `ServerSettings(...)` instantiation.

---

## Step 12: Update MCPServer instantiation in main()

Change:
```python
mcp_server: MCPServer = MCPServer(settings, device=args.device)
```

To:
```python
mcp_server: MCPServer = MCPServer(settings)
```

**Verify:** No `TypeError` about unexpected keyword argument.

---

## Step 13: Remove FAISS-specific error handling in main()

In the `except Exception as e:` block near the end of `main()`, delete the FAISS-specific help logic:

```python
if "FAISS index file not found" in str(e) or "Failed to initialize" in str(e):
    print("\n" + "="*80)
    print("HELP: Server initialization failed. Please check the following:")
    print("="*80)
    parser.print_help()
```

Replace with just:
```python
logger.error(f"Server error: {e}")
return 1
```

(Keep the `logger.error(f"Server error: {e}")` line that's already there — just delete the if-block beneath it.)

**Verify:** No reference to "FAISS" anywhere in the file.

---

## Step 14: Clean up unused imports

After all the above changes, the following imports are no longer needed. Remove them from the top of the file:

- `from datetime import datetime, timedelta` — not used anywhere in the shell
- `from mcp.server.auth.settings import AuthSettings` — not used
- `from mcp.types import PromptMessage, TextContent` — not used

Keep all other imports. The final import block should be:

```python
import argparse
import asyncio
import contextlib
from dotenv import load_dotenv
import json
import logging
import logging.handlers
import os
from pathlib import Path
from pydantic import BaseModel, Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
import sys
from typing import Any, Dict, List, Optional, Union, Annotated, Callable, Tuple, Type
import yaml

# Load environment variables
load_dotenv()

# MCP imports
from mcp.server.fastmcp import FastMCP

# FastAPI imports for HTTP mode
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, RedirectResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
```

**Note:** Some of these (`asyncio`, `json`, `yaml`, `BaseModel`, `Union`, `Callable`, `Tuple`, `Type`, `List`) may not be used *yet* but will be used in later phases. It's fine to keep them or remove them — your choice. The key requirement is: remove the three lines listed above that reference deleted modules.

**Verify:** `python -c "import ast; ast.parse(open('ai-mcp-server.py').read())"` succeeds. No `ImportError` on startup.

---

## Step 15: Final verification

Run:
```
python ai-mcp-server.py --help
```

Expected: shows AI MCP Server help with the new arguments, no reference to embeddings/FAISS/documentation.

Run:
```
python ai-mcp-server.py --transport stdio
```

Expected: server starts, waits for MCP protocol input on stdin. No crash. Ctrl+C exits cleanly.

---

## Summary of what the file looks like after Phase 1

```
ai-mcp-server.py
├── Module docstring (AI MCP Server)
├── Imports (no search_engine_faiss, no models)
├── setup_logger() → logger named "ai-mcp-server"
├── ServerSettings (transport, host, port, ssl, server_name, description, config_path, prompt_dir, save_debug_prompts, debug_dir)
├── FastAppSettings (unchanged structurally)
├── FastApp (help page says "AI MCP Server")
├── MCPServer
│   ├── __init__(settings) — no device, no search_engine
│   ├── create_public_server() — stub init + health_check tool only
│   ├── run_stdio_server() — unchanged
│   ├── run_http_server_with_mcp() — unchanged
│   └── run_http_server() — unchanged
└── main() — new args, new settings, new log message, no FAISS help
```
