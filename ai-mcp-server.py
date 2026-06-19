#!/usr/bin/env python3
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

import argparse
import contextlib
from dotenv import load_dotenv
import logging
import os
from pathlib import Path
from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Annotated

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

load_dotenv(PROJECT_ROOT / ".env")

from prompt_engine import (
    LLMRouter,
    combine_user_prompt,
    load_yaml_config,
    render_template,
    resolve_prompt_path,
    save_debug_prompt,
    sha256_text,
)

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


# Configure logging following reference pattern
def setup_logger() -> logging.Logger:
    """Configure logging for the MCP server to stderr."""
    logger = logging.getLogger("ai-mcp-server")

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create stderr handler for console output
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Add handler to logger
    logger.addHandler(stderr_handler)

    # Set level
    logger.setLevel(logging.INFO)

    return logger


logger: logging.Logger = setup_logger()


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


class FastAppSettings:
    """Application settings for HTTP/HTTPS mode following reference pattern."""
    
    def __init__(self, settings: ServerSettings, mcp_app: FastMCP) -> None:
        self.expose_url: str = str(settings.server_url)
        self.dns: Optional[str] = None
        self.settings: ServerSettings = settings
        self.servers: List[FastMCP] = [mcp_app]  # Use the provided FastMCP app

class FastApp:
    """FastAPI application for HTTP/HTTPS mode following reference pattern."""
    
    def __init__(self, fast_app_settings: FastAppSettings) -> None:
        self.app_settings: FastAppSettings = fast_app_settings

    def create_app(self) -> FastAPI:
        """Create FastAPI application with MCP server mounting."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
            
        servers = self.app_settings.servers

        # Create a combined lifespan to manage session managers
        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI) -> Any:
            async with contextlib.AsyncExitStack() as stack:
                for server in servers:
                    await stack.enter_async_context(server.session_manager.run())
                yield

        app = FastAPI(lifespan=lifespan)
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount the FastMCP app at root so its internal routes
        # (e.g., /mcp, /.well-known/*, /register) are exposed exactly
        # at those paths. This supports reverse proxies routing only
        # /mcp to this upstream.
        # Mount each MCP server at its named path
        # Expose the canonical "/mcp" route expected by tests by mounting
        # each FastMCP app at both "/mcp" (for backwards-compat) and at its
        # human-readable server name.
        for server in servers:
            # Mount canonical path
            app.mount("/mcp", server.streamable_http_app())
            # Also mount by server name for multi-server deployments
            app.mount(f"/{server.name}", server.streamable_http_app())
            
        @app.get("/", include_in_schema=False)
        async def redirect_to_help() -> RedirectResponse:
            return RedirectResponse(url="/help")

        @app.get("/help", include_in_schema=False)
        async def help() -> HTMLResponse:
            """Generate help page with server information."""
            try:
                help_text = """
                <html>
                <head><title>AI MCP Server</title></head>
                <body>
                <h1>AI MCP Server</h1>
                <p>This server provides generic LLM routing through the Model Context Protocol.</p>
                <h2>Available Servers</h2>
                """
                
                for server in servers:
                    tools = []
                    try:
                        tools = [tool.name for tool in server._tool_manager.list_tools()]
                    except Exception:
                        try:
                            maybe_tools = getattr(server, 'tools', [])
                            tools = [getattr(t, 'name', str(t)) for t in maybe_tools]
                        except Exception:
                            tools = []
                    help_text += f"""
                    <h3>{server.name}</h3>
                    <p>{server.instructions}</p>
                    <p><strong>Endpoint:</strong> <code>{self.app_settings.expose_url}/mcp</code></p>
                    <p><strong>Tools:</strong> {', '.join(tools) if tools else 'None'}</p>
                    """

                help_text += "</body></html>"
                return HTMLResponse(content=help_text, status_code=200)

            except Exception as e:
                logger.error(f"Error generating help page: {str(e)}")
                raise HTTPException(status_code=500, detail="Error generating help page")

        return app


class MCPServer:
    """
    MCP Server class that encapsulates AI routing and server creation
    following MCP specification requirements.
    """
    
    def __init__(self, settings: ServerSettings) -> None:
        self._settings: ServerSettings = settings
        self._logger: logging.Logger = logger
        self._is_ready: bool = False
        self._initialization_error: Optional[str] = None
        self._config: Optional[dict] = None
        self._router: Optional[LLMRouter] = None
        self._prompt_dir: Optional[Path] = None
       
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

        # --- Tools ---

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
                sys_path = resolve_prompt_path(self._prompt_dir, system_prompt_template)
                usr_path = resolve_prompt_path(self._prompt_dir, user_prompt_template)

                sys_template_text = sys_path.read_text(encoding="utf-8")
                usr_template_text = usr_path.read_text(encoding="utf-8")

                system_prompt, sys_warnings = render_template(sys_template_text, replacement_values)
                user_prompt_rendered, usr_warnings = render_template(usr_template_text, replacement_values)
                warnings = sys_warnings + usr_warnings

                user_prompt = combine_user_prompt(user_prompt_rendered, additional_user_prompt)

                system_prompt_hash = sha256_text(system_prompt)
                user_prompt_hash = sha256_text(user_prompt)

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

        # Tool compatibility patch
        try:
            for tool in mcp._tool_manager.list_tools():
                if not hasattr(tool, "function"):
                    func = (
                        getattr(tool, "_callback", None)
                        or getattr(tool, "callback", None)
                        or getattr(tool, "call", None)
                        or getattr(tool, "fn", None)
                    )
                    if func is not None:
                        setattr(tool, "function", func)
        except Exception:
            pass

        return mcp

    def run_stdio_server(self) -> None:
        """Run MCP server in stdio mode. Now synchronous since FastMCP handles async internally."""
        
        self._logger.debug("Step 7: Creating FastMCP server...")
        public_server = self.create_public_server()
        if public_server is None:
            self._logger.error("Failed to create public server")
            if self._initialization_error:
                raise Exception(self._initialization_error)
            return
        
        self._logger.debug("Step 8: Starting server in stdio mode...")
        self._logger.info("Starting MCP server in stdio mode")

        # Let FastMCP itself spin up the stdio transport and session manager:
        try:
            self._logger.debug("Step 9: Server running in stdio mode...")
            # Type assertion to help linter understand public_server is not None
            assert public_server is not None
            
            # Use FastMCP's default run() method (stdio mode)
            self._logger.debug("Step 10: Starting FastMCP default mode (stdio)...")
            # Use the synchronous run() method without transport parameter (defaults to stdio)
            public_server.run()
            self._logger.debug("Step 11: Server finished.")
        except Exception as e:
            self._logger.error(f"Failed to run public server: {e}")

    def run_http_server_with_mcp(self) -> int:
        """Run MCP server in HTTP mode following reference pattern."""
        
        self._logger.debug("Step 7: Creating FastMCP server...")
        
        # Create public server
        public_app = self.create_public_server()
        if public_app is None:
            self._logger.error("Failed to create public server")
            if self._initialization_error:
                raise Exception(self._initialization_error)
            return 1
        
        self._logger.debug(f"Step 8: Starting server in {self._settings.transport.upper()} mode...")
        
        # Start HTTP server with public app
        self.run_http_server(public_app)
        self._logger.debug("Step 9: HTTP/HTTPS server finished.")
        return 0

    def run_http_server(self, mcp_app: FastMCP) -> None:
        """Run MCP server in HTTP/HTTPS mode following reference pattern."""
        if not FASTAPI_AVAILABLE:
            self._logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
            return

        try:
            # Create list of servers to mount
            servers = [mcp_app]
                
            app_settings = FastAppSettings(self._settings, mcp_app)
            fast_app = FastApp(app_settings)
            app = fast_app.create_app()
            
            # Configure SSL if HTTPS
            ssl_keyfile = self._settings.ssl_key_path if self._settings.transport == "https" else None
            ssl_certfile = self._settings.ssl_cert_path if self._settings.transport == "https" else None
            
            if self._settings.transport == "https" and (not ssl_keyfile or not ssl_certfile):
                self._logger.warning("SECURITY WARNING: HTTPS requested but SSL cert/key not provided. Falling back to HTTP. This connection will NOT be encrypted!")
                ssl_keyfile = ssl_certfile = None
            
            self._logger.info(f"Starting MCP server on {self._settings.host}:{self._settings.port} ({'HTTPS' if ssl_keyfile else 'HTTP'})")

            uvicorn.run(
                app,
                host=self._settings.host,
                port=self._settings.port,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                log_level="info"
            )
            
        except Exception as e:
            self._logger.error(f"Failed to start HTTP server: {e}")
            raise


def main() -> int:
    """Parse CLI args, create settings, and start the MCP server."""
    logger.info("AI MCP Server initializing...")
    
    logger.debug("Step 1: Parsing arguments...")
    
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
    
    logger.debug(f"Step 2: Args parsed - debug: {args.debug}, transport: {args.transport}")
    
    logger.debug("Step 3: Setting up logging...")
    
    # Set up logging with debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.debug("Step 4: Logging configured")
    
    logger.debug("Step 5: Creating settings...")

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
    
    logger.debug(f"Step 6: Settings created - {settings}")
    
    # Handle different transport modes using MCPServer class
    try:
        # Create MCP server instance
        mcp_server: MCPServer = MCPServer(settings)
        
        if settings.transport == "stdio":
            # Call the stdio server directly
            mcp_server.run_stdio_server()
                
        elif settings.transport in ["http", "https"]:
            result: int = mcp_server.run_http_server_with_mcp()
            return result
        else:
            logger.error(f"Unknown transport mode '{settings.transport}'")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
