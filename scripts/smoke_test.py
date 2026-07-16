#!/usr/bin/env python3
"""Phase 6 smoke tests for the AI MCP Server.

Runs programmatic checks and optional live API calls when keys are set.
Exit code 0 = all runnable checks passed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

_server_path = ROOT / "ai-mcp-server.py"
_spec = importlib.util.spec_from_file_location("ai_mcp_server", _server_path)
assert _spec and _spec.loader
_server_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_server_mod)

ServerSettings = _server_mod.ServerSettings
MCPServer = _server_mod.MCPServer


def _get_tool_fn(public_server, name: str):
    for tool in public_server._tool_manager.list_tools():
        if tool.name == name:
            return (
                getattr(tool, "function", None)
                or getattr(tool, "fn", None)
                or getattr(tool, "_callback", None)
            )
    raise RuntimeError(f"Tool not found: {name}")


def check_health() -> None:
    settings = ServerSettings()
    server = MCPServer(settings)
    public_server = server.create_public_server()
    assert public_server is not None, server._initialization_error

    health = _get_tool_fn(public_server, "health_check")()
    assert health["status"] == "ready", health
    for provider in ("chatgpt", "gemini", "claude"):
        assert provider in health["providers"], health
    print("health_check:", json.dumps(health, indent=2))
    print("health_check: OK")


def check_debug_prompt_save() -> None:
    debug_dir = ROOT / ".debug" / "ai-mcp-server"
    for old in debug_dir.glob("query-ai-*.json"):
        old.unlink()

    settings = ServerSettings(save_debug_prompts=True, debug_dir=str(debug_dir.relative_to(ROOT)))
    server = MCPServer(settings)
    public_server = server.create_public_server()
    assert public_server is not None

    query_ai = _get_tool_fn(public_server, "query_ai")
    result = query_ai(
        provider="chatgpt",
        intelligence="low",
        system_prompt_template="test-system",
        user_prompt_template="test-user",
        replacement_values={"name": "Aaron"},
    )

    files = sorted(debug_dir.glob("query-ai-*.json"))
    assert files, "Expected a debug prompt file to be written before the API call"
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    assert "system_prompt" in data
    assert "user_prompt" in data
    assert data["provider"] == "chatgpt"
    print(f"debug prompt file: {files[-1].name}")
    print("debug prompt save: OK")
    if result.get("status") != "success":
        print(f"note: live query after debug save returned {result.get('status')}: {result.get('message', '')[:120]}")


def check_live_chatgpt() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("live ChatGPT smoke test: SKIPPED (OPENAI_API_KEY not set)")
        return

    settings = ServerSettings()
    server = MCPServer(settings)
    public_server = server.create_public_server()
    assert public_server is not None

    query_ai = _get_tool_fn(public_server, "query_ai")
    result = query_ai(
        provider="chatgpt",
        intelligence="low",
        system_prompt_template="test-system",
        user_prompt_template="test-user",
        replacement_values={"name": "Aaron"},
    )
    print(json.dumps(result, indent=2))

    assert result["status"] == "success", result
    assert result["model"] == "gpt-4.1-mini", result
    assert "Aaron" in result["response_text"], result
    assert result["elapsed_ms"] > 0
    assert len(result["system_template_sha256"]) == 64
    assert len(result["user_template_sha256"]) == 64
    assert len(result["rendered_system_prompt_sha256"]) == 64
    assert len(result["rendered_user_prompt_sha256"]) == 64
    print("live ChatGPT smoke test: OK")


def check_live_provider(provider: str, env_var: str) -> None:
    if not os.getenv(env_var):
        print(f"live {provider} smoke test: SKIPPED ({env_var} not set)")
        return

    settings = ServerSettings()
    server = MCPServer(settings)
    public_server = server.create_public_server()
    query_ai = _get_tool_fn(public_server, "query_ai")
    result = query_ai(
        provider=provider,
        intelligence="low",
        system_prompt_template="test-system",
        user_prompt_template="test-user",
        replacement_values={"name": "Aaron"},
    )
    assert result["status"] == "success", result
    print(f"live {provider} smoke test: OK ({result['model']})")


def main() -> int:
    print("=== AI MCP Server Phase 6 smoke tests ===\n")
    check_health()
    print()
    check_debug_prompt_save()
    print()
    check_live_chatgpt()
    print()
    check_live_provider("gemini", "GEMINI_API_KEY")
    print()
    check_live_provider("claude", "ANTHROPIC_API_KEY")
    print("\n=== All runnable smoke tests passed ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
