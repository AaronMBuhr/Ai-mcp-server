"""Tests for the query_ai MCP tool function with mocked router."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from llm_providers import LLMResult


def _get_server_and_tools():
    """Create MCPServer and return tool functions by name."""
    exec_globals = {}
    code = Path("ai-mcp-server.py").read_text(encoding="utf-8")
    code = code.split("if __name__")[0]
    exec(code, exec_globals)
    ServerSettings = exec_globals["ServerSettings"]
    MCPServer = exec_globals["MCPServer"]
    s = ServerSettings()
    m = MCPServer(s)
    ps = m.create_public_server()
    tools = {}
    for tool in ps._tool_manager.list_tools():
        fn = getattr(tool, "function", None) or getattr(tool, "fn", None)
        tools[tool.name] = fn
    return m, tools


class TestQueryAiTool:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.server, self.tools = _get_server_and_tools()
        self.query_ai = self.tools["query_ai"]

    def test_missing_template_returns_error(self):
        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="nonexistent",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert r["error_type"] == "FileNotFoundError"
        assert "not found" in r["message"]

    def test_missing_replacement_returns_error(self):
        # The test-system/test-user pair has no placeholders, so use the job
        # pair (which does) to exercise the missing-replacement path.
        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="example/review-system",
            user_prompt_template="example/review-user",
            replacement_values={},
        )
        assert r["status"] == "error"
        assert r["failure_type"] == "missing_replacement"
        assert r["provider_called"] is False
        assert "Missing replacement values" in r["message"]

    def test_unsafe_path_returns_error(self):
        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="../etc/passwd",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert r["error_type"] == "ValueError"

    def test_unknown_provider_returns_error(self):
        r = self.query_ai(
            provider="invalid", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert "Unknown provider" in r["message"]

    def test_unknown_intelligence_returns_error(self):
        r = self.query_ai(
            provider="chatgpt", intelligence="ultra",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "test"},
        )
        assert r["status"] == "error"
        assert "Unknown intelligence" in r["message"]

    def test_successful_query_with_mocked_router(self):
        fake_result = LLMResult(text="Hello Aaron!", model="gpt-4.1-mini")
        self.server._router.query = MagicMock(return_value=fake_result)

        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "Aaron"},
        )
        assert r["status"] == "success"
        assert r["response_text"] == "Hello Aaron!"
        assert r["model"] == "gpt-4.1-mini"
        assert r["provider"] == "chatgpt"
        assert r["intelligence"] == "low"
        assert "elapsed_ms" in r
        assert isinstance(r["elapsed_ms"], int)
        assert r["system_prompt_template"] == "test-system"
        assert r["user_prompt_template"] == "test-user"
        assert len(r["system_template_sha256"]) == 64
        assert len(r["user_template_sha256"]) == 64
        assert len(r["rendered_system_prompt_sha256"]) == 64
        assert len(r["rendered_user_prompt_sha256"]) == 64
        # name is used in user template but may warn as unused in system template
        assert not any("unused_key" in w for w in r["warnings"])

    def test_warnings_included_in_response(self):
        fake_result = LLMResult(text="Hi", model="gpt-4.1-mini")
        self.server._router.query = MagicMock(return_value=fake_result)

        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={"name": "Aaron", "unused_key": "value"},
        )
        assert r["status"] == "success"
        assert len(r["warnings"]) > 0
        assert any("unused_key" in w for w in r["warnings"])

    def test_additional_user_prompt_appended(self):
        fake_result = LLMResult(text="response", model="gpt-4.1-mini")
        self.server._router.query = MagicMock(return_value=fake_result)

        self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            additional_user_prompt="Be extra concise.",
            replacement_values={"name": "Aaron"},
        )

        call_args = self.server._router.query.call_args
        user_prompt_sent = call_args[0][3]
        assert "Be extra concise." in user_prompt_sent
        assert "---" in user_prompt_sent
