"""End-to-end tests for the hardened query_ai pipeline (P2/P3/P4/P8).

Uses the real server module with a mocked router.query so no provider is
ever contacted.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_providers import LLMResult
from prompt_engine import LLMRouter, ModelRouteMissingError


def _get_server_and_tools():
    exec_globals = {}
    code = Path("ai-mcp-server.py").read_text(encoding="utf-8")
    code = code.split("if __name__")[0]
    exec(code, exec_globals)
    ServerSettings = exec_globals["ServerSettings"]
    MCPServer = exec_globals["MCPServer"]
    s = ServerSettings(audit_log_enabled=False)
    m = MCPServer(s)
    ps = m.create_public_server()
    tools = {}
    for tool in ps._tool_manager.list_tools():
        fn = getattr(tool, "function", None) or getattr(tool, "fn", None)
        tools[tool.name] = fn
    return m, tools


EXAMPLE_VALUES = {
    "product": "Acme",
    "audience": "Senior Backend Engineer",
    "specification": "We are hiring a senior backend engineer to own ingestion. " * 20,
    "reference_doc": "# Aaron Buhr\n\nExtensive backend and data-pipeline experience. " * 40,
}


class TestHardenedQueryAi:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.server, self.tools = _get_server_and_tools()
        self.query_ai = self.tools["query_ai"]
        self.router_query = MagicMock(
            return_value=LLMResult(text="analysis text", model="gemini-2.5-flash")
        )
        self.server._router.query = self.router_query

    def _call(self, **overrides):
        kwargs = dict(
            provider="gemini", intelligence="middle",
            system_prompt_template="example/review-system",
            user_prompt_template="example/review-user",
            replacement_values=dict(EXAMPLE_VALUES),
        )
        kwargs.update(overrides)
        return self.query_ai(**kwargs)

    # --- acceptance test step 1-2: reference rejected before provider ---

    def test_read_from_file_rejected_before_provider(self):
        values = dict(EXAMPLE_VALUES)
        values["reference_doc"] = r"READ_FROM_FILE:C:\fake.md"
        r = self._call(replacement_values=values)
        assert r["status"] == "error"
        assert r["failure_type"] == "invalid_content_transport"
        assert r["field"] == "reference_doc"
        assert r["provider_called"] is False
        self.router_query.assert_not_called()

    def test_missing_replacement_rejected_before_provider(self):
        values = {k: v for k, v in EXAMPLE_VALUES.items() if k != "reference_doc"}
        r = self._call(replacement_values=values)
        assert r["status"] == "error"
        assert r["failure_type"] == "missing_replacement"
        assert "reference_doc" in r["detail"]
        assert r["provider_called"] is False
        # legacy keys preserved:
        assert "Missing replacement values" in r["message"] or "reference_doc" in r["message"]
        self.router_query.assert_not_called()

    def test_wrong_key_name_rejected(self):
        values = {k: v for k, v in EXAMPLE_VALUES.items() if k != "reference_doc"}
        values["master_resume"] = EXAMPLE_VALUES["reference_doc"]
        r = self._call(replacement_values=values)
        assert r["status"] == "error"
        assert r["failure_type"] == "missing_replacement"
        self.router_query.assert_not_called()

    def test_unexpected_key_rejected(self):
        values = dict(EXAMPLE_VALUES)
        values["surprise"] = "x"
        r = self._call(replacement_values=values)
        assert r["status"] == "error"
        assert r["failure_type"] == "unexpected_replacement"
        self.router_query.assert_not_called()

    # --- P4: resolved model + hash metadata ---

    def test_success_includes_verification_metadata(self):
        r = self._call()
        assert r["status"] == "success"
        assert r["provider_requested"] == "gemini"
        assert r["intelligence_requested"] == "middle"
        assert r["provider_resolved"] == "gemini"
        assert r["model_resolved"] == "gemini-2.5-flash"
        assert len(r["response_sha256"]) == 64
        assert len(r["rendered_system_prompt_sha256"]) == 64
        assert len(r["rendered_user_prompt_sha256"]) == 64
        meta = r["replacement_metadata"]
        assert set(meta.keys()) == set(EXAMPLE_VALUES.keys())
        for entry in meta.values():
            assert set(entry.keys()) == {"characters", "bytes", "sha256"}
        # legacy keys intact:
        assert r["response_text"] == "analysis text"
        assert r["model"] == "gemini-2.5-flash"
        assert len(r["system_template_sha256"]) == 64
        assert "request_id" in r and len(r["request_id"]) == 36

    def test_template_hashes_differ_from_rendered_when_substituted(self):
        """Raw-template and rendered-prompt hashes must be distinct values.

        They were once the same number under two names, which made it look like
        substitution had failed on every call. The raw hash fingerprints the
        template revision; the rendered hash is what the provider received.
        Equality is the real signal that nothing was substituted.
        """
        r = self._call()

        assert r["status"] == "success"
        assert r["user_template_sha256"] != r["rendered_user_prompt_sha256"]
        # The analyze system template has no placeholders, so it renders to
        # itself -- its two hashes legitimately match.
        assert r["system_template_sha256"] == r["rendered_system_prompt_sha256"]

    def test_uncontracted_templates_still_work(self):
        """The test-system/test-user pair has no contract and must keep working."""
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="OK", model="gpt-5.4-mini")
        )
        r = self.query_ai(
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            replacement_values={},
        )
        assert r["status"] == "success"
        assert r["response_text"] == "OK"

    # --- routing metadata (P4/P9) ---

    def test_chatgpt_high_resolves_expected_model(self):
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="resume", model="")
        )
        r = self._call(provider="chatgpt", intelligence="high")
        assert r["status"] == "success"
        assert r["model_resolved"] == "gpt-5.5"
        assert r["provider_resolved"] == "openai"

    def test_claude_low_resolves_expected_model(self):
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="analysis", model="")
        )
        r = self._call(provider="claude", intelligence="low")
        assert r["status"] == "success"
        assert r["model_resolved"] == "claude-haiku-4-5"
        assert r["provider_resolved"] == "anthropic"

    def test_missing_route_fails_closed(self):
        router = LLMRouter({"providers": {"broken": {"models": {"low": {}}}}})
        with pytest.raises(ModelRouteMissingError):
            router.resolve_route("broken", "low")

    def test_unknown_provider_maps_failure_type(self):
        r = self._call(provider="nonexistent")
        assert r["status"] == "error"
        assert r["failure_type"] == "unknown_provider"
        assert r["provider_called"] is False

    def test_unknown_intelligence_maps_failure_type(self):
        r = self._call(intelligence="ultra")
        assert r["status"] == "error"
        assert r["failure_type"] == "unknown_intelligence"

    # --- P8: stable envelope ---

    def test_empty_response_is_error(self):
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="   ", model="gemini-2.5-flash")
        )
        r = self._call()
        assert r["status"] == "error"
        assert r["failure_type"] == "empty_response"
        assert r["provider_called"] is True
        assert r["retryable"] is True

    def test_provider_timeout_classified(self):
        self.server._router.query = MagicMock(side_effect=TimeoutError("request timed out"))
        r = self._call()
        assert r["status"] == "error"
        assert r["failure_type"] == "provider_timeout"
        assert r["retryable"] is True
        assert r["provider_called"] is True

    def test_health_check_exposes_routes(self):
        r = self.tools["health_check"]()
        assert r["status"] == "ready"
        assert r["routes"]["chatgpt"]["high"] == "gpt-5.5"
        assert r["routes"]["claude"]["low"] == "claude-haiku-4-5"
        assert "example-review" in r["prompt_contracts"]
        assert "docx_document_program_v1" in r["response_contracts"]
