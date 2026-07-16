"""Tests for the concurrent query_ai_batch tool (hardening P5)."""

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_providers import LLMResult


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


SHARED_VALUES = {
    "product": "Acme",
    "audience": "Senior Backend Engineer",
    "specification": "We are hiring a senior backend engineer to own ingestion. " * 20,
    "reference_doc": "# Aaron Buhr\n\nExtensive backend and data-pipeline experience. " * 40,
}


def _three_requests():
    return [
        {
            "id": provider,
            "provider": provider,
            "intelligence": "middle",
            "system_prompt_template": "example/review-system",
            "user_prompt_template": "example/review-user",
            "additional_user_prompt": "",
        }
        for provider in ("gemini", "chatgpt", "claude")
    ]


class TestQueryAiBatch:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.server, self.tools = _get_server_and_tools()
        self.batch = self.tools["query_ai_batch"]

    def test_all_three_run_concurrently(self):
        """All calls must start without waiting for one another: each provider
        call blocks on a 3-party barrier, so serialized execution deadlocks
        (barrier timeout) and the test fails."""
        barrier = threading.Barrier(3, timeout=10)

        def blocking_query(provider, intelligence, system_prompt, user_prompt):
            barrier.wait()
            return LLMResult(text=f"analysis from {provider}", model=f"{provider}-model")

        self.server._router.query = MagicMock(side_effect=blocking_query)

        r = self.batch(requests=_three_requests(), shared_replacement_values=SHARED_VALUES)
        assert r["status"] == "success"
        assert [res["id"] for res in r["results"]] == ["gemini", "chatgpt", "claude"]
        assert all(res["status"] == "success" for res in r["results"])

    def test_results_in_request_order_with_ids(self):
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="ok", model="m")
        )
        r = self.batch(requests=_three_requests(), shared_replacement_values=SHARED_VALUES)
        assert [res["id"] for res in r["results"]] == ["gemini", "chatgpt", "claude"]
        assert "batch_id" in r and len(r["batch_id"]) == 36

    def test_identical_shared_hashes_across_results(self):
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="ok", model="m")
        )
        r = self.batch(requests=_three_requests(), shared_replacement_values=SHARED_VALUES)

        shared = r["shared_replacement_metadata"]
        assert set(shared.keys()) == set(SHARED_VALUES.keys())
        for res in r["results"]:
            for key, meta in shared.items():
                assert res["replacement_metadata"][key]["sha256"] == meta["sha256"]

    def test_one_failure_produces_partial_success(self):
        def flaky_query(provider, intelligence, system_prompt, user_prompt):
            if provider == "claude":
                raise TimeoutError("model timed out")
            return LLMResult(text=f"analysis from {provider}", model=f"{provider}-model")

        self.server._router.query = MagicMock(side_effect=flaky_query)

        r = self.batch(requests=_three_requests(), shared_replacement_values=SHARED_VALUES)
        assert r["status"] == "partial_success"
        by_id = {res["id"]: res for res in r["results"]}
        assert by_id["gemini"]["status"] == "success"
        assert by_id["chatgpt"]["status"] == "success"
        assert by_id["claude"]["status"] == "error"
        assert by_id["claude"]["failure_type"] == "provider_timeout"

    def test_invalid_shared_input_prevents_all_provider_calls(self):
        self.server._router.query = MagicMock()
        bad_shared = dict(SHARED_VALUES)
        bad_shared["reference_doc"] = r"READ_FROM_FILE:C:\fake.md"

        r = self.batch(requests=_three_requests(), shared_replacement_values=bad_shared)
        assert r["status"] == "error"
        assert r["provider_called"] is False
        assert r["failure_type"] == "invalid_content_transport"
        assert len(r["validation_failures"]) == 3
        self.server._router.query.assert_not_called()

    def test_empty_batch_rejected(self):
        r = self.batch(requests=[], shared_replacement_values={})
        assert r["status"] == "error"
        assert r["provider_called"] is False

    def test_duplicate_ids_rejected(self):
        requests = _three_requests()
        requests[1]["id"] = "gemini"
        self.server._router.query = MagicMock()
        r = self.batch(requests=requests, shared_replacement_values=SHARED_VALUES)
        assert r["status"] == "error"
        self.server._router.query.assert_not_called()

    def test_all_failures_produce_error_status(self):
        self.server._router.query = MagicMock(side_effect=RuntimeError("boom"))
        r = self.batch(requests=_three_requests(), shared_replacement_values=SHARED_VALUES)
        assert r["status"] == "error"
        assert all(res["status"] == "error" for res in r["results"])
