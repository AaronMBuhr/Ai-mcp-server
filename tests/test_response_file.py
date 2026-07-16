"""Tests for response_file / max_response_bytes (server-side response transport).

The response is written to disk verbatim -- raw text, no wrapper -- so hashing
the file reproduces response_sha256. max_response_bytes caps only the inline
copy, which is what lets a client with a small tool-output budget receive a
short answer inline and fall back to the file for a long one.
"""
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_providers import LLMResult


def _load():
    exec_globals = {}
    code = Path("ai-mcp-server.py").read_text(encoding="utf-8")
    code = code.split("if __name__")[0]
    exec(code, exec_globals)
    return exec_globals


G = _load()
resolve_alias_write_path = G["resolve_alias_write_path"]
truncate_utf8 = G["truncate_utf8"]
ServerSettings = G["ServerSettings"]
MCPServer = G["MCPServer"]

LONG_RESPONSE = "The ingestion path must be idempotent. " * 200  # ~7.6k chars


class TestResolveAliasWritePath:
    def _alias(self, tmp_path):
        return {"out": tmp_path.resolve()}

    def test_disabled_when_no_aliases(self, tmp_path):
        out = resolve_alias_write_path("out/x.md", aliases={})
        assert "error" in out and "disabled" in out["error"]

    def test_resolves_nonexistent_file(self, tmp_path):
        """Unlike the read side, the target need not exist yet."""
        out = resolve_alias_write_path("out/new/file.md", aliases=self._alias(tmp_path))
        assert "error" not in out
        assert out["path"] == (tmp_path / "new" / "file.md").resolve()

    def test_unknown_alias(self, tmp_path):
        out = resolve_alias_write_path("nope/x.md", aliases=self._alias(tmp_path))
        assert "error" in out and "unknown directory alias" in out["error"]

    def test_rejects_dotdot_traversal(self, tmp_path):
        out = resolve_alias_write_path("out/../escape.md", aliases=self._alias(tmp_path))
        assert "error" in out and ".." in out["error"]

    def test_requires_alias_and_relpath(self, tmp_path):
        assert "error" in resolve_alias_write_path("out", aliases=self._alias(tmp_path))
        assert "error" in resolve_alias_write_path("out/", aliases=self._alias(tmp_path))

    def test_rejects_directory_target(self, tmp_path):
        (tmp_path / "adir").mkdir()
        out = resolve_alias_write_path("out/adir", aliases=self._alias(tmp_path))
        assert "error" in out and "not a regular file" in out["error"]


class TestTruncateUtf8:
    def test_under_limit_untouched(self):
        text, cut = truncate_utf8("hello", 100)
        assert text == "hello" and cut is False

    def test_zero_returns_nothing(self):
        text, cut = truncate_utf8("hello", 0)
        assert text == "" and cut is True

    def test_never_splits_a_character(self):
        """A byte limit can land mid-character; the result must stay valid."""
        text = "é" * 10          # 2 bytes each
        out, cut = truncate_utf8(text, 5)   # lands mid-character
        assert cut is True
        assert out == "éé"        # 4 bytes, partial char dropped
        assert len(out.encode("utf-8")) <= 5
        out.encode("utf-8").decode("utf-8")  # must not raise


class TestQueryAiResponseFile:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.root = tmp_path.resolve()
        s = ServerSettings(audit_log_enabled=False)
        self.server = MCPServer(s)
        orig = self.server.create_public_server

        def patched():
            ps = orig()
            self.server._directory_aliases = {"out": self.root}
            return ps
        ps = patched()
        self.tools = {}
        for tool in ps._tool_manager.list_tools():
            fn = getattr(tool, "function", None) or getattr(tool, "fn", None)
            self.tools[tool.name] = fn
        self.server._router.query = MagicMock(
            return_value=LLMResult(text=LONG_RESPONSE, model="gpt-4.1-mini")
        )

    def _call(self, **kw):
        return self.tools["query_ai"](
            provider="chatgpt", intelligence="low",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            **kw,
        )

    def test_file_receives_response_verbatim(self):
        r = self._call(response_file="out/analysis.md")
        assert r["status"] == "success"
        written = Path(r["response_path"]).read_text(encoding="utf-8")
        assert written == LONG_RESPONSE
        # The file IS the response: hashing it reproduces response_sha256.
        digest = hashlib.sha256(written.encode("utf-8")).hexdigest()
        assert digest == r["response_sha256"]

    def test_reports_file_size_and_no_truncation_by_default(self):
        r = self._call(response_file="out/analysis.md")
        assert r["response_file_bytes"] == len(LONG_RESPONSE.encode("utf-8"))
        assert r["response_bytes"] == len(LONG_RESPONSE.encode("utf-8"))
        assert r["response_characters"] == len(LONG_RESPONSE)
        # No cap requested: inline is complete.
        assert r["response_truncated"] is False
        assert r["response_text"] == LONG_RESPONSE

    def test_zero_bytes_is_file_only(self):
        r = self._call(response_file="out/analysis.md", max_response_bytes=0)
        assert r["response_text"] == ""
        assert r["response_truncated"] is True
        # The full response still reached disk, and the hash still describes it.
        assert Path(r["response_path"]).read_text(encoding="utf-8") == LONG_RESPONSE
        assert r["response_file_bytes"] == len(LONG_RESPONSE.encode("utf-8"))

    def test_cap_truncates_inline_but_not_file(self):
        r = self._call(response_file="out/analysis.md", max_response_bytes=1024)
        assert r["response_truncated"] is True
        assert len(r["response_text"].encode("utf-8")) <= 1024
        assert r["response_text"] == LONG_RESPONSE[:len(r["response_text"])]
        assert Path(r["response_path"]).read_text(encoding="utf-8") == LONG_RESPONSE
        # response_sha256 describes the whole response, not the inline slice.
        assert r["response_sha256"] == hashlib.sha256(
            LONG_RESPONSE.encode("utf-8")).hexdigest()

    def test_short_answer_under_cap_is_not_truncated(self):
        """Short answer: returned inline AND written to the file. The caller may
        ignore the file, but the response is there regardless."""
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="OK", model="gpt-4.1-mini")
        )
        r = self._call(response_file="out/short.md", max_response_bytes=1024)
        assert r["response_truncated"] is False
        assert r["response_text"] == "OK"
        assert r["response_file_bytes"] == 2
        assert Path(r["response_path"]).read_text(encoding="utf-8") == "OK"

    @pytest.mark.parametrize("text", ["OK", LONG_RESPONSE])
    @pytest.mark.parametrize("cap", [None, 0, 1024, 10_000_000])
    def test_file_always_gets_the_complete_response(self, text, cap, tmp_path):
        """The file is never conditional. Whatever the cap and whatever the
        length, a requested file holds the whole response -- so 'always request
        the file, always ignore the inline' is a valid way to use this."""
        self.server._router.query = MagicMock(
            return_value=LLMResult(text=text, model="gpt-4.1-mini")
        )
        kw = {"response_file": "out/always.md"}
        if cap is not None:
            kw["max_response_bytes"] = cap
        r = self._call(**kw)
        assert r["status"] == "success"
        assert Path(r["response_path"]).read_text(encoding="utf-8") == text
        assert r["response_file_bytes"] == len(text.encode("utf-8"))
        assert r["response_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_creates_parent_directories(self):
        r = self._call(response_file="out/2026-07-16_acme/analysis-chatgpt.md")
        assert r["status"] == "success"
        assert Path(r["response_path"]).is_file()
        assert Path(r["response_path"]).parent.name == "2026-07-16_acme"

    def test_bad_alias_rejected_before_provider(self):
        r = self._call(response_file="nope/analysis.md")
        assert r["status"] == "error"
        assert r["failure_type"] == "file_write_failed"
        assert r["provider_called"] is False
        self.server._router.query.assert_not_called()

    def test_traversal_rejected_before_provider(self):
        r = self._call(response_file="out/../escape.md")
        assert r["status"] == "error"
        assert r["provider_called"] is False
        self.server._router.query.assert_not_called()

    def test_cap_without_file_is_honored(self):
        """A cap with no file is a legitimate choice. Some MCP clients reject an
        oversized tool result outright rather than truncating it, and a rejected
        result loses everything -- text, hashes, model_resolved. Capping keeps the
        result deliverable. The loss is never silent: response_truncated reports
        it, and response_bytes / response_sha256 still describe the complete
        response."""
        r = self._call(max_response_bytes=1024)
        assert r["status"] == "success"
        assert r["response_truncated"] is True
        assert len(r["response_text"].encode("utf-8")) <= 1024
        assert "response_path" not in r
        # The caller can always detect the truncation.
        assert r["response_bytes"] == len(LONG_RESPONSE.encode("utf-8"))
        assert r["response_sha256"] == hashlib.sha256(
            LONG_RESPONSE.encode("utf-8")).hexdigest()
        assert hashlib.sha256(
            r["response_text"].encode("utf-8")).hexdigest() != r["response_sha256"]

    def test_cap_without_file_not_truncated_when_short(self):
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="OK", model="gpt-4.1-mini")
        )
        r = self._call(max_response_bytes=1024)
        assert r["status"] == "success"
        assert r["response_truncated"] is False
        assert r["response_text"] == "OK"
        # Untruncated inline copy does match the hash of the whole response.
        assert hashlib.sha256(
            r["response_text"].encode("utf-8")).hexdigest() == r["response_sha256"]

    def test_no_response_file_keeps_old_shape(self):
        """Omitting both params leaves the previous behavior untouched."""
        r = self._call()
        assert r["status"] == "success"
        assert r["response_text"] == LONG_RESPONSE
        assert r["response_truncated"] is False
        assert "response_path" not in r
        assert "response_file_bytes" not in r


class TestQueryAiBatchResponseFile:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.root = tmp_path.resolve()
        s = ServerSettings(audit_log_enabled=False)
        self.server = MCPServer(s)
        orig = self.server.create_public_server

        def patched():
            ps = orig()
            self.server._directory_aliases = {"out": self.root}
            return ps
        ps = patched()
        self.tools = {}
        for tool in ps._tool_manager.list_tools():
            fn = getattr(tool, "function", None) or getattr(tool, "fn", None)
            self.tools[tool.name] = fn
        self.server._router.query = MagicMock(
            return_value=LLMResult(text=LONG_RESPONSE, model="gemini-2.5-flash")
        )

    def _reqs(self, **extra):
        base = {
            "provider": "gemini", "intelligence": "middle",
            "system_prompt_template": "test-system",
            "user_prompt_template": "test-user",
        }
        return [dict(base, id="a", **extra)]

    def test_per_request_file_and_cap(self):
        r = self.tools["query_ai_batch"](
            requests=self._reqs(response_file="out/a.md", max_response_bytes=0),
        )
        assert r["status"] == "success"
        res = r["results"][0]
        assert res["response_text"] == ""
        assert res["response_truncated"] is True
        assert Path(res["response_path"]).read_text(encoding="utf-8") == LONG_RESPONSE
        assert res["response_file_bytes"] == len(LONG_RESPONSE.encode("utf-8"))

    def test_bad_path_fails_whole_batch_before_any_provider(self):
        """Phase-1 validation is all-or-nothing: no provider is paid."""
        r = self.tools["query_ai_batch"](requests=self._reqs(response_file="nope/a.md"))
        assert r["status"] == "error"
        assert r["provider_called"] is False
        assert r["failure_type"] == "file_write_failed"
        self.server._router.query.assert_not_called()

    def test_cap_without_file_is_honored_per_request(self):
        r = self.tools["query_ai_batch"](requests=self._reqs(max_response_bytes=512))
        assert r["status"] == "success"
        res = r["results"][0]
        assert res["response_truncated"] is True
        assert len(res["response_text"].encode("utf-8")) <= 512
        assert "response_path" not in res
