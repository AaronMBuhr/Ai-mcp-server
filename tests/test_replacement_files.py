"""Tests for server-side file transport via directory aliases.

Covers the pure resolver/merge helpers and the end-to-end query_ai path with a
mocked router, so no provider is contacted. File references use the
'alias/relative/path' form; each alias maps to one real directory via config.
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
resolve_replacement_files = G["resolve_replacement_files"]
resolve_alias_reference = G["resolve_alias_reference"]
merge_replacement_files = G["merge_replacement_files"]
resolve_attached_files = G["resolve_attached_files"]
parse_directory_aliases = G["parse_directory_aliases"]
ServerSettings = G["ServerSettings"]
MCPServer = G["MCPServer"]

EXAMPLE_VALUES = {
    "product": "Acme",
    "audience": "Senior Backend Engineer",
    "specification": "We are hiring a senior backend engineer to own ingestion. " * 20,
    "reference_doc": "# Aaron Buhr\n\nExtensive backend and data-pipeline experience. " * 40,
}


class TestParseDirectoryAliases:
    def test_alias_maps_to_one_directory(self, tmp_path):
        a = tmp_path / "a"; a.mkdir()
        aliases = parse_directory_aliases({"single": str(a)})
        assert aliases["single"] == a.resolve()

    def test_rejects_superseded_list_form(self, tmp_path):
        """A list was the old file_families shape; an alias names exactly one
        directory, so an old-format entry is dropped rather than guessed at."""
        a = tmp_path / "a"; a.mkdir()
        b = tmp_path / "b"; b.mkdir()
        aliases = parse_directory_aliases({"multi": [str(a), str(b)], "ok": str(a)})
        assert "multi" not in aliases
        assert aliases["ok"] == a.resolve()

    def test_rejects_alias_name_with_slash(self, tmp_path):
        a = tmp_path / "a"; a.mkdir()
        aliases = parse_directory_aliases({"bad/name": str(a)})
        assert aliases == {}

    def test_non_dict_is_empty(self):
        assert parse_directory_aliases(None) == {}
        assert parse_directory_aliases("nope") == {}


class TestResolveAliasReference:
    def _alias(self, tmp_path):
        return {"apps": tmp_path.resolve()}

    def test_disabled_when_no_aliases(self, tmp_path):
        out = resolve_alias_reference("apps/x.md", aliases={}, max_bytes=1_000_000)
        assert "error" in out and "disabled" in out["error"]

    def test_resolves_and_reads(self, tmp_path):
        sub = tmp_path / "2026-07-13"; sub.mkdir()
        f = sub / "resume.md"
        f.write_text("REAL BODY", encoding="utf-8")
        out = resolve_alias_reference(
            "apps/2026-07-13/resume.md", aliases=self._alias(tmp_path), max_bytes=1_000_000
        )
        assert out.get("text") == "REAL BODY"
        assert out["path"] == f.resolve()

    def test_unknown_alias(self, tmp_path):
        out = resolve_alias_reference("nope/x.md", aliases=self._alias(tmp_path), max_bytes=1_000_000)
        assert "error" in out and "unknown directory alias" in out["error"]

    def test_rejects_dotdot_traversal(self, tmp_path):
        (tmp_path.parent / "secret.md").write_text("secret", encoding="utf-8")
        out = resolve_alias_reference(
            "apps/../secret.md", aliases=self._alias(tmp_path), max_bytes=1_000_000
        )
        assert "error" in out and ".." in out["error"]

    def test_requires_alias_and_relpath(self, tmp_path):
        out = resolve_alias_reference("apps", aliases=self._alias(tmp_path), max_bytes=1_000_000)
        assert "error" in out
        out2 = resolve_alias_reference("apps/", aliases=self._alias(tmp_path), max_bytes=1_000_000)
        assert "error" in out2

    def test_missing_file(self, tmp_path):
        out = resolve_alias_reference("apps/nope.md", aliases=self._alias(tmp_path), max_bytes=1_000_000)
        assert "error" in out and "not found" in out["error"]

    def test_oversize(self, tmp_path):
        (tmp_path / "big.md").write_text("z" * 5000, encoding="utf-8")
        out = resolve_alias_reference("apps/big.md", aliases=self._alias(tmp_path), max_bytes=1000)
        assert "error" in out and "byte limit" in out["error"]


class TestResolveReplacementFiles:
    def test_reads_field_from_alias(self, tmp_path):
        f = tmp_path / "source-resume.md"
        f.write_text(EXAMPLE_VALUES["reference_doc"], encoding="utf-8")
        out = resolve_replacement_files(
            {"reference_doc": "apps/source-resume.md"},
            aliases={"apps": tmp_path.resolve()}, max_bytes=1_000_000,
        )
        assert out["contents"]["reference_doc"] == EXAMPLE_VALUES["reference_doc"]
        assert out["source_paths"]["reference_doc"] == str(f.resolve())

    def test_empty_is_noop(self):
        assert resolve_replacement_files({}, aliases={}, max_bytes=10) == {
            "contents": {}, "source_paths": {}}

    def test_bad_ref_reports_field(self, tmp_path):
        out = resolve_replacement_files(
            {"reference_doc": "apps/missing.md"},
            aliases={"apps": tmp_path.resolve()}, max_bytes=1_000_000,
        )
        assert "error" in out
        assert "reference_doc" in out["error"]["detail"]


class TestMergeReplacementFiles:
    def test_merges_disjoint(self):
        out = merge_replacement_files({"a": "1"}, {"b": "2"})
        assert out["merged"] == {"a": "1", "b": "2"}

    def test_rejects_collision(self):
        out = merge_replacement_files({"a": "1"}, {"a": "2"})
        assert "error" in out
        assert out["error"]["failure_type"] == "unexpected_replacement"


class TestResolveAttachedFiles:
    def test_empty_is_noop(self):
        out = resolve_attached_files([], aliases={}, max_bytes=10)
        assert out == {"block": "", "metadata": {}}

    def test_reads_named_file(self, tmp_path):
        f = tmp_path / "source-resume.md"
        f.write_text("MASTER RESUME BODY", encoding="utf-8")
        out = resolve_attached_files(
            [{"name": "source-resume.md", "path": "apps/source-resume.md"}],
            aliases={"apps": tmp_path.resolve()}, max_bytes=1_000_000,
        )
        assert "error" not in out
        assert "=== FILE: source-resume.md ===" in out["block"]
        assert "MASTER RESUME BODY" in out["block"]
        assert out["metadata"]["source-resume.md"]["source_path"] == str(f.resolve())

    def test_inline_content_allowed(self):
        out = resolve_attached_files(
            [{"name": "note", "content": "inline note text"}], aliases={}, max_bytes=10)
        assert "error" not in out
        assert "inline note text" in out["block"]
        assert "source_path" not in out["metadata"]["note"]

    def test_both_path_and_content_rejected(self, tmp_path):
        (tmp_path / "a.md").write_text("x", encoding="utf-8")
        out = resolve_attached_files(
            [{"name": "a", "path": "apps/a.md", "content": "y"}],
            aliases={"apps": tmp_path.resolve()}, max_bytes=1_000_000)
        assert "error" in out

    def test_missing_name_rejected(self):
        out = resolve_attached_files([{"content": "x"}], aliases={}, max_bytes=10)
        assert "error" in out

    def test_duplicate_name_rejected(self):
        out = resolve_attached_files(
            [{"name": "a", "content": "1"}, {"name": "a", "content": "2"}],
            aliases={}, max_bytes=10)
        assert "error" in out
        assert out["error"]["failure_type"] == "unexpected_replacement"


class TestQueryAiFileTransport:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.root = tmp_path.resolve()
        s = ServerSettings(audit_log_enabled=False)
        self.server = MCPServer(s)
        # Inject an alias pointing at the temp dir via the loaded config.
        orig = self.server.create_public_server

        def patched():
            ps = orig()
            self.server._directory_aliases = {"apps": self.root}
            return ps
        ps = patched()
        self.tools = {}
        for tool in ps._tool_manager.list_tools():
            fn = getattr(tool, "function", None) or getattr(tool, "fn", None)
            self.tools[tool.name] = fn
        self.server._router.query = MagicMock(
            return_value=LLMResult(text="analysis text", model="gemini-2.5-flash")
        )

    def test_file_sourced_field_reaches_provider(self):
        (self.root / "source-resume.md").write_text(EXAMPLE_VALUES["reference_doc"], encoding="utf-8")
        values = {k: v for k, v in EXAMPLE_VALUES.items() if k != "reference_doc"}
        r = self.tools["query_ai"](
            provider="gemini", intelligence="middle",
            system_prompt_template="example/review-system",
            user_prompt_template="example/review-user",
            replacement_values=values,
            replacement_files={"reference_doc": "apps/source-resume.md"},
        )
        assert r["status"] == "success"
        meta = r["replacement_metadata"]["reference_doc"]
        assert meta["source_path"] == str((self.root / "source-resume.md").resolve())
        assert meta["sha256"] == hashlib.sha256(
            EXAMPLE_VALUES["reference_doc"].encode("utf-8")).hexdigest()

    def test_collision_between_value_and_file_rejected(self):
        (self.root / "source-resume.md").write_text(EXAMPLE_VALUES["reference_doc"], encoding="utf-8")
        r = self.tools["query_ai"](
            provider="gemini", intelligence="middle",
            system_prompt_template="example/review-system",
            user_prompt_template="example/review-user",
            replacement_values=dict(EXAMPLE_VALUES),
            replacement_files={"reference_doc": "apps/source-resume.md"},
        )
        assert r["status"] == "error"
        assert r["failure_type"] == "unexpected_replacement"
        assert r["provider_called"] is False

    def test_unknown_alias_rejected_before_provider(self):
        values = {k: v for k, v in EXAMPLE_VALUES.items() if k != "reference_doc"}
        r = self.tools["query_ai"](
            provider="gemini", intelligence="middle",
            system_prompt_template="example/review-system",
            user_prompt_template="example/review-user",
            replacement_values=values,
            replacement_files={"reference_doc": "nope/source-resume.md"},
        )
        assert r["status"] == "error"
        assert r["failure_type"] == "file_read_failed"
        assert r["provider_called"] is False
        self.server._router.query.assert_not_called()

    def test_attached_files_appended_to_prompt(self):
        (self.root / "source-resume.md").write_text("ATTACHED MASTER RESUME CONTENT", encoding="utf-8")
        (self.root / "job-description.md").write_text("ATTACHED JOB POSTING CONTENT", encoding="utf-8")
        r = self.tools["query_ai"](
            provider="gemini", intelligence="middle",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            additional_user_prompt="Use source-resume.md as the master resume and job-description.md as the target job.",
            files=[
                {"name": "source-resume.md", "path": "apps/source-resume.md"},
                {"name": "job-description.md", "path": "apps/job-description.md"},
            ],
        )
        assert r["status"] == "success"
        sent_user_prompt = self.server._router.query.call_args.args[3]
        assert "=== FILE: source-resume.md ===" in sent_user_prompt
        assert "ATTACHED MASTER RESUME CONTENT" in sent_user_prompt
        assert "ATTACHED JOB POSTING CONTENT" in sent_user_prompt
        meta = r["attachment_metadata"]["source-resume.md"]
        assert meta["source_path"] == str((self.root / "source-resume.md").resolve())
        assert meta["sha256"] == hashlib.sha256(b"ATTACHED MASTER RESUME CONTENT").hexdigest()

    def test_attached_file_unknown_alias_rejected(self):
        r = self.tools["query_ai"](
            provider="gemini", intelligence="middle",
            system_prompt_template="test-system",
            user_prompt_template="test-user",
            files=[{"name": "x", "path": "nope/x.md"}],
        )
        assert r["status"] == "error"
        assert r["failure_type"] == "file_read_failed"
        assert r["provider_called"] is False
        self.server._router.query.assert_not_called()
