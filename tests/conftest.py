"""Isolate every test from the machine it runs on.

ServerSettings reads MCP_* environment variables and defaults config_path to
the live ai-mcp-server.yaml, which is gitignored and edited freely (swapping a
model, adding a directory alias). Tests that inherited it failed whenever that
file changed. This fixture pins the suite to a config it owns, and keeps audit
and debug output out of the real .audit/ and .debug/ directories.
"""
import os
from pathlib import Path

import pytest

FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "ai-mcp-server.test.yaml"


@pytest.fixture(autouse=True)
def isolated_server_environment(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.upper().startswith("MCP_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("MCP_CONFIG_PATH", str(FIXTURE_CONFIG))
    monkeypatch.setenv("MCP_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("MCP_DEBUG_DIR", str(tmp_path / "debug"))
