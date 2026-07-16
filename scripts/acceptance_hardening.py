#!/usr/bin/env python3
"""Acceptance test for AI_MCP_SERVER_HARDENING.md ("Acceptance Test for the
Whole Workflow").

Default mode is OFFLINE: the router is mocked so no provider is contacted and
no API keys are needed.  Pass --live to run step 3-5 against the real
providers with small synthetic inputs (costs a few cheap calls).

Run from the repo root:
    C:\\Users\\aaron\\anaconda3\\envs\\mypython312\\python.exe scripts/acceptance_hardening.py [--live]

Exit code 0 means every check passed; the real workflow may run.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from llm_providers import LLMResult  # noqa: E402

CHECKS: list = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))
    marker = "PASS" if passed else "FAIL"
    print(f"[{marker}] {name}" + (f" — {detail}" if detail and not passed else ""))


def get_server_and_tools(audit: bool = False):
    exec_globals = {}
    code = (REPO_ROOT / "ai-mcp-server.py").read_text(encoding="utf-8")
    code = code.split("if __name__")[0]
    exec(code, exec_globals)
    settings = exec_globals["ServerSettings"](audit_log_enabled=audit)
    server = exec_globals["MCPServer"](settings)
    public = server.create_public_server()
    tools = {}
    for tool in public._tool_manager.list_tools():
        tools[tool.name] = getattr(tool, "function", None) or getattr(tool, "fn", None)
    return server, tools


SYNTHETIC_JOB = (
    "Acme Inc is hiring a Senior Backend Engineer to own our ingestion "
    "pipeline. You will design resilient async services, debug production "
    "failures, and work with Python, PostgreSQL, and message queues. "
) * 5

SYNTHETIC_RESUME = (
    "# Test Candidate\n\n## Summary\n\nSenior backend engineer with deep "
    "production experience in Python services, data pipelines, and "
    "operational debugging across distributed systems.\n\n## Experience\n\n"
    "* Built async ingestion services processing malformed input safely.\n"
    "* Owned reliability and observability for production data flows.\n"
) * 8

SHARED = {
    "product": "Acme",
    "audience": "Senior Backend Engineer",
    "specification": SYNTHETIC_JOB,
    "reference_doc": SYNTHETIC_RESUME,
}

THREE_REQUESTS = [
    {
        "id": provider,
        "provider": provider,
        "intelligence": "low",
        "system_prompt_template": "example/review-system",
        "user_prompt_template": "example/review-user",
    }
    for provider in ("gemini", "chatgpt", "claude")
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Run the batch step against real providers")
    args = parser.parse_args()

    server, tools = get_server_and_tools()
    query_ai = tools["query_ai"]
    query_ai_batch = tools["query_ai_batch"]

    # ------------------------------------------------------------------
    # Steps 1-2: READ_FROM_FILE rejected before any provider contact.
    # ------------------------------------------------------------------
    sentinel = MagicMock()
    server._router.query = sentinel
    values = dict(SHARED)
    values["reference_doc"] = r"READ_FROM_FILE:C:\fake.md"
    r = query_ai(
        provider="chatgpt", intelligence="low",
        system_prompt_template="example/review-system",
        user_prompt_template="example/review-user",
        replacement_values=values,
    )
    check(
        "1-2. READ_FROM_FILE rejected before provider",
        r["status"] == "error"
        and r["failure_type"] == "invalid_content_transport"
        and r["provider_called"] is False
        and not sentinel.called,
        json.dumps({k: r.get(k) for k in ("status", "failure_type", "provider_called")}),
    )

    # ------------------------------------------------------------------
    # Steps 3-5: valid batch -> identical shared hashes + resolved models.
    # ------------------------------------------------------------------
    if args.live:
        server, tools = get_server_and_tools(audit=True)
        query_ai_batch = tools["query_ai_batch"]
    else:
        server._router.query = MagicMock(
            side_effect=lambda p, i, s, u: LLMResult(text=f"analysis from {p}", model=f"{p}-model")
        )

    rb = query_ai_batch(requests=THREE_REQUESTS, shared_replacement_values=SHARED)
    ok_status = rb.get("status") in ("success", "partial_success")
    check("3. batch ran" + (" (live)" if args.live else " (mocked)"),
          ok_status, rb.get("detail") or rb.get("status", ""))

    if ok_status:
        shared_meta = rb["shared_replacement_metadata"]
        hashes_ok = all(
            res["status"] != "success"
            or all(
                res["replacement_metadata"][key]["sha256"] == meta["sha256"]
                for key, meta in shared_meta.items()
            )
            for res in rb["results"]
        )
        check("4. all results report identical source hashes", hashes_ok)
        models_ok = all(
            res["status"] != "success" or bool(res.get("model_resolved"))
            for res in rb["results"]
        )
        check("5. each result reports its resolved model", models_ok,
              str([(res["id"], res.get("model_resolved")) for res in rb["results"]]))

    # ------------------------------------------------------------------
    # Steps 6-7: renderer fixture with trailing JSON tail is rejected.
    # ------------------------------------------------------------------
    js_plus_tail = (
        'import fs from "node:fs";\n'
        "const doc = buildDocument();\n"
        "fs.writeFileSync(process.env.OUTPUT_DOCX_PATH, buffer);\n"
        '\n{"renderer_type": "node", "notes": ["trailing structured data"]}\n'
    )
    server._router.query = MagicMock(
        return_value=LLMResult(text=js_plus_tail, model="claude-haiku-4-5")
    )
    query_ai = tools["query_ai"]
    r = query_ai(
        provider="claude", intelligence="low",
        system_prompt_template="example/compose-system",
        user_prompt_template="example/compose-user",
        additional_user_prompt="Return a docx document program for the attached document.",
        response_contract="docx_document_program_v1",
    )
    check(
        "6-7. docx_document_program_v1 rejects source-plus-JSON-tail",
        r["status"] == "error" and r["failure_type"] == "response_contract_failed",
        json.dumps({k: r.get(k) for k in ("status", "failure_type")}),
    )

    failed = [name for name, passed, _ in CHECKS if not passed]
    print()
    if failed:
        print(f"{len(failed)} check(s) FAILED: {', '.join(failed)}")
        print("Do NOT run the real workflow.")
        return 1
    print(f"All {len(CHECKS)} checks passed. Step 8: the real workflow may run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
