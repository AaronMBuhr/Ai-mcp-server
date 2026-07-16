# Implementation Phases: AI MCP Server

## Phase 1: Strip and Scaffold the MCP Shell
**Sections covered: 2, 3, 4, 8, 9, 24, 25**

Scope:
- Take the existing `ai-mcp-server.py`, remove all FAISS/search/embedding imports, settings, tools, and initialization
- Revise `ServerSettings` with the new AI-specific fields
- Revise CLI arguments (remove `--device`/`--model`/`--embedding-*`, add `--config`/`--prompt-dir`/`--save-debug-prompts`/`--debug-dir`)
- Revise `main()` labels and help page text
- Strip `MCPServer.__init__` and `create_public_server()` of search internals (leave them as stubs)
- Confirm the server still starts in stdio and http mode without crashing

**Deliverable:** A clean, runnable shell that starts up with the new settings but has no functional tools yet (just a stub `health_check`).

**Why one phase:** These are all "subtractive" edits to a single file. Low risk, and the AI can hold the full file context.

---

## Phase 2: LLM Providers Module
**Sections covered: 5, 6, 15, 16, 17**

Scope:
- Create `llm_providers.py` by extracting from the existing `llm.py`
- Implement `LLMOptions` and `LLMResult` dataclasses
- Implement `LLMProvider` ABC with `generate(system_prompt, user_prompt, options)`
- Implement `ChatGPTProvider`, `ClaudeProvider`, `GeminiProvider` with the new interface
- Implement `create_llm_provider()` factory
- Keep retry logic, rate-limit handling, auth error handling
- Remove all job/resume/batch functions

**Deliverable:** A standalone `llm_providers.py` that can be imported and tested in isolation.

**Why separate:** This is the most complex module. It needs careful attention to each provider's API differences, and the existing `llm.py` is 1700+ lines, so the AI needs focused context on just the provider extraction.

---

## Phase 3: Config Loading + Prompt Engine
**Sections covered: 10, 12, 13, 14, 23**

Scope:
- Create the `ai-mcp-server.yaml` configuration file
- Implement `load_yaml_config()` with validation (providers exist, models have required tiers)
- Implement `resolve_prompt_path()` with security checks
- Implement `render_template()` with explicit placeholder replacement (not `str.format()`)
- Implement `combine_user_prompt()`
- Implement `sha256_text()` utility
- Implement debug prompt file saving logic
- Create `prompts/test-system.md` and `prompts/test-user.md`

**Deliverable:** All the "pure function" utilities needed by `query_ai`, plus the YAML config and test prompts.

**Why grouped:** These are all stateless utility functions with no provider dependencies. They can be fully unit-tested without API keys or network access.

---

## Phase 4: Router + Tool Integration
**Sections covered: 18, 19, 20, 21, 22, 26**

Scope:
- Implement `LLMRouter` class (provider/intelligence resolution, config merging, provider caching)
- Wire up `create_public_server()` to load config, create router, register tools
- Implement the full `query_ai` tool function (validation, template loading, rendering, routing, response normalization)
- Implement the revised `health_check` tool
- Keep the tool compatibility patch (Section 26)

**Deliverable:** A fully wired MCP server where `query_ai` works end-to-end if you have API keys set.

**Why this is the integration phase:** It depends on Phases 1-3 being complete. It ties all the pieces together into the working tool.

---

## Phase 5: Tests
**Sections covered: 30 (all subsections)**

Scope:
- `tests/test_config_loading.py` — YAML loading, validation, merging
- `tests/test_prompt_rendering.py` — path safety, template rendering, combine logic
- `tests/test_llm_routing.py` — router resolution with fake providers
- `tests/test_query_ai.py` — MCP tool function with mocked router

**Deliverable:** A comprehensive pytest suite that validates all logic without real API calls.

**Why last before smoke test:** Tests written after integration can cover real code paths and catch wiring bugs. Fake providers avoid API costs.

---

## Phase 6: Smoke Test + Claude Code Wiring
**Sections covered: 28, 31, 32, 33**

Scope:
- Run a real `query_ai` call with ChatGPT low tier
- Verify `health_check` returns expected data
- Confirm stderr logging / stdout MCP protocol separation
- Set up Claude Code MCP configuration
- Validate the "Definition of Done" checklist

**Deliverable:** Proven working system, documented MCP config.

**Why last:** This requires real API keys and manual verification. It's the acceptance test.

---

## Summary Table

| Phase | Focus | Key Files | Testable Without API? |
|-------|-------|-----------|----------------------|
| 1 | Strip shell, add settings | `ai-mcp-server.py` | Yes (startup only) |
| 2 | Provider module | `llm_providers.py` | Partially (structure) |
| 3 | Config + prompts | `ai-mcp-server.yaml`, utility fns, `prompts/` | Yes (fully) |
| 4 | Router + tool wiring | `ai-mcp-server.py` | With mocks |
| 5 | Tests | `tests/` | Yes |
| 6 | Smoke test | Config/manual | Requires API keys |

---

## Dependencies

- Phases 2 and 3 are independent of each other; both depend only on Phase 1.
- Phase 4 requires Phases 1, 2, and 3.
- Phase 5 requires Phase 4.
- Phase 6 requires Phase 5.

Phases 2 and 3 can be parallelized if desired.
