# Phase 6: Smoke Test + Claude Code Wiring — Discrete Implementation Steps

This phase validates the server works end-to-end with real API keys and documents the MCP configuration for Cursor/Claude Code usage.

**Prerequisites:**
- Phases 1–5 complete (all 58 tests passing)
- At least one API key available (OPENAI_API_KEY recommended for cheapest smoke test)

---

## Step 1: Fix the stdio transport compatibility issue

In Phase 1 testing, we observed that `public_server.run()` fails with `'function' object is not subscriptable` due to an anyio/mcp version incompatibility on the system Python (3.11). The `mypython312` conda env now has `mcp 1.27.2` installed (from Phase 5), which should resolve this.

Verify stdio starts cleanly:

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 ai-mcp-server.py --debug --transport stdio
```

Expected: Server starts, prints initialization logs to stderr, then waits for MCP protocol input on stdin. Press Ctrl+C to exit.

If it fails, investigate the traceback. The most likely issue is anyio version mismatch — ensure `anyio>=4.5` is installed.

---

## Step 2: Verify health_check via direct function call

Before wiring up an MCP client, verify the tool function works programmatically:

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 -c "
import sys; sys.path.insert(0, '.')
exec(open('ai-mcp-server.py', encoding='utf-8').read().split('if __name__')[0])
s = ServerSettings()
m = MCPServer(s)
ps = m.create_public_server()
for tool in ps._tool_manager.list_tools():
    if tool.name == 'health_check':
        fn = getattr(tool, 'function', None) or getattr(tool, 'fn', None)
        print(fn())
"
```

Expected output:
```
{'status': 'ready', 'server_name': 'ai-mcp-server', 'providers': ['chatgpt', 'claude', 'gemini'], 'prompt_dir': '...\\prompts', 'config_path': 'ai-mcp-server.yaml'}
```

---

## Step 3: Run a real ChatGPT low smoke test

Set the API key and call `query_ai` with the test templates:

```powershell
$env:OPENAI_API_KEY="sk-..."

conda run -n mypython312 --no-capture-output python -X utf8 -c "
import sys, os; sys.path.insert(0, '.')
exec(open('ai-mcp-server.py', encoding='utf-8').read().split('if __name__')[0])
s = ServerSettings()
m = MCPServer(s)
ps = m.create_public_server()
for tool in ps._tool_manager.list_tools():
    if tool.name == 'query_ai':
        fn = getattr(tool, 'function', None) or getattr(tool, 'fn', None)
        result = fn(
            provider='chatgpt',
            intelligence='low',
            system_prompt_template='test-system',
            user_prompt_template='test-user',
            replacement_values={'name': 'Aaron'},
        )
        import json
        print(json.dumps(result, indent=2))
"
```

Expected response:
```json
{
  "status": "success",
  "provider": "chatgpt",
  "intelligence": "low",
  "model": "gpt-4.1-mini",
  "response_text": "...(greeting to Aaron)...",
  "warnings": [...],
  "usage": null,
  "elapsed_ms": ...,
  "system_prompt_template": "test-system",
  "user_prompt_template": "test-user",
  "system_prompt_sha256": "...",
  "user_prompt_sha256": "..."
}
```

**What to verify:**
- `status` is `"success"`
- `model` is `"gpt-4.1-mini"` (matches config)
- `response_text` contains a greeting to Aaron
- `elapsed_ms` is a positive integer
- SHA256 hashes are 64-char hex strings

---

## Step 4: Verify debug prompt saving

Run with `--save-debug-prompts` to confirm debug file is written:

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 -c "
import sys, os; sys.path.insert(0, '.')
exec(open('ai-mcp-server.py', encoding='utf-8').read().split('if __name__')[0])
s = ServerSettings(save_debug_prompts=True, debug_dir='.debug/ai-mcp-server')
m = MCPServer(s)
ps = m.create_public_server()
for tool in ps._tool_manager.list_tools():
    if tool.name == 'query_ai':
        fn = getattr(tool, 'function', None) or getattr(tool, 'fn', None)
        fn(
            provider='chatgpt',
            intelligence='low',
            system_prompt_template='test-system',
            user_prompt_template='test-user',
            replacement_values={'name': 'Aaron'},
        )
from pathlib import Path
debug_files = list(Path('.debug/ai-mcp-server').glob('query-ai-*.json'))
print(f'Debug files found: {len(debug_files)}')
if debug_files:
    import json
    data = json.loads(debug_files[-1].read_text())
    print(f'Keys: {list(data.keys())}')
    print(f'Provider: {data[\"provider\"]}')
    assert 'system_prompt' in data
    assert 'user_prompt' in data
    print('Debug prompt save: OK')
"
```

Expected: At least one debug file written, containing `system_prompt`, `user_prompt`, `provider`, etc.

---

## Step 5: Verify stderr/stdout separation

Run the server in stdio mode briefly and confirm:
- Log messages go to stderr (the terminal)
- stdout is clean (reserved for MCP protocol JSON)

```powershell
conda run -n mypython312 --no-capture-output python -X utf8 ai-mcp-server.py --debug --transport stdio 2>stderr.log
```

Then inspect `stderr.log` — it should contain `"AI MCP Server initializing..."` and other log messages.

stdout should have nothing (or only MCP protocol messages if a client is connected).

---

## Step 6: Configure Cursor MCP server

Create or update the Cursor MCP configuration. The exact location depends on your setup:

**For Cursor IDE (`.cursor/mcp.json` in the workspace or user settings):**

```json
{
  "mcpServers": {
    "ai": {
      "command": "conda",
      "args": [
        "run", "-n", "mypython312", "--no-capture-output",
        "python", "-X", "utf8",
        "E:\\Source\\Mine\\AIMcpServer\\ai-mcp-server.py",
        "--transport", "stdio"
      ],
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "GEMINI_API_KEY": "...",
        "ANTHROPIC_API_KEY": "..."
      }
    }
  }
}
```

**Alternative if API keys are already in system environment variables:**

```json
{
  "mcpServers": {
    "ai": {
      "command": "conda",
      "args": [
        "run", "-n", "mypython312", "--no-capture-output",
        "python", "-X", "utf8",
        "E:\\Source\\Mine\\AIMcpServer\\ai-mcp-server.py",
        "--transport", "stdio"
      ]
    }
  }
}
```

**Alternative using system Python (if deps are installed there):**

```json
{
  "mcpServers": {
    "ai": {
      "command": "python",
      "args": [
        "E:\\Source\\Mine\\AIMcpServer\\ai-mcp-server.py",
        "--transport", "stdio"
      ]
    }
  }
}
```

---

## Step 7: Verify from Cursor/Claude Code MCP client

After configuring the MCP server in Cursor:

1. Call `health_check` — verify it returns `status: "ready"` with providers listed
2. Call `query_ai` with:
   ```json
   {
     "provider": "chatgpt",
     "intelligence": "low",
     "system_prompt_template": "test-system",
     "user_prompt_template": "test-user",
     "additional_user_prompt": "Add one short sentence saying the AI MCP server worked.",
     "replacement_values": {
       "name": "Aaron"
     }
   }
   ```
3. Verify the response has `status: "success"` and `response_text` contains a greeting

---

## Step 8: Walk the Definition of Done checklist

Review each item from Section 33 and mark it off:

| # | Requirement | Status |
|---|-------------|--------|
| 1 | MCP shell runs in stdio mode | Verify in Step 1 |
| 2 | HTTP/HTTPS mode works or fails gracefully | Already handled in Phase 1 (FASTAPI_AVAILABLE guard) |
| 3 | Documentation search imports are gone | Verified in Phase 1 (grep showed 0 matches) |
| 4 | FAISS/search initialization is gone | Verified in Phase 1 |
| 5 | search_for_chunks/documents are gone | Verified in Phase 1 |
| 6 | query_ai MCP tool exists | Verified in Phase 4+5 |
| 7 | health_check is AI-specific | Verified in Phase 4+5 (shows providers list) |
| 8 | ai-mcp-server.yaml controls providers | Verified in Phase 3+5 |
| 9 | provider + intelligence resolves to model | Verified in Phase 5 (14 routing tests) |
| 10 | prompt files loaded from prompts/ | Verified in Phase 5 |
| 11 | template replacements work | Verified in Phase 5 (11 rendering tests) |
| 12 | missing replacements error | Verified in Phase 5 |
| 13 | unused replacement values warn | Verified in Phase 5 |
| 14 | system/user prompts sent separately | Verified in Phase 5 (routing tests) |
| 15 | ChatGPT provider works | Verify in Step 3 (real API call) |
| 16 | Gemini provider is wired | Verify with Gemini key if available |
| 17 | Claude provider is wired | Verify with Anthropic key if available |
| 18 | No job/resume logic in MCP server | Verified — no imports from find-jobs |
| 19 | Real low-cost ChatGPT smoke test succeeds | Verify in Step 3 |
| 20 | Claude Code can call via MCP stdio | Verify in Step 7 |

---

## Step 9: (Optional) Test Gemini and Claude providers

If you have keys for the other providers:

```powershell
$env:GEMINI_API_KEY="..."

conda run -n mypython312 --no-capture-output python -X utf8 -c "
import sys; sys.path.insert(0, '.')
exec(open('ai-mcp-server.py', encoding='utf-8').read().split('if __name__')[0])
s = ServerSettings()
m = MCPServer(s)
ps = m.create_public_server()
for tool in ps._tool_manager.list_tools():
    if tool.name == 'query_ai':
        fn = getattr(tool, 'function', None) or getattr(tool, 'fn', None)
        result = fn(provider='gemini', intelligence='low', system_prompt_template='test-system', user_prompt_template='test-user', replacement_values={'name': 'Aaron'})
        print(result['status'], result.get('model'), result.get('response_text', '')[:100])
"
```

Repeat for `provider='claude'` with `ANTHROPIC_API_KEY`.

---

## Step 10: Clean up and document

1. Delete the `.debug/ai-mcp-server/` directory if created during testing (or `.gitignore` it)
2. Add `.debug/` to `.gitignore` if not already there
3. Optionally create a brief `README.md` describing how to run and configure the server

---

## Summary

Phase 6 is primarily a **manual verification phase**. The code is already complete from Phases 1–5. This phase proves:

- The server starts and stays alive in stdio mode
- A real LLM API call works end-to-end through `query_ai`
- Debug file saving works when enabled
- The Cursor MCP client can connect and invoke tools
- All 20 Definition of Done items are satisfied
