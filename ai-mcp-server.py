#!/usr/bin/env python3
"""
AI MCP Server
=============

A generic Model Context Protocol (MCP) server that routes prompt templates
to configured LLM providers (ChatGPT, Gemini, Claude).

Features:
- Prompt template loading and placeholder rendering
- Provider/intelligence-tier routing via YAML config
- Support for stdio, HTTP, and HTTPS transports
"""

import argparse
import contextlib
import copy
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import logging
import os
from pathlib import Path
from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
import sys
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional, Annotated, Set, Tuple

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

load_dotenv(PROJECT_ROOT / ".env")

from prompt_engine import (
    LLMRouter,
    RoutingError,
    combine_user_prompt,
    load_yaml_config,
    render_template,
    resolve_prompt_path,
    save_debug_prompt,
    sha256_text,
    template_placeholders,
)
from prompt_contracts import (
    ContractError,
    PromptContract,
    check_contract_against_templates,
    load_contracts,
    validate_replacements,
)
from content_validation import replacement_metadata
from response_contracts import (
    KNOWN_CONTRACTS,
    ResponseContractError,
    parse_docx_document_program_v1,
)
from audit_log import AuditLogger

# MCP imports
from mcp.server.fastmcp import FastMCP

# FastAPI imports for HTTP mode
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, RedirectResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


# Configure logging following reference pattern
def setup_logger() -> logging.Logger:
    """Configure logging for the MCP server to stderr."""
    logger = logging.getLogger("ai-mcp-server")

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create stderr handler for console output
    stderr_handler = logging.StreamHandler(sys.stderr)

    # Add handler to logger
    logger.addHandler(stderr_handler)

    # Set level
    logger.setLevel(logging.INFO)

    return logger


logger: logging.Logger = setup_logger()


class ServerSettings(BaseSettings):
    """Settings for the AI MCP Server."""

    model_config = SettingsConfigDict(env_prefix="MCP_", extra="allow")

    # Transport settings
    transport: str = Field(default="stdio", description="Transport mode: stdio, http, or https")
    host: str = Field(default="localhost", description="Server host address")
    port: int = Field(default=8000, description="Server port number")
    server_url: AnyHttpUrl = Field(default="http://localhost:8000", description="Full server URL")

    # SSL settings
    ssl_cert_path: Optional[str] = Field(default=None, description="Path to SSL certificate file")
    ssl_key_path: Optional[str] = Field(default=None, description="Path to SSL private key file")

    # Server identity
    server_name: str = Field(default="ai-mcp-server", description="Name of the MCP server")
    description: str = Field(default="generic AI provider routing server", description="Server description")

    # AI-specific settings
    config_path: str = Field(default="ai-mcp-server.yaml", description="Path to YAML config file")
    prompt_dir: str = Field(default="prompts", description="Directory containing prompt templates")
    save_debug_prompts: bool = Field(default=False, description="Save rendered prompts to debug files")
    debug_dir: str = Field(default=".debug/ai-mcp-server", description="Directory for debug prompt files")

    # Hardening settings
    require_prompt_contracts: bool = Field(
        default=False,
        description="Reject calls to template pairs that have no prompt contract",
    )
    audit_log_enabled: bool = Field(default=True, description="Write JSONL audit records per call")
    audit_dir: str = Field(default=".audit/ai-mcp-server", description="Directory for audit JSONL files")
    audit_retention_days: int = Field(default=30, description="Days of audit files to retain (0 = keep forever)")
    batch_max_workers: int = Field(default=8, description="Max concurrent provider calls in query_ai_batch")

    # Server-side file transport. Directory aliases are configured in the
    # YAML config under 'directory_aliases:' (name -> dir); requests reference
    # files as 'alias/relative/path'. Empty disables the feature (fail closed).
    max_replacement_file_bytes: int = Field(
        default=10_000_000,
        description="Maximum size of a single file read via replacement_files/files",
    )


# ---------------------------------------------------------------------------
# Result envelope + request pipeline (AI_MCP_SERVER_HARDENING.md P8, P2-P4)
# ---------------------------------------------------------------------------

def error_result(
    failure_type: str,
    detail: str,
    *,
    request_id: str,
    provider_called: bool,
    retryable: bool = False,
    error_type: Optional[str] = None,
    message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Stable error envelope.

    Keeps the legacy 'error_type'/'message' keys (existing clients branch on
    them) and adds the stable 'failure_type'/'detail'/'provider_called'/
    'retryable' fields from the hardening spec.
    """
    result: Dict[str, Any] = {
        "status": "error",
        "failure_type": failure_type,
        "detail": detail,
        "request_id": request_id,
        "provider_called": provider_called,
        "retryable": retryable,
        # Legacy keys, kept for backward compatibility:
        "error_type": error_type or failure_type,
        "message": message or detail,
    }
    if extra:
        result.update(extra)
    return result


def prepare_query(
    *,
    prompt_dir: Path,
    contracts: Dict[Tuple[str, str], PromptContract],
    router: LLMRouter,
    require_contracts: bool,
    provider: str,
    intelligence: str,
    system_prompt_template: str,
    user_prompt_template: str,
    additional_user_prompt: str,
    replacement_values: Dict[str, Any],
    request_id: str,
    attached_files_block: str = "",
) -> Dict[str, Any]:
    """Validate and render a query without contacting any provider.

    Returns {"error": <error envelope>} on any validation failure, else
    {"prepared": {...}} with everything the execution step needs.  Every
    rejection happens before provider invocation (fail closed).
    """
    def _fail(failure_type: str, detail: str, *, error_type: str = None,
              extra: Dict[str, Any] = None) -> Dict[str, Any]:
        return {"error": error_result(
            failure_type, detail,
            request_id=request_id, provider_called=False,
            error_type=error_type, extra=extra,
        )}

    # 1. Resolve templates safely.
    try:
        sys_path = resolve_prompt_path(prompt_dir, system_prompt_template)
        usr_path = resolve_prompt_path(prompt_dir, user_prompt_template)
    except FileNotFoundError as e:
        return _fail("prompt_not_found", str(e), error_type="FileNotFoundError")
    except ValueError as e:
        return _fail("prompt_not_found", str(e), error_type="ValueError")

    sys_template_text = sys_path.read_text(encoding="utf-8")
    usr_template_text = usr_path.read_text(encoding="utf-8")

    placeholders: Set[str] = (
        template_placeholders(sys_template_text)
        | template_placeholders(usr_template_text)
    )

    # 2. Contract enforcement (P1/P2/P3).
    contract = contracts.get((system_prompt_template, user_prompt_template))
    if contract is not None:
        mismatch = check_contract_against_templates(contract, placeholders)
        if mismatch is not None:
            return _fail(mismatch["failure_type"], mismatch["detail"],
                         error_type="ValueError")
        violation = validate_replacements(contract, replacement_values)
        if violation is not None:
            extra = {k: v for k, v in violation.items()
                     if k not in ("failure_type", "detail")}
            return _fail(violation["failure_type"], violation["detail"],
                         error_type="ValueError", extra=extra)
    elif require_contracts:
        return _fail(
            "prompt_contract_missing",
            f"No prompt contract found for template pair "
            f"('{system_prompt_template}', '{user_prompt_template}') and "
            f"require_prompt_contracts is enabled.",
            error_type="ValueError",
        )

    # 3. Render templates (belt-and-suspenders re-check of placeholders).
    try:
        system_prompt, sys_warnings = render_template(
            sys_template_text, replacement_values,
            template_label=f"the system template '{system_prompt_template}'",
        )
        user_prompt_rendered, usr_warnings = render_template(
            usr_template_text, replacement_values,
            template_label=f"the user template '{user_prompt_template}'",
        )
    except ValueError as e:
        msg = str(e)
        failure_type = ("missing_replacement" if "Missing replacement values" in msg
                        else "unexpected_replacement")
        return _fail(failure_type, msg, error_type="ValueError")

    warnings = sys_warnings + usr_warnings
    user_prompt = combine_user_prompt(user_prompt_rendered, additional_user_prompt)
    if attached_files_block:
        user_prompt = user_prompt.rstrip() + "\n\n" + attached_files_block

    # 4. Resolve the provider route before contacting anything (P4).
    try:
        route = router.resolve_route(provider, intelligence)
    except RoutingError as e:
        return _fail(e.failure_type, str(e), error_type="ValueError")

    return {"prepared": {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "warnings": warnings,
        # Raw template hashes: constant for a given template revision, so they
        # fingerprint WHICH template version produced this call. They differ from
        # the rendered hashes below whenever substitution actually occurred --
        # equality means nothing was substituted.
        "system_template_sha256": sha256_text(sys_template_text),
        "user_template_sha256": sha256_text(usr_template_text),
        # Rendered prompt hashes: exactly what the provider received, including
        # additional_user_prompt and any attached files block.
        "rendered_system_prompt_sha256": sha256_text(system_prompt),
        "rendered_user_prompt_sha256": sha256_text(user_prompt),
        "replacement_metadata": {
            key: replacement_metadata(value)
            for key, value in sorted(replacement_values.items())
        },
        "provider_resolved": route["provider_resolved"],
        "model_resolved": route["model_resolved"],
    }}


def parse_directory_aliases(raw: Any) -> Dict[str, Path]:
    """Parse the config's directory_aliases mapping (name -> dir) into resolved Paths.

    An alias name is a simple identifier (no path separators) naming exactly one
    real source directory; a request references a file as 'alias/relative/path'
    and the server resolves it against that directory.
    """
    aliases: Dict[str, Path] = {}
    if not isinstance(raw, dict):
        return aliases
    for name, directory in raw.items():
        name = str(name).strip()
        if not name or "/" in name or "\\" in name:
            continue
        if isinstance(directory, (list, tuple)):
            # The superseded file_families shape mapped a name to a list of
            # directories. An alias names exactly one, so an old-format entry is
            # config to fix rather than something to guess at.
            logger.warning(
                "Ignoring directory alias %r: expected a single directory path, "
                "got a list. Update the config to 'name: C:\\path'.", name,
            )
            continue
        text = str(directory).strip() if directory is not None else ""
        if text:
            aliases[name] = Path(text).resolve()
    return aliases


def resolve_alias_reference(
    ref: str,
    *,
    aliases: Dict[str, Path],
    max_bytes: int,
) -> Dict[str, Any]:
    """Resolve an 'alias/relative/path' reference to an on-disk file, safely.

    Security (fail closed): the feature is disabled unless directory_aliases are
    configured; the reference must name a known alias and a relative path;
    '..' segments are rejected; the resolved path (following links) must stay
    inside the alias's directory; the file must be a regular UTF-8 file within
    max_bytes.

    Returns {"path": Path, "text": str} on success, else {"error": <detail>}.
    """
    if not aliases:
        return {"error": "no directory_aliases are configured on this server; "
                         "server-side file transport is disabled."}
    if not isinstance(ref, str) or not ref.strip():
        return {"error": f"file reference is not a usable string: {ref!r}."}
    normalized = ref.strip().replace("\\", "/")
    if "/" not in normalized:
        return {"error": f"file reference {ref!r} must be of the form "
                         "'alias/relative/path'."}
    alias_name, rel = normalized.split("/", 1)
    alias_name = alias_name.strip()
    rel = rel.strip("/")
    if not alias_name or not rel:
        return {"error": f"file reference {ref!r} must name a directory alias "
                         "and a relative path."}
    if alias_name not in aliases:
        return {"error": f"unknown directory alias {alias_name!r}; configured "
                         f"aliases: {sorted(aliases)}."}
    if ".." in rel.split("/"):
        return {"error": f"file reference {ref!r} may not contain '..' segments."}

    root = aliases[alias_name]
    candidate = (root / rel).resolve()
    root_cmp = os.path.normcase(str(root))
    cand_cmp = os.path.normcase(str(candidate))
    if not (cand_cmp == root_cmp or cand_cmp.startswith(root_cmp + os.sep)):
        # Escaped the root (defensive; '..' already blocked).
        return {"error": f"file reference {ref!r} resolves outside the directory "
                         f"for alias {alias_name!r}."}
    if not candidate.is_file():
        return {"error": f"file reference {ref!r} not found in the directory for "
                         f"alias {alias_name!r}."}
    size = candidate.stat().st_size
    if size > max_bytes:
        return {"error": f"{ref!r} is {size:,} bytes, exceeding the "
                         f"{max_bytes:,} byte limit."}
    try:
        text = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return {"error": f"{ref!r} could not be read as UTF-8: {e}"}
    return {"path": candidate, "text": text}


def resolve_alias_write_path(
    ref: str,
    *,
    aliases: Dict[str, Path],
) -> Dict[str, Any]:
    """Resolve an 'alias/relative/path' reference to a writable on-disk path.

    The read-side twin (resolve_alias_reference) requires the file to exist;
    this one does not, since it names a file about to be created. The same
    containment rules apply: known alias, relative path, no '..' segments, and
    the resolved path must stay inside the alias's directory.

    Resolution happens before the provider is contacted, so a bad path costs no
    provider call — the caller finds out for free rather than after paying for a
    response that cannot be delivered.

    Returns {"path": Path} on success, else {"error": <detail>}.
    """
    if not aliases:
        return {"error": "no directory_aliases are configured on this server; "
                         "server-side file transport is disabled."}
    if not isinstance(ref, str) or not ref.strip():
        return {"error": f"response_file is not a usable string: {ref!r}."}
    normalized = ref.strip().replace("\\", "/")
    if "/" not in normalized:
        return {"error": f"response_file {ref!r} must be of the form "
                         "'alias/relative/path'."}
    alias_name, rel = normalized.split("/", 1)
    alias_name = alias_name.strip()
    rel = rel.strip("/")
    if not alias_name or not rel:
        return {"error": f"response_file {ref!r} must name a directory alias "
                         "and a relative path."}
    if alias_name not in aliases:
        return {"error": f"unknown directory alias {alias_name!r}; configured "
                         f"aliases: {sorted(aliases)}."}
    if ".." in rel.split("/"):
        return {"error": f"response_file {ref!r} may not contain '..' segments."}

    root = aliases[alias_name]
    candidate = (root / rel)
    # Resolve the parent rather than the file: the file may not exist yet, and
    # a non-existent tail must not defeat containment checking.
    parent = candidate.parent.resolve()
    root_cmp = os.path.normcase(str(root))
    par_cmp = os.path.normcase(str(parent))
    if not (par_cmp == root_cmp or par_cmp.startswith(root_cmp + os.sep)):
        return {"error": f"response_file {ref!r} resolves outside the directory "
                         f"for alias {alias_name!r}."}
    if candidate.exists() and not candidate.is_file():
        return {"error": f"response_file {ref!r} exists and is not a regular file."}
    return {"path": parent / candidate.name}


def truncate_utf8(text: str, max_bytes: int) -> Tuple[str, bool]:
    """Cut text to at most max_bytes of UTF-8, never splitting a character.

    Returns (text, was_truncated). A byte limit can land mid-character, so the
    tail is decoded with errors='ignore' to drop a partial one; the result is
    always valid UTF-8 and always within the limit.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def resolve_replacement_files(
    replacement_files: Dict[str, str],
    *,
    aliases: Dict[str, Path],
    max_bytes: int,
) -> Dict[str, Any]:
    """Server-side literal file transport: read each field's file from disk.

    This is the explicit, typed alternative to the (still forbidden) pattern
    of smuggling a path inside a content-bearing field. Each value is an
    'alias/relative/path' reference the server resolves and reads itself, so
    downstream contract validation and hash metadata operate on the real
    literal content.

    Returns {"contents": {field: text}, "source_paths": {field: str}} on
    success, else {"error": {"failure_type": ..., "detail": ...}}.
    """
    def _fail(detail: str) -> Dict[str, Any]:
        return {"error": {"failure_type": "file_read_failed", "detail": detail}}

    if not replacement_files:
        return {"contents": {}, "source_paths": {}}

    contents: Dict[str, str] = {}
    source_paths: Dict[str, str] = {}
    for field_name, ref in replacement_files.items():
        out = resolve_alias_reference(ref, aliases=aliases, max_bytes=max_bytes)
        if "error" in out:
            return _fail(f"replacement_files['{field_name}']: {out['error']}")
        contents[field_name] = out["text"]
        source_paths[field_name] = str(out["path"])
    return {"contents": contents, "source_paths": source_paths}


def merge_replacement_files(
    replacement_values: Dict[str, Any],
    file_contents: Dict[str, str],
) -> Dict[str, Any]:
    """Merge file-sourced contents into replacement values.

    A key present in both is a caller bug (ambiguous source of truth) and
    returns an error rather than silently preferring one.
    """
    collisions = sorted(set(replacement_values.keys()) & set(file_contents.keys()))
    if collisions:
        return {"error": {
            "failure_type": "unexpected_replacement",
            "detail": (
                "These keys were supplied in BOTH replacement_values and "
                "replacement_files (ambiguous source of truth): "
                + ", ".join(collisions)
            ),
        }}
    merged = dict(replacement_values)
    merged.update(file_contents)
    return {"merged": merged}


def annotate_source_paths(
    metadata: Optional[Dict[str, Any]],
    source_paths: Dict[str, str],
) -> None:
    """Record where file-sourced fields came from in replacement metadata."""
    if not metadata or not source_paths:
        return
    for field_name, path in source_paths.items():
        entry = metadata.get(field_name)
        if isinstance(entry, dict):
            entry["source_path"] = path


def resolve_attached_files(
    files: List[Dict[str, str]],
    *,
    aliases: Dict[str, Path],
    max_bytes: int,
) -> Dict[str, Any]:
    """Read named file attachments for inclusion in the prompt.

    Unlike replacement_files (which fills a specific template placeholder),
    attachments are labeled blocks appended to the user prompt; the caller's
    additional_user_prompt refers to them by name to control how each is used.

    Each entry is {"name": <label>, "path": "alias/relative/path"} (server
    reads the file, alias-resolved) or {"name": <label>, "content": <text>}
    (inline; use only for genuinely generated content, not file contents).

    Returns {"block": <text to append or "">, "metadata": {name: {...}}} on
    success, else {"error": {"failure_type": ..., "detail": ...}}.
    """
    if not files:
        return {"block": "", "metadata": {}}

    parts: List[str] = []
    metadata: Dict[str, Any] = {}
    seen: Set[str] = set()
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            return {"error": {"failure_type": "file_read_failed",
                              "detail": f"files[{i}] is not an object."}}
        name = str(entry.get("name") or "").strip()
        if not name:
            return {"error": {"failure_type": "file_read_failed",
                              "detail": f"files[{i}] is missing a non-empty 'name'."}}
        if name in seen:
            return {"error": {"failure_type": "unexpected_replacement",
                              "detail": f"Duplicate attachment name: {name!r}."}}
        seen.add(name)

        has_path = bool(str(entry.get("path") or "").strip())
        has_content = entry.get("content") is not None
        if has_path == has_content:
            return {"error": {"failure_type": "file_read_failed",
                              "detail": (f"files[{i}] ({name!r}) must supply exactly "
                                         "one of 'path' or 'content'.")}}

        source_path: Optional[str] = None
        if has_content:
            content = entry["content"]
            if not isinstance(content, str):
                return {"error": {"failure_type": "file_read_failed",
                                  "detail": f"files[{i}] ({name!r}) 'content' must be a string."}}
        else:
            # Reuse the same alias-resolution security as replacement_files.
            single = resolve_replacement_files(
                {name: entry["path"]}, aliases=aliases, max_bytes=max_bytes
            )
            if "error" in single:
                return single
            content = single["contents"][name]
            source_path = single["source_paths"][name]

        parts.append(f"=== FILE: {name} ===\n{content}\n=== END FILE: {name} ===")
        meta = replacement_metadata(content)
        if source_path:
            meta["source_path"] = source_path
        metadata[name] = meta

    block = (
        "--- Attached files (each is delimited below and referenced by name in the "
        "instructions above; use them exactly as those instructions specify) ---\n\n"
        + "\n\n".join(parts)
    )
    return {"block": block, "metadata": metadata}


def classify_provider_exception(exc: Exception) -> Tuple[str, bool]:
    """Map a provider exception to (failure_type, retryable)."""
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "provider_timeout", True
    if "rate" in text or "429" in text or "resource_exhausted" in text:
        return "provider_error", True
    return "provider_error", False


def execute_prepared_query(
    *,
    router: LLMRouter,
    provider: str,
    intelligence: str,
    system_prompt_template: str,
    user_prompt_template: str,
    prepared: Dict[str, Any],
    request_id: str,
    response_contract: Optional[str] = None,
    debug_dir: Optional[str] = None,
    response_path: Optional[Path] = None,
    max_response_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Contact the provider for an already-validated request and build the
    success/error envelope (P4, P6, P8)."""
    start_time = time.time()
    try:
        result = router.query(
            provider, intelligence,
            prepared["system_prompt"], prepared["user_prompt"],
        )
    except Exception as e:
        failure_type, retryable = classify_provider_exception(e)
        return error_result(
            failure_type, str(e),
            request_id=request_id, provider_called=True, retryable=retryable,
            error_type=type(e).__name__,
        )
    elapsed_ms = int((time.time() - start_time) * 1000)

    response_text = result.text or ""
    if not response_text.strip():
        return error_result(
            "empty_response",
            f"Provider '{provider}' returned an empty response.",
            request_id=request_id, provider_called=True, retryable=True,
            extra={"elapsed_ms": elapsed_ms, "model_resolved": prepared["model_resolved"]},
        )

    # Write the response to disk before building the envelope. A requested file
    # ALWAYS receives the complete response -- never conditioned on length or on
    # max_response_bytes -- so a caller may always request the file and ignore it
    # whenever the inline copy suffices. The file gets the response verbatim and
    # nothing else (no wrapper, no metadata), so hashing it reproduces
    # response_sha256 exactly. That is what makes the file trustworthy as the
    # response rather than a report about it.
    written_bytes: Optional[int] = None
    if response_path is not None:
        try:
            response_path.parent.mkdir(parents=True, exist_ok=True)
            data = response_text.encode("utf-8")
            response_path.write_bytes(data)
            written_bytes = len(data)
        except OSError as e:
            # The path was validated before the provider call, so this is rare
            # (disk full, permissions changed mid-flight). The provider was
            # already paid for, so return the full text inline rather than
            # discarding it -- an error the caller can still salvage.
            return error_result(
                "response_write_failed",
                f"Could not write response_file: {e}. The complete response is "
                "returned inline as response_text.",
                request_id=request_id, provider_called=True, retryable=False,
                error_type=type(e).__name__,
                extra={
                    "response_text": response_text,
                    "response_sha256": sha256_text(response_text),
                    "response_characters": len(response_text),
                    "elapsed_ms": elapsed_ms,
                    "model_resolved": result.model or prepared["model_resolved"],
                },
            )

    # max_response_bytes caps only the inline copy; the file always holds the
    # complete response. 0 means "file only".
    inline_text = response_text
    truncated = False
    if max_response_bytes is not None:
        inline_text, truncated = truncate_utf8(response_text, max_response_bytes)

    response: Dict[str, Any] = {
        "status": "success",
        "request_id": request_id,
        "failure_type": None,
        "provider_called": True,
        "retryable": False,
        # Legacy keys:
        "provider": provider,
        "intelligence": intelligence,
        "model": result.model,
        "response_text": inline_text,
        "warnings": list(prepared["warnings"]),
        "usage": result.usage,
        "elapsed_ms": elapsed_ms,
        "system_prompt_template": system_prompt_template,
        "user_prompt_template": user_prompt_template,
        # Raw template hashes (constant per template revision). Compare against
        # the rendered_* hashes below to prove substitution occurred.
        "system_template_sha256": prepared["system_template_sha256"],
        "user_template_sha256": prepared["user_template_sha256"],
        # Hardening metadata (P4):
        "provider_requested": provider,
        "intelligence_requested": intelligence,
        "provider_resolved": prepared["provider_resolved"],
        "model_resolved": result.model or prepared["model_resolved"],
        "rendered_system_prompt_sha256": prepared["rendered_system_prompt_sha256"],
        "rendered_user_prompt_sha256": prepared["rendered_user_prompt_sha256"],
        # Always describes the COMPLETE response, never the inline copy. Hash the
        # response_file and compare against this to prove the file is intact.
        "response_sha256": sha256_text(response_text),
        "response_characters": len(response_text),
        "response_bytes": len(response_text.encode("utf-8")),
        # True when max_response_bytes cut the inline copy. The caller must read
        # response_file for the rest; response_text alone is then incomplete.
        "response_truncated": truncated,
        "replacement_metadata": prepared["replacement_metadata"],
    }
    if response_path is not None:
        response["response_path"] = str(response_path)
        response["response_file_bytes"] = written_bytes

    # Optional structured response contract (P6).
    if response_contract:
        if response_contract not in KNOWN_CONTRACTS:
            return error_result(
                "response_contract_failed",
                f"Unknown response_contract '{response_contract}'. "
                f"Known contracts: {', '.join(KNOWN_CONTRACTS)}",
                request_id=request_id, provider_called=True,
            )
        try:
            parsed = parse_docx_document_program_v1(response_text)
        except ResponseContractError as e:
            if debug_dir:
                _save_raw_response(debug_dir, request_id, response_text)
            return error_result(
                "response_contract_failed", str(e),
                request_id=request_id, provider_called=True, retryable=True,
                extra={
                    "response_contract": response_contract,
                    "raw_response_sha256": sha256_text(response_text),
                    "elapsed_ms": elapsed_ms,
                    "model_resolved": response["model_resolved"],
                },
            )
        # Lift envelope repairs out of the payload: `parsed` stays exactly the
        # shape the workflow contract documents, and a repair shows up in the
        # run's warnings instead of passing silently.
        envelope_warnings = parsed.pop("envelope_warnings", [])
        if envelope_warnings:
            response["warnings"] = list(response["warnings"]) + envelope_warnings
            response["envelope_repaired"] = True
        response["response_contract"] = response_contract
        response["raw_response_sha256"] = sha256_text(response_text)
        response["parsed"] = parsed

    return response


def _save_raw_response(debug_dir: str, request_id: str, response_text: str) -> None:
    """Persist the raw provider response for diagnostics on contract failure."""
    try:
        dir_path = Path(debug_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / f"raw-response-{request_id}.txt").write_text(
            response_text, encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not save raw response for %s: %s", request_id, exc)


def audit_record_for(
    *,
    request_id: str,
    batch_id: Optional[str],
    provider: str,
    intelligence: str,
    system_prompt_template: str,
    user_prompt_template: str,
    prepared: Optional[Dict[str, Any]],
    result: Dict[str, Any],
    duration_ms: int,
) -> Dict[str, Any]:
    """Build a compact audit record (P7) — lengths and hashes, never content."""
    record: Dict[str, Any] = {
        "request_id": request_id,
        "provider_requested": provider,
        "intelligence_requested": intelligence,
        "system_prompt_template": system_prompt_template,
        "user_prompt_template": user_prompt_template,
        "status": result.get("status"),
        "failure_type": result.get("failure_type"),
        "provider_called": result.get("provider_called"),
        "duration_ms": duration_ms,
    }
    if batch_id:
        record["batch_id"] = batch_id
    if prepared:
        record["model_resolved"] = prepared.get("model_resolved")
        record["system_template_sha256"] = prepared.get("system_template_sha256")
        record["user_template_sha256"] = prepared.get("user_template_sha256")
        record["rendered_system_prompt_sha256"] = prepared.get("rendered_system_prompt_sha256")
        record["rendered_user_prompt_sha256"] = prepared.get("rendered_user_prompt_sha256")
        record["replacement_fields"] = {
            key: {"characters": meta["characters"], "sha256": meta["sha256"]}
            for key, meta in (prepared.get("replacement_metadata") or {}).items()
        }
    response_text = result.get("response_text")
    if isinstance(response_text, str):
        record["response_characters"] = len(response_text)
        record["response_sha256"] = result.get("response_sha256") or sha256_text(response_text)
    return record


class FastAppSettings:
    """Application settings for HTTP/HTTPS mode following reference pattern."""
    
    def __init__(self, settings: ServerSettings, mcp_app: FastMCP) -> None:
        self.expose_url: str = str(settings.server_url)
        self.dns: Optional[str] = None
        self.settings: ServerSettings = settings
        self.servers: List[FastMCP] = [mcp_app]  # Use the provided FastMCP app

class FastApp:
    """FastAPI application for HTTP/HTTPS mode following reference pattern."""
    
    def __init__(self, fast_app_settings: FastAppSettings) -> None:
        self.app_settings: FastAppSettings = fast_app_settings

    def create_app(self) -> FastAPI:
        """Create FastAPI application with MCP server mounting."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
            
        servers = self.app_settings.servers

        # Create a combined lifespan to manage session managers
        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI) -> Any:
            async with contextlib.AsyncExitStack() as stack:
                for server in servers:
                    await stack.enter_async_context(server.session_manager.run())
                yield

        app = FastAPI(lifespan=lifespan)
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount the FastMCP app at root so its internal routes
        # (e.g., /mcp, /.well-known/*, /register) are exposed exactly
        # at those paths. This supports reverse proxies routing only
        # /mcp to this upstream.
        # Mount each MCP server at its named path
        # Expose the canonical "/mcp" route expected by tests by mounting
        # each FastMCP app at both "/mcp" (for backwards-compat) and at its
        # human-readable server name.
        for server in servers:
            # Mount canonical path
            app.mount("/mcp", server.streamable_http_app())
            # Also mount by server name for multi-server deployments
            app.mount(f"/{server.name}", server.streamable_http_app())
            
        @app.get("/", include_in_schema=False)
        async def redirect_to_help() -> RedirectResponse:
            return RedirectResponse(url="/help")

        @app.get("/help", include_in_schema=False)
        async def help() -> HTMLResponse:
            """Generate help page with server information."""
            try:
                help_text = """
                <html>
                <head><title>AI MCP Server</title></head>
                <body>
                <h1>AI MCP Server</h1>
                <p>This server provides generic LLM routing through the Model Context Protocol.</p>
                <h2>Available Servers</h2>
                """
                
                for server in servers:
                    tools = []
                    try:
                        tools = [tool.name for tool in server._tool_manager.list_tools()]
                    except Exception:
                        try:
                            maybe_tools = getattr(server, 'tools', [])
                            tools = [getattr(t, 'name', str(t)) for t in maybe_tools]
                        except Exception:
                            tools = []
                    help_text += f"""
                    <h3>{server.name}</h3>
                    <p>{server.instructions}</p>
                    <p><strong>Endpoint:</strong> <code>{self.app_settings.expose_url}/mcp</code></p>
                    <p><strong>Tools:</strong> {', '.join(tools) if tools else 'None'}</p>
                    """

                help_text += "</body></html>"
                return HTMLResponse(content=help_text, status_code=200)

            except Exception as e:
                logger.error(f"Error generating help page: {str(e)}")
                raise HTTPException(status_code=500, detail="Error generating help page")

        return app


class MCPServer:
    """
    MCP Server class that encapsulates AI routing and server creation
    following MCP specification requirements.
    """
    
    def __init__(self, settings: ServerSettings) -> None:
        self._settings: ServerSettings = settings
        self._logger: logging.Logger = logger
        self._is_ready: bool = False
        self._initialization_error: Optional[str] = None
        self._config: Optional[dict] = None
        self._router: Optional[LLMRouter] = None
        self._prompt_dir: Optional[Path] = None
        self._contracts: Dict[Tuple[str, str], PromptContract] = {}
        self._directory_aliases: Dict[str, Path] = {}
        self._audit: Optional[AuditLogger] = None

    def create_public_server(self) -> Optional[FastMCP]:
        """Create the AI MCP server with tool registration."""
        try:
            self._config = load_yaml_config(Path(self._settings.config_path))
            self._prompt_dir = Path(self._settings.prompt_dir).resolve()
            self._router = LLMRouter(self._config)
            # Fail closed on malformed/duplicate contracts (P1).
            self._contracts = load_contracts(self._prompt_dir)
            self._directory_aliases = parse_directory_aliases(
                (self._config or {}).get("directory_aliases")
            )
            self._audit = AuditLogger(
                directory=self._settings.audit_dir,
                enabled=self._settings.audit_log_enabled,
                retention_days=self._settings.audit_retention_days,
            )

            mcp = FastMCP(
                name=self._settings.server_name,
                instructions=(
                    "Generic AI provider routing server. "
                    "Renders prompt templates and queries configured LLM providers."
                ),
                debug=True,
                stateless_http=True,
            )

            self._is_ready = True

        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"Failed to initialize AI MCP server: {e}")
            return None

        # --- Tools ---

        @mcp.tool()
        def health_check() -> Dict[str, Any]:
            """Check the health status of the AI MCP server."""
            if not self._is_ready:
                return {
                    "status": "error",
                    "message": self._initialization_error or "Server not initialized",
                }

            providers = sorted((self._config.get("providers") or {}).keys())

            return {
                "status": "ready",
                "server_name": self._settings.server_name,
                "providers": providers,
                "prompt_dir": str(self._prompt_dir),
                "config_path": self._settings.config_path,
                # Safe route summary (P4): prove which model each
                # provider/tier resolves to, without exposing secrets.
                "routes": self._router.route_map(),
                "prompt_contracts": sorted(
                    contract.name for contract in self._contracts.values()
                ),
                "require_prompt_contracts": self._settings.require_prompt_contracts,
                "response_contracts": list(KNOWN_CONTRACTS),
                "audit_log_enabled": self._settings.audit_log_enabled,
                # Server-side file transport (replacement_files / files):
                "file_params_enabled": bool(self._directory_aliases),
                "directory_aliases": {
                    name: str(directory)
                    for name, directory in self._directory_aliases.items()
                },
            }

        @mcp.tool(
            name="query_ai",
            description="Render system/user prompt templates and query a configured LLM provider."
        )
        def query_ai(
            provider: Annotated[str, Field(description="Provider name from YAML config (e.g., chatgpt, gemini, claude)")],
            intelligence: Annotated[str, Field(description="Model tier: low, middle, or high")],
            system_prompt_template: Annotated[str, Field(description="System prompt template path under prompts/, without .md extension")],
            user_prompt_template: Annotated[str, Field(description="User prompt template path under prompts/, without .md extension")],
            additional_user_prompt: Annotated[str, Field(description="Extra user instructions appended after the rendered user template")] = "",
            replacement_values: Dict[str, Any] = {},
            replacement_files: Annotated[Dict[str, str], Field(
                description=(
                    "Optional server-side file transport: map of replacement "
                    "field name -> an 'alias/relative/path' reference. The "
                    "server resolves the reference against the directory "
                    "configured for that alias in directory_aliases and uses the "
                    "file's complete literal content for the field. A key must "
                    "not appear in both replacement_values and replacement_files."
                ),
            )] = {},
            files: Annotated[List[Dict[str, str]], Field(
                description=(
                    "Optional named file attachments. Each entry is "
                    "{name, path} where path is an 'alias/relative/path' "
                    "reference the server reads from disk, or {name, content} "
                    "(inline text). They are appended to the user prompt as "
                    "labeled '=== FILE: name ===' blocks; refer to them by "
                    "name in additional_user_prompt to control how each is "
                    "used (e.g. 'use source-resume.md as the master resume'). "
                    "Unlike replacement_files, attachments do not fill a "
                    "template placeholder."
                ),
            )] = [],
            response_contract: Annotated[Optional[str], Field(
                description="Optional structured-response contract to enforce, e.g. 'docx_document_program_v1'",
            )] = None,
            response_file: Annotated[Optional[str], Field(
                description=(
                    "Optional 'alias/relative/path' reference. When given, the "
                    "server ALWAYS writes the complete response to that file, "
                    "verbatim -- raw text only, no wrapper and no metadata -- so "
                    "hashing the file reproduces response_sha256. This never "
                    "depends on max_response_bytes or on the response's length. "
                    "Parent directories are created. The path is validated before "
                    "the provider is contacted."
                ),
            )] = None,
            max_response_bytes: Annotated[Optional[int], Field(
                description=(
                    "Optional cap on the INLINE response_text only. It never "
                    "affects response_file, which always receives the complete "
                    "response. 0 returns no inline text at all. A small value "
                    "(e.g. 1024) returns the response inline when it fits and a "
                    "truncated prefix when it does not. Omit for the full "
                    "response inline. response_truncated reports whether the cap "
                    "was hit; response_bytes and response_sha256 always describe "
                    "the complete response, so a truncated inline copy will not "
                    "match that hash.\n"
                    "Usable with or without response_file. Without one, anything "
                    "past the cap is discarded -- but that can still be the right "
                    "call: some MCP clients REJECT an oversized tool result "
                    "outright rather than truncating it, and a rejected result "
                    "loses everything, including the hashes and model_resolved "
                    "needed to verify the call happened. Capping under such a "
                    "client's limit is what makes the result deliverable at all. "
                    "Pair it with response_file to keep the full text as well."
                ),
                ge=0,
            )] = None,
        ) -> Dict[str, Any]:
            """Render prompt templates, query the configured LLM, return normalized response."""
            request_id = str(uuid.uuid4())
            if not self._is_ready or self._router is None or self._prompt_dir is None:
                return error_result(
                    "health_check_failed",
                    self._initialization_error or "Server not initialized",
                    request_id=request_id, provider_called=False,
                    error_type="ServerNotReady",
                )

            # Server-side file transport (fail closed before any provider work).
            file_outcome = resolve_replacement_files(
                replacement_files,
                aliases=self._directory_aliases,
                max_bytes=self._settings.max_replacement_file_bytes,
            )
            if "error" in file_outcome:
                err = file_outcome["error"]
                return error_result(
                    err["failure_type"], err["detail"],
                    request_id=request_id, provider_called=False,
                    error_type="ValueError",
                )
            merge_outcome = merge_replacement_files(
                replacement_values, file_outcome["contents"]
            )
            if "error" in merge_outcome:
                err = merge_outcome["error"]
                return error_result(
                    err["failure_type"], err["detail"],
                    request_id=request_id, provider_called=False,
                    error_type="ValueError",
                )
            replacement_values = merge_outcome["merged"]
            file_source_paths = file_outcome["source_paths"]

            # Named file attachments appended to the prompt (fail closed too).
            attach_outcome = resolve_attached_files(
                files,
                aliases=self._directory_aliases,
                max_bytes=self._settings.max_replacement_file_bytes,
            )
            if "error" in attach_outcome:
                err = attach_outcome["error"]
                return error_result(
                    err["failure_type"], err["detail"],
                    request_id=request_id, provider_called=False,
                    error_type="ValueError",
                )
            attached_files_block = attach_outcome["block"]
            attachment_metadata = attach_outcome["metadata"]

            # Response destination, resolved before any provider work: a bad
            # alias or path must cost nothing.
            #
            # max_response_bytes is honored with or without response_file.
            # Capping without a file discards the remainder, but never silently:
            # response_truncated reports it, and response_bytes/response_sha256
            # still describe the complete response. It is also sometimes the only
            # way to get anything back -- clients differ on overflow, and one that
            # REJECTS an oversized result rather than truncating it loses the
            # whole envelope, hashes included. Capping is what keeps the result
            # deliverable; the caller knows its own client's limit.
            response_path: Optional[Path] = None
            if response_file:
                write_outcome = resolve_alias_write_path(
                    response_file, aliases=self._directory_aliases,
                )
                if "error" in write_outcome:
                    return error_result(
                        "file_write_failed", write_outcome["error"],
                        request_id=request_id, provider_called=False,
                        error_type="ValueError",
                    )
                response_path = write_outcome["path"]

            call_start = time.time()
            prepared: Optional[Dict[str, Any]] = None
            try:
                outcome = prepare_query(
                    prompt_dir=self._prompt_dir,
                    contracts=self._contracts,
                    router=self._router,
                    require_contracts=self._settings.require_prompt_contracts,
                    provider=provider,
                    intelligence=intelligence,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                    additional_user_prompt=additional_user_prompt,
                    replacement_values=replacement_values,
                    request_id=request_id,
                    attached_files_block=attached_files_block,
                )
                if "error" in outcome:
                    result = outcome["error"]
                else:
                    prepared = outcome["prepared"]
                    # Record file provenance so both the response and the
                    # audit record show where file-sourced fields came from.
                    annotate_source_paths(
                        prepared.get("replacement_metadata"), file_source_paths
                    )

                    if self._settings.save_debug_prompts:
                        save_debug_prompt(
                            debug_dir=self._settings.debug_dir,
                            provider=provider,
                            intelligence=intelligence,
                            system_prompt_template=system_prompt_template,
                            user_prompt_template=user_prompt_template,
                            replacement_keys=sorted(replacement_values.keys()),
                            system_prompt=prepared["system_prompt"],
                            user_prompt=prepared["user_prompt"],
                            warnings=prepared["warnings"],
                        )

                    result = execute_prepared_query(
                        router=self._router,
                        provider=provider,
                        intelligence=intelligence,
                        system_prompt_template=system_prompt_template,
                        user_prompt_template=user_prompt_template,
                        prepared=prepared,
                        request_id=request_id,
                        response_contract=response_contract,
                        debug_dir=self._settings.debug_dir,
                        response_path=response_path,
                        max_response_bytes=max_response_bytes,
                    )
            except Exception as e:
                logger.error(f"Unexpected error in query_ai: {traceback.format_exc()}")
                result = error_result(
                    "internal_error", str(e),
                    request_id=request_id, provider_called=True,
                    error_type=type(e).__name__,
                )

            # Surface attachment provenance/hashes on a successful call so the
            # caller can verify what was appended, mirroring replacement_metadata.
            if attachment_metadata and result.get("status") == "success":
                result["attachment_metadata"] = attachment_metadata

            if self._audit is not None:
                self._audit.record(audit_record_for(
                    request_id=request_id,
                    batch_id=None,
                    provider=provider,
                    intelligence=intelligence,
                    system_prompt_template=system_prompt_template,
                    user_prompt_template=user_prompt_template,
                    prepared=prepared,
                    result=result,
                    duration_ms=int((time.time() - call_start) * 1000),
                ))
            return result

        @mcp.tool(
            name="query_ai_batch",
            description=(
                "Validate a set of query_ai requests up front, then run them "
                "concurrently against their providers with identical shared "
                "replacement values. Returns per-request results in order, "
                "plus shared-input hashes proving all calls used the same content."
            ),
        )
        def query_ai_batch(
            requests: Annotated[List[Dict[str, Any]], Field(
                description=(
                    "List of request objects: {id, provider, intelligence, "
                    "system_prompt_template, user_prompt_template, "
                    "additional_user_prompt?, replacement_values?, "
                    "replacement_files?, response_contract?, response_file?, "
                    "max_response_bytes?}. response_file is an "
                    "'alias/relative/path'; when given, the server always writes "
                    "that request's complete response to it verbatim. "
                    "max_response_bytes caps only that request's inline "
                    "response_text (0 = no inline text), never affects the file, "
                    "and may be used with or without response_file."
                ),
            )],
            shared_replacement_values: Dict[str, Any] = {},
            shared_replacement_files: Annotated[Dict[str, str], Field(
                description=(
                    "Optional server-side file transport for shared fields: "
                    "map of field name -> an 'alias/relative/path' reference. "
                    "Each file is read once (resolved via directory_aliases) and "
                    "its complete literal content is shared by every request, "
                    "so the shared hash block covers it."
                ),
            )] = {},
        ) -> Dict[str, Any]:
            """Concurrent multi-provider fan-out with all-or-nothing validation (P5)."""
            batch_id = str(uuid.uuid4())
            if not self._is_ready or self._router is None or self._prompt_dir is None:
                return error_result(
                    "health_check_failed",
                    self._initialization_error or "Server not initialized",
                    request_id=batch_id, provider_called=False,
                    error_type="ServerNotReady",
                )

            if not requests:
                return error_result(
                    "missing_replacement", "Batch contains no requests.",
                    request_id=batch_id, provider_called=False,
                    error_type="ValueError",
                )

            ids = [str(req.get("id") or f"request-{i}") for i, req in enumerate(requests)]
            if len(set(ids)) != len(ids):
                return error_result(
                    "unexpected_replacement",
                    "Batch request ids must be unique.",
                    request_id=batch_id, provider_called=False,
                    error_type="ValueError",
                )

            # Server-side file transport for shared fields: read each file
            # exactly ONCE, before freezing, so every request provably shares
            # identical content and the shared hash block covers it.
            shared_file_outcome = resolve_replacement_files(
                shared_replacement_files,
                aliases=self._directory_aliases,
                max_bytes=self._settings.max_replacement_file_bytes,
            )
            if "error" in shared_file_outcome:
                err = shared_file_outcome["error"]
                return error_result(
                    err["failure_type"], err["detail"],
                    request_id=batch_id, provider_called=False,
                    error_type="ValueError",
                )
            shared_merge = merge_replacement_files(
                shared_replacement_values or {}, shared_file_outcome["contents"]
            )
            if "error" in shared_merge:
                err = shared_merge["error"]
                return error_result(
                    err["failure_type"], err["detail"],
                    request_id=batch_id, provider_called=False,
                    error_type="ValueError",
                )

            # Freeze the shared inputs so every request provably uses
            # identical content.
            frozen_shared = copy.deepcopy(shared_merge["merged"])
            shared_metadata = {
                key: replacement_metadata(value)
                for key, value in sorted(frozen_shared.items())
            }
            annotate_source_paths(shared_metadata, shared_file_outcome["source_paths"])

            # Phase 1: validate every request before contacting any provider.
            prepared_requests: List[Dict[str, Any]] = []
            validation_failures: List[Dict[str, Any]] = []
            for req_id, req in zip(ids, requests):
                request_uuid = str(uuid.uuid4())
                merged_values = dict(frozen_shared)
                merged_values.update(copy.deepcopy(req.get("replacement_values") or {}))
                req_file_outcome = resolve_replacement_files(
                    req.get("replacement_files") or {},
                    aliases=self._directory_aliases,
                    max_bytes=self._settings.max_replacement_file_bytes,
                )
                if "error" in req_file_outcome:
                    failure = dict(req_file_outcome["error"])
                    failure["id"] = req_id
                    validation_failures.append(failure)
                    continue
                req_merge = merge_replacement_files(
                    merged_values, req_file_outcome["contents"]
                )
                if "error" in req_merge:
                    failure = dict(req_merge["error"])
                    failure["id"] = req_id
                    validation_failures.append(failure)
                    continue
                merged_values = req_merge["merged"]

                # Per-request response destination, validated here in phase 1 so
                # one bad path fails the whole batch before any provider is paid.
                # max_response_bytes needs no file: see the note in query_ai.
                req_response_file = req.get("response_file")
                req_max_bytes = req.get("max_response_bytes")
                req_response_path: Optional[Path] = None
                if req_response_file:
                    write_outcome = resolve_alias_write_path(
                        str(req_response_file), aliases=self._directory_aliases,
                    )
                    if "error" in write_outcome:
                        validation_failures.append({
                            "id": req_id,
                            "failure_type": "file_write_failed",
                            "detail": write_outcome["error"],
                        })
                        continue
                    req_response_path = write_outcome["path"]

                entry = {
                    "id": req_id,
                    "request_id": request_uuid,
                    "provider": str(req.get("provider") or ""),
                    "intelligence": str(req.get("intelligence") or ""),
                    "system_prompt_template": str(req.get("system_prompt_template") or ""),
                    "user_prompt_template": str(req.get("user_prompt_template") or ""),
                    "response_contract": req.get("response_contract"),
                    "response_path": req_response_path,
                    "max_response_bytes": req_max_bytes,
                }
                outcome = prepare_query(
                    prompt_dir=self._prompt_dir,
                    contracts=self._contracts,
                    router=self._router,
                    require_contracts=self._settings.require_prompt_contracts,
                    provider=entry["provider"],
                    intelligence=entry["intelligence"],
                    system_prompt_template=entry["system_prompt_template"],
                    user_prompt_template=entry["user_prompt_template"],
                    additional_user_prompt=str(req.get("additional_user_prompt") or ""),
                    replacement_values=merged_values,
                    request_id=request_uuid,
                )
                if "error" in outcome:
                    failure = dict(outcome["error"])
                    failure["id"] = req_id
                    validation_failures.append(failure)
                else:
                    entry["prepared"] = outcome["prepared"]
                    annotate_source_paths(
                        entry["prepared"].get("replacement_metadata"),
                        {**shared_file_outcome["source_paths"],
                         **req_file_outcome["source_paths"]},
                    )
                    prepared_requests.append(entry)

            if validation_failures:
                # Invalid input prevents ALL provider calls (fail closed).
                return {
                    "status": "error",
                    "failure_type": validation_failures[0]["failure_type"],
                    "detail": (
                        "Batch validation failed; no provider was contacted. "
                        "First failure: " + validation_failures[0]["detail"]
                    ),
                    "request_id": batch_id,
                    "batch_id": batch_id,
                    "provider_called": False,
                    "retryable": False,
                    "error_type": "ValueError",
                    "message": validation_failures[0]["detail"],
                    "shared_replacement_metadata": shared_metadata,
                    "validation_failures": validation_failures,
                }

            # Phase 2: run all requests concurrently; a failure in one must
            # not cancel the others.
            call_start = time.time()

            def _run(entry: Dict[str, Any]) -> Dict[str, Any]:
                return execute_prepared_query(
                    router=self._router,
                    provider=entry["provider"],
                    intelligence=entry["intelligence"],
                    system_prompt_template=entry["system_prompt_template"],
                    user_prompt_template=entry["user_prompt_template"],
                    prepared=entry["prepared"],
                    request_id=entry["request_id"],
                    response_contract=entry["response_contract"],
                    debug_dir=self._settings.debug_dir,
                    response_path=entry["response_path"],
                    max_response_bytes=entry["max_response_bytes"],
                )

            max_workers = max(1, min(len(prepared_requests), self._settings.batch_max_workers))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_run, entry) for entry in prepared_requests]
                raw_results = []
                for future in futures:
                    try:
                        raw_results.append(future.result())
                    except Exception as e:  # defensive: _run shouldn't raise
                        logger.error(f"Batch worker error: {traceback.format_exc()}")
                        raw_results.append(error_result(
                            "internal_error", str(e),
                            request_id=batch_id, provider_called=True,
                            error_type=type(e).__name__,
                        ))
            duration_ms = int((time.time() - call_start) * 1000)

            results = []
            success_count = 0
            for entry, result in zip(prepared_requests, raw_results):
                tagged = dict(result)
                tagged["id"] = entry["id"]
                if tagged.get("status") == "success":
                    success_count += 1
                results.append(tagged)
                if self._audit is not None:
                    self._audit.record(audit_record_for(
                        request_id=entry["request_id"],
                        batch_id=batch_id,
                        provider=entry["provider"],
                        intelligence=entry["intelligence"],
                        system_prompt_template=entry["system_prompt_template"],
                        user_prompt_template=entry["user_prompt_template"],
                        prepared=entry["prepared"],
                        result=result,
                        duration_ms=duration_ms,
                    ))

            if success_count == len(results):
                status = "success"
            elif success_count > 0:
                status = "partial_success"
            else:
                status = "error"

            return {
                "status": status,
                "batch_id": batch_id,
                "request_id": batch_id,
                "failure_type": None if success_count else "provider_error",
                "provider_called": True,
                "retryable": success_count < len(results),
                "elapsed_ms": duration_ms,
                "shared_replacement_metadata": shared_metadata,
                "results": results,
            }

        # Tool compatibility patch
        try:
            for tool in mcp._tool_manager.list_tools():
                if not hasattr(tool, "function"):
                    func = (
                        getattr(tool, "_callback", None)
                        or getattr(tool, "callback", None)
                        or getattr(tool, "call", None)
                        or getattr(tool, "fn", None)
                    )
                    if func is not None:
                        setattr(tool, "function", func)
        except Exception:
            pass

        return mcp

    def run_stdio_server(self) -> None:
        """Run MCP server in stdio mode. Now synchronous since FastMCP handles async internally."""
        
        self._logger.debug("Step 7: Creating FastMCP server...")
        public_server = self.create_public_server()
        if public_server is None:
            self._logger.error("Failed to create public server")
            if self._initialization_error:
                raise Exception(self._initialization_error)
            return
        
        self._logger.debug("Step 8: Starting server in stdio mode...")
        self._logger.info("Starting MCP server in stdio mode")

        # Let FastMCP itself spin up the stdio transport and session manager:
        try:
            self._logger.debug("Step 9: Server running in stdio mode...")
            # Type assertion to help linter understand public_server is not None
            assert public_server is not None
            
            # Use FastMCP's default run() method (stdio mode)
            self._logger.debug("Step 10: Starting FastMCP default mode (stdio)...")
            # Use the synchronous run() method without transport parameter (defaults to stdio)
            public_server.run()
            self._logger.debug("Step 11: Server finished.")
        except Exception as e:
            self._logger.error(f"Failed to run public server: {e}")

    def run_http_server_with_mcp(self) -> int:
        """Run MCP server in HTTP mode following reference pattern."""
        
        self._logger.debug("Step 7: Creating FastMCP server...")
        
        # Create public server
        public_app = self.create_public_server()
        if public_app is None:
            self._logger.error("Failed to create public server")
            if self._initialization_error:
                raise Exception(self._initialization_error)
            return 1
        
        self._logger.debug(f"Step 8: Starting server in {self._settings.transport.upper()} mode...")
        
        # Start HTTP server with public app
        self.run_http_server(public_app)
        self._logger.debug("Step 9: HTTP/HTTPS server finished.")
        return 0

    def run_http_server(self, mcp_app: FastMCP) -> None:
        """Run MCP server in HTTP/HTTPS mode following reference pattern."""
        if not FASTAPI_AVAILABLE:
            self._logger.error("FastAPI not available. Install with: pip install fastapi uvicorn")
            return

        try:
            # Create list of servers to mount
            servers = [mcp_app]
                
            app_settings = FastAppSettings(self._settings, mcp_app)
            fast_app = FastApp(app_settings)
            app = fast_app.create_app()
            
            # Configure SSL if HTTPS
            ssl_keyfile = self._settings.ssl_key_path if self._settings.transport == "https" else None
            ssl_certfile = self._settings.ssl_cert_path if self._settings.transport == "https" else None
            
            if self._settings.transport == "https" and (not ssl_keyfile or not ssl_certfile):
                self._logger.warning("SECURITY WARNING: HTTPS requested but SSL cert/key not provided. Falling back to HTTP. This connection will NOT be encrypted!")
                ssl_keyfile = ssl_certfile = None
            
            self._logger.info(f"Starting MCP server on {self._settings.host}:{self._settings.port} ({'HTTPS' if ssl_keyfile else 'HTTP'})")

            uvicorn.run(
                app,
                host=self._settings.host,
                port=self._settings.port,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                log_level="info"
            )
            
        except Exception as e:
            self._logger.error(f"Failed to start HTTP server: {e}")
            raise


def main() -> int:
    """Parse CLI args, create settings, and start the MCP server."""
    logger.info("AI MCP Server initializing...")
    
    logger.debug("Step 1: Parsing arguments...")
    
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="AI MCP Server - A generic Model Context Protocol server for LLM routing",
        epilog="""
Environment Variables:
  All command line arguments can be set via environment variables with MCP_ prefix.
  Command line arguments override environment variables.

  Examples:
    export MCP_TRANSPORT=http
    export MCP_HOST=127.0.0.1
    export MCP_PORT=8008
    export MCP_SERVER_NAME=ai-mcp-server
    export MCP_CONFIG_PATH=ai-mcp-server.yaml
    export MCP_PROMPT_DIR=prompts

    """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-d", "--debug", action="store_true",
                       help="Enable debug logging output.")
    parser.add_argument("--transport", choices=["stdio", "http", "https"],
                       default=os.getenv("MCP_TRANSPORT", "stdio"),
                       help="Transport protocol: 'stdio' for MCP clients, 'http'/'https' for web servers. Default: stdio.")
    parser.add_argument("--host", type=str,
                       default=os.getenv("MCP_HOST", "localhost"),
                       help="Host address to bind server to (only for http/https transport). Default: localhost.")
    parser.add_argument("--port", type=int,
                       default=int(os.getenv("MCP_PORT", "8000")),
                       help="Port number to bind server to (only for http/https transport). Default: 8000.")
    parser.add_argument("--ssl-cert", type=str,
                       default=os.getenv("MCP_SSL_CERT_PATH"),
                       help="Path to SSL certificate file for HTTPS transport.")
    parser.add_argument("--ssl-key", type=str,
                       default=os.getenv("MCP_SSL_KEY_PATH"),
                       help="Path to SSL private key file for HTTPS transport.")
    parser.add_argument("-n", "--server-name", type=str,
                       default=os.getenv("MCP_SERVER_NAME", "ai-mcp-server"),
                       help="Name identifier for the MCP server instance. Default: ai-mcp-server.")
    parser.add_argument("--description", type=str,
                       default=os.getenv("MCP_DESCRIPTION", "generic AI provider routing server"),
                       help="Human-readable description of the MCP server.")
    parser.add_argument("--config", type=str,
                       default=os.getenv("MCP_CONFIG_PATH", "ai-mcp-server.yaml"),
                       help="Path to ai-mcp-server YAML config.")
    parser.add_argument("--prompt-dir", type=str,
                       default=os.getenv("MCP_PROMPT_DIR", "prompts"),
                       help="Directory containing prompt templates.")
    parser.add_argument("--save-debug-prompts", action="store_true",
                       default=os.getenv("MCP_SAVE_DEBUG_PROMPTS", "").lower() in ("1", "true", "yes"),
                       help="Save rendered prompts to debug files.")
    parser.add_argument("--debug-dir", type=str,
                       default=os.getenv("MCP_DEBUG_DIR", ".debug/ai-mcp-server"),
                       help="Directory for debug prompt files.")
    args: argparse.Namespace = parser.parse_args()
    
    logger.debug(f"Step 2: Args parsed - debug: {args.debug}, transport: {args.transport}")
    
    logger.debug("Step 3: Setting up logging...")
    
    # Set up logging with debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    logger.debug("Step 4: Logging configured")
    
    logger.debug("Step 5: Creating settings...")

    settings: ServerSettings = ServerSettings(
        transport=args.transport,
        host=args.host,
        port=args.port,
        server_url=f"http{'s' if args.transport == 'https' else ''}://{args.host}:{args.port}",
        ssl_cert_path=args.ssl_cert,
        ssl_key_path=args.ssl_key,
        server_name=args.server_name,
        description=args.description,
        config_path=args.config,
        prompt_dir=args.prompt_dir,
        save_debug_prompts=args.save_debug_prompts,
        debug_dir=args.debug_dir,
    )
    
    logger.debug(f"Step 6: Settings created - {settings}")
    
    # Handle different transport modes using MCPServer class
    try:
        # Create MCP server instance
        mcp_server: MCPServer = MCPServer(settings)
        
        if settings.transport == "stdio":
            # Call the stdio server directly
            mcp_server.run_stdio_server()
                
        elif settings.transport in ["http", "https"]:
            result: int = mcp_server.run_http_server_with_mcp()
            return result
        else:
            logger.error(f"Unknown transport mode '{settings.transport}'")
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
