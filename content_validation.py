"""Content-transport validation for document replacement values.

The failed workflow run showed that ``query_ai`` accepted arbitrary strings
for document placeholders, letting ``READ_FROM_FILE:<path>`` pass through as
though it were resume content.  This module detects values that *function as
references* (paths, read-file directives, deferral phrases) rather than
literal document text.

Design rule from AI_MCP_SERVER_HARDENING.md Priority 3: do not reject
ordinary document text merely because it contains a path somewhere inside it.
Only the whole value, or its leading control token, counts as a reference.
"""

import hashlib
import re
from typing import Any, Dict, Optional

WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
UNC_PATH = re.compile(r"^\\\\[^\\]+\\[^\\]+")
UNIX_ABSOLUTE_PATH = re.compile(r"^/")
RELATIVE_PATH_PREFIX = re.compile(r"^\.\.?[\\/]")
REFERENCE_PREFIX = re.compile(
    r"^(READ_FROM_FILE|FILE_FROM_PATH|LOAD_FILE|file)\s*:",
    re.IGNORECASE,
)

# A value that is nothing but a single filename token (e.g. "resume.md").
_FILENAME_ONLY = re.compile(
    r"^[\w.\- ]+\.(md|txt|docx|doc|pdf|json|yaml|yml|csv|html|rtf)$",
    re.IGNORECASE,
)

# Whole-value deferral phrases that mean "content omitted".
_DEFERRAL_PHRASES = frozenset({
    "see file",
    "see attached",
    "see attachment",
    "same as previous",
    "same as above",
    "same as before",
    "content omitted",
    "omitted",
    "as before",
    "unchanged",
})


def looks_like_reference(value: str) -> Optional[str]:
    """Return a human-readable reason if *value* functions as a reference
    to content rather than the content itself, else ``None``.

    Checks apply to the whole (stripped) value or its leading control token
    only; paths appearing inside a longer document are fine.
    """
    stripped = value.strip().strip('"').strip("'")

    if REFERENCE_PREFIX.match(stripped):
        return "read-from-file style directive prefix"

    # Path checks: only meaningful when the value is short and pathlike as a
    # whole.  Real documents are long and contain whitespace/newlines; a bare
    # path is a single token.
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    whole_is_single_token = stripped == first_line and " " not in stripped

    if whole_is_single_token:
        if WINDOWS_ABSOLUTE_PATH.match(stripped):
            return "Windows absolute path"
        if UNC_PATH.match(stripped):
            return "UNC path"
        if UNIX_ABSOLUTE_PATH.match(stripped):
            return "Unix absolute path"
        if RELATIVE_PATH_PREFIX.match(stripped):
            return "relative path"
        if _FILENAME_ONLY.match(stripped):
            return "bare filename"
    else:
        # Multi-token single-line values that are still clearly paths,
        # e.g. 'C:\My Documents\resume.md' (space inside a Windows path).
        if stripped == first_line and (
            WINDOWS_ABSOLUTE_PATH.match(stripped) or UNC_PATH.match(stripped)
        ) and len(stripped) < 260 and "\n" not in stripped:
            return "Windows path"

    if stripped.lower() in _DEFERRAL_PHRASES:
        return "content-deferral phrase"

    return None


def validate_document_value(
    field: str,
    value: Any,
    *,
    min_characters: int = 0,
    allow_empty: bool = False,
) -> Optional[Dict[str, Any]]:
    """Validate one ``kind: document`` replacement value.

    Returns ``None`` when valid, otherwise a structured error dict with
    ``failure_type`` ``invalid_content_transport`` ready to merge into the
    result envelope.
    """
    def _err(detail: str) -> Dict[str, Any]:
        return {
            "failure_type": "invalid_content_transport",
            "field": field,
            "detail": detail,
        }

    if value is None:
        if allow_empty:
            return None
        return _err("Expected literal document text but received null.")

    if not isinstance(value, str):
        return _err(
            f"Expected literal document text but received {type(value).__name__}."
        )

    stripped = value.strip()
    if not stripped:
        if allow_empty:
            return None
        return _err("Expected literal document text but received an empty value.")

    reason = looks_like_reference(value)
    if reason is not None:
        return _err(
            f"Expected literal document text but received a path-like reference ({reason})."
        )

    if min_characters and len(stripped) < min_characters:
        return _err(
            f"Document value is {len(stripped)} characters; "
            f"minimum is {min_characters}. Send the complete literal text."
        )

    return None


def validate_scalar_value(
    field: str,
    value: Any,
    *,
    allow_empty: bool = False,
) -> Optional[Dict[str, Any]]:
    """Validate one ``kind: scalar`` replacement value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_empty:
            return None
        return {
            "failure_type": "invalid_content_transport",
            "field": field,
            "detail": "Required scalar value is empty.",
        }
    return None


def replacement_metadata(value: Any) -> Dict[str, Any]:
    """Return {characters, bytes, sha256} for a replacement value."""
    if isinstance(value, str):
        text = value
    else:
        import json
        text = json.dumps(value, ensure_ascii=False)
    encoded = text.encode("utf-8")
    return {
        "characters": len(text),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
