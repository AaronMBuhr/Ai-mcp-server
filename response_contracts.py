"""Response contracts: server-side validation of structured model responses.

``docx_document_program_v1`` carries a **docx document program**: a complete,
self-contained program whose execution produces exactly one .docx. Content and
formatting are fused into it (any character may carry its own formatting, so a
template cannot express it), and it is written against the docx object model,
so it is specific to that output format. The JSON object is only an envelope
for shipping that program through a model intact.

The failed run produced a ``render-document.mjs`` containing JavaScript plus
trailing structured data, which the orchestrator then "repaired" by hand.
This contract prevents that at the server boundary
(AI_MCP_SERVER_HARDENING.md Priority 6): the model response must be exactly
one JSON object of the shape::

    {
      "renderer_type": "node",
      "source_filename": "render-document.mjs",
      "source_code": "<the complete docx document program>"
    }

``renderer_type`` names the runtime that executes the program (``node`` or
``python``); it is not part of the program's identity.

A single markdown code fence wrapping the object is tolerated (models add
one persistently despite instructions). A short run of structurally
meaningless close-punctuation after the object is likewise tolerated and
reported as a warning; any other prose or trailing content outside the
object is rejected.
"""

import hashlib
import json
import re
from typing import Any, Dict

DOCX_DOCUMENT_PROGRAM_V1 = "docx_document_program_v1"
KNOWN_CONTRACTS = (DOCX_DOCUMENT_PROGRAM_V1,)

_REQUIRED_FIELDS = ("renderer_type", "source_filename", "source_code")

_FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_-]*\s*\n")
_FENCE_CLOSE = re.compile(r"\n```\s*$")

# Trailing junk that cannot carry meaning: a few stray quotes/brackets the model
# emitted after already closing the envelope. Deliberately narrow. Anything that
# could be a second value, more source, or prose -- anything containing a letter,
# digit, or opening brace -- falls outside this and is still rejected, because
# substantive trailing content can mean the program was split or truncated and
# accepting it would yield a silently corrupt .docx. See _MAX_TRAILING_JUNK.
_TOLERABLE_TRAILING = re.compile(r"^[\s\"'}\]]+$")
_MAX_TRAILING_JUNK = 8


class ResponseContractError(ValueError):
    """The model response violated the requested response contract."""


def _strip_single_fence(text: str) -> str:
    """Strip exactly one wrapping markdown code fence, if present."""
    stripped = text.strip()
    open_match = _FENCE_OPEN.match(stripped)
    if open_match and _FENCE_CLOSE.search(stripped):
        inner = stripped[open_match.end():]
        inner = _FENCE_CLOSE.sub("", inner)
        return inner.strip()
    return stripped


def parse_docx_document_program_v1(raw_response: str, strict: bool = True) -> Dict[str, Any]:
    """Parse and validate a ``docx_document_program_v1`` model response.

    Returns the parsed payload::

        {
          "renderer_type": ...,
          "source_filename": ...,
          "source_code": ...,
          "source_code_sha256": ...,
          "envelope_warnings": [...],
        }

    Raises ResponseContractError with a precise reason on any violation.
    The caller must never concatenate the whole model response into a source
    file; only ``source_code`` from a successful parse may be used.
    """
    if not raw_response or not raw_response.strip():
        raise ResponseContractError("Model response was empty.")

    candidate = _strip_single_fence(raw_response)

    if not candidate.startswith("{"):
        raise ResponseContractError(
            "Docx document program response did not start with a JSON object "
            "(prose or code found outside the object)."
        )

    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise ResponseContractError(
            f"Docx document program response is not valid JSON: {exc}"
        ) from exc

    envelope_warnings = []
    trailing = candidate[end:].strip()
    if trailing:
        if len(trailing) <= _MAX_TRAILING_JUNK and _TOLERABLE_TRAILING.match(trailing):
            # raw_decode already proved the envelope complete and well-formed, so
            # these characters follow a finished object and cannot belong to it.
            # A truncated response fails raw_decode above instead of landing here.
            envelope_warnings.append(
                f"Discarded {len(trailing)} stray trailing character(s) "
                f"({trailing!r}) after the JSON object. The envelope parsed "
                f"cleanly without them and source_code is unmodified."
            )
        else:
            raise ResponseContractError(
                "Docx document program response contained trailing content after the JSON object."
            )

    if not isinstance(parsed, dict):
        raise ResponseContractError("Docx document program response JSON is not an object.")

    missing = [f for f in _REQUIRED_FIELDS if f not in parsed]
    if missing:
        raise ResponseContractError(
            "Docx document program response is missing required fields: " + ", ".join(missing)
        )

    if strict:
        extra = sorted(set(parsed.keys()) - set(_REQUIRED_FIELDS))
        if extra:
            raise ResponseContractError(
                "Docx document program response has unexpected top-level fields: " + ", ".join(extra)
            )

    source_code = parsed.get("source_code")
    if not isinstance(source_code, str) or not source_code.strip():
        raise ResponseContractError("Docx document program response has empty source_code.")

    for text_field in ("renderer_type", "source_filename"):
        if not isinstance(parsed.get(text_field), str) or not parsed[text_field].strip():
            raise ResponseContractError(
                f"Docx document program response field '{text_field}' must be a non-empty string."
            )

    return {
        "renderer_type": parsed["renderer_type"],
        "source_filename": parsed["source_filename"],
        "source_code": source_code,
        "source_code_sha256": hashlib.sha256(source_code.encode("utf-8")).hexdigest(),
        # Empty on a clean parse. The caller lifts this out of the payload so a
        # repaired envelope is visible in the run's warnings rather than silent.
        "envelope_warnings": envelope_warnings,
    }
