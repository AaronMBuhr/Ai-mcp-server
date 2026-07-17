"""Tests for the docx_document_program_v1 response contract (hardening P6)."""

import json

import pytest

from response_contracts import ResponseContractError, parse_docx_document_program_v1

VALID_PROGRAM = (
    'import fs from "node:fs";\n'
    'import { Document, Packer, Paragraph } from "docx";\n'
    "const doc = new Document({ sections: [] });\n"
    "const buffer = await Packer.toBuffer(doc);\n"
    "fs.writeFileSync(process.env.OUTPUT_DOCX_PATH, buffer);\n"
)

VALID_RESPONSE = json.dumps({
    "renderer_type": "node",
    "source_filename": "render-document.mjs",
    "source_code": VALID_PROGRAM,
})


class TestDocxDocumentProgramV1:

    def test_valid_response_parses_and_returns_exact_program(self):
        parsed = parse_docx_document_program_v1(VALID_RESPONSE)
        assert parsed["renderer_type"] == "node"
        assert parsed["source_filename"] == "render-document.mjs"
        assert parsed["source_code"] == VALID_PROGRAM
        assert len(parsed["source_code_sha256"]) == 64

    def test_single_fence_wrapper_tolerated(self):
        fenced = f"```json\n{VALID_RESPONSE}\n```"
        parsed = parse_docx_document_program_v1(fenced)
        assert parsed["source_code"] == VALID_PROGRAM

    def test_source_plus_trailing_json_rejected(self):
        """The exact failure from the bad run: JS followed by structured tail."""
        with pytest.raises(ResponseContractError, match="did not start with a JSON object"):
            parse_docx_document_program_v1(VALID_PROGRAM + "\n" + VALID_RESPONSE)

    def test_trailing_content_after_object_rejected(self):
        with pytest.raises(ResponseContractError, match="trailing content"):
            parse_docx_document_program_v1(VALID_RESPONSE + "\nAll done!")

    def test_clean_parse_reports_no_envelope_warnings(self):
        parsed = parse_docx_document_program_v1(VALID_RESPONSE)
        assert parsed["envelope_warnings"] == []

    @pytest.mark.parametrize("junk", ['"}}', '"}', "}}", "}", '"', "]", "}\n}", "  }} "])
    def test_stray_closing_punctuation_tolerated(self, junk):
        """The observed 2026-07-17 defect: a complete envelope, then stray closers.

        claude-haiku-4-5 emitted these on three consecutive attempts with the
        anti-trailing-content instructions already in the prompt. raw_decode has
        already proved the object complete, so these characters cannot belong to it.
        """
        parsed = parse_docx_document_program_v1(VALID_RESPONSE + junk)
        assert parsed["source_code"] == VALID_PROGRAM
        assert len(parsed["envelope_warnings"]) == 1
        assert "Discarded" in parsed["envelope_warnings"][0]

    def test_tolerated_junk_does_not_alter_source_code_hash(self):
        clean = parse_docx_document_program_v1(VALID_RESPONSE)
        repaired = parse_docx_document_program_v1(VALID_RESPONSE + '"}}')
        assert repaired["source_code_sha256"] == clean["source_code_sha256"]

    @pytest.mark.parametrize("tail", [
        '"}}}}}}}}}}',          # too long to be a stray tic
        "}x",                   # contains a letter
        "} 1",                  # contains a digit
        "}{",                   # an opening brace: a second value may follow
        '} {"renderer_type": "node"}',   # an actual second envelope
        "}\nAll done!",         # prose
        '}\nconst x = 1;',      # more source code
    ])
    def test_substantive_trailing_content_still_rejected(self, tail):
        """Tolerance must stay narrow.

        Substantive trailing content can mean the program was split or truncated;
        accepting it would yield a silently corrupt .docx. That is the failure this
        contract was written for, and it must keep failing.
        """
        with pytest.raises(ResponseContractError, match="trailing content"):
            parse_docx_document_program_v1(VALID_RESPONSE + tail)

    def test_fence_and_stray_closer_together_tolerated(self):
        fenced = f"```json\n{VALID_RESPONSE}\n```"
        parsed = parse_docx_document_program_v1(fenced.replace("\n```", '"}\n```'))
        assert parsed["source_code"] == VALID_PROGRAM

    def test_fenced_prose_outside_object_rejected(self):
        text = f"Here is your docx document program:\n```json\n{VALID_RESPONSE}\n```"
        with pytest.raises(ResponseContractError):
            parse_docx_document_program_v1(text)

    def test_empty_source_code_rejected(self):
        response = json.dumps({
            "renderer_type": "node",
            "source_filename": "render-document.mjs",
            "source_code": "",
        })
        with pytest.raises(ResponseContractError, match="empty source_code"):
            parse_docx_document_program_v1(response)

    def test_missing_fields_rejected(self):
        response = json.dumps({"renderer_type": "node"})
        with pytest.raises(ResponseContractError, match="missing required fields"):
            parse_docx_document_program_v1(response)

    def test_extra_fields_rejected_in_strict_mode(self):
        response = json.dumps({
            "renderer_type": "node",
            "source_filename": "render-document.mjs",
            "source_code": VALID_PROGRAM,
            "notes": ["extra"],
        })
        with pytest.raises(ResponseContractError, match="unexpected top-level fields"):
            parse_docx_document_program_v1(response)
        # tolerated when strict mode is off:
        parsed = parse_docx_document_program_v1(response, strict=False)
        assert parsed["source_code"] == VALID_PROGRAM

    def test_empty_response_rejected(self):
        with pytest.raises(ResponseContractError, match="empty"):
            parse_docx_document_program_v1("   ")

    def test_non_object_json_rejected(self):
        with pytest.raises(ResponseContractError):
            parse_docx_document_program_v1('["not", "an", "object"]')
