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
