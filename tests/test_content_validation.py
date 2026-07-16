"""Tests for path-like / undersized document value rejection (hardening P3)."""

from content_validation import (
    looks_like_reference,
    replacement_metadata,
    validate_document_value,
)

VALID_RESUME = (
    "# Aaron Buhr\n\n## Professional Summary\n\n"
    "Senior backend engineer with production experience across Python, "
    "C++, and distributed log-ingestion systems.\n\n"
    "## Experience\n\n"
    "* Built async ingestion pipelines handling malformed input safely.\n"
    "* Reduced production incident rates through observability work.\n"
) * 10  # comfortably above min_characters thresholds


class TestReferenceDetection:

    def test_read_from_file_directive_rejected(self):
        err = validate_document_value("source_resume", r"READ_FROM_FILE:C:\resume.md")
        assert err is not None
        assert err["failure_type"] == "invalid_content_transport"
        assert err["field"] == "source_resume"
        assert "path-like reference" in err["detail"]

    def test_load_file_and_file_prefixes_rejected(self):
        for value in (r"FILE_FROM_PATH: C:\x.md", "LOAD_FILE:/home/u/x.md", "file: ./x.md"):
            assert validate_document_value("doc", value) is not None

    def test_bare_windows_path_rejected(self):
        assert validate_document_value("source_resume", r"C:\resume.md") is not None

    def test_windows_path_with_spaces_rejected(self):
        assert validate_document_value(
            "source_resume", r"C:\My Documents\resume.md"
        ) is not None

    def test_bare_unix_path_rejected(self):
        assert validate_document_value("source_resume", "/home/user/resume.md") is not None

    def test_relative_paths_rejected(self):
        assert validate_document_value("doc", "./resume.md") is not None
        assert validate_document_value("doc", "../inputs/resume.md") is not None

    def test_unc_path_rejected(self):
        assert validate_document_value("doc", r"\\server\share\resume.md") is not None

    def test_bare_filename_rejected(self):
        assert validate_document_value("doc", "resume.md") is not None

    def test_deferral_phrases_rejected(self):
        for phrase in ("see file", "same as previous", "content omitted", '"see file"'):
            assert validate_document_value("doc", phrase) is not None, phrase

    def test_normal_resume_accepted(self):
        assert validate_document_value(
            "source_resume", VALID_RESUME, min_characters=1000
        ) is None

    def test_path_inside_document_body_not_rejected(self):
        text = VALID_RESUME + "\n* Deployed configs to C:\\deploy\\prod and /etc/app.conf.\n"
        assert validate_document_value("source_resume", text, min_characters=1000) is None
        assert looks_like_reference(text) is None


class TestSizeChecks:

    def test_empty_document_rejected(self):
        err = validate_document_value("job_description", "")
        assert err is not None
        assert "empty" in err["detail"]

    def test_whitespace_only_rejected(self):
        assert validate_document_value("job_description", "   \n\t ") is not None

    def test_empty_allowed_when_contract_permits(self):
        assert validate_document_value("job_analysis", "", allow_empty=True) is None
        assert validate_document_value("job_analysis", None, allow_empty=True) is None

    def test_undersized_document_rejected(self):
        err = validate_document_value("job_description", "Too short.", min_characters=300)
        assert err is not None
        assert "minimum is 300" in err["detail"]

    def test_non_string_rejected(self):
        assert validate_document_value("doc", {"text": "hi"}) is not None


class TestReplacementMetadata:

    def test_metadata_shape(self):
        meta = replacement_metadata("hello")
        assert meta["characters"] == 5
        assert meta["bytes"] == 5
        assert len(meta["sha256"]) == 64

    def test_identical_content_identical_hash(self):
        assert replacement_metadata(VALID_RESUME) == replacement_metadata(VALID_RESUME)
