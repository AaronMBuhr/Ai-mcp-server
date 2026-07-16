"""Tests for prompt contracts and placeholder validation (hardening P1/P2)."""

from pathlib import Path

import pytest

from prompt_contracts import (
    ContractError,
    FieldSpec,
    PromptContract,
    check_contract_against_templates,
    load_contracts,
    validate_replacements,
)


def _example_contract() -> PromptContract:
    return PromptContract(
        name="example-review",
        system_template="example/review-system",
        user_template="example/review-user",
        required_replacements={
            "product": FieldSpec(kind="scalar", allow_empty=True),
            "audience": FieldSpec(kind="scalar", allow_empty=True),
            "specification": FieldSpec(kind="document", min_characters=300),
            "reference_doc": FieldSpec(kind="document", min_characters=1000),
        },
    )


VALID_VALUES = {
    "product": "Acme Ingest",
    "audience": "Senior Backend Engineer",
    "specification": "The service must accept batched events over HTTP. " * 20,
    "reference_doc": "# Reference\n\nThe platform guarantees at-least-once delivery. " * 40,
}


class TestContractLoading:

    def test_shipped_contracts_load(self):
        """The example prompt set is what ships; a real prompt set lives in the
        caller's own --prompt-dir."""
        contracts = load_contracts(Path("prompts").resolve())
        names = sorted(c.name for c in contracts.values())
        assert names == ["example-compose", "example-review"]

    def test_contracts_match_real_templates(self):
        """Every shipped contract must agree with its actual templates."""
        prompt_dir = Path("prompts").resolve()
        contracts = load_contracts(prompt_dir)
        from prompt_engine import template_placeholders
        for contract in contracts.values():
            placeholders = set()
            for template in (contract.system_template, contract.user_template):
                text = (prompt_dir / f"{template}.md").read_text(encoding="utf-8")
                placeholders |= template_placeholders(text)
            assert check_contract_against_templates(contract, placeholders) is None, (
                f"Contract '{contract.name}' disagrees with its templates"
            )

    def test_placeholder_free_validation_behavior(self):
        """example-compose is placeholder-free: no replacement_values expected,
        and any stray key is rejected (allow_extra_keys is false)."""
        contracts = load_contracts(Path("prompts").resolve())
        compose = next(c for c in contracts.values() if c.name == "example-compose")
        assert compose.required_replacements == {}
        assert compose.allow_extra_keys is False
        # Empty replacement_values is valid (task comes from additional_user_prompt).
        assert validate_replacements(compose, {}) is None
        # A stray key is an unexpected_replacement (fail closed on caller mistake).
        violation = validate_replacements(compose, {"anything": "x"})
        assert violation is not None
        assert violation["failure_type"] == "unexpected_replacement"

    def test_malformed_contract_raises(self, tmp_path):
        (tmp_path / "bad.contract.yaml").write_text("name: x\n", encoding="utf-8")
        with pytest.raises(ContractError):
            load_contracts(tmp_path)

    def test_duplicate_pair_raises(self, tmp_path):
        body = (
            "name: {n}\nsystem_template: a/sys\nuser_template: a/usr\n"
            "required_replacements: {{}}\n"
        )
        (tmp_path / "one.contract.yaml").write_text(body.format(n="one"), encoding="utf-8")
        (tmp_path / "two.contract.yaml").write_text(body.format(n="two"), encoding="utf-8")
        with pytest.raises(ContractError):
            load_contracts(tmp_path)


class TestTemplateAgreement:

    def test_contract_template_mismatch_detected(self):
        contract = _example_contract()
        err = check_contract_against_templates(
            contract, {"specification", "reference_doc", "product", "audience", "surprise"},
        )
        assert err is not None
        assert err["failure_type"] == "prompt_contract_mismatch"
        assert "surprise" in err["detail"]

        err = check_contract_against_templates(
            contract, {"specification", "product", "audience"},
        )
        assert err is not None
        assert "reference_doc" in err["detail"]


class TestReplacementValidation:

    def test_valid_values_pass(self):
        assert validate_replacements(_example_contract(), VALID_VALUES) is None

    def test_missing_document_rejected(self):
        values = {k: v for k, v in VALID_VALUES.items() if k != "reference_doc"}
        err = validate_replacements(_example_contract(), values)
        assert err is not None
        assert err["failure_type"] == "missing_replacement"
        assert "reference_doc" in err["detail"]

    def test_near_miss_key_does_not_satisfy_required_key(self):
        """A plausible synonym is still a missing_replacement, not a match."""
        values = {k: v for k, v in VALID_VALUES.items() if k != "reference_doc"}
        values["reference_document"] = VALID_VALUES["reference_doc"]
        err = validate_replacements(_example_contract(), values)
        assert err is not None
        assert err["failure_type"] == "missing_replacement"
        assert "reference_doc" in err["detail"]

    def test_unexpected_key_rejected(self):
        values = dict(VALID_VALUES)
        values["extra_field"] = "boo"
        err = validate_replacements(_example_contract(), values)
        assert err is not None
        assert err["failure_type"] == "unexpected_replacement"
        assert "extra_field" in err["detail"]

    def test_path_like_document_rejected(self):
        values = dict(VALID_VALUES)
        values["reference_doc"] = r"READ_FROM_FILE:C:\fake.md"
        err = validate_replacements(_example_contract(), values)
        assert err is not None
        assert err["failure_type"] == "invalid_content_transport"
        assert err["field"] == "reference_doc"

    def test_undersized_document_rejected(self):
        values = dict(VALID_VALUES)
        values["specification"] = "short"
        err = validate_replacements(_example_contract(), values)
        assert err is not None
        assert err["failure_type"] == "invalid_content_transport"
