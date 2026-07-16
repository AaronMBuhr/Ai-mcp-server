"""Prompt contracts: machine-readable replacement schemas beside prompt pairs.

Each contract is a YAML file living next to its prompt pair, e.g.::

    prompts/job/analyze.contract.yaml

    name: job-analysis
    system_template: job/analyze-system
    user_template: job/analyze-user
    required_replacements:
      job_description:
        kind: document
        allow_empty: false
        min_characters: 300

The server loads all ``*.contract.yaml`` files under the prompt directory at
startup, indexes them by (system_template, user_template), and enforces them
before any provider call (AI_MCP_SERVER_HARDENING.md Priorities 1-2).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from content_validation import validate_document_value, validate_scalar_value

_VALID_KINDS = ("scalar", "document")


@dataclass
class FieldSpec:
    kind: str = "scalar"
    allow_empty: bool = False
    min_characters: int = 0


@dataclass
class PromptContract:
    name: str
    system_template: str
    user_template: str
    required_replacements: Dict[str, FieldSpec] = field(default_factory=dict)
    allow_extra_keys: bool = False
    source_path: Optional[Path] = None

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.system_template, self.user_template)


class ContractError(ValueError):
    """A contract file is malformed or self-inconsistent."""


def _parse_contract(path: Path) -> PromptContract:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractError(f"Contract '{path}' must be a YAML mapping")

    for key in ("name", "system_template", "user_template"):
        if not data.get(key) or not isinstance(data[key], str):
            raise ContractError(f"Contract '{path}' is missing string key '{key}'")

    raw_fields = data.get("required_replacements") or {}
    if not isinstance(raw_fields, dict):
        raise ContractError(f"Contract '{path}': required_replacements must be a mapping")

    specs: Dict[str, FieldSpec] = {}
    for fname, fconf in raw_fields.items():
        fconf = fconf or {}
        if not isinstance(fconf, dict):
            raise ContractError(f"Contract '{path}': field '{fname}' must be a mapping")
        kind = fconf.get("kind", "scalar")
        if kind not in _VALID_KINDS:
            raise ContractError(
                f"Contract '{path}': field '{fname}' has invalid kind '{kind}'"
            )
        specs[fname] = FieldSpec(
            kind=kind,
            allow_empty=bool(fconf.get("allow_empty", False)),
            min_characters=int(fconf.get("min_characters", 0)),
        )

    return PromptContract(
        name=data["name"],
        system_template=data["system_template"],
        user_template=data["user_template"],
        required_replacements=specs,
        allow_extra_keys=bool(data.get("allow_extra_keys", False)),
        source_path=path,
    )


def load_contracts(prompt_dir: Path) -> Dict[Tuple[str, str], PromptContract]:
    """Load every ``*.contract.yaml`` under *prompt_dir*, keyed by template pair.

    Raises ContractError on a malformed file or a duplicate pair so that
    misconfiguration fails closed at startup rather than silently at call time.
    """
    contracts: Dict[Tuple[str, str], PromptContract] = {}
    for path in sorted(prompt_dir.rglob("*.contract.yaml")):
        contract = _parse_contract(path)
        if contract.pair in contracts:
            raise ContractError(
                f"Duplicate contract for template pair {contract.pair}: "
                f"'{path}' and '{contracts[contract.pair].source_path}'"
            )
        contracts[contract.pair] = contract
    return contracts


def check_contract_against_templates(
    contract: PromptContract,
    template_placeholders: Set[str],
) -> Optional[Dict[str, Any]]:
    """Confirm the contract and the actual template placeholders agree.

    *template_placeholders* is the union of placeholders found in the system
    and user templates.  Any disagreement is a deployment bug, reported as
    ``prompt_contract_mismatch`` (fail closed).
    """
    declared = set(contract.required_replacements.keys())
    missing_from_contract = sorted(template_placeholders - declared)
    missing_from_templates = sorted(declared - template_placeholders)
    if missing_from_contract or missing_from_templates:
        parts = []
        if missing_from_contract:
            parts.append(
                "placeholders not declared in contract: "
                + ", ".join(missing_from_contract)
            )
        if missing_from_templates:
            parts.append(
                "contract fields not present in templates: "
                + ", ".join(missing_from_templates)
            )
        return {
            "failure_type": "prompt_contract_mismatch",
            "detail": f"Contract '{contract.name}' disagrees with templates: "
                      + "; ".join(parts),
        }
    return None


def validate_replacements(
    contract: PromptContract,
    replacement_values: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate a request's replacement values against a contract.

    Returns ``None`` when valid, else a structured error dict carrying one of
    the stable failure types: ``missing_replacement``, ``unexpected_replacement``,
    or ``invalid_content_transport``.
    """
    declared = contract.required_replacements

    missing = sorted(set(declared.keys()) - set(replacement_values.keys()))
    if missing:
        return {
            "failure_type": "missing_replacement",
            "detail": "Missing replacement values: " + ", ".join(missing),
            "fields": missing,
        }

    if not contract.allow_extra_keys:
        extra = sorted(set(replacement_values.keys()) - set(declared.keys()))
        if extra:
            return {
                "failure_type": "unexpected_replacement",
                "detail": "Unexpected replacement keys: " + ", ".join(extra),
                "fields": extra,
            }

    for fname, spec in declared.items():
        value = replacement_values.get(fname)
        if spec.kind == "document":
            err = validate_document_value(
                fname,
                value,
                min_characters=spec.min_characters,
                allow_empty=spec.allow_empty,
            )
        else:
            err = validate_scalar_value(fname, value, allow_empty=spec.allow_empty)
        if err is not None:
            return err

    return None


def document_fields(contract: PromptContract) -> List[str]:
    """Names of the contract's ``kind: document`` fields."""
    return [
        fname
        for fname, spec in contract.required_replacements.items()
        if spec.kind == "document"
    ]
