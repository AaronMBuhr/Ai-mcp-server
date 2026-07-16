"""Tests for prompt path resolution, template rendering, and utility functions."""

import pytest
from pathlib import Path

from prompt_engine import (
    resolve_prompt_path,
    render_template,
    combine_user_prompt,
    sha256_text,
)


class TestResolvePromptPath:

    @pytest.fixture
    def prompt_dir(self, tmp_path):
        d = tmp_path / "prompts"
        d.mkdir()
        (d / "test-system.md").write_text("system prompt")
        (d / "test-user.md").write_text("hello {name}")
        nested = d / "resume"
        nested.mkdir()
        (nested / "draft-system.md").write_text("nested system")
        return d

    def test_loads_simple_template(self, prompt_dir):
        path = resolve_prompt_path(prompt_dir, "test-system")
        assert path.exists()
        assert path.name == "test-system.md"

    def test_loads_nested_template(self, prompt_dir):
        path = resolve_prompt_path(prompt_dir, "resume/draft-system")
        assert path.exists()
        assert path.name == "draft-system.md"

    def test_rejects_dotdot(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "../secret")

    def test_rejects_absolute_path(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name|outside prompt directory"):
            resolve_prompt_path(prompt_dir, "/etc/passwd")

    def test_rejects_windows_drive_path(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "C:/Windows/file")

    def test_rejects_backslash(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "foo\\bar")

    def test_rejects_invalid_chars(self, prompt_dir):
        with pytest.raises(ValueError, match="Unsafe prompt template name"):
            resolve_prompt_path(prompt_dir, "test system")

    def test_fails_on_missing_template(self, prompt_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            resolve_prompt_path(prompt_dir, "nonexistent")

    def test_loads_from_real_prompts_dir(self):
        prompt_dir = Path("prompts").resolve()
        path = resolve_prompt_path(prompt_dir, "test-system")
        assert path.exists()


class TestRenderTemplate:

    def test_replaces_single_placeholder(self):
        rendered, warnings = render_template("Hello {name}!", {"name": "Aaron"})
        assert rendered == "Hello Aaron!"
        assert warnings == []

    def test_replaces_multiple_occurrences(self):
        rendered, _ = render_template("{name} says hi to {name}", {"name": "Aaron"})
        assert rendered == "Aaron says hi to Aaron"

    def test_replaces_multiple_keys(self):
        template = "{greeting}, {name}!"
        rendered, _ = render_template(template, {"greeting": "Hello", "name": "World"})
        assert rendered == "Hello, World!"

    def test_renders_dict_as_json(self):
        template = "Data: {payload}"
        rendered, _ = render_template(template, {"payload": {"key": "value"}})
        assert '"key"' in rendered
        assert '"value"' in rendered

    def test_renders_list_as_json(self):
        template = "Items: {items}"
        rendered, _ = render_template(template, {"items": [1, 2, 3]})
        assert "1" in rendered
        assert "2" in rendered
        assert "3" in rendered

    def test_errors_on_missing_placeholder(self):
        with pytest.raises(ValueError, match="Missing replacement values"):
            render_template("Hello {name}!", {})

    def test_warns_on_unused_key(self):
        rendered, warnings = render_template("Hello!", {"unused": "value"})
        assert len(warnings) == 1
        assert "unused" in warnings[0]

    def test_unused_warning_names_its_template(self):
        """Warnings must identify which template did not use the key.

        A prompt pair is rendered against one shared replacement_values dict, so
        a placeholder-free system template reports every key as unused. Without
        the template name in the warning, that reads as though the documents
        never reached the model.
        """
        _, sys_warnings = render_template(
            "You are a technical document reviewer.",
            {"specification": "...", "reference_doc": "..."},
            template_label="the system template 'example/review-system'",
        )
        _, usr_warnings = render_template(
            "Spec:\n{specification}\n\nReference:\n{reference_doc}",
            {"specification": "...", "reference_doc": "..."},
            template_label="the user template 'example/review-user'",
        )

        # The user template consumed both keys: no warnings at all.
        assert usr_warnings == []
        # The system template has no placeholders, so both keys are unused there
        # -- but each warning must say so, and say which template.
        assert len(sys_warnings) == 2
        for w in sys_warnings:
            assert "the system template 'example/review-system'" in w
            assert "it has no placeholders" in w
            assert "user template" not in w

    def test_does_not_treat_json_braces_as_placeholders(self):
        template = 'Example: {"key": "value"} and {name}'
        rendered, warnings = render_template(template, {"name": "Aaron"})
        assert '{"key": "value"}' in rendered
        assert "Aaron" in rendered
        assert len(warnings) == 0

    def test_does_not_use_str_format_behavior(self):
        template = "Index {0} and name {name}"
        rendered, warnings = render_template(template, {"name": "Aaron"})
        assert "{0}" in rendered
        assert "Aaron" in rendered

    def test_rejects_invalid_replacement_key(self):
        with pytest.raises(ValueError, match="Invalid replacement key"):
            render_template("Hi", {"bad-key": "value"})

    def test_rejects_key_starting_with_digit(self):
        with pytest.raises(ValueError, match="Invalid replacement key"):
            render_template("Hi", {"0key": "value"})


class TestCombineUserPrompt:

    def test_appends_additional_with_separator(self):
        result = combine_user_prompt("Main prompt.", "Be brief.")
        assert "---" in result
        assert "Additional user instructions:" in result
        assert "Be brief." in result
        assert result.startswith("Main prompt.")

    def test_returns_plain_when_additional_empty(self):
        result = combine_user_prompt("Main prompt.", "")
        assert "---" not in result
        assert result == "Main prompt."

    def test_returns_plain_when_additional_whitespace(self):
        result = combine_user_prompt("Main prompt.", "   \n  ")
        assert "---" not in result

    def test_strips_leading_trailing_whitespace(self):
        result = combine_user_prompt("  Main prompt.  ", "")
        assert result == "Main prompt."


class TestSha256Text:

    def test_returns_64_char_hex(self):
        h = sha256_text("hello")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        assert sha256_text("test") == sha256_text("test")

    def test_different_inputs_different_hashes(self):
        assert sha256_text("a") != sha256_text("b")
