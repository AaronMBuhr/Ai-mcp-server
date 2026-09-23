"""A response cut off at the output limit, or refused, is an error, not success.

Before this, no provider recorded why generation stopped, so a response that hit
max_tokens came back as status "success" with incomplete text.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_providers import (
    FINISH_MAX_TOKENS,
    FINISH_REFUSAL,
    FINISH_STOP,
    ChatGPTProvider,
    LLMOptions,
    LLMResult,
    normalize_finish_reason,
)
from tests.test_hardened_query_ai import EXAMPLE_VALUES, _get_server_and_tools


@pytest.mark.parametrize("raw, expected", [
    ("stop", FINISH_STOP), ("length", FINISH_MAX_TOKENS),
    ("content_filter", FINISH_REFUSAL),
    ("end_turn", FINISH_STOP), ("max_tokens", FINISH_MAX_TOKENS),
    ("model_context_window_exceeded", FINISH_MAX_TOKENS), ("refusal", FINISH_REFUSAL),
    (SimpleNamespace(name="MAX_TOKENS"), FINISH_MAX_TOKENS),
    (SimpleNamespace(name="SAFETY"), FINISH_REFUSAL),
    ("FinishReason.STOP", FINISH_STOP),
    ("tool_calls", "tool_calls"), (None, None),
])
def test_finish_reasons_are_normalized(raw, expected):
    assert normalize_finish_reason(raw) == expected


def _chunk(content=None, finish=None, usage=None, choices=True):
    choice = SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish)
    return SimpleNamespace(choices=[choice] if choices else [], usage=usage)


def test_chatgpt_stream_records_finish_reason_and_usage():
    provider = ChatGPTProvider.__new__(ChatGPTProvider)
    provider.config = {}
    provider._openai = MagicMock()
    provider._client = MagicMock()
    usage = SimpleNamespace(model_dump=lambda: {"completion_tokens": 7})
    provider._client.chat.completions.create.return_value = iter([
        _chunk("par"), _chunk("tial", finish="length"),
        _chunk(choices=False, usage=usage),
    ])
    result = provider.generate("s", "u", LLMOptions(model="gpt-x"))
    assert result.text == "partial"
    assert result.finish_reason == FINISH_MAX_TOKENS
    assert result.usage == {"completion_tokens": 7}
    kwargs = provider._client.chat.completions.create.call_args.kwargs
    assert kwargs["stream_options"] == {"include_usage": True}


class TestEnvelope:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.server, self.tools = _get_server_and_tools()

    def _call(self, result):
        self.server._router.query = MagicMock(return_value=result)
        return self.tools["query_ai"](
            provider="gemini", intelligence="middle",
            system_prompt_template="example/review-system",
            user_prompt_template="example/review-user",
            replacement_values=dict(EXAMPLE_VALUES),
        )

    def test_truncated_output_is_a_non_retryable_error(self):
        r = self._call(LLMResult(
            text='{"selections": [', model="m",
            finish_reason=FINISH_MAX_TOKENS, usage={"output_tokens": 32000},
        ))
        assert r["status"] == "error"
        assert r["failure_type"] == "output_truncated"
        assert r["retryable"] is False
        assert r["provider_called"] is True
        assert r["partial_response_characters"] == len('{"selections": [')
        assert r["usage"] == {"output_tokens": 32000}
        assert "response_text" not in r

    def test_refusal_is_an_error_even_when_empty(self):
        r = self._call(LLMResult(text="", model="m", finish_reason=FINISH_REFUSAL))
        assert r["failure_type"] == "refused"
        assert r["retryable"] is False

    def test_a_complete_response_reports_why_it_stopped(self):
        r = self._call(LLMResult(text="done", model="m", finish_reason=FINISH_STOP))
        assert r["status"] == "success"
        assert r["finish_reason"] == FINISH_STOP
