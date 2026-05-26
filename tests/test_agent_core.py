"""
Unit tests for agent core utilities.
Tests LLM factory, JSON extraction, text truncation, and memory management.
"""
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.core.llm import get_llm, extract_json
from src.agent.core.tools import truncate_text
from src.agent.core.memory import build_memory_context, build_history_string


class TestLLMFactory:
    """Tests for the LLM factory function."""

    def test_get_llm_fast(self):
        """Fast model should be instantiated without errors."""
        llm = get_llm("fast")
        assert llm is not None
        assert "8b" in llm.model_name.lower()

    def test_get_llm_powerful(self):
        """Powerful model should be instantiated without errors."""
        llm = get_llm("powerful")
        assert llm is not None
        assert "70b" in llm.model_name.lower()

    def test_get_llm_unknown_falls_back_to_fast(self):
        """Unknown purpose should default to the fast model."""
        llm = get_llm("nonexistent_purpose")
        assert "8b" in llm.model_name.lower()


class TestJSONExtraction:
    """Tests for robust JSON extraction from LLM output."""

    def test_clean_json(self):
        result = extract_json('{"clarity_status": "clear"}')
        assert result == {"clarity_status": "clear"}

    def test_json_with_markdown_fences(self):
        result = extract_json('```json\n{"clarity_status": "clear"}\n```')
        assert result == {"clarity_status": "clear"}

    def test_json_with_surrounding_text(self):
        result = extract_json('Here is the result: {"score": 8} as requested.')
        assert result == {"score": 8}

    def test_invalid_json_returns_empty(self):
        result = extract_json("This is not JSON at all")
        assert result == {}

    def test_empty_string_returns_empty(self):
        result = extract_json("")
        assert result == {}


class TestTruncation:
    """Tests for text truncation utility."""

    def test_short_text_not_truncated(self):
        text = "Short text"
        assert truncate_text(text, 100) == text

    def test_long_text_truncated(self):
        text = "A" * 500
        result = truncate_text(text, 100)
        assert len(result) < 500
        assert "[...truncated for brevity]" in result

    def test_exact_limit_not_truncated(self):
        text = "A" * 100
        assert truncate_text(text, 100) == text


class TestMemoryManagement:
    """Tests for sliding window memory."""

    def test_small_history_returns_all(self):
        msgs = [HumanMessage(content="Hello"), AIMessage(content="Hi")]
        window, summary = build_memory_context(msgs, "")
        assert len(window) == 2
        assert summary == ""

    def test_history_string_with_summary(self):
        msgs = [HumanMessage(content="Tell me about Tesla")]
        result = build_history_string(msgs, "Previous discussion about SpaceX")
        assert "Previous discussion about SpaceX" in result
        assert "Tesla" in result

    def test_history_string_without_summary(self):
        msgs = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
        result = build_history_string(msgs)
        assert "User: Hello" in result
        assert "Assistant: Hi there" in result
