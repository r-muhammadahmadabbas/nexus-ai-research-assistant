"""
Unit tests for configuration module.
Validates that all required config values are present and correctly typed.
"""
import pytest
from src.core.config import (
    GROQ_API_KEY, TAVILY_API_KEY, MODELS,
    CONFIDENCE_THRESHOLD, MAX_RESEARCH_ATTEMPTS,
    MAX_RESEARCH_CHARS, TAVILY_MAX_RESULTS,
    SLIDING_WINDOW_SIZE, LLM_RETRY_ATTEMPTS,
)


class TestConfigValues:
    """Tests for centralized configuration."""

    def test_api_keys_exist(self):
        """API keys must be set in the environment."""
        assert GROQ_API_KEY is not None, "GROQ_API_KEY is not set"
        assert TAVILY_API_KEY is not None, "TAVILY_API_KEY is not set"

    def test_model_registry_has_required_keys(self):
        """Model registry must define both fast and powerful models."""
        assert "fast" in MODELS
        assert "powerful" in MODELS
        assert isinstance(MODELS["fast"], str)
        assert isinstance(MODELS["powerful"], str)

    def test_thresholds_are_valid(self):
        """Agent thresholds must be within sensible ranges."""
        assert 0 < CONFIDENCE_THRESHOLD <= 10
        assert 1 <= MAX_RESEARCH_ATTEMPTS <= 10
        assert MAX_RESEARCH_CHARS > 0
        assert TAVILY_MAX_RESULTS > 0

    def test_memory_parameters(self):
        """Memory parameters must be positive integers."""
        assert SLIDING_WINDOW_SIZE > 0
        assert LLM_RETRY_ATTEMPTS > 0
