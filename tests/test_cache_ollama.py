"""
Tests for ChromaDB Semantic Cache and Ollama Local LLM strategy integration.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

from src.core.config import LLM_PROVIDER
from src.agent.core.llm import get_llm
from src.agent.core.cache import (
    get_semantic_cache, add_to_semantic_cache, _get_collection, CACHE_DIR
)


def test_ollama_llm_factory_instantiation():
    """Verify that get_llm instantiates ChatOllama when LLM_PROVIDER is set to 'ollama'."""
    with patch("src.agent.core.llm.LLM_PROVIDER", "ollama"), \
         patch("src.agent.core.llm.OLLAMA_BASE_URL", "http://localhost:11434"), \
         patch("src.agent.core.llm.MODELS", {"fast": "llama3", "powerful": "llama3"}):
         
        llm = get_llm("fast")
        assert isinstance(llm, ChatOllama)
        assert llm.model == "llama3"
        assert llm.base_url == "http://localhost:11434"


def test_groq_llm_factory_instantiation():
    """Verify that get_llm instantiates ChatGroq when LLM_PROVIDER is set to 'groq'."""
    with patch("src.agent.core.llm.LLM_PROVIDER", "groq"), \
         patch("src.agent.core.llm.GROQ_API_KEY", "gsk_test_key"):
         
        llm = get_llm("fast")
        assert isinstance(llm, ChatGroq)


def test_semantic_cache_lifecycle():
    """Verify write, exact hit, semantic hit, and miss workflows in semantic cache."""
    # Ensure local persistence behaves nicely for testing
    collection = _get_collection()
    if not collection:
        pytest.skip("ChromaDB persistent client not available or disabled")
        
    test_query = "What is the capital city of France?"
    test_data = "Paris is the capital and most populous city of France."
    
    # Clean up existing entry if any
    try:
        collection.delete(ids=[test_query])
    except Exception:
        pass
        
    # Test 1: Cache Miss
    miss_result = get_semantic_cache(test_query)
    assert miss_result is None
    
    # Test 2: Add to Cache
    add_to_semantic_cache(test_query, test_data)
    
    # Test 3: Exact Match Hit
    hit_result = get_semantic_cache(test_query)
    assert hit_result == test_data
    
    # Test 4: Semantically Similar Match Hit (resolving spelling/phrasing)
    similar_query = "What's the capital of France?"
    similar_hit_result = get_semantic_cache(similar_query)
    assert similar_hit_result == test_data
    
    # Test 5: Irrelevant Query (Cache Miss)
    irrelevant_query = "How do you build a multi-agent system?"
    irrelevant_result = get_semantic_cache(irrelevant_query)
    assert irrelevant_result is None
    
    # Clean up
    try:
        collection.delete(ids=[test_query])
    except Exception:
        pass
