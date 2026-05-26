"""
Centralized configuration for the Nexus AI multi-agent system.
All tunable parameters live here — single source of truth.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys & Providers ──────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq")  # "groq" or "ollama"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Model Registry (Multi-Model Strategy) ────────────────
MODELS = {
    "fast": "llama-3.1-8b-instant" if LLM_PROVIDER == "groq" else os.environ.get("OLLAMA_FAST_MODEL", "llama3"),
    "powerful": "llama-3.3-70b-versatile" if LLM_PROVIDER == "groq" else os.environ.get("OLLAMA_POWERFUL_MODEL", "llama3"),
}

# ─── Semantic Cache Configurations ────────────────────────
SEMANTIC_CACHE_ENABLED = os.environ.get("SEMANTIC_CACHE_ENABLED", "True").lower() == "true"
SEMANTIC_CACHE_THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", 0.85))

# ─── Agent Parameters ─────────────────────────────────────
CONFIDENCE_THRESHOLD = 6
MAX_RESEARCH_ATTEMPTS = 3
MAX_RESEARCH_CHARS = 3000
TAVILY_MAX_RESULTS = 3
TAVILY_SEARCH_DEPTH = "advanced"

# ─── Memory Parameters ────────────────────────────────────
SLIDING_WINDOW_SIZE = 4
SUMMARY_MAX_CHARS = 500

# ─── Retry / Resilience ──────────────────────────────────
LLM_RETRY_ATTEMPTS = 2
LLM_RETRY_DELAY_SEC = 2
FALLBACK_MODEL = "llama-3.1-8b-instant"
