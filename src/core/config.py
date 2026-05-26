"""
Centralized configuration for the Nexus AI multi-agent system.
All tunable parameters live here — single source of truth.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# ─── Model Registry (Multi-Model Strategy) ────────────────
MODELS = {
    "fast": "llama-3.1-8b-instant",
    "powerful": "llama-3.3-70b-versatile",
}

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
