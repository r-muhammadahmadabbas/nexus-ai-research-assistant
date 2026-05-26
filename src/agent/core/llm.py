"""
LLM factory and resilient invocation logic.
"""
import time
import json
from langchain_groq import ChatGroq
from src.core.config import (
    GROQ_API_KEY, MODELS, LLM_RETRY_ATTEMPTS, 
    LLM_RETRY_DELAY_SEC, FALLBACK_MODEL
)
from src.core.logging import setup_logger

logger = setup_logger("nexus_ai.llm")

def get_llm(purpose: str = "fast", temperature: float = 0) -> ChatGroq:
    """Creates LLM instances based on task purpose (Strategy Pattern)."""
    model_name = MODELS.get(purpose, MODELS["fast"])
    return ChatGroq(model=model_name, temperature=temperature, api_key=GROQ_API_KEY)

def invoke_llm_with_retry(llm: ChatGroq, messages: list, purpose: str = "fast") -> str:
    """Invokes the LLM with automatic retry (exponential backoff) and model fallback."""
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "413" in error_str or "429" in error_str or "rate_limit" in error_str
            
            # Handle rate limits by sleeping and retrying
            if is_rate_limit and attempt < LLM_RETRY_ATTEMPTS - 1:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}), retrying in {LLM_RETRY_DELAY_SEC * (attempt + 1)}s...")
                time.sleep(LLM_RETRY_DELAY_SEC * (attempt + 1))
                continue
                
            # If the powerful model fails (e.g., context too large), fallback to the fast model
            if purpose == "powerful":
                logger.warning("Powerful model failed, falling back to fast model...")
                fallback_llm = get_llm("fast")
                try:
                    response = fallback_llm.invoke(messages)
                    return response.content
                except Exception:
                    pass
            raise e

def extract_json(text: str) -> dict:
    """Robustly extracts a JSON object from LLM output, handling markdown blocks and conversational padding."""
    cleaned = text.strip()
    
    # Attempt to parse from markdown blocks
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except (json.JSONDecodeError, ValueError):
                continue
                
    # Attempt direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
        
    # Attempt to extract substring between braces
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
            
    return {}
