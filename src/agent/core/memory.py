"""
Sliding window memory management for conversations.
"""
from langchain_core.messages import HumanMessage
from src.core.config import SLIDING_WINDOW_SIZE, SUMMARY_MAX_CHARS
from src.agent.core.llm import get_llm, invoke_llm_with_retry

def build_memory_context(messages: list, existing_summary: str = "") -> tuple[list, str]:
    """Implements sliding window + summary memory management to prevent token overflow."""
    # If messages fit in window, no summary needed yet
    if len(messages) <= SLIDING_WINDOW_SIZE:
        return list(messages), existing_summary

    # Split into recent context (window) and older messages to be summarized
    window = list(messages[-SLIDING_WINDOW_SIZE:])
    older = messages[:-SLIDING_WINDOW_SIZE]

    # Generate a new summary if we have older messages and no existing summary
    if older and not existing_summary:
        history_lines = []
        for msg in older:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {msg.content[:200]}")
        history_text = "\n".join(history_lines)

        summary_prompt = f"""Summarize this conversation history in 2-3 concise sentences. 
Focus on: what entities/companies were discussed, what was asked, and key findings.

{history_text}

Summary:"""
        try:
            llm = get_llm("fast")
            summary = invoke_llm_with_retry(llm, [HumanMessage(content=summary_prompt)], "fast")
            summary = summary.strip()[:SUMMARY_MAX_CHARS]
        except Exception:
            # Fallback if summarization fails
            summary = existing_summary or "Previous conversation context available."
    else:
        summary = existing_summary

    return window, summary

def build_history_string(messages: list, summary: str = "") -> str:
    """Builds a formatted text representation of conversation history for LLM prompts."""
    parts = []
    if summary:
        parts.append(f"[Previous Context Summary]: {summary}")
        
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Truncate individual messages to prevent prompt injection or bloat
        parts.append(f"{role}: {msg.content[:300]}")
        
    return "\n".join(parts)
