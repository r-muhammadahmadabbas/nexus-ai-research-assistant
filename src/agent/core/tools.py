"""
External tools integration (e.g., Tavily Search).
"""
from tavily import TavilyClient
from src.core.config import (
    TAVILY_API_KEY, MAX_RESEARCH_CHARS, 
    TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def truncate_text(text: str, max_chars: int = MAX_RESEARCH_CHARS) -> str:
    """Truncates text to fit within LLM token budget."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated for brevity]"

def search_tavily(query: str) -> tuple[str, str]:
    """Performs an optimized Tavily search and formats the results."""
    # Execute search with predefined depth and max results from config
    result = tavily_client.search(
        query=query,
        search_depth=TAVILY_SEARCH_DEPTH,
        max_results=TAVILY_MAX_RESULTS,
        include_answer=True,
    )

    # Extract the AI-generated direct answer (if available)
    direct_answer = result.get("answer", "")
    
    # Format individual search results into a readable markdown string
    detailed_parts = []
    for r in result.get("results", []):
        title = r.get("title", "No Title")
        content = r.get("content", "")
        detailed_parts.append(f"**{title}**\n{content}")

    detailed_context = "\n\n".join(detailed_parts)
    return direct_answer, detailed_context
