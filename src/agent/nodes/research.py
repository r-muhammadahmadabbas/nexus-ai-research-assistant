from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.agent.core.llm import get_llm, invoke_llm_with_retry, extract_json
from src.agent.core.memory import build_history_string
from src.agent.core.tools import search_tavily, truncate_text
from src.agent.core.cache import get_semantic_cache


def research_node(state: AgentState) -> dict:
    """Builds a context-aware search query and fetches data."""
    messages = state.get("messages", [])
    summary = state.get("conversation_summary", "")
    recent_message = messages[-1].content
    attempts = state.get("attempts", 0)

    # Reconstruct query context using history (resolves pronouns and topic changes)
    if len(messages) > 1 or summary:
        history = build_history_string(messages[:-1], summary)
        query_prompt = f"""Given this conversation and the latest user query, generate a single concise search query.

Rules:
- If the latest query names a NEW entity, search for THAT entity only (ignore previous topics).
- If the latest query uses pronouns ("their", "it"), resolve them using conversation history.
- Return ONLY the search query string. No explanation.

{history}

Latest Query: {recent_message}"""

        llm = get_llm("fast")
        search_query = invoke_llm_with_retry(llm, [HumanMessage(content=query_prompt)], "fast")
        search_query = search_query.strip().strip('"').strip("'")
    else:
        # No history available, use query verbatim
        search_query = recent_message

    # ─── Semantic Cache Layer ─────────────────────────────────
    cached_data = get_semantic_cache(search_query)
    if cached_data:
        past_research = state.get("research_data", "")
        new_research = f"{past_research}\n\n--- Cache Hit (Query: '{search_query}') ---\n{cached_data}"
        return {
            "research_data": new_research,
            "confidence_score": 9.0,  # High confidence for cached high-quality findings
            "search_query": f"[CACHED] {search_query}",
            "attempts": attempts + 1,
            "active_agent": "research",
        }

    # Execute web search using Tavily API
    try:
        direct_answer, detailed_context = search_tavily(search_query)
        context = f"Direct Answer: {direct_answer}\n\nDetailed Sources:\n{detailed_context}" if direct_answer else detailed_context
    except Exception as e:
        context = f"Search failed: {str(e)}"

    # Prevent token overflow by truncating data before LLM evaluation
    truncated_context = truncate_text(context, 1500)
    
    # Evaluate search quality using a calibrated confidence scale
    confidence_prompt = f"""Rate how well this research data answers the user's query. Use this calibrated scale:

- 9-10: Data directly and comprehensively answers the query with specific facts, numbers, or details.
- 7-8: Data answers the core question well but may miss some secondary aspects.
- 5-6: Data is partially relevant but has notable gaps or lacks specificity.
- 3-4: Data is loosely related but does not adequately answer the query.
- 0-2: Data is irrelevant, missing, or a search failure.

User Query: {recent_message}
Search Query Used: {search_query}
Research Data (excerpt): {truncated_context}

Return JSON: {{"confidence_score": <integer 0-10>, "reasoning": "one sentence"}}"""

    llm = get_llm("fast")
    conf_content = invoke_llm_with_retry(llm, [HumanMessage(content=confidence_prompt)], "fast")
    conf_result = extract_json(conf_content)
    confidence = float(conf_result.get("confidence_score", 5.0))

    # Append new findings to any existing research data (supports looping)
    past_research = state.get("research_data", "")
    new_research = f"{past_research}\n\n--- Attempt {attempts + 1} (Query: '{search_query}') ---\n{context}"

    return {
        "research_data": new_research,
        "confidence_score": confidence,
        "search_query": search_query,
        "attempts": attempts + 1,
        "active_agent": "research",
    }
