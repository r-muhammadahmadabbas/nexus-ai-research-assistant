from langchain_core.messages import HumanMessage
from src.agent.state import AgentState
from src.agent.core.llm import get_llm, invoke_llm_with_retry, extract_json
from src.agent.core.tools import truncate_text

def validator_node(state: AgentState) -> dict:
    """Reviews research data quality and determines if it's sufficient."""
    messages = state.get("messages", [])
    recent_message = messages[-1].content
    
    # Truncate research data to prevent token limits during validation
    research_data = truncate_text(state.get("research_data", ""), 2000)

    # Prompt LLM to evaluate if the data objectively answers the query
    prompt = f"""You are the Validator Agent. Assess whether the research data adequately answers the user's query.

User Query: {recent_message}
Research Data: {research_data}

Evaluate: relevance, completeness, and whether key aspects of the query are addressed.
Return JSON: {{"validation_result": "sufficient" | "insufficient", "missing_aspects": "what's missing, if anything"}}"""

    llm = get_llm("fast")
    content = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)], "fast")
    result = extract_json(content)

    val_result = result.get("validation_result", "insufficient")
    
    # ─── Cache Successful Research Findings ───────────────────
    if val_result == "sufficient":
        search_query = state.get("search_query", "")
        research_data = state.get("research_data", "")
        if search_query and research_data and not search_query.startswith("[CACHED]"):
            try:
                from src.agent.core.cache import add_to_semantic_cache
                add_to_semantic_cache(search_query, research_data)
            except Exception:
                pass  # Resilient fallback: fail silently if cache write fails
                
    # Return assessment outcome to trigger loop or advance to synthesis
    return {
        "validation_result": val_result,
        "active_agent": "validator",
    }
