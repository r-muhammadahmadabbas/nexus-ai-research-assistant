from langchain_core.messages import HumanMessage, AIMessage
from src.agent.state import AgentState
from src.agent.core.llm import get_llm, invoke_llm_with_retry
from src.agent.core.memory import build_memory_context
from src.agent.core.tools import truncate_text
from src.core.config import MAX_RESEARCH_CHARS

def synthesis_node(state: AgentState) -> dict:
    """Consumes research findings and generates a coherent summary."""
    messages = state.get("messages", [])
    summary = state.get("conversation_summary", "")
    
    # Strictly enforce token limits on raw research data before passing to LLM
    research_data = truncate_text(state.get("research_data", ""), MAX_RESEARCH_CHARS)

    # Rebuild conversation memory using the sliding window approach
    recent_msgs, updated_summary = build_memory_context(messages, summary)

    # Prompt enforces topic focus to prevent context bleed from previous queries
    prompt = f"""You are the Synthesis Agent. Synthesize a coherent, well-structured response that directly answers the user's LATEST query.

Rules:
- Focus ONLY on the topic the user is currently asking about. Do NOT blend in unrelated topics from earlier conversation.
- Use clear Markdown: headings (##), bullet points, **bold** for key terms
- Be informative yet concise — aim for depth over breadth
- Cite specific facts, numbers, or dates from the research data when available
- If data is limited or missing, acknowledge gaps honestly rather than guessing

Research Data:
{research_data}"""

    # Combine recent conversation history with the final synthesis prompt
    full_prompt = recent_msgs + [HumanMessage(content=prompt)]

    # Use the powerful model (e.g., Llama 70B) for high-quality text generation
    llm = get_llm("powerful")
    content = invoke_llm_with_retry(llm, full_prompt, "powerful")

    # Append response to messages and update the sliding window summary
    return {
        "messages": [AIMessage(content=content)],
        "conversation_summary": updated_summary,
        "active_agent": "synthesis",
    }
