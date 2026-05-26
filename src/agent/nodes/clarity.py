from langchain_core.messages import HumanMessage, AIMessage
from src.agent.state import AgentState
from src.agent.core.llm import get_llm, invoke_llm_with_retry, extract_json
from src.agent.core.memory import build_history_string

def clarity_node(state: AgentState) -> dict:
    """Determines if the user's query is specific enough for research."""
    # Extract current conversation state
    messages = state.get("messages", [])
    summary = state.get("conversation_summary", "")
    recent_message = messages[-1].content if messages else ""

    # Build history to resolve pronouns or ambiguous terms in the prompt
    history = build_history_string(messages[:-1], summary) if len(messages) > 1 else ""

    # Principle-based prompt preventing overfitting to specific test cases
    prompt = f"""You are the Clarity Agent. Your job is to decide whether the user's query is specific enough to begin research.

Conversation History:
{history}

Latest Query: {recent_message}

Decision Framework:
- A query is "clear" if a reasonable person could identify exactly what entity or topic to research — either from the query itself or from conversation history.
- A query "needs_clarification" only when the intended subject is genuinely ambiguous and cannot be resolved from any available context.
- Users naturally switch topics mid-conversation. A new topic is not ambiguity.
- Users naturally use pronouns to refer to previously discussed entities. Resolve them using history.

Bias toward "clear". Only ask for clarification when you truly cannot determine the research target.

Return JSON: {{"clarity_status": "clear" | "needs_clarification", "clarification_question": "your question here or empty string"}}"""

    # Execute the fast model and safely parse JSON
    llm = get_llm("fast")
    content = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)], "fast")
    result = extract_json(content)

    clarity = result.get("clarity_status", "needs_clarification")
    question = result.get("clarification_question", "Could you please specify which company or entity you are asking about?")

    # Return updated state, explicitly resetting research cache for new queries
    return {
        "clarity_status": clarity,
        "clarification_question": question,
        "active_agent": "clarity",
        "attempts": 0,
        "research_data": "",
    }

def ask_clarification(state: AgentState) -> dict:
    """Handles ambiguous queries with a context-aware clarification question."""
    question = state.get(
        "clarification_question",
        "Could you please specify which company or entity you are asking about?"
    )
    return {
        "messages": [AIMessage(content=question)],
        "active_agent": "clarification",
    }
